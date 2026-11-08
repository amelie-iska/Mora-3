import importlib.util,pathlib,json,multiprocessing,os,time,sys
import pytest
ROOT=pathlib.Path.cwd();sys.path.insert(0,str(ROOT/'scripts'))
def module(name):
    p=ROOT/('scripts/'+name+'.py');assert p.is_file(), 'reviewed actual-wakeup/hash/checkpoint guards not implemented'
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_delayed_wakeups_recheck_actual_shared_time_and_do_not_bunch_reserved_slots():
    m=module('data_sources_quickgo_resume_v2');g=m.SharedLimits(multiprocessing.get_context('fork'),3,{'annotation':(1000,0)})
    assert g.try_start(10)==(True,0)
    for _ in range(3):assert g.try_start(10)[0] is False
    # Three waiters all wake at20, long after any formerly booked slots expired.
    assert [g.try_start(20)[0] for _ in range(3)]==[True,False,False]
    assert g.snapshot()['request_starts']==2
    clock=[20.0];waits=[]
    def sleeper(delay):waits.append(delay);clock[0]=max(clock[0]+delay,25.0)
    admitted=g.wait_for_start(clock=lambda:clock[0],sleeper=sleeper)
    assert admitted==25 and waits and g.try_start(25)[0] is False
    clock[0]=25+1/3;assert g.try_start(clock[0])[0] is True

def review_files(tmp_path):
    m=module('data_sources_quickgo_parallel_v2');cp=tmp_path/'checkpoint.json';cp.write_text('{"owned_processes":[]}');r=tmp_path/'retirement.json';r.write_text(json.dumps(dict(all_owned_old_writers_retired=True,checkpoint_sha256=m.sha(cp))));a=tmp_path/'approval.json';a.write_text(json.dumps(dict(approved=True,action='QuickGO_checkpoint_parallel_handoff_v2',reviewed_code_sha256={p:m.sha(p) for p in m.REVIEWED_FILES})));return m,cp,r,a

def test_changed_imported_ontology_gate_and_nonexact_approval_map_are_rejected(tmp_path):
    m,cp,r,a=review_files(tmp_path);assert m.verify_review(a,cp,r)['owned_processes']==[];approval=json.loads(a.read_text());assert 'scripts/data_sources_go_ontology.py' in approval['reviewed_code_sha256'];approval['reviewed_code_sha256']['scripts/data_sources_go_ontology.py']='modified';a.write_text(json.dumps(approval))
    with pytest.raises(ValueError,match='changed'):m.verify_review(a,cp,r)
    approval['reviewed_code_sha256']={p:m.sha(p) for p in m.REVIEWED_FILES};approval['reviewed_code_sha256']['unreviewed.py']='extra';a.write_text(json.dumps(approval))
    with pytest.raises(ValueError,match='exact'):m.verify_review(a,cp,r)

def test_duplicate_different_runIDs_cannot_claim_same_checkpoint(tmp_path):
    m=module('data_sources_quickgo_resume_v2');p=tmp_path/'checkpoint.json';p.write_text('{"pending":[1,2]}');first=m.CheckpointClaim(tmp_path,p,'run-one')
    try:
        with pytest.raises(ValueError,match='checkpoint'):m.CheckpointClaim(tmp_path,p,'run-two')
    finally:first.close()
    assert len(list((tmp_path/'audits/checkpoint-execution-locks').glob('*.lock')))==1

def test_claim_recovers_after_prior_owner_exited_without_deleting_history(tmp_path):
    m=module('data_sources_quickgo_resume_v2');p=tmp_path/'checkpoint.json';p.write_text('{"pending":[1]}');ctx=multiprocessing.get_context('fork');parent,child=ctx.Pipe()
    def owner():
        claim=m.CheckpointClaim(tmp_path,p,'old-owner');child.send(str(claim.receipt));child.close();os._exit(0)
    proc=ctx.Process(target=owner);proc.start();assert parent.poll(5);receipt=pathlib.Path(parent.recv());proc.join(5);assert proc.exitcode==0 and receipt.exists()
    claim=m.CheckpointClaim(tmp_path,p,'recovery-owner')
    try:assert claim.prior_owners_checked and receipt.exists()
    finally:claim.close()

def test_forked_worker_keeps_checkpoint_locked_after_coordinator_closes_its_fd(tmp_path):
    m=module('data_sources_quickgo_resume_v2');p=tmp_path/'checkpoint.json';p.write_text('{"pending":[1]}');claim=m.CheckpointClaim(tmp_path,p,'coordinator');ctx=multiprocessing.get_context('fork');ready=ctx.Event();release=ctx.Event()
    def inherited_worker():ready.set();release.wait(5)
    proc=ctx.Process(target=inherited_worker);proc.start();assert ready.wait(5);claim.close()
    try:
        with pytest.raises(ValueError,match='checkpoint'):m.CheckpointClaim(tmp_path,p,'other-coordinator')
    finally:release.set();proc.join(5)
    assert proc.exitcode==0
