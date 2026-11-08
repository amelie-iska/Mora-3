import sys,gzip,subprocess,os,time
from pathlib import Path
import pytest
sys.path.insert(0,str(Path.cwd()/"scripts"))
from data_sources_mmseqs_createdb import shuffle_key,verify_native_database,guard_decision,tree_bytes,process_identity,stop_owned_group,verify_approval
TOOL=Path("/home/iska/Documents/amelie/bio/MMseqs2/mmseqs/bin/mmseqs")

@pytest.mark.parametrize("n",[1,20,1001])
def test_pinned32way_shuffle_is_a_bijection_and_preserves_source_ordinals(n):
    values=[shuffle_key(i,n,32) for i in range(n)]
    assert sorted(values)==list(range(n))
    if n>32:assert shuffle_key(32,n,32)==1 and shuffle_key(1,n,32)>1

@pytest.fixture
def native_db(tmp_path):
    if not TOOL.exists():pytest.skip("existing remote MMseqs tool notavailable")
    source=tmp_path/"native.fasta.gz"
    raw="".join(f">9606.test_{i} originalheader\n"+("ACDEFGHIKLMNPQRSTVWY"*(1+i%3) if i else "AXUBJOZ")+"\n" for i in range(1001))
    source.write_bytes(gzip.compress(raw.encode(),mtime=0));prefix=tmp_path/"DB100"
    subprocess.run([str(TOOL),"createdb",str(source),str(prefix),"--shuffle","1","--createdb-mode","0","--threads","2"],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=20)
    return source,prefix

def test_full_native_headers_sequences_and_lookup_roundtrip_after_default_shuffle(native_db):
    source,prefix=native_db;r=verify_native_database(source,prefix,1001,32)
    assert r["native_records_compared"]==1001 and r["all_native_headers_sequences_and_provider_lookup_exact"]
    assert r["database_keys_are_shuffled_not_source_ordinals"]

def test_changed_native_database_bytes_rejected(native_db):
    source,prefix=native_db
    with prefix.open("r+b") as f:f.write(b"Q")
    with pytest.raises(ValueError):verify_native_database(source,prefix,1001,32)

def test_guard_counts_hidden_nested_artifacts_and_refuses_external_links(tmp_path):
    (tmp_path/".tmp").mkdir();(tmp_path/".tmp"/"file").write_bytes(b"abc")
    assert tree_bytes(tmp_path)==3
    (tmp_path/"link").symlink_to(tmp_path/".tmp"/"file")
    with pytest.raises(ValueError):tree_bytes(tmp_path)

def test_cap_free_and_time_guards_include_stop_margin():
    p=dict(stage_cap_bytes=1000,stop_margin_bytes=100,minimum_free_bytes=200,max_seconds=10)
    assert guard_decision(899,301,9,p) is None
    assert guard_decision(900,301,9,p)=="stage_byte_highwater"
    assert guard_decision(1,299,9,p)=="free_reserve_highwater"
    assert guard_decision(1,1000,10,p)=="finite_walltime"

def test_changed_hash_or_missing_explicit_approval_rejected():
    codes={"a":"1"};approval=dict(approved=True,action="MMseqs_StageB_full_createdb_v1",plan_sha256="p",code_sha256=codes)
    verify_approval(approval,"p",codes)
    with pytest.raises(ValueError):verify_approval(approval,"q",codes)
    with pytest.raises(ValueError):verify_approval(approval,"p",{"a":"2"})
    approval["approved"]=False
    with pytest.raises(ValueError):verify_approval(approval,"p",codes)

def test_owned_child_stop_rejects_changed_identity_and_preserves_fixture(tmp_path):
    marker=tmp_path/"protected-fixture";marker.write_bytes(b"unchanged")
    p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(30)"],start_new_session=True)
    try:
        from data_sources_mmseqs_createdb import wait_process_identity
        expected=wait_process_identity(p.pid,[sys.executable,"-c","import time;time.sleep(30)"],timeout=5);assert expected is not None
        wrong=dict(expected,start_ticks=expected["start_ticks"]+1)
        with pytest.raises(ValueError):stop_owned_group(p.pid,wrong,lambda:None)
        assert p.poll() is None
        called=[];result=stop_owned_group(p.pid,expected,lambda:called.append(marker.read_bytes()))
        p.wait(timeout=5);assert p.returncode<0 and called==[b"unchanged"] and result["owned_group_stopped"]
        assert marker.read_bytes()==b"unchanged"
    finally:
        if p.poll() is None:p.kill();p.wait()

@pytest.fixture
def compressed_native_db(native_db):
    source,prefix=native_db;compressed=prefix.parent/"compressedDB"
    subprocess.run([str(TOOL),"createdb",str(source),str(compressed),"--shuffle","1","--createdb-mode","0","--compressed","1","--threads","2"],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=20)
    return source,compressed

def test_compressed_and_raw_short_frames_roundtrip_every_native_header_sequence(compressed_native_db):
    source,prefix=compressed_native_db;r=verify_native_database(source,prefix,1001,32)
    assert r["native_records_compared"]==1001 and r["native_compressed_storage_losslessly_decoded"]

def test_oversized_compressed_frame_rejected_before_allocation(compressed_native_db):
    import struct
    source,prefix=compressed_native_db
    with prefix.open("r+b") as f:f.write(struct.pack("<I",2**30))
    with pytest.raises(ValueError):verify_native_database(source,prefix,1001,32)

def test_small_native_proof_reports_both_actual_index_and_replay_progress(native_db):
    source,prefix=native_db;events=[]
    verify_native_database(source,prefix,1001,32,on_progress=lambda n,total,phase:events.append((n,total,phase)))
    assert (0,1001,"native_index_scan") in events and (1001,1001,"native_index_scan") in events
    assert (0,1001,"pointwise_source_replay") in events and (1001,1001,"pointwise_source_replay") in events

def test_createdb_command_whitelists_two_CPU_lossless_compressed_default_shuffle(tmp_path):
    from data_sources_mmseqs_createdb import createdb_command
    plan=dict(binary=str(TOOL),source_path="/source/native.fa.gz",new_output=str(tmp_path),threads=2)
    cmd=createdb_command(plan)
    assert cmd[1]=="createdb" and "cluster" not in cmd
    assert cmd[cmd.index("--threads")+1]=="2" and cmd[cmd.index("--compressed")+1]=="1"
    assert cmd[cmd.index("--shuffle")+1]=="1" and cmd[cmd.index("--createdb-mode")+1]=="0"
    plan["threads"]=32
    with pytest.raises(ValueError):createdb_command(plan)

def test_runner_rejects_unapproved_execution_before_creating_stage(tmp_path):
    import json
    runner=Path.cwd()/"scripts/data_sources_mmseqs_createdb_run.py"
    assert runner.exists()
    output=tmp_path/"not-created";plan=tmp_path/"plan.json";approval=tmp_path/"approval.json"
    plan.write_text(json.dumps(dict(new_output=str(output))));approval.write_text(json.dumps(dict(approved=False)))
    result=subprocess.run([sys.executable,str(runner),"--plan",str(plan),"--approval",str(approval)],capture_output=True,text=True,timeout=10)
    assert result.returncode!=0 and "execution review required" in result.stderr
    assert not output.exists()

class ReadinessClock:
    def __init__(self):self.value=0.
    def now(self):return self.value
    def sleep(self,delay):self.value+=delay

def ready_snapshot(argv=None,start=100,pgid=123):
    return dict(pid=123,start_ticks=start,pgid=pgid,argv=["python","child.py"] if argv is None else argv)

def test_empty_fork_exec_argv_is_not_bound_before_two_stable_expected_snapshots():
    from data_sources_mmseqs_createdb import wait_process_identity
    clock=ReadinessClock();seq=iter([ready_snapshot([]),ready_snapshot(),ready_snapshot()]);calls=[]
    def read(pid):calls.append(pid);return next(seq)
    result=wait_process_identity(123,["python","child.py"],timeout=.1,poll=.01,_read=read,_clock=clock.now,_sleep=clock.sleep)
    assert result==ready_snapshot() and len(calls)==3 and clock.value==.02

def test_matching_snapshot_then_transient_exec_change_requires_stability_again():
    from data_sources_mmseqs_createdb import wait_process_identity
    clock=ReadinessClock();seq=iter([ready_snapshot(),ready_snapshot([]),ready_snapshot(),ready_snapshot()]);calls=[]
    def read(pid):calls.append(pid);return next(seq)
    result=wait_process_identity(123,["python","child.py"],timeout=.1,poll=.01,_read=read,_clock=clock.now,_sleep=clock.sleep)
    assert result==ready_snapshot() and len(calls)==4

def test_stable_wrong_argv_times_out_without_becoming_owned():
    from data_sources_mmseqs_createdb import wait_process_identity
    clock=ReadinessClock()
    with pytest.raises(ValueError):wait_process_identity(123,["python","child.py"],timeout=.03,poll=.01,_read=lambda _:ready_snapshot(["foreign"]),_clock=clock.now,_sleep=clock.sleep)
    assert .03<=clock.value<=.04

def test_reused_PID_start_during_exec_readiness_is_rejected():
    from data_sources_mmseqs_createdb import wait_process_identity
    clock=ReadinessClock();seq=iter([ready_snapshot([]),ready_snapshot(start=101)])
    with pytest.raises(ValueError):wait_process_identity(123,["python","child.py"],timeout=.1,poll=.01,_read=lambda _:next(seq),_clock=clock.now,_sleep=clock.sleep)

def test_exclusive_process_group_must_be_ready_and_stable_too():
    from data_sources_mmseqs_createdb import wait_process_identity
    clock=ReadinessClock();seq=iter([ready_snapshot(pgid=99),ready_snapshot(),ready_snapshot()])
    assert wait_process_identity(123,["python","child.py"],timeout=.1,poll=.01,_read=lambda _:next(seq),_clock=clock.now,_sleep=clock.sleep)==ready_snapshot()

def test_empty_expected_command_is_never_permission_to_signal(monkeypatch):
    import data_sources_mmseqs_createdb as module
    expected=ready_snapshot([]);signals=[]
    monkeypatch.setattr(module,"process_identity",lambda _:expected)
    monkeypatch.setattr(module.os,"killpg",lambda *args:signals.append(args))
    with pytest.raises(ValueError):module.stop_owned_group(123,expected,lambda:None)
    assert not signals
