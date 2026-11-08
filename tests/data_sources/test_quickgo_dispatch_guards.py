import importlib.util,pathlib,json,os,sys
import pytest
ROOT=pathlib.Path.cwd();sys.path.insert(0,str(ROOT/'scripts'))
def module(name):
    p=ROOT/('scripts/'+name+'.py');s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def files(tmp_path,cp):
    m=module('data_sources_quickgo_parallel');c=tmp_path/'checkpoint.json';c.write_text(json.dumps(cp));r=tmp_path/'retirement.json';r.write_text(json.dumps(dict(all_owned_old_writers_retired=True,checkpoint_sha256=m.sha(c))));a=tmp_path/'approval.json';a.write_text(json.dumps(dict(approved=True,action='QuickGO_checkpoint_parallel_handoff_v1',reviewed_code_sha256={p:m.sha(p) for p in ['scripts/data_sources_quickgo_resume.py','scripts/data_sources_quickgo_client.py','scripts/data_sources_quickgo_checkpoint.py','scripts/data_sources_quickgo_parallel.py']})));return m,c,r,a

def test_exact_review_hashes_and_retirement_checkpoint_are_required(tmp_path):
    m,c,r,a=files(tmp_path,{'owned_processes':[]});assert m.verify_review(a,c,r)=={'owned_processes':[]}
    approval=json.loads(a.read_text());approval['reviewed_code_sha256']['scripts/data_sources_quickgo_client.py']='wrong';a.write_text(json.dumps(approval))
    with pytest.raises(ValueError,match='helper changed'):m.verify_review(a,c,r)

def test_running_old_owned_writer_refuses_dispatch_before_any_new_namespace(tmp_path):
    stat=pathlib.Path('/proc/'+str(os.getpid())+'/stat').read_text().split();root=tmp_path/'must-not-create';cp=dict(root=str(root),owned_processes=[dict(pid=os.getpid(),start_ticks=int(stat[21]))]);m,c,r,a=files(tmp_path,cp)
    with pytest.raises(ValueError,match='old owned source writer remains'):m.run(c,r,a,'quickgo-parallel-test')
    assert not root.exists()

def test_changed_protected_old_bytes_refuse_dispatch_before_any_network_or_output(tmp_path):
    p=tmp_path/'original.response';p.write_text('original');root=tmp_path/'must-not-create';cp=dict(root=str(root),owned_processes=[],old_raw_hashes=[dict(path=str(p),bytes=p.stat().st_size,sha256='incorrect')]);m,c,r,a=files(tmp_path,cp)
    with pytest.raises(ValueError,match='oldsource changed'):m.run(c,r,a,'quickgo-parallel-test')
    assert p.read_text()=='original' and not root.exists()

def test_checkpoint_builder_sends_no_signals_and_refuses_active_process(monkeypatch):
    m=module('data_sources_quickgo_checkpoint');stat=pathlib.Path('/proc/'+str(os.getpid())+'/stat').read_text().split();cmd=pathlib.Path('/proc/'+str(os.getpid())+'/cmdline').read_bytes().replace(b'\0',b' ').decode().strip()
    monkeypatch.setattr(m.os,'kill',lambda *a:pytest.fail('checkpoint builder sent a signal'))
    with pytest.raises(ValueError):m.inspect_owned(dict(pid=os.getpid(),start_ticks=int(stat[21]),command=cmd))
