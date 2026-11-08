"""Supervisor tests: missing evidence, fabricated completion and runaway children."""
import importlib.util
import json
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]

def module():
    path = ROOT / 'scripts/implementation_loop.py'
    assert path.is_file(), 'Implementation supervisor is missing'
    spec = importlib.util.spec_from_file_location('mora3_loop', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def report(**updates):
    value = dict(status='progress', summary='Reviewed primary methods',
                 completed_task='R1a', next_task='R1b', evidence=['outputs/evidence.txt'])
    value.update(updates)
    return value


def test_progress_requires_real_evidence(tmp_path):
    loop = module()
    with pytest.raises(ValueError, match='evidence'):
        loop.validate_report(report(), tmp_path, [])
    (tmp_path / 'outputs').mkdir()
    (tmp_path / 'outputs/evidence.txt').write_text('measured source notes')
    loop.validate_report(report(), tmp_path, [])


@pytest.mark.parametrize('name', ['../outside.txt', '/etc/passwd'])
def test_evidence_must_stay_inside_project(tmp_path, name):
    with pytest.raises(ValueError, match='evidence'):
        module().validate_report(report(evidence=[name]), tmp_path, [])


def test_completion_needs_verified_readiness_and_live_training(tmp_path):
    loop = module()
    (tmp_path / 'outputs').mkdir()
    (tmp_path / 'outputs/evidence.txt').write_text('evidence')
    (tmp_path / 'readiness.json').write_text(json.dumps({'verified':False}))
    with pytest.raises(ValueError, match='verified'):
        loop.validate_report(report(status='complete'), tmp_path, ['readiness.json'])
    (tmp_path / 'readiness.json').write_text(json.dumps({'verified':True,'evidence_files':['outputs/evidence.txt']}))
    (tmp_path / 'launch_report.json').write_text(json.dumps({'verified':True,'training_pid':99999999,'evidence_files':['outputs/evidence.txt']}))
    with pytest.raises(ValueError, match='training'):
        loop.validate_report(report(status='complete'), tmp_path, ['readiness.json','launch_report.json'])


def test_blocked_can_report_without_fabricated_completed_task(tmp_path):
    module().validate_report(report(status='blocked', completed_task='', next_task='', evidence=[]), tmp_path, [])


def test_codex_invocation_disables_every_configured_mcp(tmp_path):
    command = module().worker_command(tmp_path, tmp_path/'result.json', ['MCP_DOCKER','other'])
    assert 'mcp_servers.MCP_DOCKER.enabled=false' in command
    assert 'mcp_servers.other.enabled=false' in command
    assert 'workspace-write' in command
    assert '--dangerously-bypass-approvals-and-sandbox' not in command
    assert 'approval_policy="never"' in command


def test_worker_failure_is_retained(tmp_path):
    loop = module()
    code = loop.run_child([sys.executable,'-c','import sys; print("failed"); sys.exit(7)'], 'test', tmp_path, 5)
    assert code == 7
    assert 'failed' in (tmp_path/'events.jsonl').read_text()


def test_worker_timeout_terminates_process_group(tmp_path):
    loop = module()
    code = loop.run_child([sys.executable,'-c','import time; time.sleep(60)'], 'test', tmp_path, 0.05)
    assert code == 124
    assert (tmp_path/'stderr.log').exists()


def setup_loop(tmp_path):
    planning = tmp_path / 'planning'
    planning.mkdir()
    (planning/'progress.json').write_text('{"task":"initial"}')
    (planning/'worker_prompt.md').write_text('Do a bounded task.')
    settings = dict(max_iterations=2, task_timeout_seconds=5, max_no_progress=1,
                    minimum_free_gib=0, test_command=[sys.executable,'-c','pass'],
                    completion_evidence=['readiness.json','launch_report.json'])
    return settings


def test_real_worker_subprocess_blocker_stops_loop_and_retains_report(tmp_path, monkeypatch):
    loop = module()
    settings = setup_loop(tmp_path)
    script = tmp_path/'fake_worker.py'
    script.write_text('import json,sys\nfrom pathlib import Path\nPath(sys.argv[1]).write_text(json.dumps('+repr(report(status='blocked', completed_task='', next_task='', evidence=[]))+'))\n')
    monkeypatch.setattr(loop, 'worker_command', lambda root,output,names,sandbox="workspace-write":[sys.executable,str(script),str(output)])
    assert loop.run(tmp_path,settings) == 2
    state = json.loads((tmp_path/'.local/implementation-loop/state.json').read_text())
    assert state['status'] == 'blocked'
    assert state['iteration'] == 1
    assert state['report']['summary'] == 'Reviewed primary methods'


def test_repeated_unchanged_progress_stops_instead_of_running_forever(tmp_path, monkeypatch):
    loop = module()
    settings = setup_loop(tmp_path)
    (tmp_path/'outputs').mkdir()
    (tmp_path/'outputs/evidence.txt').write_text('actual notes')
    script = tmp_path/'fake_worker.py'
    script.write_text('import json,sys\nfrom pathlib import Path\nPath(sys.argv[1]).write_text(json.dumps('+repr(report())+'))\n')
    monkeypatch.setattr(loop, 'worker_command', lambda root,output,names,sandbox="workspace-write":[sys.executable,str(script),str(output)])
    assert loop.run(tmp_path,settings) == 2
    state = json.loads((tmp_path/'.local/implementation-loop/state.json').read_text())
    assert state['status'] == 'blocked'
    assert 'progress' in state['reason']


def test_duplicate_supervisor_cannot_start_worker(tmp_path):
    import fcntl
    loop = module()
    settings = setup_loop(tmp_path)
    home = tmp_path/'.local/implementation-loop'
    home.mkdir(parents=True)
    with (home/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(SystemExit,match='already running'):
            loop.run(tmp_path,settings)


def process_is_active(pid):
    path = Path('/proc')/str(pid)/'stat'
    return path.exists() and path.read_text().split()[2] != 'Z'


def test_timeout_kills_descendant_that_ignores_term(tmp_path):
    import os
    import signal
    loop = module()
    child_pid = tmp_path/'child.pid'
    child_code = 'import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)'
    leader_code = ('import subprocess,sys,time; from pathlib import Path; '
                   'p=subprocess.Popen([sys.executable,"-c",'+repr(child_code)+']); '
                   'Path('+repr(str(child_pid))+').write_text(str(p.pid)); time.sleep(60)')
    try:
        assert loop.run_child([sys.executable,'-c',leader_code], '', tmp_path, 0.5) == 124
        pid = int(child_pid.read_text())
        import time
        time.sleep(0.1)
        assert not process_is_active(pid), 'Timeout left a worker descendant running'
    finally:
        if child_pid.exists() and process_is_active(int(child_pid.read_text())):
            os.kill(int(child_pid.read_text()),signal.SIGKILL)


def test_supervisor_termination_cleans_active_worker(tmp_path):
    import os
    import signal
    import subprocess
    import time
    loop = module()
    script = tmp_path/'supervisor.py'
    worker_pid = tmp_path/'worker.pid'
    worker_code = 'import os,time; from pathlib import Path; Path('+repr(str(worker_pid))+').write_text(str(os.getpid())); time.sleep(60)'
    script.write_text('import importlib.util,sys\nfrom pathlib import Path\nspec=importlib.util.spec_from_file_location("loop",'+repr(str(ROOT/'scripts/implementation_loop.py'))+')\nloop=importlib.util.module_from_spec(spec)\nspec.loader.exec_module(loop)\nloop.install_signal_handlers()\nsys.exit(loop.run_child([sys.executable,"-c",'+repr(worker_code)+'], "", Path('+repr(str(tmp_path))+'), 60))\n')
    proc = subprocess.Popen([sys.executable,str(script)])
    try:
        for _ in range(100):
            if worker_pid.exists() or proc.poll() is not None:
                break
            time.sleep(0.02)
        assert worker_pid.exists(), 'Supervisor failed to start worker'
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=15)
        time.sleep(0.1)
        assert not process_is_active(int(worker_pid.read_text()))
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        if worker_pid.exists() and process_is_active(int(worker_pid.read_text())):
            os.kill(int(worker_pid.read_text()),signal.SIGKILL)


def test_blocker_survives_when_worker_could_not_write_evidence(tmp_path):
    module().validate_report(report(status='blocked', completed_task='', next_task='repair sandbox', evidence=['Sandbox prevented all file writes']),tmp_path,[])


def test_explicit_project_approval_selects_unrestricted_worker(tmp_path):
    command = module().worker_command(tmp_path,tmp_path/'result.json',['MCP_DOCKER'],sandbox='danger-full-access')
    assert command[command.index('--sandbox')+1] == 'danger-full-access'
    assert 'mcp_servers.MCP_DOCKER.enabled=false' in command
    assert '--dangerously-bypass-approvals-and-sandbox' not in command
