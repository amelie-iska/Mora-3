#!/usr/bin/env python3
"""Sequential, resumable Mora-3 task supervisor; Python stdlib only.

Logs are private project-local artifacts. A stop request is honored between tasks.
The CLI's existing authentication is used; project key files are never accessed.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def stamp():
    return datetime.now(timezone.utc).isoformat()


def evidence_path(root, name):
    path = Path(name)
    if path.is_absolute() or not (root.resolve() / path).resolve().is_relative_to(root.resolve()):
        raise ValueError('evidence must be inside project')
    path = root / path
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError('missing or empty evidence: ' + name)
    return path


def validate_report(report, root, completion_evidence):
    required = {'status', 'summary', 'completed_task', 'next_task', 'evidence'}
    if set(report) != required or report['status'] not in {'progress', 'blocked', 'complete'}:
        raise ValueError('invalid worker report schema')
    if any(not isinstance(report[k], str) for k in required - {'evidence'}):
        raise ValueError('invalid worker report strings')
    if not isinstance(report['evidence'], list) or any(not isinstance(x, str) for x in report['evidence']):
        raise ValueError('invalid evidence list')
    if not report['summary'].strip():
        raise ValueError('empty worker summary')
    if report['status'] != 'blocked' and (not report['completed_task'] or not report['evidence']):
        raise ValueError('progress requires task and evidence')
    if report['status'] == 'progress' and not report['next_task']:
        raise ValueError('progress requires next task')
    if report['status'] == 'blocked':
        # A broken execution environment may prevent evidence-file creation.
        # Retain its diagnostic report and supervisor-captured logs, without
        # accepting this as progress or completion.
        return
    for name in report['evidence']:
        evidence_path(root, name)
    if report['status'] == 'complete':
        if not completion_evidence:
            raise ValueError('completion requires verified readiness evidence')
        for name in completion_evidence:
            value = json.loads(evidence_path(root, name).read_text())
            if value.get('verified') is not True:
                raise ValueError('completion evidence not verified: ' + name)
            if not value.get('evidence_files'):
                raise ValueError('completion evidence needs supporting files')
            for supporting in value['evidence_files']:
                evidence_path(root, supporting)
            if Path(name).name == 'launch_report.json':
                pid = value.get('training_pid')
                if not isinstance(pid, int) or pid <= 1:
                    raise ValueError('training pid missing')
                try:
                    os.kill(pid, 0)
                    proc = Path('/proc') / str(pid)
                    command = (proc / 'cmdline').read_bytes()
                    cwd = (proc / 'cwd').resolve()
                    if b'train' not in command.lower() or not cwd.is_relative_to(root.resolve()):
                        raise ValueError('training process does not match project')
                except (OSError, ProcessLookupError):
                    raise ValueError('training process is not live') from None


def worker_command(root, output, mcp_names, sandbox="workspace-write"):
    if sandbox not in {"workspace-write", "danger-full-access"}:
        raise ValueError("Unsupported worker sandbox setting")
    command = ['codex', 'exec', '-C', str(root), '--sandbox', sandbox,
               '-c', 'approval_policy="never"',
               '-c', 'sandbox_workspace_write.network_access=true',
               '-c', 'web_search="live"', '--json', '--color', 'never',
               '--output-schema', str(root / 'planning/worker_result.schema.json'),
               '--output-last-message', str(output)]
    for name in mcp_names:
        if not re.fullmatch(r'[A-Za-z0-9_-]+', name):
            raise ValueError('unsupported MCP server identifier')
        command.extend(['-c', 'mcp_servers.' + name + '.enabled=false'])
    return command + ['-']


def install_signal_handlers():
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    for signum in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, interrupted)


def run_child(command, prompt, directory, timeout, output_name='events.jsonl', error_name='stderr.log'):
    """Capture one child; terminate its process group on timeout or interruption."""
    with (directory / output_name).open('w') as out, (directory / error_name).open('w') as err:
        proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=out, stderr=err,
                                text=True, start_new_session=True, cwd=ROOT)
        try:
            proc.communicate(prompt, timeout=timeout)
            return proc.returncode
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=10)
            except (subprocess.TimeoutExpired, ProcessLookupError):
                pass
            # The group may outlive its leader, including children ignoring TERM.
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
            return 124


def save_state(path, **fields):
    # Replace only supervisor-owned state, retaining append-only history.
    value = dict(updated_at=stamp(), **fields)
    temp = path.with_suffix('.pending')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)
    with path.with_suffix('.history.jsonl').open('a') as stream:
        stream.write(json.dumps(value) + '\n')


def run(root, settings):
    home = root / '.local/implementation-loop'
    home.mkdir(parents=True, exist_ok=True, mode=0o700)
    state = home / 'state.json'
    with (home / 'supervisor.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('A Mora-3 supervisor is already running')
        run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + str(os.getpid())
        run_dir = home / run_id
        run_dir.mkdir(mode=0o700)
        stop = run_dir / 'STOP'
        config_path = Path.home() / '.codex/config.toml'
        config = tomllib.loads(config_path.read_text()) if config_path.exists() else {}
        mcp_names = list(config.get('mcp_servers', {}))
        (run_dir / 'invocation_policy.json').write_text(json.dumps({
            'sandbox':settings.get('sandbox_mode','workspace-write'), 'mcp_disabled':mcp_names,
            'model':'inherited from existing CLI configuration',
            'task_timeout_seconds':settings['task_timeout_seconds']}, indent=2))
        previous_task = None
        unchanged = 0
        for iteration in range(1, settings['max_iterations'] + 1):
            if stop.exists():
                save_state(state, status='stopped', run_id=run_id, iteration=iteration, pid=os.getpid())
                return 0
            if shutil.disk_usage(root).free < settings['minimum_free_gib'] * 1024**3:
                save_state(state, status='blocked', reason='Disk headroom below configured minimum', run_id=run_id)
                return 2
            task_dir = run_dir / ('task-%04d' % iteration)
            task_dir.mkdir()
            before = hashlib.sha256((root / 'planning/progress.json').read_bytes()).hexdigest()
            prompt = (root / 'planning/worker_prompt.md').read_text()
            prompt += '\nCurrent iteration: %d. Save evidence in project-relative paths.\n' % iteration
            command = worker_command(root, task_dir / 'result.json', mcp_names,
                                     sandbox=settings.get('sandbox_mode','workspace-write'))
            save_state(state, status='running', run_id=run_id, iteration=iteration,
                       pid=os.getpid(), task_dir=str(task_dir.relative_to(root)))
            code = run_child(command, prompt, task_dir, settings['task_timeout_seconds'])
            if code:
                save_state(state, status='error', run_id=run_id, iteration=iteration,
                           reason='Worker exit code %d; inspect retained logs' % code)
                return code
            try:
                result = json.loads((task_dir / 'result.json').read_text())
                validate_report(result, root, settings['completion_evidence'])
            except (OSError, ValueError, TypeError) as error:
                save_state(state, status='error', run_id=run_id, iteration=iteration, reason=str(error))
                return 2
            if result['status'] == 'blocked':
                save_state(state, status='blocked', run_id=run_id, iteration=iteration, report=result)
                return 2
            test_code = run_child(settings['test_command'], '', task_dir, 600,
                                  output_name='verification.log', error_name='verification.stderr.log')
            if test_code:
                save_state(state, status='error', run_id=run_id, iteration=iteration,
                           reason='Post-task tests failed or timed out; inspect verification.log', report=result)
                return test_code
            after = hashlib.sha256((root / 'planning/progress.json').read_bytes()).hexdigest()
            if before == after or result['completed_task'] == previous_task:
                unchanged += 1
            else:
                unchanged = 0
            previous_task = result['completed_task']
            if unchanged >= settings['max_no_progress']:
                save_state(state, status='blocked', run_id=run_id, iteration=iteration,
                           reason='Repeated iterations without distinct recorded progress', report=result)
                return 2
            if result['status'] == 'complete':
                save_state(state, status='complete', run_id=run_id, iteration=iteration, report=result)
                return 0
            save_state(state, status='progress', run_id=run_id, iteration=iteration, report=result)
        save_state(state, status='iteration_limit', run_id=run_id, reason='Configured finite iteration budget reached')
        return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['run', 'status', 'stop'])
    args = parser.parse_args()
    home = ROOT / '.local/implementation-loop'
    state = home / 'state.json'
    if args.action == 'status':
        print(state.read_text() if state.exists() else json.dumps({'status':'not_started'}))
        return 0
    if args.action == 'stop':
        if not state.exists():
            print('No running supervisor')
            return 0
        value = json.loads(state.read_text())
        if value.get('status') not in {'running', 'progress'}:
            print('Supervisor is already stopped: ' + value.get('status', 'unknown'))
            return 0
        (home / value['run_id'] / 'STOP').touch(exist_ok=True)
        print('Stop requested after current task; training processes are unaffected')
        return 0
    install_signal_handlers()
    os.umask(0o077)
    settings = json.loads((ROOT / 'planning/loop_config.json').read_text())
    return run(ROOT, settings)


if __name__ == '__main__':
    sys.exit(main())
