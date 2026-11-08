"""Fresh-process proof for zero-step RGB allocation, not training recovery.

The RGB image is a synthetic fixture; LINCS backing records are source-verified
by the existing fixture. No synthetic image/response pairing is asserted.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from test_color_optimizer import boundary
from test_color_schedule import make, cases
from mora3.model.profile_reconstruction import _digest


CHILD = r'''
import json, os, sys
from pathlib import Path
import torch
from test_color_optimizer import boundary, fingerprint
from test_color_schedule import make
from test_lincs_native import cases
from mora3.model.profile_reconstruction import _digest
root, archive, report, device, corruption = sys.argv[1:]
b = boundary(make(Path(root), cases.__wrapped__(), device))
# Deliberately differ from the producing process, including runtime state.
with torch.no_grad():
    for p in b.model.parameters(): p.add_(.375)
b.model.eval(); b.model.color.train()
next(b.model.parameters()).grad = torch.ones_like(next(b.model.parameters()))
state = torch.load(archive, weights_only=True)
original = _digest(b.model.state_dict())
ids = [id(p) for p in b.model.parameters()]
runtime = _digest(([p.grad for p in b.model.parameters()],
    [m.training for m in b.model.modules()], torch.get_rng_state(),
    torch.cuda.get_rng_state() if torch.cuda.is_available() else None))
if corruption == 'provider': state['provider']['weight'] = 99.
if corruption == 'alias': state['parameter_aliases'] = []
if corruption == 'late_nan':
    key = [k for k,v in state['model'].items() if isinstance(v,torch.Tensor) and v.is_floating_point()][-1]
    state['model'][key].fill_(float('nan'))
if corruption == 'none':
    assert original != _digest(state['model'])
    b.load_allocation_state_dict(state)
    assert _digest(b.model.state_dict()) == _digest(state['model'])
else:
    before = fingerprint(b)
    try: b.load_allocation_state_dict(state)
    except ValueError: pass
    else: raise AssertionError('corrupt allocation accepted')
    assert fingerprint(b) == before
assert ids == [id(p) for p in b.model.parameters()]
assert runtime == _digest(([p.grad for p in b.model.parameters()],
    [m.training for m in b.model.modules()], torch.get_rng_state(),
    torch.cuda.get_rng_state() if torch.cuda.is_available() else None))
assert not b.optimizer.state and b.steps == 0
before = fingerprint(b)
try: b.step()
except ValueError: pass
else: raise AssertionError('training gate opened')
assert fingerprint(b) == before
with torch.no_grad():
    output = b.schedule.evaluate(0)
Path(report).write_text(json.dumps(dict(pid=os.getpid(), device=device,
    output_digest=_digest(output),
    corruption=corruption, model_digest=_digest(b.model.state_dict()),
    optimizer_updates=0, training_admission=False, execution_supported=False)))
'''


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('corruption', ['none', 'provider', 'alias', 'late_nan'])
def test_fresh_process_allocation_restores_or_refuses_transactionally(tmp_path, cases, device, corruption):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    b = boundary(make(tmp_path, cases, device))
    state = b.allocation_state_dict()
    archive, report = tmp_path/'allocation.pt', tmp_path/'child.json'
    torch.save(state, archive)
    expected = _digest(state['model'])
    with torch.no_grad():
        expected_output = _digest(b.schedule.evaluate(0))
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join((str(Path('src').resolve()), str(Path('tests/model').resolve()), str(Path('tests/data').resolve())))
    env['OMP_NUM_THREADS'] = env['MKL_NUM_THREADS'] = '1'
    child = subprocess.run([sys.executable, '-c', CHILD, str(tmp_path), str(archive),
        str(report), device, corruption], env=env, capture_output=True, text=True, timeout=120)
    assert child.returncode == 0, child.stdout + child.stderr
    result = json.loads(report.read_text())
    assert result['pid'] != os.getpid()
    assert result['device'] == device and result['corruption'] == corruption
    assert result['optimizer_updates'] == 0 and result['training_admission'] is False
    if corruption == 'none':
        assert result['model_digest'] == expected
        assert result['output_digest'] == expected_output
    if os.environ.get('MORA3_COLOR_PROOF_OUTPUT'):
        target = Path(os.environ['MORA3_COLOR_PROOF_OUTPUT']) / (device+'-'+corruption+'.json')
        with target.open('x') as stream:
            json.dump(dict(result, parent_pid=os.getpid(), expected_model_digest=expected,
                expected_output_digest=expected_output, fixture_image='synthetic',
                substantive_training=False), stream, indent=2)
