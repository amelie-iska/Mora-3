"""Finite subprocess reconstruction proof; synthetic construction is not training."""
import os
import subprocess
import sys
from pathlib import Path
import json
import pytest

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_fresh_process_allocation_roundtrip(tmp_path,device):
    import torch
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    script=Path('outputs/implementation/i246-instance-fresh-process/worker.py')
    assert script.is_file(), 'missing finite fresh-process diagnostic'
    env=dict(os.environ,PYTHONPATH='src:tests/model:tests/data:tests/imaging',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    for phase in ('produce','consume'):
        result=subprocess.run([sys.executable,str(script),phase,str(tmp_path),device],env=env,capture_output=True,text=True,timeout=120)
        assert result.returncode==0,result.stdout+result.stderr
    a=json.loads((tmp_path/'produce.json').read_text());b=json.loads((tmp_path/'consume.json').read_text())
    assert a['pid']!=b['pid'] and a['state_digest']==b['state_digest']
    assert b['runtime_preserved'] and b['late_rejection_unchanged'] and b['admission_refused']
    assert b['neural_calls']==b['optimizer_updates']==0
