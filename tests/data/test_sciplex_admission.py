import copy
import hashlib
import pytest
import torch
from mora3.data.connected_sciplex import SciPlexTask
from mora3.data.gperturb_sciplex import array_task
from mora3.data.sciplex_admission import ObservationalSciPlexTask


def fixture(direction='response', **changes):
    row=dict(cell_id='a',source_index=0,intervention='Dex',dose_coordinate=.1,
             dose_units='source_scaled_coordinate',counts=[1,2],n_genes=2,n_counts=3)
    row.update(changes)
    task=SciPlexTask.from_sidecar(array_task(row,['A','B'],direction=direction),source_sha256='a'*64)
    receipt=dict(version=1,policy='author-export-observational-train-only-v1',
        split='train_only',leakage_group='study:GSE139944:GSM4150377',
        audit_sha256='b'*64,source_sha256='a'*64,
        task_sha256=hashlib.sha256(task.canonical_bytes()).hexdigest())
    return task,receipt


def test_narrow_admission_preserves_quarantine_and_both_masks():
    for direction in ('response','intervention'):
        task,r=fixture(direction)
        with pytest.raises(ValueError): task.project(for_training=True)
        admitted=ObservationalSciPlexTask.from_receipt(task,r)
        for group,values in task.project().items():
            for key,value in values.items():
                assert torch.equal(value,admitted.project(for_training=True)[group][key])
        assert admitted.input_metadata()['policy']==r['policy']
        assert 'task_sha256' not in admitted.input_metadata()
        restored=ObservationalSciPlexTask.from_bytes(admitted.canonical_bytes())
        admitted.assert_resume_matches(restored)


@pytest.mark.parametrize('change',[dict(split='test'),dict(policy='physical-dose'),
    dict(source_sha256='c'*64),dict(task_sha256='d'*64),dict(audit_sha256='bad'),
    dict(leakage_group='cell:0'),dict(version=2),dict(extra=True)])
def test_fail_closed_receipts(change):
    task,r=fixture();r.update(change)
    with pytest.raises(ValueError): ObservationalSciPlexTask.from_receipt(task,r)


def test_hidden_target_qc_identity_do_not_change_inputs_but_resume_rejects():
    a,ra=fixture();b,rb=fixture(counts=[100,200],cell_id='hidden',n_counts=999)
    x=ObservationalSciPlexTask.from_receipt(a,ra); y=ObservationalSciPlexTask.from_receipt(b,rb)
    assert x.input_metadata()==y.input_metadata()
    for k,v in x.project()['inputs'].items(): assert torch.equal(v,y.project()['inputs'][k])
    with pytest.raises(ValueError): x.assert_resume_matches(y)
    with pytest.raises(ValueError): ObservationalSciPlexTask.from_receipt(b,ra)


def test_receipt_snapshot_and_cpu_cuda_projection():
    t,r=fixture();a=ObservationalSciPlexTask.from_receipt(t,r)
    r['split']='test'
    assert a.input_metadata()['split']=='train_only'
    if torch.cuda.is_available():
        for group,values in a.project(for_training=True).items():
            for key,v in values.items(): assert torch.equal(v,a.project(device='cuda',for_training=True)[group][key].cpu())
