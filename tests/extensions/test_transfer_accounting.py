"""Partitioned independent qualification checks; no training admission."""
import pytest
from mora3.extensions.scri.transfer_audit import comparison_design,check_run_states,recovery_checklist


def test_exact_44_runs_and_no_guessed_transfer_settings():
    plan=comparison_design(baseline_lrs=('0.00003','0.0001','0.0003','0.001'),
        baseline_seeds=(2026110201,2026110202),transfer_widths=(128,256,512),
        transfer_seeds=(2026110801,2026110802,2026110803),train_sha256='a'*64,validation_sha256='b'*64)
    assert len(plan['runs'])==44 and len({r['run_id'] for r in plan['runs']})==44
    assert plan['counts']=={'primary':24,'baseline':8,'transfer':9,'naive_sp':3}
    assert plan['launch_authorized'] is False
    assert all(r['hyperparameters'] is None for r in plan['runs'] if r['arm']=='transfer')
    with pytest.raises(ValueError):comparison_design(baseline_lrs=('0.001',)*4,baseline_seeds=(1,2),
        transfer_widths=(2,4,8),transfer_seeds=(1,2,3),train_sha256='a'*64,validation_sha256='b'*64)


def test_launched_terminal_measured_selected_are_not_interchangeable():
    assert check_run_states(launched=True,terminal=False,measured=False,selected=False,artifact_sha256=None)['state']=='launched'
    for args in ((False,False,True,False),(True,False,False,True),(True,True,True,True)):
        with pytest.raises(ValueError):check_run_states(launched=args[0],terminal=args[1],measured=args[2],selected=args[3],artifact_sha256=None)
    assert 'exact_model_and_optimizer_aliases' in recovery_checklist()
