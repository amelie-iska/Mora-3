"""Independent 44-run accounting and recovery requirements; never a launcher.

Uses the primary owner's existing validation-only selector. The standard-
parameterization comparator still requires a separately qualified native engine;
relabeling a MuAdamW run as SP is not an acceptable baseline.
"""
from decimal import Decimal
from hashlib import sha256
from itertools import product
from .contracts import canonical,digest
from mora3.research.mup_sweep_selection import planned_proxy_trials,select_proxy_transfer


def _seeds(values,count):
    if type(values)is not tuple or len(values)!=count or len(set(values))!=count or any(type(v)is not int or v<0 for v in values):
        raise ValueError('exact distinct nonnegative run seeds required')


def comparison_design(*,baseline_lrs,baseline_seeds,transfer_widths,transfer_seeds,train_sha256,validation_sha256):
    """Freeze caller choices structurally; scientific admissions remain external."""
    _seeds(baseline_seeds,2);_seeds(transfer_seeds,3)
    digest(train_sha256,'train commitment');digest(validation_sha256,'validation commitment')
    if train_sha256==validation_sha256:raise ValueError('train and validation commitments must be distinct')
    if (type(transfer_widths)is not tuple or len(transfer_widths)!=3
            or any(type(w)is not int or w<2 or w%2 for w in transfer_widths)
            or not transfer_widths[0]<transfer_widths[1]<transfer_widths[2]):
        raise ValueError('three increasing explicit even transfer widths required')
    if type(baseline_lrs)is not tuple or len(baseline_lrs)!=4 or any(type(x)is not str for x in baseline_lrs):
        raise ValueError('four explicit decimal-string baseline learning rates required')
    rates=tuple(Decimal(x)for x in baseline_lrs)
    if any(not x.is_finite()or x<=0 for x in rates)or len(set(rates))!=4:
        raise ValueError('four unique positive finite baseline learning rates required')
    runs=[]
    for t in planned_proxy_trials():
        runs.append(dict(run_id=t.trial_id,arm='primary',seed=t.seed,width=None,
            hyperparameters=dict(lr=str(t.learning_rate),readout=str(t.readout_multiplier),attention=str(t.attention_key_multiplier)),
            max_updates=5000,max_seconds=1800))
    for i,(lr,seed)in enumerate(product(rates,baseline_seeds)):
        runs.append(dict(run_id=f'sp-proxy-{i:02d}',arm='baseline',seed=seed,width=None,
            hyperparameters=dict(lr=str(lr),readout='standard-unscaled',attention='inverse-square-root-head-dimension'),
            max_updates=5000,max_seconds=1800))
    for w,seed in product(transfer_widths,transfer_seeds):
        runs.append(dict(run_id=f'mup-transfer-{w}-{seed}',arm='transfer',width=w,seed=seed,
            hyperparameters=None,hyperparameter_source='validated-primary-selector-result',max_updates=1000,max_seconds=1800))
    for seed in transfer_seeds:
        runs.append(dict(run_id=f'sp-target-{transfer_widths[-1]}-{seed}',arm='naive_sp',width=transfer_widths[-1],seed=seed,
            hyperparameters=None,hyperparameter_source='separate-SP-proxy-validation-selection-no-target-tuning',
            max_updates=1000,max_seconds=1800))
    return dict(schema='Mora3.AlternateComparisonDesign/v1',counts={'primary':24,'baseline':8,'transfer':9,'naive_sp':3},
        runs=runs,train_sha256=train_sha256,validation_sha256=validation_sha256,maximum_gpu_hours=22.,
        validation_metric='predeclared source-family macro normalized native loss; exact per-task observed denominators',
        primary_selection='native select_proxy_transfer: seed-mean normalized validation, then GPU time, then HP ID',
        baseline_definition='same data/depth/finite axes/objectives; standard initialization/readout/attention and AdamW LR; limited 8-run search',
        transfer_hyperparameters_selected=False,baseline_implementation_qualified=False,
        source_snapshot_admitted=False,launch_authorized=False)


def check_run_states(*,launched,terminal,measured,selected,artifact_sha256,selection_sha256=None):
    flags=(launched,terminal,measured,selected)
    if any(type(x)is not bool for x in flags):raise ValueError('exact run-state booleans required')
    if terminal and not launched or measured and not terminal or selected and not measured:
        raise ValueError('launched, terminal, measured and selected are distinct ordered states')
    if terminal or measured:
        digest(artifact_sha256,'terminal artifact')
    elif artifact_sha256 is not None:
        raise ValueError('an unclosed attempt cannot carry a terminal artifact')
    if selected:digest(selection_sha256,'validation selector receipt')
    elif selection_sha256 is not None:raise ValueError('selection receipt without selection')
    return dict(state='selected'if selected else'measured'if measured else'terminal'if terminal else'launched'if launched else'planned',
        launch_authorized=False)


def recovery_checklist():
    return ('exact_train_and_validation_commitments','all_source_rights_groups_and_masks',
        'complete_source_native_axes_and_operator_policies','exact_model_and_optimizer_aliases',
        'initialization_readout_attention_and_decay_scaling','per_width_numerical_coordinate_witnesses',
        'model_optimizer_scheduler_rng_and_reader_cursor_state','fresh_process_resume_equivalence',
        'atomic_checkpoint_publication_and_partial_write_refusal','missing_or_corrupted_alias_preflight',
        'finite_loss_gradient_and_oom_failure_dispositions','same_comparison_update_and_observed_denominators',
        'measured_cpu_ram_gpu_disk_throughput_and_context_bounds','exclusive_existing_worker_release',
        'immutable_run_trial_attempt_and_segment_identifiers','all_44_terminal_dispositions_and_artifact_digests',
        'validation_only_selection_no_target_retuning','all_metrics_and_failures_retained_offline',
        'distinct_same_snapshot_construction_and_model_attestations')


def audit_optimizer_groups(model,family,roles,*,base_lr=0.0003,weight_decay=0.01,member='target'):
    """Independently recompute every expected LR/decay ratio; no updates."""
    from mora3.model.mu_optimizer import build_mu_adamw_groups
    names=tuple(n for n,p in model.named_parameters(remove_duplicate=False)if p.requires_grad)
    groups=build_mu_adamw_groups(model,family,roles,base_lr=base_lr,weight_decay=weight_decay,
        decay_aliases=names,member=member,decoupled_wd=False)
    base={e.aliases:e for e in family.base.parameters};current=getattr(family,member)
    expected={}
    for e in current.parameters:
        if not e.trainable:continue
        idx=[i for i,a in enumerate(e.axes)if a.kind=='width']
        ratio=e.shape[idx[-1]]/base[e.aliases].shape[idx[-1]] if len(idx)==2 else 1.
        expected[e.aliases]=(base_lr/ratio,weight_decay*ratio)
    if len({id(g['params'][0])for g in groups})!=len(groups)or len(groups)!=len(expected):
        raise ValueError('optimizer unique owner coverage mismatch')
    for g in groups:
        lr,wd=expected[g['mup_aliases']]
        if g['lr']!=lr or g['weight_decay']!=wd:raise ValueError('reference optimizer LR/decay mismatch')
    return dict(groups=len(groups),all_ratios_match=True,optimizer_updates=0,coordinate_check_passed=False)


def replay_primary_selection(contract,outcomes):
    """Use the existing owner selector, retaining its complete terminal/failure rules."""
    result=select_proxy_transfer(contract,outcomes)
    if result['target_validation_or_test_used'] or result['launch_authorized']:
        raise ValueError('selector scope violation')
    return result
