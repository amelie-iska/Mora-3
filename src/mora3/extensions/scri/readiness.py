"""Read-only project audit and a finite, explicitly gated sweep manifest.

This module prepares evidence for the existing owner. It neither replaces the
primary campaign runner nor promotes ledger claims into runtime admission.
"""
import ast
from collections import Counter
from dataclasses import asdict
from datetime import datetime,timezone
from hashlib import sha256
from itertools import product
import json
from pathlib import Path

from .contracts import canonical
from .curation import file_digest,inspect_catalog,regular_file,strict_json


TASKS=('identity_state','state_transition','perturbation_response','spatial_neighborhood',
       'temporal_trajectory','bioelectric_state','functional_phenotype','localized_inverse_design')


def _read(path,max_bytes=8*1024*1024):
    return strict_json(regular_file(path,max_bytes=max_bytes).read_bytes())


def _optional(path):
    return _read(path) if path.exists() else None


def inspect_primary_admission(path):
    """Report the actual current implementation; never import or patch it."""
    path=Path(path)
    if not path.is_file(): return dict(status='missing',ready=False)
    raw=regular_file(path,max_bytes=2*1024*1024).read_bytes();tree=ast.parse(raw)
    for node in tree.body:
        if isinstance(node,ast.ClassDef) and node.name=='SourceBoundOwner':
            for method in node.body:
                if isinstance(method,ast.FunctionDef) and method.name=='require_training_admission':
                    body=[x for x in method.body if not isinstance(x,ast.Expr) or not isinstance(x.value,ast.Constant) or not isinstance(x.value.value,str)]
                    closed=bool(body) and isinstance(body[0],ast.Raise)
                    return dict(status='unconditional_refusal' if closed else 'runtime_verification_required',
                                ready=False,line=method.lineno,source_sha256=sha256(raw).hexdigest())
    return dict(status='admission_method_missing',ready=False,source_sha256=sha256(raw).hexdigest())


def audit_workspace(root, *, include_inventory=False):
    root=Path(root).resolve();planning=root/'planning'
    if not planning.is_dir(): raise ValueError('explicit active workspace with planning/ required')
    inventory=[];sizes=Counter();formats=Counter();parse_failures=[]
    for path in sorted(planning.rglob('*')):
        if path.is_symlink() or not path.is_file(): continue
        # The immutable private original specification is represented by metadata
        # only in this report; its content is never exported or committed here.
        rel=str(path.relative_to(root));size=path.stat().st_size
        if size>16*1024*1024:
            inventory.append(dict(path=rel,bytes=size,review='oversize-not-read'));continue
        raw=path.read_bytes();record=dict(path=rel,bytes=len(raw),sha256=sha256(raw).hexdigest())
        formats[path.suffix]+=1;sizes[path.suffix]+=len(raw)
        if path.suffix=='.json':
            try:
                value=strict_json(raw)
                if type(value) is dict:
                    record['schema']=value.get('schema')
                    record['status']=str(value.get('status',''))[:240]
                    record['keys']=len(value)
                record['review']='parsed-structure-not-semantic-approval'
            except (ValueError,UnicodeError) as exc:
                record['review']='parse-failed';parse_failures.append(dict(path=rel,error=type(exc).__name__))
        else: record['review']='content-fingerprinted'
        inventory.append(record)
    progress=_optional(planning/'progress.json') or {}
    intake=_optional(planning/'data_intake_progress.json') or {}
    sources=_optional(planning/'data_source_expansion_progress.json') or {}
    coordination=_optional(planning/'data_worker_coordination.json') or {}
    plan=_optional(planning/'mup_hyperparameter_transfer_plan.json') or {}
    report=dict(schema='Mora3.SCRIActiveWorkspaceAudit/v1',observed_at_utc=datetime.now(timezone.utc).isoformat(),
        active_root=str(root),planning_file_count=len(inventory),planning_bytes=sum(x['bytes'] for x in inventory),
        file_formats=dict(formats),format_bytes=dict(sizes),json_parse_failures=parse_failures,
        current_model_task=progress.get('current_task'),completed_ledger_entries=len(progress.get('completed',[])),
        corpus_counts=intake.get('current_corpus_counts'),corpus_manifest=intake.get('current_corpus_manifest'),
        corpus_manifest_sha256=intake.get('current_corpus_manifest_sha256'),
        construction_status=intake.get('status'),source_status=sources.get('status'),
        full_dataset_training_gate=intake.get('full_dataset_training_gate'),
        full_dataset_ready=coordination.get('full_dataset_ready'),
        current_muP_campaign_gate=intake.get('current_muP_campaign_gate'),
        plan_status=plan.get('status'),plan_gates=plan.get('gates'),
        primary_admission=inspect_primary_admission(root/'src/mora3/model/full_owner.py'),
        inventory_semantics='Inventory/JSON parsing is not a claim that every historical artifact was scientifically reviewed.',
        actual_sweep_started_by_this_extension=False)
    if include_inventory: report['planning_inventory']=inventory
    return report


def build_sweep_manifest(plan, *, catalog_sha256,task_names=TASKS):
    """Expand the agreed grid without choosing target HPs or claiming readiness."""
    if (type(plan) is not dict or plan.get('schema')!='mora3.mup_hyperparameter_transfer.USER_plan.v1'
            or tuple(task_names)!=TASKS):
        raise ValueError('exact existing user plan and complete SCRI objective family required')
    from .contracts import digest
    digest(catalog_sha256,'catalog SHA')
    proxy=plan['proposed_primary_sweep'];transfer=plan['transfer'];baseline=plan['proposed_baseline']
    if (proxy['master_learning_rates']!=[0.0001,0.0003,0.001]
            or proxy['readout_multipliers']!=[0.5,1]
            or proxy['ordinary_attention_key_multipliers']!=[0.5,1]
            or proxy['seeds']!=[2026110201,2026110202] or proxy['runs']!=24):
        raise ValueError('existing registered proxy grid changed; explicit reconciliation required')
    if (proxy['max_updates_each']!=5000 or proxy['max_seconds_each']!=1800 or proxy['max_gpu_hours']!=12
            or baseline['proxy_runs']!=8 or baseline['max_gpu_hours']!=4
            or transfer['seeds_per_width']!=3 or transfer['max_updates_each']!=1000
            or transfer['max_seconds_each']!=1800 or transfer['max_gpu_hours']!=4.5
            or transfer['naive_SP_target_runs']!=3 or transfer['naive_SP_max_gpu_hours']!=1.5):
        raise ValueError('finite registered campaign budget changed')
    trials=[]
    for n,(lr,readout,attention) in enumerate(product(proxy['master_learning_rates'],proxy['readout_multipliers'],proxy['ordinary_attention_key_multipliers']),1):
        for seed in proxy['seeds']:
            trials.append(dict(trial_id=f'hp-{n:02d}-{seed}',hp_id=f'hp-{n:02d}',seed=seed,
                learning_rate=str(lr),readout_multiplier=str(readout),attention_key_multiplier=str(attention),
                max_updates=5000,max_seconds=1800))
    return dict(schema='Mora3.SCRIExtendedSweepPreparation/v1',source_catalog_sha256=catalog_sha256,
        original_plan_sha256=sha256(canonical(plan)).hexdigest(),required_task_families=list(TASKS),
        primary_trials=trials,baseline_proxy_runs=8,transfer_runs=9,naive_sp_target_runs=3,
        total_planned_runs=44,maximum_gpu_hours=22.0,
        widths=None,all_head_parameter_counts=None,selected_transfer_hyperparameters=None,
        ready_to_run=False,optimizer_execution_authorized=False,
        native_runner='mora3.research.mup_campaign_worker.execute_one_primary_proxy_trial',
        selection='mora3.research.mup_sweep_selection.select_proxy_transfer',
        missing_bindings=['complete native source and split admission','all required source-bound heads and routes',
            'measured full-owner width family and alias ledger','coordinate-check qualification for every active operator',
            'matching validation cohort and native target denominators','recovery and bounded hardware evidence',
            'existing exclusive worker admission','baseline trial specifications and transfer widths/seeds'],
        selection_uses_test=False,target_retuning_under_zero_shot_label=False,
        synthetic_fixture_results_count_as_data_readiness=False)


def preflight(active_root,catalog_path,*,tasmorpho_revision=None):
    audit=audit_workspace(active_root);catalog=inspect_catalog(catalog_path)
    blockers=list(catalog['blockers'])
    gate=audit.get('full_dataset_training_gate')
    if not isinstance(gate,dict) or gate.get('ready') is not True:
        blockers.append('active full-dataset training gate is not ready')
    if audit['full_dataset_ready'] is not True:
        blockers.append('active coordinated expanded-dataset readiness is not COMPLETE')
    admission=audit['primary_admission']['status']
    if admission=='unconditional_refusal':
        blockers.append('SourceBoundOwner.require_training_admission unconditionally refuses training')
    else:
        blockers.append('actual owner/operator/recovery/runtime admission still requires the existing native runner')
    if tasmorpho_revision is None:
        blockers.append('tasmorpho code/export revision not supplied')
    else:
        from .tasmorpho import SUPPORTED_SOURCE_COMMIT
        if tasmorpho_revision!=SUPPORTED_SOURCE_COMMIT:
            blockers.append('unreviewed tasmorpho revision; adapter compatibility review required')
    blockers.extend(['SCRI native datasets, axes and routes are not admitted by this preparation report',
                     'all-head width/coordinate/optimizer qualification remains an active-owner gate'])
    return dict(schema='Mora3.SCRIPreflight/v1',observed_at_utc=audit['observed_at_utc'],
        ready_to_run=False,status='BLOCKED',blockers=blockers,
        active_workspace=audit,source_catalog=catalog,tasmorpho_revision=tasmorpho_revision,
        source_mutations=0,optimizer_updates=0,sweep_launches=0)


def write_new_json(path,value):
    p=Path(path).absolute()
    if any(x.is_symlink() for x in (p,*p.parents)):
        raise ValueError('output cannot traverse symlinks')
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as stream: stream.write(canonical(value)+b'\n')
    return p
