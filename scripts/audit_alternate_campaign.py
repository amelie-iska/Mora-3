#!/usr/bin/env python3
"""Read-only campaign accounting, blocker ownership and proposed comparator manifest.

This produces no launch command or approval and does not freeze the coordinator's
train/validation source, architecture widths or baseline implementation.
"""
import argparse,datetime,json,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from mora3.extensions.scri.curation import strict_json,file_digest,regular_file
from mora3.extensions.scri.readiness import build_sweep_manifest,write_new_json,inspect_primary_admission
from mora3.extensions.scri.transfer_audit import recovery_checklist


def pin(path):
    p=regular_file(path,max_bytes=16*1024*1024)
    return dict(bytes=p.stat().st_size,sha256=file_digest(p))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--active-root',required=True);p.add_argument('--previous-extension',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    active=Path(a.active_root).resolve();prior=Path(a.previous_extension).resolve();out=Path(a.output).absolute()
    if out.exists()or any(x.is_symlink()for x in(out,*out.parents)):raise ValueError('new nonsymlink receipt directory required')
    planpath=active/'planning/mup_hyperparameter_transfer_plan.json'
    oldpath=prior/'outputs/scri-20261108/preflight.json'
    plan=strict_json(planpath.read_bytes());old=strict_json(oldpath.read_bytes())
    blocks=[]
    for message in old['blockers']:
        if ': native assets'in message or ': source rights'in message or ': source revision'in message:
            owner='source-owner';kind='original-assets-rights-revision'
        elif ': native source/task admission'in message:
            owner='construction-and-model-owners';kind='task-admission'
        elif 'unconditionally refuses'in message:
            owner='coordinator';kind='native-production-admission'
        elif 'all-head'in message:
            owner='model-owner';kind='full-operator-width-optimizer-qualification'
        elif 'SCRI native'in message:
            owner='alternate-agent-and-coordinator';kind='source-route-registration'
        else:
            owner='construction-model-and-coordinator';kind='same-snapshot-full-readiness'
        blocks.append(dict(original_blocker=message,owner=owner,kind=kind,
            status='not-cleared-by-preparation-report',original_evidence_sha256=pin(oldpath)['sha256']))
    sourcefiles=('src/mora3/model/full_owner.py','src/mora3/model/full_owner_mup.py',
        'src/mora3/model/mu_optimizer.py','src/mora3/research/primary_training.py',
        'src/mora3/research/mup_sweep_selection.py','src/mora3/research/mup_campaign_worker.py',
        'src/mora3/research/primary_native_routes.py')
    sources={name:pin(active/name)for name in sourcefiles}
    grid=build_sweep_manifest(plan,catalog_sha256=file_digest(ROOT/'configs/scri_sources.json'))
    output=dict(schema='Mora3.AlternateCampaignAudit/v1',observed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        plan=pin(planpath),primary_implementation=sources,planned_counts={'primary':24,'baseline':8,'transfer':9,'naive_sp':3},
        primary_trials=grid['primary_trials'],maximum_gpu_hours=22.,
        baseline_proposal=dict(status='UNFROZEN_COORDINATOR_REVIEW_REQUIRED',runs=8,
            learning_rates=['0.00003','0.0001','0.0003','0.001'],seeds=[2026110201,2026110202],
            initialization='standard fan-in initialization, no muP readout rescaling',
            attention='standard inverse-square-root head dimension, not the muP inverse-head-dimension rule',
            optimizer='ordinary AdamW with declared LR and identical fixed decay policy',
            matching='same source train/evaluation commitments, depth, finite axes, objectives and update budgets',
            limitation='eight-trial limited baseline search; current primary update primitive demands exact MuAdamW'),
        transfer_proposal=dict(runs=9,seeds_per_width=3,candidate_seeds=[2026110801,2026110802,2026110803],
            seeds_frozen=False,widths=None,hyperparameters=None,rule='existing native validation-only paired-seed selector'),
        naive_sp_proposal=dict(runs=3,source='SP proxy winner without target retuning',implementation_qualified=False),
        native_selection_rule=dict(tasks='frozen source-family macro normalized native losses',
            normalization='(mean loss - initial reference loss)/max(abs(initial reference loss),minimum normalization scale)',
            seed_aggregation='mean of both fixed seeds; one failed seed makes that HP point ineligible',
            tie_break='total GPU seconds then hyperparameter ID',
            required_artifacts=24,comparison_exposure='same predeclared validation sources, observed denominators and update',
            test_or_target_selection=False),
        recovery_checklist=recovery_checklist(),required_record_formats={
            'planned':'immutable run/trial/arm/source/operator/grid/budget commitment',
            'launched':'actual PID/start/exclusive-owner release and immutable attempt ID',
            'terminal':'actual exit, disposition, logs and checkpoint digests',
            'measured':'validated native source denominators, comparable evaluation, resource measurements',
            'selected':'native complete-24 artifact selector receipt; never a launch release'},
        current_primary_admission=inspect_primary_admission(active/'src/mora3/model/full_owner.py'),
        complete_source_and_owner_commitments_frozen=False,baseline_path_qualified=False,
        full_44_run_qualification_complete=False,launch_authorized=False)
    out.mkdir();write_new_json(out/'campaign-audit.json',output)
    write_new_json(out/'blocker-ownership.json',dict(schema='Mora3.SCRIBlockerOwnership/v1',original_report=pin(oldpath),
        blocker_count=len(blocks),by_owner=dict(Counter(x['owner']for x in blocks)),blockers=blocks,
        new_branch_capability='independent exact spatial incidence with shared transformer; 64-cell CPU diagnostics only',
        active_owner_changed=False,source_admission_changed=False))
    print(json.dumps(dict(blockers=len(blocks),planned_runs=sum(output['planned_counts'].values()),launch_authorized=False)))

if __name__=='__main__':main()
