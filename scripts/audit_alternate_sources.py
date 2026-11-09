#!/usr/bin/env python3
"""Read-only independent source qualification; writes new receipts, never data."""
import argparse,ast,datetime,hashlib,json,os,resource,shutil,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from mora3.extensions.scri.curation import strict_json,file_digest,regular_file
from mora3.extensions.scri.readiness import write_new_json,inspect_primary_admission
from mora3.extensions.scri.grh_intake import GRHRelease,REVISION


def pin(path):
    p=regular_file(path,max_bytes=8*1024*1024)
    return dict(bytes=p.stat().st_size,sha256=file_digest(p))


def reconciliation(active):
    witness=active/'planning/research/data_intake/expanded-readiness-current-witness-reconciliation-v1/CURRENT-owner-witness-and-gap-delta.json'
    w=strict_json(witness.read_bytes())
    facts={r['id']:r for r in w['unmet']}
    mapping={'X-Pert':'XPERT','small-molecule-retrosynthesis':'RETRO_SMALL_MOLECULE',
        'peptide-retrosynthesis':'RETRO_PEPTIDE','MINT':'FULL_MINT_COMPONENTS','ProCyon':'PROCYON',
        'scCello':'SCCELLO','LINCS':'LINCS_WHOLE4','ADMET':'ADMET_FULL_SCOPE'}
    lanes=[]
    for name,identity in mapping.items():
        item=facts[identity];evidence=[]
        for ref in item['evidence']:
            p=Path(ref['path']);p=p if p.is_absolute()else active/p
            actual=pin(p)
            evidence.append(dict(sha256=actual['sha256'],bytes=actual['bytes'],matches_recorded=actual['sha256']==ref['sha256']and actual['bytes']==ref['bytes']))
        lanes.append(dict(lane=name,required_complete_denominator=None,eligible_numerator=None,
            new_admissions_by_this_package=0,prior_authority='pinned existing expanded witness, not a new source census',
            existing_evidence=evidence,unresolved=item['fact'],rights_groups_masks_qualified=False))
    for name,rel in [('transition-state-enzyme','transition_state_enzyme_design_contract.json'),
                     ('GRH','grh_conformer_response_data_model_contract.json'),
                     ('UMA-eSEN','uma_esen_training_contract-20261106-v1.json')]:
        p=active/'planning'/rel
        lanes.append(dict(lane=name,contract=pin(p),required_complete_denominator=None,eligible_numerator=None,
            new_admissions_by_this_package=0,scope='existing source/model owners retained; no shared state changes',
            unresolved='original source/method/frame/units/roles, complete groups, masks and same-snapshot native admission'))
    state=strict_json((active/'planning/data_worker_coordination.json').read_bytes())
    intake=strict_json((active/'planning/data_intake_progress.json').read_bytes())
    return dict(schema='Mora3.AlternateExpandedLaneReconciliation/v1',witness=pin(witness),lanes=lanes,
        full_dataset_ready=state.get('full_dataset_ready'),
        construction_owner_signed=intake.get('full_dataset_training_gate',{}).get('construction_owner_signature')is not None,
        model_owner_signed=intake.get('full_dataset_training_gate',{}).get('model_owner_signature')is not None,
        primary_admission=inspect_primary_admission(active/'src/mora3/model/full_owner.py'),
        shared_ledgers_modified=False,restricted_coordinates_or_payloads_exported=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--active-root',required=True);p.add_argument('--output',required=True);p.add_argument('--grh',action='store_true');a=p.parse_args()
    active=Path(a.active_root).resolve();out=Path(a.output).absolute()
    if out.exists()or any(x.is_symlink()for x in(out,*out.parents)):raise ValueError('new nonsymlink output directory required')
    if shutil.disk_usage(out.parent).free<200*1024**3+32*1024**2:raise ValueError('200 GiB floor')
    out.mkdir();t=time.monotonic()
    rec=reconciliation(active);write_new_json(out/'lane-reconciliation.json',rec)
    if a.grh:
        source=active/'data/raw/GRH-Mora-ConformerResponse-20261104'/REVISION
        result=GRHRelease(source).audit();write_new_json(out/'grh-candidate-audit.json',result)
        print(json.dumps({k:result[k]for k in('source_rows','native_mmff94_ef_candidates','other_method_or_dynamic_candidates','rejected_schema_rows','declared_group_cross_split_conflicts')}))
    write_new_json(out/'receipt.json',dict(schema='Mora3.AlternateSourceExecution/v1',
        observed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-t,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,optimizer_updates=0,
        source_mutations=0,source_downloads=0,trained_rows=0,
        output_pins={p.name:pin(p)for p in out.glob('*.json')},source_verification_is_not_admission=True))
    print('READ_ONLY_SOURCE_AUDIT_COMPLETE')

if __name__=='__main__':main()
