#!/usr/bin/env python3
"""Finite SCRI audit, source curation and sweep-preparation commands."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from mora3.extensions.scri.curation import read_native_jsonl,read_long_cell_csv,curate_records,inspect_catalog
from mora3.extensions.scri.readiness import audit_workspace,build_sweep_manifest,preflight,write_new_json


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);subs=parser.add_subparsers(dest='command',required=True)
    audit=subs.add_parser('audit');audit.add_argument('--active-root',required=True)
    audit.add_argument('--output',required=True);audit.add_argument('--inventory',action='store_true')
    catalog=subs.add_parser('catalog');catalog.add_argument('--catalog',required=True);catalog.add_argument('--output',required=True)
    curate=subs.add_parser('curate');curate.add_argument('--input',required=True);curate.add_argument('--output-directory',required=True)
    curate.add_argument('--format',choices=('jsonl','long-csv'),default='jsonl')
    curate.add_argument('--holdout-axes',nargs='+',default=['study','donor']);curate.add_argument('--seed',type=int,default=20261108)
    for arg in ('dataset-id','revision','license-id','evidence-kind'):curate.add_argument('--'+arg)
    plan=subs.add_parser('plan');plan.add_argument('--active-root',required=True);plan.add_argument('--catalog',required=True);plan.add_argument('--output',required=True)
    ready=subs.add_parser('preflight');ready.add_argument('--active-root',required=True);ready.add_argument('--catalog',required=True)
    ready.add_argument('--tasmorpho-revision');ready.add_argument('--output',required=True)
    tas=subs.add_parser('tasmorpho-summary');tas.add_argument('--input',required=True)
    tas.add_argument('--source-commit',required=True);tas.add_argument('--run-id',required=True)
    tas.add_argument('--expected-sha256',required=True);tas.add_argument('--output',required=True)
    args=parser.parse_args(argv)
    if args.command=='curate':
        if args.format=='long-csv':
            if any(getattr(args,name) is None for name in ('dataset_id','revision','license_id','evidence_kind')):
                parser.error('long-csv requires explicit dataset, revision, license, and evidence-kind')
            rows=read_long_cell_csv(args.input,dataset_id=args.dataset_id,revision=args.revision,
                license_id=args.license_id,evidence_kind=args.evidence_kind)
        else: rows=read_native_jsonl(args.input)
        result=curate_records(rows,source_path=args.input,output_directory=args.output_directory,
            holdout_axes=tuple(args.holdout_axes),seed=args.seed)
    elif args.command=='audit': result=audit_workspace(args.active_root,include_inventory=args.inventory)
    elif args.command=='catalog': result=inspect_catalog(args.catalog)
    elif args.command=='plan':
        from mora3.extensions.scri.curation import strict_json,regular_file
        path=regular_file(Path(args.active_root)/'planning/mup_hyperparameter_transfer_plan.json',max_bytes=2*1024*1024)
        doc=strict_json(path.read_bytes())
        result=build_sweep_manifest(doc,catalog_sha256=inspect_catalog(args.catalog)['catalog_sha256'])
    elif args.command=='tasmorpho-summary':
        from mora3.extensions.scri.tasmorpho import load_summary
        result=load_summary(args.input,source_commit=args.source_commit,run_id=args.run_id,
                            expected_sha256=args.expected_sha256).wire()
    else: result=preflight(args.active_root,args.catalog,tasmorpho_revision=args.tasmorpho_revision)
    if args.command!='curate': write_new_json(args.output,result)
    print(json.dumps({k:v for k,v in result.items() if k in ('schema','status','ready','ready_to_run',
        'candidates','record_count','split_counts','planning_file_count','total_planned_runs','maximum_gpu_hours',
        'evidence_kind','observation_scope','training_admitted')},indent=2))
    return 2 if args.command=='preflight' and not result['ready_to_run'] else 0


if __name__=='__main__': raise SystemExit(main())
