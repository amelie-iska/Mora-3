#!/usr/bin/env python3
"""Prepare an immutable tasmorpho replay packet without unsafe upstream execution.

The audited upstream main deletes its output directory and deserializes dill.
This preparer does neither. Missing engine/version/seed authority produces a
blocked receipt, not a fabricated regeneration result or an automatic install.
"""
import argparse,datetime,hashlib,json,shutil,subprocess,sys
from dataclasses import asdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from mora3.extensions.scri.readiness import write_new_json
from mora3.extensions.scri.curation import file_digest,regular_file
from mora3.extensions.scri.simulation_contracts import installed_simulation_readiness,tasmorpho_interventions
from mora3.extensions.scri.tasmorpho import load_summary,SUPPORTED_SOURCE_COMMIT,CASES

SUMMARY_SHA='657aec89b9f828c355dc7af968a38bd3e1332accdc18f71d512b9d0e18552734'
BETSEE_REVISION='5cbe6d5f2d06f0752661d909f2eb0a32c35064d1'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    ref=Path(a.reference).resolve();out=Path(a.output).absolute()
    if out.exists()or any(x.is_symlink()for x in(out,*out.parents)):raise ValueError('fresh nonsymlink replay directory required')
    if shutil.disk_usage(out.parent).free<200*1024**3+16*1024**2:raise ValueError('200 GiB disk floor')
    head=subprocess.check_output(['git','--no-optional-locks','-C',str(ref),'rev-parse','HEAD'],text=True).strip()
    if head!=SUPPORTED_SOURCE_COMMIT:raise ValueError('unreviewed tasmorpho revision')
    names=['LICENSE','scripts/01_betse_local_grounding.py','scripts/betse_tas_example_run/betse_summary.csv']
    names+=['scripts/betse_tas_example_run/demo/'+('minimal_control.yml'if c=='control'else f'case_{c}.yml')for c in CASES]
    pins=[]
    for name in names:
        path=regular_file(ref/name,max_bytes=1024*1024);raw=path.read_bytes()
        committed=subprocess.check_output(['git','--no-optional-locks','-C',str(ref),'show',head+':'+name])
        if raw!=committed:raise ValueError('pinned template/source working bytes differ from original commit')
        pins.append(dict(path=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
    source=ref/'scripts/betse_tas_example_run/betse_summary.csv'
    summary=load_summary(source,source_commit=head,run_id='committed-example',expected_sha256=SUMMARY_SHA).wire()
    out.mkdir()
    rows=[dict(case=c,source_revision=head,source_sha256=SUMMARY_SHA,scope='aggregate-simulation-case',
        evidence_kind='simulation',declared_interventions=[asdict(x)for x in tasmorpho_interventions(c)],
        independently_observed_experimental_support=None,biological_system=None,
        uncertainty=None,engine_version=None,seed=None,training_admitted=False,
        control_arm='control',declared_simulation_window_seconds=[0.,.02],
        native_response_units=summary['native_units'],
        native_summary=summary['cases'][i])for i,c in enumerate(CASES)]
    write_new_json(out/'simulation-candidates.json',dict(schema='Mora3.TasmorphoTypedCandidates/v1',cases=rows,
        candidate_rows=7,independent_simulation_runs=1,newly_admitted_rows=0,
        split_policy='one shared-control lineage; no case-level random split',
        retry_policy='no automatic retries; preserve failures and every requested seed/configuration',
        conditioning_policy='declared intervention/context only; summary responses are targets, not inputs'))
    installed=installed_simulation_readiness()
    receipt=dict(schema='Mora3.TasmorphoReplayPreparation/v1',observed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_revision=head,inputs=pins,source_template_bytes_verified=True,installed=installed,
        tasmorpho_software_license='MIT; copyright 2026 Marcel Blattner @ btec.ai; retain notice',
        betsee_fork=dict(repository='amelie-iska/betsee',revision=BETSEE_REVISION,license='BSD-2-Clause',
            license_git_blob='569485840ff49f2332c02f82cf0bd3ac6c493e95',
            scope='GUI code rights only; engine, input and calibration provenance are separate'),
        fresh_output_directory=True,simulations_executed=0,exact_example_regenerated=False,
        source_run_redistribution_rights_fully_qualified=False,experimental_rows=0,
        blocker='engine absent or exact historical engine/seed binding unavailable; installation not performed',
        next_gate='coordinator-approved engine availability and versioned seeded export adapter; new simulation and independent comparison',
        launch_authorized=False,status='BLOCKED',
        output_sha256=file_digest(out/'simulation-candidates.json'))
    write_new_json(out/'receipt.json',receipt)
    print(json.dumps({'status':'BLOCKED','candidate_rows':7,'simulations_executed':0,'engine_available':installed['engine_available']}))
    return 2

if __name__=='__main__':raise SystemExit(main())
