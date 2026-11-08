"""NEW approval-gated index launcher; frozen original export controller reused read-only."""
from __future__ import annotations
import argparse,json,os
from pathlib import Path
from datetime import datetime,timezone
import data_sources_xpert_export_run as frozen
import data_sources_xpert_row_index_phase as phase
import data_sources_xpert_row_index as core
from data_sources_hardware_storage import load_policy,hardware_snapshot
from data_sources_mint_full_resources import pressure_snapshot
ACTION="XPERT_Norman_original_ordinal_row_index_v1"

def required_code_paths():
    b=Path(__file__).resolve().parent
    return {str(b/n) for n in ("data_sources_xpert_row_index.py","data_sources_xpert_row_index_phase.py",
      "data_sources_xpert_row_index_run.py","data_sources_xpert_arrays.py","data_sources_xpert_export_run.py",
      "data_sources_array_export_guard.py","data_sources_hardware_storage.py","data_sources_mint_full_resources.py")}
def validate_code_map(p):
    expected=required_code_paths()
    if set(p["code_sha256"])!=expected or not expected<=set(p["bindings"]):
        raise ValueError("exact complete index/imported code map required")
    if any(p["code_sha256"][k]!=p["bindings"][k] for k in expected):raise ValueError("code bindings differ")
def validate_index_approval(path,expected_sha,plan_path,p):
    if p["action"]!=ACTION:raise ValueError("exact index action required")
    return frozen.validate_approval(path,expected_sha,plan_path,p)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--plan",required=True);ap.add_argument("--approval",required=True);ap.add_argument("--approval-sha256",required=True)
    a=ap.parse_args();frozen.require_isolated_wrapper();pp=Path(a.plan).resolve();p=json.loads(pp.read_text())
    validate_index_approval(a.approval,a.approval_sha256,pp,p)
    if p.get("fixture_mode") is not False:raise ValueError("production launcher refuses fixture")
    load_policy(p["execution_policy"],p["execution_policy_sha256"]);frozen.assert_bindings(p["bindings"]);validate_code_map(p)
    frozen.validate_resource_envelope(p["resources"]);phase.validate_source_plan(p)
    phase.validate_toolchain(p)
    frozen.assert_no_writers([Path(__file__).resolve(),p["phase_script"],Path(frozen.__file__).resolve(),
       Path(__file__).resolve().parent/"data_sources_xpert_export_phase.py"])
    with core.Source(p["source_spec"]) as src:src.check()
    r=p["resources"];space=hardware_snapshot(Path(p["new_output"]).parent,r["minimum_free_bytes"])
    if space["available_bytes"]-p["expected_payload_bytes"]-p["expected_metadata_bytes_estimate"]<=r["minimum_free_bytes"]:
        raise ValueError("hardware cannot retain proposed index and64GiB floor")
    mem=pressure_snapshot(os.getsid(0))
    if mem["host_mem_available_bytes"]<r["physical_memory_policy"]["min_host_available_bytes"]:
        raise ValueError("host available RAM floor not met")
    lease=frozen.claim_destination(p["new_output"],p["lease_path"],{"utc":datetime.now(timezone.utc).isoformat(),
       "identity":frozen.guard.workflow_identity(os.getpid()),"plan_sha256":frozen.sha(pp),"approval_sha256":a.approval_sha256})
    try:
        frozen.write_new(Path(p["new_output"])/"approved-launch.json",{"utc":datetime.now(timezone.utc).isoformat(),
          "wrapper":frozen.guard.workflow_identity(os.getpid()),"plan_path":str(pp),"plan_sha256":frozen.sha(pp),
          "approval":str(Path(a.approval).resolve()),"approval_sha256":a.approval_sha256,
          "hardware":space,"memory":mem,"source_task_admitted":0,"model_use":False})
        frozen.execute_phases(p,pp,Path(p["new_output"]))
    finally:os.close(lease)
if __name__=="__main__":main()
