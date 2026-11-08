"""NEW numeric-only producer/verifier phases for original-offset row access."""
from __future__ import annotations
import argparse,json,os,sys,time,mmap
from pathlib import Path
import numpy as np
import data_sources_xpert_arrays as native
import data_sources_xpert_row_index as core
from data_sources_xpert_row_controller_v2 import write_new,sha,assert_bindings

def canonical_spec(store):
    datasets=[x for x in store["datasets"] if x["matrix_encoding"]=="csc_matrix"]
    if len(datasets)!=1:raise ValueError("one exact native CSC source required")
    d=datasets[0]
    return {"logical_shape":d["logical_shape"],"arrays":{k:store["array_transports"][v] for k,v in d["arrays"].items()},"fixture_only":False},d

def validate_source_plan(p):
    assert_bindings(p["bindings"])
    if p.get("required_kernel_page_bytes",4096)!=4096 or mmap.PAGESIZE!=4096:
        raise ValueError("exact tested4KiB kernel page size required")
    spec=p["source_spec"]
    if p.get("fixture_mode") is True:
        root=Path(p["fixture_root"]).resolve()
        paths=[Path(v["transport"][k]) for v in spec["arrays"].values() for k in ("path","validity_path")]
        if (any(not x.resolve().is_relative_to(root) for x in paths)
            or not Path(p["new_output"]).resolve().is_relative_to(root)
            or spec["arrays"]["indices"]["expected_native"]["native_shape"][0]>100000):
            raise ValueError("tiny fixture cannot access original protected sources or full arrays")
    elif p.get("fixture_mode") is False:
        ref=p["source_store_ref"]
        if sha(ref["path"])!=ref["sha256"]:raise ValueError("exact original numeric store SHA changed")
        store=json.loads(Path(ref["path"]).read_text());expected,d=canonical_spec(store)
        if (spec!=expected or p["source_dataset"]!=d or d["logical_shape"]!=[111445,33694]
            or d["native_stored_matrix_values"]!=361582621 or p["block_elements"]!=65536):
            raise ValueError("source CSC/full native axes or original offset schema differs from verified store")
        if (store["schema"]!="mora3.xpert.numeric-array-store.v1" or store["full_source_export"] is not True
            or store["task_admitted_records"]!=0 or store["model_use_allowed"] is not False):
            raise ValueError("exact completed source evidence/closed admission required")
    else:raise ValueError("explicit fixture/full mode required")
    return spec

def validate_plan(p,stage):
    stage=Path(stage)
    if stage.is_symlink() or not stage.is_dir() or stage.resolve()!=Path(p["new_output"]).resolve():
        raise ValueError("exact regular source-owned stage required")
    return validate_source_plan(p)

def validate_toolchain(p):
    if p.get("fixture_mode") is True:return
    t=p["toolchain"]
    if (str(Path(sys.executable).resolve())!=t["resolved_executable"]
        or sys.version.split()[0]!=t["python_version"] or np.__version__!=t["numpy_version"]
        or sha(t["resolved_executable"])!=t["executable_sha256"]):
        raise ValueError("exact reviewed numeric-only toolchain changed")
    if "h5py" in sys.modules:raise ValueError("numeric-only index must not import HDF5")

def tick_for(p):
    started=time.monotonic()
    def check():
        st=os.statvfs(p["new_output"])
        if st.f_bavail*st.f_frsize<=p["resources"]["minimum_free_bytes"]:raise ValueError("hardware free-space floor")
        if time.monotonic()-started>=p["resources"]["max_seconds"]:raise ValueError("finite native phase deadline")
    return check

def run_writer(p,stage):
    spec=validate_plan(p,stage);validate_toolchain(p);stage=Path(stage)
    out=stage/"payloads";out.mkdir(exist_ok=False)
    result=core.build_index(spec,out,p["block_elements"],tick_for(p))
    size=sum(x["transport"]["bytes"]+x["transport"]["validity_bytes"] for x in result["array_transports"].values())
    if "expected_payload_bytes" in p and size!=p["expected_payload_bytes"]:raise ValueError("exact4 derived index payload byte mismatch")
    result["payload_bytes"]=size
    write_new(stage/"writer-receipt.json",result)
    return result

def run_verifier(p,stage,writer_sha):
    spec=validate_plan(p,stage);validate_toolchain(p);stage=Path(stage);f=stage/"writer-receipt.json"
    if sha(f)!=writer_sha:raise ValueError("exact writer receipt SHA mismatch")
    w=json.loads(f.read_text())
    if w["source_spec"]!=spec or w["block_elements"]!=p["block_elements"]:raise ValueError("exact source/writer binding mismatch")
    result=core.verify_index(spec,stage/"payloads",w,p["block_elements"],tick_for(p))
    result.update(full_original_gene_axis=spec["logical_shape"][1],source_store_ref=p.get("source_store_ref"),
        original_source_dataset=p.get("source_dataset"),source_evidence_refs=p.get("evidence_refs",{}),
        complete_original_ordinal_occurrence_bijection=True,source_task_admitted=0,model_use=False,
        source_values_copied_or_rewritten=False,derived_access_index_only=True,
        native_control_split_material_and_target_gates_unchanged=True,writer_receipt_sha256=writer_sha,
        full_dataset_readiness=False,substantive_training=False)
    if "h5py" in sys.modules:raise ValueError("numeric-only verifier imported HDF5")
    write_new(stage/"numeric-store-manifest.json",result)
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--mode",choices=["writer","verifier"],required=True)
    ap.add_argument("--plan",required=True);ap.add_argument("--stage",required=True);ap.add_argument("--writer-sha")
    a=ap.parse_args();p=json.loads(Path(a.plan).read_text())
    if a.mode=="writer":run_writer(p,Path(a.stage))
    else:
        if not a.writer_sha:raise ValueError("completed writer SHA pin required")
        run_verifier(p,Path(a.stage),a.writer_sha)
if __name__=="__main__":main()
