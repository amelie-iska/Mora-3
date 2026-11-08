"""Sequential source-only HDF5 writer and numeric-only tokengt verifier phases."""
from __future__ import annotations
import argparse,hashlib,json,os,time
from pathlib import Path
from datetime import datetime,timezone
from data_sources_xpert_arrays import verify_numeric_pair

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
    return h.hexdigest()

def write_new(path,obj):
    with Path(path).open("x") as f:json.dump(obj,f,indent=2,sort_keys=True);f.write("\n");f.flush();os.fsync(f.fileno())

def refs_valid(plan):
    for p,expected in plan["bindings"].items():
        if Path(p).is_symlink() or sha(p)!=expected:raise ValueError("exact code/evidence binding SHA changed: "+p)

def validate_phase_plan(plan,stage):
    stage=Path(stage)
    if stage.is_symlink() or not stage.is_dir() or stage.resolve()!=Path(plan["new_output"]).resolve():
        raise ValueError("exact fresh regular stage path required")
    refs_valid(plan)
    keys=[e["array_id"] for source in plan["source_specs"] for e in source["datasets"]]
    if len(keys)!=len(set(keys)):raise ValueError("duplicate native array identifiers")
    if plan.get("fixture_mode") is True:
        fixture=Path(plan["fixture_root"]).resolve()
        if not stage.resolve().is_relative_to(fixture) or any(not Path(x["source_path"]).resolve().is_relative_to(fixture) for x in plan["source_specs"]):
            raise ValueError("synthetic fixture cannot access protected original source paths")
    else:
        if plan.get("fixture_mode") is not False or len(keys)!=4 or len(plan["source_specs"])!=2:
            raise ValueError("exact full2source/4native-array plan required")
        qc_ref=plan["evidence_refs"]["source_QC"];qc=json.loads(Path(qc_ref["path"]).read_text())
        if sha(qc_ref["path"])!=qc_ref["sha256"]:raise ValueError("original full source QC SHA changed")
        originals={s["source_article_ID"]:s for s in qc["sources"]}
        for s in plan["source_specs"]:
            q=originals[s["source_article_ID"]]
            if (s["source_path"]!=q["source_H5AD_path"] or s["source_sha256"]!=q["source_H5AD_SHA256"]
                or s["native_encoding"]!=q["native_matrix"]["encoding"] or s["native_logical_shape"]!=q["native_matrix"]["logical_shape"]):
                raise ValueError("source native identity/encoding/axes differs from full source QC")
            expected={d["native_dataset_path"]:d for d in q["all_native_datasets"] if d["native_dataset_path"] in q["native_matrix"]["source_dataset_paths"]}
            if set(expected)!={e["native_dataset_path"] for e in s["datasets"]}:raise ValueError("native full matrix dataset allowlist mismatch")
            for e in s["datasets"]:
                qd=expected[e["native_dataset_path"]]
                for k in ["native_shape","native_dtype","native_value_SHA256","native_validity_SHA256","chunks"]:
                    if e[k]!=qd[k]:raise ValueError("original native array/chunk/value/validity pin changed")
    return keys

def tick_for(plan):
    started=time.monotonic();floor=plan["resources"]["minimum_free_bytes"];limit=plan["resources"]["max_seconds"]
    def check():
        s=os.statvfs(plan["new_output"])
        if s.f_bavail*s.f_frsize<=floor:raise ValueError("hardware free-space buffer reached")
        if time.monotonic()-started>=limit:raise ValueError("finite native phase deadline reached")
    return check

def validate_toolchain(plan,mode):
    if plan.get("fixture_mode") is True:return
    import sys,numpy as np
    expected=plan["toolchains"][mode]
    if (str(Path(sys.executable).resolve())!=expected["resolved_executable"]
        or sha(expected["resolved_executable"])!=expected["executable_sha256"]
        or sys.version.split()[0]!=expected["python_version"] or np.__version__!=expected["numpy_version"]):
        raise ValueError("exact reviewed interpreter/numpy toolchain changed")
    if mode=="writer":
        import h5py
        if h5py.__version__!=expected["h5py_version"] or h5py.version.hdf5_version!=expected["HDF5_version"]:
            raise ValueError("exact existing read-only HDF5 toolchain changed")

def run_writer(plan,stage):
    keys=validate_phase_plan(plan,stage);validate_toolchain(plan,"writer")
    from data_sources_xpert_hdf import export_source
    stage=Path(stage);payloads=stage/"payloads";payloads.mkdir(exist_ok=False);tick=tick_for(plan)
    all_arrays={};sources=[]
    for source in plan["source_specs"]:
        result=export_source(source,payloads,checkpoint=tick)
        all_arrays.update(result["array_transports"]);sources.append(result["source_gate"])
        print(json.dumps({"phase":"source_export_complete","source":source["source_path"],"arrays":len(all_arrays)}),flush=True)
    if set(all_arrays)!=set(keys):raise ValueError("native source arrays not conserved")
    total=sum(v["transport"]["bytes"]+v["transport"]["validity_bytes"] for v in all_arrays.values())
    if total!=plan["expected_numeric_and_validity_bytes"]:raise ValueError("exact8 numeric/validity payload-byte accounting mismatch")
    result={"schema":"mora3.xpert.numeric-array-writer-receipt.v1","completed_utc":datetime.now(timezone.utc).isoformat(),
            "source_gates":sources,"array_transports":all_arrays,"exact_numeric_and_validity_bytes":total,
            "source_metadata_and_role_refs":plan["evidence_refs"],"source_task_admissions":0,"model_use_allowed":False}
    write_new(stage/"writer-receipt.json",result);return result

def run_verifier(plan,stage,writer_sha):
    keys=validate_phase_plan(plan,stage);validate_toolchain(plan,"verifier");stage=Path(stage);writer=stage/"writer-receipt.json"
    if sha(writer)!=writer_sha:raise ValueError("exact completed writer receipt SHA mismatch")
    w=json.loads(writer.read_text())
    if set(w["array_transports"])!=set(keys):raise ValueError("complete native numeric array set mismatch")
    arrays={};tick=tick_for(plan)
    approved_expected={e["array_id"]:e for source in plan["source_specs"] for e in source["datasets"]}
    for key,item in w["array_transports"].items():
        if item["expected_native"]!=approved_expected[key]:
            raise ValueError("completed writer native metadata differs from exact source plan")
        transport=item["transport"];expected=item["expected_native"]
        if (Path(transport["path"])!=stage/"payloads"/(key+".native.npy")
            or Path(transport["validity_path"])!=stage/"payloads"/(key+".finite-validity-packed.npy")):
            raise ValueError("completed writer payload path differs from exact source plan")
        proof=verify_numeric_pair(transport["path"],transport["validity_path"],expected,checkpoint=tick,transport_manifest=transport)
        arrays[key]={**item,"verified_store":proof}
        print(json.dumps({"phase":"numeric_verifier_complete","array":key,"elements":proof["all_native_elements"]}),flush=True)
    if "h5py" in __import__("sys").modules:raise ValueError("tokengt verifier unexpectedly imported HDF5 dependency")
    result={"schema":"mora3.xpert.numeric-array-store.v1","completed_utc":datetime.now(timezone.utc).isoformat(),
            "fixture_only":plan["fixture_mode"],"full_source_export":not plan["fixture_mode"],"array_transports":arrays,
            "datasets":plan["datasets"],"source_evidence_refs":plan["evidence_refs"],"writer_receipt_path":str(writer),"writer_receipt_sha256":writer_sha,
            "numeric_payload_bytes":w["exact_numeric_and_validity_bytes"],"tokengt_numeric_only_full_readback":True,
            "no_runtime_HDF5_or_pickle_dependency":True,"no_native_axis_crop_cast_normalization_reorder_or_coalescing":True,
            "native_observation_units_distinct":True,"task_admitted_records":0,"adapter_supported":False,"model_use_allowed":False,
            "reader_limit":"Norman is native CSC; arbitrary efficient row iteration requires a separately reviewed NEW derived access index, not included",
            "scientific_limits":"all original X-Pert source masks/protocol/control/placeholder/chemical/material/same-cell gates remain unchanged; store availability is not task admission"}
    write_new(stage/"numeric-store-manifest.json",result);return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--mode",choices=["writer","verifier"],required=True)
    ap.add_argument("--plan",required=True);ap.add_argument("--stage",required=True);ap.add_argument("--writer-sha")
    a=ap.parse_args();p=json.loads(Path(a.plan).read_text())
    if a.mode=="writer":run_writer(p,Path(a.stage))
    else:
        if not a.writer_sha:raise ValueError("completed writer receipt pin required")
        run_verifier(p,Path(a.stage),a.writer_sha)

if __name__=="__main__":main()
