"""Finite, exact author-bundle TS export; immutable outputs, no new transforms."""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import collections
import copy
import datetime
import hashlib
import json
import pathlib
import resource
import sys
import numpy as np
import pyarrow.parquet as pq
from data_sources_ts import restricted_numpy_load, ts_record, ts_condition, check_source_split_masks
from data_sources_build import dump_parquet, emit_json, sha

ROOT = pathlib.Path(__file__).resolve().parents[1]
BATCH = ROOT / "outputs/data-sources/20261031-expansion-v1"
OUT = BATCH / "curated/transition1x-reactot-v1"


def main():
    resource.setrlimit(resource.RLIMIT_AS, (4*2**30, 4*2**30))
    OUT.mkdir(exist_ok=False)
    receipts = json.loads((BATCH/"acquisition-receipts-0002.json").read_text())
    data = {}
    for asset in receipts:
        p = pathlib.Path(asset["path"])
        if p.stat().st_size != asset["bytes"] or sha(p) != asset["sha256"]:
            raise ValueError("TS raw asset integrity failure")
        with p.open("rb") as f:
            data["train" if asset["filename"].startswith("train") else "valid"] = restricted_numpy_load(f)
    train, valid = data["train"], data["valid"]
    n = len(train["reactant"]["num_atoms"])
    split = check_source_split_masks(train["use_ind"], valid["use_ind"], n)
    if train["reactant"]["rxn"] != valid["reactant"]["rxn"] or train["reactant"]["formula"] != valid["reactant"]["formula"]:
        raise ValueError("bundle reaction axes differ; cannot combine official masks")
    counter=collections.Counter();rows=[];chunks=[];ids=set();atom_configurations=0
    same_ts=0;same_r=0;same_p=0
    for i in range(n):
        chosen = train if split[i] == "source_train" else valid
        record = ts_record(chosen, i, split[i])
        record["selected_source_asset"] = receipts[0]["path"] if split[i] == "source_train" else receipts[1]["path"]
        record["selection_rule"] = "official use_ind index; full opposite bundle is duplicate source, never additional sample"
        if record["source_record_id"] in ids:
            raise ValueError("duplicate source formula/reaction key")
        ids.add(record["source_record_id"])
        counter[split[i]]+=1;counter["multifragment_reactions"]+=not record["single_fragment"]
        atom_configurations += len(record["atomic_numbers"]) * 3
        same_ts+=np.array_equal(train["transition_state"]["positions"][i],valid["transition_state"]["positions"][i])
        same_r+=np.array_equal(train["reactant"]["positions"][i],valid["reactant"]["positions"][i])
        same_p+=np.array_equal(train["product"]["positions"][i],valid["product"]["positions"][i])
        changed=copy.deepcopy(record)
        changed["roles"]["transition_state"]["positions"]=[[999.,999.,999.]] * len(record["atomic_numbers"])
        for role in ("reactant","transition_state","product"):
            changed["roles"][role]["wB97x_6-31G(d).energy"] = 999.
        changed["source_extra_guess_keys"]=["ts_guess_true"]
        if ts_condition(record)!=ts_condition(changed):
            raise ValueError("TS target/energy leakage into predictor condition")
        rows.append(record)
        if len(rows)>=1000:
            chunks.append(dump_parquet(OUT/("records-%03d.parquet"%len(chunks)),rows));rows=[]
    if rows:chunks.append(dump_parquet(OUT/("records-%03d.parquet"%len(chunks)),rows))
    # Independently compare serialized role fields to every selected original source value.
    checked=0
    for payload in chunks:
        for record in pq.read_table(ROOT/payload["path"]).to_pylist():
            i=record["source_row"];chosen=train if split[i]=="source_train" else valid
            for role in ("reactant","transition_state","product"):
                for key,axis in chosen[role].items():
                    left=axis[i];right=record["roles"][role][key]
                    if not np.array_equal(np.asarray(left),np.asarray(right)) if key!="fragments" else left!=right:
                        raise ValueError("independent TS native source value mismatch")
            checked+=1
    manifest=dict(schema="mora3.transition1x_author_derivative.v1",created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        inventory_id="INV-020",source="Transition1x full endpoint/TS author preprocessing; not full9.6M paths",
        available_bundle_rows=2*n,available_unique_reactions=n,quality_pass_records=n,task_admitted_records=0,
        record_unit="one paired computed reactant/TS/product reaction",underlying_source="Figshare19614657v4",
        derivative_release="Zenodo13131875v1",license="CC-BY-4.0",underlying_data_license="MIT",
        source_assets=receipts,source_split_counts=dict(counter),source_masks_disjoint_exhaustive=True,
        heldout_semantics="source file valid use_ind; retained as heldout, not assigned train or used for fitting. No test-vs-validation reinterpretation.",
        all_multifragment_reactions_retained=True,no_hydrogens_removed=True,source_atom_configurations=atom_configurations,
        duplicate_bundle_coordinates=dict(transition_state=same_ts,reactant=same_r,product=same_p),
        independent_selected_source_rows_roundtrip=checked,actual_target_mutation_invariance_checked=n,
        full_original_path_configurations_acquired=False,full_endpoint_TS_release_bundles_acquired=True,
        source_frame="author released preprocessed frame; all coordinates exactly retained; no further scaling/centering/rotation/reflection",
        role_policy="DFT source geometry and forces/energies differ from xTB endpoints and unreviewed/source-target-derived guesses. No synthetic TS counted as DFT observation.",
        energy_force_unit_gate="original ASE producer implies eV/eV-A, derivative transform not independently established; mask energy/force tasks until resolved",
        task_candidate="observed computed R/P positions and atomic numbers -> computed TS positions; unknown-topology or single-reactant generation not implied",
        condition_target_rules="use data_sources_ts.ts_condition only; no TS, force, energy, source IDs, ts_guess_true, other guesses or TS-derived topology in predictors",
        grouping_required=["exact reaction/formula/source ID", "original use_ind holdouts", "cross-source endpoint chemical identity and reaction family"],
        task_gates=["source alignment/target-frame preprocessing review", "original-to-derivative energy/force units", "chemical/graph identity across sources", "native model coordinate reader and objective", "corpus writer admission"],
        payloads=chunks,reader_support="verified plain NumPy decode and typed source Parquet; model native reader not claimed",mcp_servers_enabled=0)
    emit_json(OUT/"manifest.json",manifest)
    print(json.dumps({k:manifest[k] for k in ("available_unique_reactions","quality_pass_records","source_split_counts","independent_selected_source_rows_roundtrip","actual_target_mutation_invariance_checked")}),flush=True)

if __name__=="__main__":main()
