"""Finite immutable source-native export. Run with existing tokengt only."""
from __future__ import annotations
import collections
import csv
import datetime
import gzip
import hashlib
import json
import pathlib
import shutil
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import RDLogger
from data_sources_native import parse_cif, atom_rows, category_rows, mapped_reaction, physical_link

ROOT = pathlib.Path(__file__).resolve().parents[1]
BATCH = ROOT / "outputs/data-sources/20261031-expansion-v1"
OUT = BATCH / "curated/source-native-v1"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def emit_json(path, obj):
    with path.open("x") as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write("\n")


def dump_parquet(path, rows):
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path, compression="zstd")
    decoded = pq.read_table(path).to_pylist()
    if decoded != rows:
        raise ValueError("source record Parquet roundtrip mismatch: " + str(path))
    return dict(path=str(path.relative_to(ROOT)), rows=len(rows), bytes=path.stat().st_size, sha256=sha(path), exact_decoded_roundtrip=True)


def structures(receipts):
    out = OUT / "pdb"; out.mkdir()
    recs = []; payloads = []; atom_total = 0
    meta_receipts = json.loads((ROOT / "planning/research/data_source_expansion/pdb-metadata-receipt.json").read_text())
    metadata = {r["pdb_id"]: r for r in meta_receipts}
    for asset in receipts:
        if asset["inventory_id"] != "INV-015":
            continue
        pid = asset["filename"].split(".")[0]
        cif = parse_cif(gzip.decompress(pathlib.Path(asset["path"]).read_bytes()).decode())
        atoms = atom_rows(cif)
        if len(atoms) != metadata[pid]["atom_count"]:
            raise ValueError("independent deposited atom-count mismatch for " + pid)
        atom_total += len(atoms)
        payload = dump_parquet(out / (pid + "-atoms.parquet"), atoms);payloads.append(payload)
        emit_json(out / (pid + "-native-categories.json"), cif)
        meta = json.loads((ROOT / metadata[pid]["path"]).read_text())
        entities = category_rows(cif, "_entity.")
        polymers = category_rows(cif, "_entity_poly.")
        unp_refs = [r for r in category_rows(cif, "_struct_ref.") if r.get("db_name") == "UNP"]
        record = dict(source_record_id="PDB:"+pid, evidence_class="experimental_structure",
            evidence_detail="deposited refined atomic model fitted to experimental data; not raw diffraction/EM observations",
            source_asset=asset, metadata_receipt=metadata[pid], license="CC0-1.0",
            source_revision_history=category_rows(cif,"_pdbx_audit_revision_history."),
            source_methods=cif.get("_exptl.method",[]), resolution_angstrom=meta["rcsb_entry_info"].get("resolution_combined"),
            frame="deposited_asymmetric_unit_Cartesian", coordinate_unit="angstrom", full_atom_axis=True,
            models=sorted({r.get("pdbx_PDB_model_num","1") for r in atoms}),
            alternate_location_rows=sum(r.get("label_alt_id",".") not in (".","?") for r in atoms),
            atom_rows=len(atoms), atom_payload=payload, entities=entities, polymers=polymers,
            assembly_operators_applied=False, assembly_definitions_available=bool(cif.get("_pdbx_struct_assembly.id")),
            source_entity_keys=["pdb:"+pid+":entity:"+r["id"] for r in entities],
            ligand_ccd_keys=sorted({"ccd:"+r.get("label_comp_id","") for r in atoms if r.get("group_PDB")=="HETATM" and r.get("label_comp_id")!="HOH"}),
            uniprot_xrefs_unverified=unp_refs, uniprot_sequence_admission=False,
            task_admitted=False, condition_candidate_fields=["atom_identity_axis", "source_polymer_chemical_identity"],
            target_candidate_fields=["atom_payload.Cartn_x", "atom_payload.Cartn_y", "atom_payload.Cartn_z"],
            input_exclusion_fields=["atom_payload.coordinates", "_struct_conn", "source_metadata", "target_derived_contact_edges"],
            grouping_required=["PDB entry/assembly", "chemical_identity", "protein_homology", "nucleic_acid_sequence"],
            missing_context=["cellular_intervention_assay", "binding_affinity", "source_structure_validation_report_review", "cross_source_homology_split"])
        recs.append(record)
        print("PDB",pid,len(atoms),flush=True)
    emit_json(out / "records.json", recs)
    return dict(inventory_id="INV-015", source="PDB selected complete ASUs", available_records=len(recs), quality_pass_records=len(recs), task_admitted_records=0, atoms=atom_total, payloads=payloads, records=str((out/"records.json").relative_to(ROOT)), full_collection=False)


def reactions(receipts):
    out = OUT / "rgd1";out.mkdir()
    source = next(a for a in receipts if a["filename"]=="RGD1CHNO_smiles.csv")
    qc = collections.Counter(); chunks=[]; rows=[]; total=0; families=set(); canonical=set()
    RDLogger.DisableLog("rdApp.*")
    quarantine = (out/"quarantines.jsonl").open("x")
    started=time.time()
    with pathlib.Path(source["path"]).open() as f:
        for line,row in enumerate(csv.DictReader(f),2):
            total+=1
            try:
                rec=mapped_reaction(row)
                rec["source_csv_line"]=line
                rows.append(rec); families.add(rec["reaction_family_key"]);canonical.add(rec["canonical_reaction_group"])
                qc["quality_pass"]+=1
                if any(rec["measurements"][k]["value"]<0 for k in ("DE_F","DE_B")):
                    qc["negative_electronic_barrier_rows"]+=1
            except ValueError as exc:
                reason=str(exc);qc[reason]+=1
                quarantine.write(json.dumps(dict(source_csv_line=line,source_native=row,reason=reason))+"\n")
            if len(rows)>=4096:
                chunks.append(dump_parquet(out/("records-%04d.parquet"%len(chunks)),rows));rows=[]
            if total%20000==0:
                print("RGD1",total,"pass",qc["quality_pass"],"seconds",round(time.time()-started,1),flush=True)
    if rows:
        chunks.append(dump_parquet(out/("records-%04d.parquet"%len(chunks)),rows))
    quarantine.close()
    # The companion source file is retained without claiming its xTB/electronic/enthalpy columns are identical.
    companion = next(a for a in receipts if a["filename"]=="DFT_reaction_info.csv")
    with pathlib.Path(companion["path"]).open() as f:
        companion_ids=[r["channel"] for r in csv.DictReader(f)]
    result=dict(inventory_id="INV-019",source="Original RGD1 CHNO v6 CSV; not RitS extended",source_asset=source,
        available_records=total,quality_pass_records=qc["quality_pass"],task_admitted_records=0,
        record_unit="reaction pathway/conformation row",unique_model_reaction_ids=len(families),
        canonical_reaction_groups=len(canonical),qc=dict(qc),payloads=chunks,
        source_method="B3LYP-D3/TZVP",energy_unit="kcal/mol",geometry_available=False,
        geometry_gate="original RGD1_CHNO.h5 not acquired because no installed h5py; no TS geometry claimed",
        companion_asset=companion,companion_rows=len(companion_ids),companion_unique_channels=len(set(companion_ids)),
        full_csv_release_acquired=True,full_rgd1_geometry_release_acquired=False,
        task_gates=["source_model_reaction/conformer/chemical cross-source split", "explicit known reaction graph task adapter", "target energy exclusion", "corpus writer admission"],
        phase="phase_i_candidate_reaction_property;TS_generation_geometry_pending")
    emit_json(out/"manifest.json",result);return result


def interactions(receipts):
    out=OUT/"string-human";out.mkdir()
    source=next(a for a in receipts if "9606.protein.physical.links.detailed" in a["filename"])
    rows=[];chunks=[];count=0;qc=collections.Counter();pairs=set();proteins=set();directed=set()
    with gzip.open(source["path"],"rt") as f:
        columns=next(f).split()
        for line,text in enumerate(f,2):
            parts=text.split()
            if len(parts)!=len(columns):raise ValueError("STRING source column count mismatch")
            raw=dict(zip(columns,parts));rec=physical_link(raw)
            # Compact typed table; the schema below holds common evidence, roles, masks, and missing context.
            a,b=raw["protein1"],raw["protein2"]
            item=dict(source_row=line,protein1=a,protein2=b,**rec["score_channels"],
                unordered_pair_group=rec["unordered_pair_group"],experimental_channel_supported=rec["experimental_channel_supported"])
            rows.append(item);count+=1;proteins.update([a,b]);pairs.add(tuple(sorted([a,b])));directed.add((a,b))
            qc["experimental_channel_nonzero"]+=rec["experimental_channel_supported"]
            qc["textmining_only_nonzero"]=qc["textmining_only_nonzero"]+int(int(raw["textmining"])>0 and int(raw["experimental"])==0 and int(raw["database"])==0)
            if len(rows)>=50000:
                chunks.append(dump_parquet(out/("records-%04d.parquet"%len(chunks)),rows));rows=[]
            if count%300000==0:print("STRING",count,flush=True)
    if rows:chunks.append(dump_parquet(out/("records-%04d.parquet"%len(chunks)),rows))
    result=dict(inventory_id="INV-025",source="STRING human v12.0 physical detailed complete species file; NOT full MINT95.8M",source_asset=source,
        available_records=count,quality_pass_records=count,task_admitted_records=0,
        record_unit="source oriented row",undirected_pairs=len(pairs),proteins=len(proteins),duplicate_oriented_rows=count-len(directed),
        symmetric_reverse_rows_present=all((b,a) in directed for a,b in directed),qc=dict(qc),payloads=chunks,
        common_record_schema=physical_link(dict(protein1="9606.__schema_A",protein2="9606.__schema_B",experimental="0",database="0",textmining="0",combined_score="0")),
        schema_template_is_biological_record=False,complete_human_file=True,full_mint_collection_acquired=False,
        role="interaction_evidence_prior_candidate; not a direct causal/affinity label",
        task_gates=["species subset admitted separately from full requested collection", "exact protein-ID/sequence/annotation mapping", "entity/homology split", "source association versus direct binding", "corpus admission"],
        sequence_rows_admitted=0,missing_assays_explicit=True)
    emit_json(out/"manifest.json",result);return result


def main():
    if shutil.disk_usage(ROOT).free < 200*2**30 + 2*2**30:
        raise RuntimeError("free space gate")
    OUT.mkdir(exist_ok=False)
    receipts=json.loads((BATCH/"acquisition-receipts-0001.json").read_text())
    for asset in receipts:
        p=pathlib.Path(asset["path"])
        if asset["state"]!="verified_available" or p.stat().st_size!=asset["bytes"] or sha(p)!=asset["sha256"]:
            raise ValueError("raw source integrity failure")
    results=[structures(receipts),reactions(receipts),interactions(receipts)]
    manifest=dict(schema="mora3.source_expansion_handoff.v1",created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        owner="dataset_sources",status="source_quality_candidates_not_corpus_admitted",sources=results,
        seed=None,seed_note="deterministic lossless validation; no split/sampling/shuffling/tokenizer fitting",raw_assets=len(receipts),
        raw_bytes=sum(a["bytes"] for a in receipts),raw_receipts=str((BATCH/"acquisition-receipts-0001.json").relative_to(ROOT)),
        task_admitted_records=0,source_native_payloads=True,coordinate_transform=None,
        final_admission_owner="dataset_construction",missing_modalities_are_null_not_zero=True,
        no_fabricated_cellular_links=True,mcp_servers_enabled=0)
    emit_json(OUT/"manifest.json",manifest)
    print("COMPLETE",json.dumps({r["source"]:r["quality_pass_records"] for r in results}),flush=True)

if __name__=="__main__":main()
