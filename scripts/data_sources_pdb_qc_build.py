"""Finite PDB source quality/entity supplement; raw files immutable."""
import collections
import datetime
import gzip
import hashlib
import json
import pathlib
from rdkit import Chem, rdBase
from data_sources_native import parse_cif, category_rows
from data_sources_pdb_qc import validation_record, component_context
from data_sources_build import emit_json, sha

ROOT=pathlib.Path(__file__).resolve().parents[1]
BATCH=ROOT/"outputs/data-sources/20261031-expansion-v1"
OUT=BATCH/"curated/pdb-quality-entities-v1"


def main():
    OUT.mkdir(exist_ok=False)
    receipts=json.loads((BATCH/"acquisition-receipts-0003.json").read_text())
    count=collections.Counter();validations=[];polymers=[];components=[];contexts=[]
    source_pdb=json.loads((BATCH/"curated/source-native-v1/pdb/records.json").read_text())
    for r in source_pdb:
        cif=parse_cif(gzip.decompress(pathlib.Path(r["source_asset"]["path"]).read_bytes()).decode())
        for c in component_context(cif):
            contexts.append({"pdb_id":r["source_record_id"].split(":")[1],**c})
    for asset in receipts:
        p=pathlib.Path(asset["path"])
        if asset["state"]!="verified_available" or sha(p)!=asset["sha256"]:
            raise ValueError("PDB supplement raw integrity failure")
        count[asset["kind"]]+=1
        if asset["kind"]=="official_wwpdb_validation_report":
            pid=asset["filename"].split("-")[0]
            record=validation_record(gzip.decompress(p.read_bytes()),pid)
            source=next(r for r in source_pdb if r["source_record_id"]=="PDB:"+pid)
            source_revisions=source["source_revision_history"]
            report_date=record["entry_attributes"].get("PDB-revision-date")
            last_date=source_revisions[-1].get("revision_date") if source_revisions else None
            record.update(source_asset=asset,coordinate_source_last_revision_date=last_date,
                validation_report_revision_date=report_date,revision_date_exact_match=report_date==last_date,
                admission_note="Revision mismatch requires content comparison; no unconditional validation of newer coordinates.")
            emit_json(OUT/(pid+"-validation-native.json"),record);validations.append(record)
        elif asset["kind"]=="official_ccd_definition":
            cif=parse_cif(p.read_text());ccd=cif["_chem_comp.id"][0]
            if asset["filename"]!="CCD-"+ccd+".cif":raise ValueError("CCD source ID mismatch")
            desc=category_rows(cif,"_pdbx_chem_comp_descriptor.")
            candidates=[d for d in desc if d.get("type")=="SMILES_CANONICAL"]
            canonical=None;used=None
            for d in candidates:
                mol=Chem.MolFromSmiles(d["descriptor"])
                if mol is not None:
                    canonical=Chem.MolToSmiles(mol,isomericSmiles=True);used=d;break
            rec=dict(ccd_id=ccd,source_asset=asset,source_chem_comp=category_rows(cif,"_chem_comp."),
                source_descriptors=desc,rdkit_isomeric_canonical_smiles=canonical,derived_from_source_descriptor=used,
                transform=dict(rdkit_version=rdBase.rdkitVersion,sanitize=True,isomericSmiles=True),
                source_component_atom_count=len(category_rows(cif,"_chem_comp_atom.")),
                binding_role=None,task_admitted=False,
                chemistry_role_note="CCD definition/ideal geometry is chemical reference, not experimental bound-pose evidence or affinity label")
            emit_json(OUT/("CCD-"+ccd+"-identity.json"),rec);components.append(rec)
        else:
            raw=json.loads(p.read_text());ids=raw["rcsb_polymer_entity_container_identifiers"]
            if asset["filename"]!=ids["entry_id"]+"-polymer-"+ids["entity_id"]+".json":raise ValueError("polymer entity ID mismatch")
            rec=dict(pdb_id=ids["entry_id"],entity_id=ids["entity_id"],source_asset=asset,
                source_identifiers=ids,source_alignment=raw.get("rcsb_polymer_entity_align",[]),
                source_polymer=raw.get("entity_poly"),full_source_record=raw,
                exact_uniprot_sequence_verified=False,annotation_score_verified=False,
                new_uniprot_sequence_admitted=False,task_admitted=False,
                mapping_note="SIFTS/RCSB cross-reference and alignment provenance retained; not full-sequence identity or score>=4 proof")
            emit_json(OUT/(asset["filename"].replace(".json","-mapping.json")),rec);polymers.append(rec)
    emit_json(OUT/"component-source-contexts.json",contexts)
    refs={r["ccd_id"] for r in contexts};defs={r["ccd_id"] for r in components}
    if refs!=defs:raise ValueError("native nonwater HETATM component dictionary coverage mismatch")
    manifest=dict(schema="mora3.pdb_quality_entity_supplement.v1",created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        inventory_id="INV-015",license="CC0-1.0",source_assets=receipts,asset_counts=dict(count),
        validation_reports=len(validations),polymer_entity_records=len(polymers),ccd_definitions=len(components),
        source_component_contexts=len(contexts),ccd_descriptor_rows=sum(len(r["source_descriptors"]) for r in components),
        rdkit_isomeric_canonical_smiles_available=sum(r["rdkit_isomeric_canonical_smiles"] is not None for r in components),
        validation_revision_date_match={r["pdb_id"]:r["revision_date_exact_match"] for r in validations},
        all_requested_source_ids_verified=True,full_original_validation_attributes_and_subgroups_preserved=True,
        corrects_field="source-native-v1/pdb/records.json ligand_ccd_keys means all nonwater HETATM components; includes modified polymers, ions, solvent, cofactors/glycans. Do not infer binding role.",
        component_contexts=str((OUT/"component-source-contexts.json").relative_to(ROOT)),
        uniprot_sequence_admissions=0,task_admitted_records=0,admission_owner="dataset_construction",
        raw_bytes=sum(a["bytes"] for a in receipts),quality_thresholds_not_selected=True,
        source_role_note="Validation is quality/evaluation metadata excluded from predictors; chemical descriptors/source alignments do not create affinity or same-assay/cellular links.")
    emit_json(OUT/"manifest.json",manifest)
    print(json.dumps({k:manifest[k] for k in ("validation_reports","polymer_entity_records","ccd_definitions","ccd_descriptor_rows","rdkit_isomeric_canonical_smiles_available","validation_revision_date_match")}),flush=True)

if __name__=="__main__":main()
