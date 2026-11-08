"""Full row/native nested annotation audit of immutable, checksum-verified DNA shards."""
import argparse
import collections
import datetime
import hashlib
import json
import pathlib
import pyarrow as pa
import pyarrow.parquet as pq
from data_sources_sequences import dna_record_qc
from data_sources_fetch import digest


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--batch",required=True);parser.add_argument("--manifest-name",required=True);args=parser.parse_args()
    root=pathlib.Path(args.batch).resolve();out=root/"curated/dna-source-qc-v1";out.mkdir(parents=True,exist_ok=True)
    source_receipts=sorted((root/"audits/asset-receipts-dna").glob("*.json"));results=[]
    for receipt_path in source_receipts:
        asset=json.loads(receipt_path.read_text())
        if asset.get("state")!="verified_available":continue
        target=out/asset["id"];final=target/"manifest.json"
        if final.exists():
            results.append(json.loads(final.read_text()));continue
        target.mkdir(exist_ok=False);path=pathlib.Path(asset["path"])
        if path.stat().st_size!=asset["expected_bytes"] or digest(path)!=asset["expected_sha256"]:raise ValueError("source integrity failure")
        pf=pq.ParquetFile(path);counts=collections.Counter();rows=[];payloads=[];index=0;seqs=set();accessions=set();min_length=None;max_length=0
        def flush():
            if not rows:return
            p=target/("qc-%04d.parquet"%len(payloads));pq.write_table(pa.Table.from_pylist(rows),p,compression="zstd")
            if pq.read_table(p).to_pylist()!=rows:raise ValueError("QC record roundtrip failure")
            payloads.append(dict(path=str(p),rows=len(rows),bytes=p.stat().st_size,sha256=digest(p)))
            rows.clear()
        for batch in pf.iter_batches(batch_size=128):
            for raw in batch.to_pylist():
                qc=dna_record_qc(raw);features=qc.pop("feature_qc");length=qc["source_sequence_length"]
                counts["source_rows"]+=1;counts["dna_sequence_quality_pass"]+=qc["dna_sequence_quality_pass"]
                counts["protein_annotations"]+=qc["protein_annotation_count"];counts["protein_annotation_invalid_sequence"]+=qc["protein_invalid_annotation_count"]
                counts["rows_without_protein_annotation"]+=not qc["protein_target_available"]
                counts["source_DNA_bases"]+=length;min_length=length if min_length is None else min(min_length,length);max_length=max(max_length,length)
                for f in features:
                    counts[f["feature_type"]+"_annotations"]+=1
                    counts["region_"+f["coordinate_status"]]+=1
                    counts["region_"+f["sequence_alignment"]]+=1
                seqs.add(qc["uppercase_sequence_sha256"]);accessions.add(qc["accession"])
                qc.update(source_row_index=index,source_payload_path=str(path),source_payload_row_grouping="exact GenBank accession and sequence; cross-file and homology grouping reserved for corpus writer",feature_qc_json=json.dumps(features,sort_keys=True),source_record_id=asset["source_repo"]+":"+asset["source_revision"]+":"+asset["id"]+":"+str(index))
                rows.append(qc);index+=1
                if len(rows)>=5000:flush()
        flush()
        if index!=pf.metadata.num_rows:raise ValueError("full source footer/scan count mismatch")
        record=dict(schema="mora3.dna_source_shard_qc.v1",source_asset=asset,footer_rows=pf.metadata.num_rows,full_row_scan=True,all_source_columns_decoded=True,source_schema=str(pf.schema_arrow),counts=dict(counts),available_records=index,dna_sequence_quality_pass_records=counts["dna_sequence_quality_pass"],task_admitted_records=0,
            source_sequence_length_range=[min_length,max_length],within_shard_distinct_uppercase_sequences=len(seqs),within_shard_distinct_accessions=len(accessions),qc_payloads=payloads,
            native_payload=asset["path"],source_origin="GenBank-derived author export; underlying GenBank accessions not independently re-fetched",
            coordinate_status="candidate zero-based half-open from pinned producer; full exact forward/reverse sequence alignment measured per feature; source positions/strings unchanged",
            provenance_limit="pinned public producer emits cds/exin and strand, while published Parquet uses proteins/exons/introns and omits strand; complete producer-to-Parquet transformation is not established",
            missing_context=["CDS strand", "compound exon/CDS join", "codon_start", "genetic code", "GenBank extraction date/release"],
            annotation_policy="Keep DNA integrity, each annotation quality, task eligibility and corpus admission separate. Missing annotations do not create negative labels. No UniProt sequence/score admission inferred.",
            predictor_target_gate="region labels and proteins are targets unless explicitly selected as desired conditions; no target-derived features or topology in predictors",
            completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        with final.open("x") as f:json.dump(record,f,indent=2);f.write("\n")
        results.append(record);print("FULL_SHARD_QC",asset["id"],json.dumps(record["counts"]),flush=True)
    plan=json.loads((root/"acquisition-plan-0001.json").read_text());expected=[a for a in plan["assets"] if a["group"]=="dna"]
    manifest=dict(schema="mora3.dna_source_manifest.v1",created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),inventory_id="INV-001",source_repo="GustavoHCruz/DNA_coding_regions",source_revision="4e6860536e67a2799c206bb66b0ccce4931e78f1",license="CC-BY-4.0",
        verified_and_full_scanned_shards=len(results),registered_full_source_shards=len(expected),full_source_acquired_and_scanned=len(results)==len(expected),
        source_counts_not_historical_revision_proof=True,available_full_scanned_rows=sum(r["available_records"] for r in results),dna_sequence_quality_pass_rows=sum(r["dna_sequence_quality_pass_records"] for r in results),task_admitted_records=0,
        shards=results,pending_asset_ids=sorted(set(a["id"] for a in expected)-set(r["source_asset"]["id"] for r in results)),new_uniprot_sequences_admitted=0,
        native_reader_support="verified source-native Parquet and QC sidecars; model/corpus adapter not asserted",admission_owner="dataset_construction")
    with (out/args.manifest_name).open("x") as f:json.dump(manifest,f,indent=2);f.write("\n")
    print("MANIFEST",str(out/args.manifest_name),manifest["available_full_scanned_rows"],"full",manifest["full_source_acquired_and_scanned"],flush=True)

if __name__=="__main__":main()
