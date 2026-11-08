"""Finite Rfam15.1 source-native family/clan/confidence supplement; no corpus edits."""
import argparse
import collections
import datetime
import gzip
import json
import pathlib
import shutil
import pyarrow as pa
import pyarrow.parquet as pq
from data_sources_rfam import parse_region,open_seed_text,seed_families,clan_bindings
from data_sources_fetch import digest

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--batch',required=True);args=parser.parse_args();root=pathlib.Path(args.batch).resolve();out=root/'curated/rfam15.1-context-v1';out.mkdir(exist_ok=False)
    source={}
    for name in ['Rfam.seed.gz','Rfam.clanin','Rfam.full_region.gz']:
        rp=root/'audits/asset-receipts-rfam15.1'/('DSE2-RFAM15.1-'+name+'.json');r=json.loads(rp.read_text());p=pathlib.Path(r['path'])
        if r['state']!='verified_available' or p.stat().st_size!=r['expected_bytes'] or digest(p)!=r['sha256']:raise ValueError('original Rfammetadata sourceintegrity')
        source[name]=r
    with open_seed_text(source['Rfam.seed.gz']['path']) as f:families=list(seed_families(f))
    if len(families)!=4227 or len({x['family_accession'] for x in families})!=4227:raise ValueError('Rfam15.1familycount/schema')
    names=collections.defaultdict(list);catalog=[]
    for i,r in enumerate(families):
        names[r['family_name']].append(r['family_accession']);gf=r['source_GF_fields'];catalog.append(dict(family_accession=r['family_accession'],family_name=r['family_name'],native_GF_json=json.dumps(gf,sort_keys=True,ensure_ascii=False),native_GF_lines_json=json.dumps(r['source_GF_native_lines'],ensure_ascii=False),native_SS_cons=r['SS_cons_native'],native_RF_reference=r['RF_reference_native'],seed_sequences=r['seed_sequences'],alignment_columns=r['alignment_columns'],alignment_axis_quality_pass=r['alignment_axis_quality_pass'],native_GA_json=json.dumps(gf.get('GA',[])),native_TC_json=json.dumps(gf.get('TC',[])),native_NC_json=json.dumps(gf.get('NC',[])),structure_provenance_json=json.dumps(r['structure_provenance_native'],ensure_ascii=False),sequence_provenance_json=json.dumps(r['sequence_provenance_native'],ensure_ascii=False),source_payload=source['Rfam.seed.gz']['path'],source_family_index=i,experimental_structure_truth_inferred=False,task_admitted=False))
    p=out/'family-catalog.parquet';pq.write_table(pa.Table.from_pylist(catalog),p,compression='zstd')
    if pq.read_table(p).to_pylist()!=catalog:raise ValueError('familycatalogroundtrip')
    family_ref=dict(path=str(p),rows=len(catalog),bytes=p.stat().st_size,sha256=digest(p),exact_decoded_roundtrip=True)
    bindings=clan_bindings(pathlib.Path(source['Rfam.clanin']['path']).read_text().splitlines(),names);p=out/'clan-family-name-bindings.parquet';cs=pa.schema([('clan_id',pa.string()),('source_family_name',pa.string()),('family_accession',pa.string()),('exact_seed_family_name_bound',pa.bool_()),('candidate_family_accessions',pa.list_(pa.string()))]);pq.write_table(pa.Table.from_pylist(bindings,schema=cs),p,compression='zstd')
    if pq.read_table(p).to_pylist()!=bindings:raise ValueError('clanbindingsroundtrip')
    clan_ref=dict(path=str(p),rows=len(bindings),bytes=p.stat().st_size,sha256=digest(p),exact_decoded_roundtrip=True);known={x['family_accession'] for x in families};rows=[];payloads=[];counts=collections.Counter();fam_counts=collections.Counter();index=0;derived=family_ref['bytes']+clan_ref['bytes']
    rs=pa.schema([('source_record_id',pa.string()),('source_row_index',pa.int64()),('source_payload',pa.string()),('family_accession',pa.string()),('sequence_accession_version',pa.string()),('native_start',pa.int64()),('native_end',pa.int64()),('native_fasta_identifier_key',pa.string()),('native_coordinate_order',pa.string()),('native_bitscore_text',pa.string()),('bitscore_float64',pa.float64()),('native_evalue_text',pa.string()),('native_CM_start',pa.int64()),('native_CM_end',pa.int64()),('native_truncation_flag',pa.int64()),('source_membership_type',pa.string()),('native_fields_json',pa.string()),('family_catalog_found',pa.bool_()),('source_quality_fields_are_inference_conditions',pa.bool_()),('task_admitted',pa.bool_())])
    def flush():
        nonlocal derived
        if not rows:return
        if shutil.disk_usage(root).free<200*1024**3+8*1024**3:raise ValueError('free-space reserve')
        p=out/('full-region-%04d.parquet'%len(payloads));pq.write_table(pa.Table.from_pylist(rows,schema=rs),p,compression='zstd')
        if pq.read_table(p).to_pylist()!=rows:raise ValueError('native regionroundtrip')
        derived+=p.stat().st_size
        if derived>6*1024**3:raise ValueError('Rfamcontextcap within8GiBallocation')
        payloads.append(dict(path=str(p),rows=len(rows),bytes=p.stat().st_size,sha256=digest(p),exact_decoded_roundtrip=True));rows.clear()
    with gzip.open(source['Rfam.full_region.gz']['path'],'rt',encoding='ascii',errors='strict') as f:
        for line in f:
            if not line.strip():continue
            r=parse_region(line);found=r['family_accession'] in known;counts['native_region_rows']+=1;counts['family_catalog_found']+=found;counts['truncated_matches']+=r['native_truncation_flag'];counts['membership_'+r['source_membership_type']]+=1;counts['coordinate_order_'+r['native_coordinate_order']]+=1;fam_counts[r['family_accession']]+=1
            r.update(source_record_id='Rfam15.1:full_region:'+str(index),source_row_index=index,source_payload=source['Rfam.full_region.gz']['path'],family_catalog_found=found,task_admitted=False);rows.append(r);index+=1
            if len(rows)>=50000:flush();print('RFAM_REGION_CONTEXT_PROGRESS',index,derived,flush=True)
    flush()
    for r in source.values():
        if digest(pathlib.Path(r['path']))!=r['sha256']:raise ValueError('source changed duringcontextscan')
    manifest=dict(schema='mora3.Rfam15.1_native_context_supplement.v1',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_release='Rfam15.1 January2026',license='CC0-1.0',source_assets=source,family_catalog=family_ref,clan_name_bindings=clan_ref,full_region_payloads=payloads,counts=dict(counts),families=len(families),seed_alignment_sequences=sum(x['seed_sequences'] for x in families),seed_alignment_axis_quality_pass=sum(x['alignment_axis_quality_pass'] for x in families),all4227SS_cons_preserved=True,region_families_with_rows=len(fam_counts),region_counts_by_family=dict(fam_counts),clan_memberships=len(bindings),exact_clan_name_bindings=sum(x['exact_seed_family_name_bound'] for x in bindings),distinct_clans=len({x['clan_id'] for x in bindings}),derived_bytes=derived,coordinate_policy='release-native sequenceaccession/version orderedstart/end andCMstart/end; no0basedconversion orinventedassembly/strandcontext',confidence_policy='bitscore inbits andEvalueunitless; originaldecimal text preserved evenbelowfloat64range; no inferred probability/experimentaltruth; per-hit score/Evalue/truncation/CMpositions are sourcequality/supervision metadata, not inferenceconditions',seed_text_policy='sourceUTF8strictdecodefails; explicit reversibleLatin1byte mapping preserves everycitationbyte, no replacementcharacters; originalcompressedbytes remain authoritative; humanencoding notindependentlyasserted',structure_policy='native seedalignment SS_cons andGFSS/SEprovenance retained; sourceoftenPredictedCMfinder; no experimentalstructure labels inferred',join_policy='exact native_fasta_identifier_key toFASTA source_identifier; potentiallymany-to-many/absent; no positionalzip orfamilyIDguessedfromFASTAfileorder; clanlistscontainfamilynamesresolvedonlyviaexactseedGFID',current_corpus_edits=False,task_admitted_rows=0,reader_support=False,corpus_admission_grouping_owner='dataset_construction')
    with (out/'manifest.json').open('x') as f:json.dump(manifest,f,indent=2);f.write('\n')
    print('RFAM_CONTEXT_COMPLETE',len(families),index,derived,flush=True)
if __name__=='__main__':main()
