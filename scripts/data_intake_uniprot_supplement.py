"""Versioned physical/form projection; full immutable UniProt JSON remains authoritative."""
import json,hashlib,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from data_intake_uniprot_annotations import normalize_feature,validate_entry,evidence_bindings

def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()

def physical_records(e):
    rows=[]
    def add(kind,value,source,pointer,unit=None,role='source_biophysical_annotation',molecule=None,**extra):
        rows.append(dict(accession=e['primaryAccession'],parameter_type=kind,value=value,unit=unit,source_pointer=pointer,source_value_json=canonical(source).decode(),source_assertion_sha256=hashlib.sha256(canonical(source)).hexdigest(),evidence_bindings_json=canonical(list(evidence_bindings(source,pointer))).decode(),evidence_role=role,molecule_scope=molecule,canonical_property_supervision=molecule is None,numeric_objective_eligible=False,assay_pH=None,assay_temperature=None,**extra))
    for key,kind,unit in [('length','sequence_length','residue'),('molWeight','sequence_mass','Da')]:
        if key in e['sequence']:add(kind,e['sequence'][key],e['sequence'][key],'/sequence/'+key,unit,role='provider_computed_from_canonical_sequence')
    for i,c in enumerate(e.get('comments',[])):
        ptr='/comments/'+str(i);scope=c.get('molecule');typ=c.get('commentType')
        if typ=='MASS SPECTROMETRY':
            add('mass_spectrometry_source_mass',c.get('molWeight'),c,ptr,c.get('unit'),role='source_reported_mass_spectrometry_annotation',molecule=scope,method=c.get('method'),error=c.get('molWeightError'),unit_status='explicit_source_unit' if c.get('unit') else 'unit_not_explicit_in_JSON;numeric_objective_gated')
        elif typ=='BIOPHYSICOCHEMICAL PROPERTIES':
            for key,val in c.items():
                if key in ['commentType','molecule']:continue
                if key=='kineticParameters':
                    for array_key,kind,valuekey in [('michaelisConstants','MichaelisConstant','constant'),('maximumVelocities','maximumVelocity','velocity')]:
                        for j,p in enumerate(val.get(array_key,[])):
                            add(kind,p.get(valuekey),p,ptr+'/kineticParameters/'+array_key+'/'+str(j),p.get('unit'),molecule=scope,role='source_reported_kinetic_annotation',substrate=p.get('substrate'),enzyme=p.get('enzyme'),converted_to_kcat=False)
                    # Preserve the whole container too: notes and future fields must not disappear.
                    add('kineticParameters_container',None,val,ptr+'/'+key,molecule=scope)
                else:add(key,None,val,ptr+'/'+key,molecule=scope,role='source_biophysical_structured_or_text_annotation')
    return rows

def form_records(e):
    rows=[];seq=e['sequence']['value'];acc=e['primaryAccession']
    def add(form_id,kind,source,pointer,status,verified=None,interval=None,sequence_ids=None):
        rows.append(dict(accession=acc,form_id=form_id,form_kind=kind,source_pointer=pointer,source_value_json=canonical(source).decode(),source_assertion_sha256=hashlib.sha256(canonical(source)).hexdigest(),sequence_status=status,verified_sequence=verified,verified_sequence_sha256=hashlib.sha256(verified.encode()).hexdigest() if verified is not None else None,canonical_interval0=interval,sequence_feature_ids=sequence_ids or [],canonical_labels_inherited=False,form_specific_training_admitted=False))
    for i,c in enumerate(e.get('comments',[])):
        if c.get('commentType')=='ALTERNATIVE PRODUCTS':
            for j,iso in enumerate(c.get('isoforms',[])):
                for form_id in iso.get('isoformIds',[]):
                    status=iso.get('isoformSequenceStatus');add(form_id,'alternative_product',iso,f'/comments/{i}/isoforms/{j}',status,verified=seq if status=='Displayed' else None,sequence_ids=iso.get('sequenceIds'))
    for j,f in enumerate(e.get('features',[])):
        if f.get('featureId','').startswith('PRO_'):
            n=normalize_feature(f,seq,acc);interval=n['interval0'] if n['coordinate_supervision'] else None
            add(f['featureId'],'processed_'+f.get('type','unknown'),f,f'/features/{j}',n['coordinate_status'],verified=seq[interval[0]:interval[1]] if interval else None,interval=interval)
    return rows

def main():
    import argparse,gzip,collections,time,shutil
    import pyarrow as pa,pyarrow.parquet as pq
    ap=argparse.ArgumentParser();ap.add_argument('--output-name',required=True);a=ap.parse_args();root=Path('/home/iska/Documents/amelie/bio/Mora-3');batch=root/'outputs/data-intake/20261031-uniprot-full-annotations-v1';out=batch/a.output_name;out.mkdir();(out/'physical').mkdir();(out/'forms').mkdir();(out/'code').mkdir();(out/'code'/Path(__file__).name).write_bytes(Path(__file__).read_bytes());assert shutil.disk_usage(root).free>200*2**30+2*2**30
    manifest=json.loads((batch/'acquisition-manifest.json').read_text());assert manifest['source_entries']==107778;counts=collections.Counter();phys_entries=collections.defaultdict(set);formstatus=collections.Counter();files=[];seen=set();start=time.perf_counter()
    def resolve(e,p):
        x=e
        for token in p.split('/')[1:]:x=x[int(token)] if isinstance(x,list) else x[token.replace('~1','/').replace('~0','~')]
        return x
    for i,receipt in enumerate(manifest['page_receipts']):
        raw=(root/receipt['path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==receipt['sha256'];body=json.loads(gzip.decompress(raw) if receipt['compression']=='gzip' else raw);tables={'physical':[],'forms':[]}
        for j,e in enumerate(body['results']):
            validate_entry(e);assert e['primaryAccession'] not in seen;seen.add(e['primaryAccession'])
            for kind,rows in [('physical',physical_records(e)),('forms',form_records(e))]:
                for row in rows:
                    original=resolve(e,row['source_pointer']);assert json.loads(row['source_value_json'])==original;assert hashlib.sha256(canonical(original)).hexdigest()==row['source_assertion_sha256'];row.update(source_ref_json=canonical(dict(raw_file=receipt['path'],raw_file_sha256=receipt['sha256'],results_index=j,source_release='2026_03')).decode());tables[kind].append(row)
                    if kind=='physical':counts['physical:'+row['parameter_type']]+=1;phys_entries[row['parameter_type']].add(e['primaryAccession'])
                    else:formstatus[row['sequence_status']]+=1;counts['forms:'+row['form_kind']]+=1
        for kind,rows in tables.items():
            if rows:
                # JSON envelope keeps fields introduced after the first row lossless.
                envelope=[dict(accession=r['accession'],record_id=hashlib.sha256(canonical(r)).hexdigest(),record_json=canonical(r).decode()) for r in rows];p=out/kind/f'{i:04d}.parquet';pq.write_table(pa.Table.from_pylist(envelope),p,compression='zstd');assert pq.read_table(p).to_pylist()==envelope;files.append(dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),rows=len(rows),kind=kind))
        if i%20==0:print('SUPPLEMENT_PAGE',i+1,'entries',len(seen),'rows',sum(counts.values()),'seconds',round(time.perf_counter()-start,2),flush=True)
    assert len(seen)==107778
    result=dict(schema='mora3.UniProtPhysicalFormSupplement/v1',status='full_source_projection_verified',source_entries=len(seen),source_release='2026_03',source_manifest=str((batch/'acquisition-manifest.json').relative_to(root)),record_counts=counts,physical_entries_by_type={k:len(v) for k,v in phys_entries.items()},form_sequence_status_counts=formstatus,files=files,bytes=sum(f['bytes'] for f in files),source_pointer_hash_roundtrip_all_rows=True,task_admitted=0,physical_numeric_objectives='not_admitted;source_context_and_unit_review_required',old_mass_spectrometry_projection='typed-full-v1/physical facts have value/method null;this version supersedes only normalized projection;raw targets were preserved',isoform_complete_sequence_payloads='not_acquired_yet;Displayed maps canonical;Described/External unknown',canonical_labels_inherited=False,protein_homology_split='unresolved;no_newheldouts',seconds=time.perf_counter()-start)
    (out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n');print('COMPLETE',str(out/'manifest.json'),sum(counts.values()),result['bytes'],flush=True)
if __name__=='__main__':main()
