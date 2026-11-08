"""Explicit UniProt isoform-ID-list equivalence, exact sequences and scoped coordinates."""
from pathlib import Path
import json,hashlib,sys,copy
sys.path.insert(0,str(Path(__file__).resolve().parent))
from data_intake_uniprot_annotations import normalize_feature
from data_intake_corpus import canonical

def bind_isoform_descriptor(iso,canonical_sequence,native_sequences):
 ids=iso.get('isoformIds',[]);status=iso.get('isoformSequenceStatus');verified_ids=sorted(i for i in ids if i in native_sequences);seqs={native_sequences[i] for i in verified_ids}
 if status=='Displayed':seqs.add(canonical_sequence)
 if len(seqs)>1:seq=None;state='conflicting_source_sequences'
 elif status in ['Displayed','Described'] and len(seqs)==1:seq=next(iter(seqs));state='verified_source_form'
 elif status=='External':seq=None;state='external_reference_requires_own_scope_qualification'
 else:seq=None;state='unresolved_source_form_sequence'
 return dict(declared_ids=ids,source_sequence_status=status,source_verified_ids=verified_ids,verified_sequence=seq,verified_sequence_sha256=hashlib.sha256(seq.encode()).hexdigest() if seq is not None else None,status=state,alias_relation='same_source_ALTERNATIVE_PRODUCTS_isoformIds_object;notprefix/suffixinference',canonical_labels_inherited=False)

def bind_feature(feature,entry,bindings):
 ref=feature.get('location',{}).get('sequence') or entry['primaryAccession'];canonical_ref=ref==entry['primaryAccession'];binding=bindings.get(ref);seq=entry['sequence']['value'] if canonical_ref else binding.get('verified_sequence') if binding and binding.get('status')=='verified_source_form' else None
 if seq is None:return dict(reference_id=ref,coordinate_supervision=False,interval0=None,coordinate_status='unverified_reference_sequence',required_predictor_sequence_sha256=None,canonical_entry_input_coordinate_supervision=False,requires_exact_reference_sequence_input=True,source_feature_json=canonical(feature).decode())
 f=copy.deepcopy(feature);f.get('location',{}).pop('sequence',None);norm=normalize_feature(f,seq,ref)
 return dict(reference_id=ref,coordinate_supervision=norm['coordinate_supervision'],interval0=norm['interval0'],coordinate_status=norm['coordinate_status'],required_predictor_sequence_sha256=hashlib.sha256(seq.encode()).hexdigest(),canonical_entry_input_coordinate_supervision=canonical_ref and norm['coordinate_supervision'],requires_exact_reference_sequence_input=True,normalization=norm,source_feature_json=canonical(feature).decode())

def main():
 import gzip,collections,time
 import pyarrow as pa,pyarrow.parquet as pq
 R=Path('/home/iska/Documents/amelie/bio/Mora-3');B=R/'outputs/data-intake/20261031-uniprot-full-annotations-v1';out=B/'form-sequence-coordinate-bindings-v1';out.mkdir();(out/'bindings').mkdir();(out/'feature-delta').mkdir();(out/'code').mkdir();(out/'code'/Path(__file__).name).write_bytes(Path(__file__).read_bytes());start=time.perf_counter();pool={};poolrefs=collections.defaultdict(list)
 def verify(rec):
  p=R/rec['path'];assert p.stat().st_size==rec['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==rec['sha256'];return p
 def add_native(rec,iso_key):
  p=verify(rec)
  for r in pq.read_table(p).to_pylist():
   iscanonical=r.get('is_canonical_record',r.get('is_canonical',False))
   if iscanonical:continue
   sid=r['source_accession'];seq=r['sequence'];assert hashlib.sha256(seq.encode()).hexdigest()==r['sequence_sha256']
   if sid in pool:assert pool[sid]==seq,'same-release native form ID sequence conflict'
   pool[sid]=seq;poolrefs[sid].append(dict(native_file=rec['path'],native_file_sha256=rec['sha256']))
 initial=json.loads((B/'isoform-sequences-v2/manifest.json').read_text())
 for r in initial['native_shards']:add_native(r,'is_canonical_record')
 second=json.loads((B/'isoform-exact-parents-v1/manifest.json').read_text())
 for mr in second['complete_batches']:
  part=json.loads(verify(mr).read_text());add_native(part['native_sequences'],'is_canonical_record')
 alias=json.loads((B/'accession-alias-and-form-source-v1/manifest.json').read_text());add_native(alias['native_sequences'],'is_canonical')
 source=json.loads((B/'acquisition-manifest.json').read_text());counts=collections.Counter();featurecounts=collections.Counter();bindingfiles=[];featurefiles=[];global_ids=collections.defaultdict(set);described_ids=set();described_covered=set();nonprimary_ids=set();nonprimary_covered=set();features_total=0;entries_seen=0
 def save(rows,path,kind):
  if not rows:return
  envelope=[dict(primary_accession=r['primary_accession'],record_json=canonical(r).decode()) for r in rows];pq.write_table(pa.Table.from_pylist(envelope),path,compression='zstd');assert pq.read_table(path).to_pylist()==envelope;rec=dict(path=str(path.relative_to(R)),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),rows=len(rows),kind=kind);(bindingfiles if kind=='binding' else featurefiles).append(rec)
 for i,receipt in enumerate(source['page_receipts']):
  p=verify(receipt);raw=p.read_bytes();entries=json.loads(gzip.decompress(raw) if receipt['compression']=='gzip' else raw)['results'];rows=[];features=[]
  for j,e in enumerate(entries):
   entries_seen+=1;acc=e['primaryAccession'];local={};source_ref=dict(raw_file=receipt['path'],raw_file_sha256=receipt['sha256'],results_index=j,source_release='2026_03')
   for ci,c in enumerate(e.get('comments',[])):
    if c.get('commentType')=='ALTERNATIVE PRODUCTS':
     for k,iso in enumerate(c.get('isoforms',[])):
      binding=bind_isoform_descriptor(iso,e['sequence']['value'],pool)
      for fid in binding['declared_ids']:
       old=local.get(fid)
       if old and old['verified_sequence_sha256']!=binding['verified_sequence_sha256']:binding=dict(binding,verified_sequence=None,verified_sequence_sha256=None,status='conflicting_source_sequences')
       local[fid]=binding;counts[binding['source_sequence_status']+':'+binding['status']]+=1
       row=dict(primary_accession=acc,declared_form_id=fid,source_pointer=f'/comments/{ci}/isoforms/{k}',source_isoform_descriptor_json=canonical(iso).decode(),source_ref_json=canonical(source_ref).decode(),entry_annotation_score=e['annotationScore'],entry_reviewed=True,entry_score_is_not_perform_score=True,**binding);row['native_sequence_refs_json']=canonical([r for sid in binding['source_verified_ids'] for r in poolrefs[sid]]).decode();rows.append(row)
       if binding['verified_sequence'] is not None:global_ids[fid].add(binding['verified_sequence_sha256'])
       if binding['source_sequence_status']=='Described':
        described_ids.add(fid)
        if binding['verified_sequence'] is not None:described_covered.add(fid)
        if fid.split('-',1)[0]!=acc:
         nonprimary_ids.add(fid)
         if binding['verified_sequence'] is not None:nonprimary_covered.add(fid)
   for fi,f in enumerate(e.get('features',[])):
    scope=f.get('location',{}).get('sequence')
    if scope and scope!=acc:
     b=bind_feature(f,e,local);features_total+=1;featurecounts[b['coordinate_status']]+=1;features.append(dict(primary_accession=acc,source_pointer=f'/features/{fi}',source_ref_json=canonical(source_ref).decode(),training_task_admitted=False,eligibility='feature target only when task predictor sequence hash equals required reference;notcanonicalinheritance',**b))
  save(rows,out/'bindings'/f'{i:04d}.parquet','binding');save(features,out/'feature-delta'/f'{i:04d}.parquet','feature_delta')
  if i%30==0:print('FORM_BINDING_PAGE',i+1,'entries',entries_seen,'describedcovered',len(described_covered),'featureviews',features_total,'seconds',round(time.perf_counter()-start,2),flush=True)
 assert entries_seen==107778
 conflicts={k:sorted(v) for k,v in global_ids.items() if len(v)>1};cp=out/'global_form_ID_multiple_sequence_hashes.json';cp.write_text(json.dumps(conflicts,indent=2)+'\n');result=dict(schema='mora3.UniProtExactFormSequenceBinding/v1',status='wholefullKBsourceandnativeformbindingverified',source_release='2026_03',source_license='CC-BY-4.0',entry_records=entries_seen,native_unique_isoform_sequences=len(pool),binding_state_counts=counts,Described_distinct_IDs=len(described_ids),Described_IDs_sequence_covered=len(described_covered),Described_IDs_missing=sorted(described_ids-described_covered),nonprimary_prefix_Described_IDs=len(nonprimary_ids),nonprimary_prefix_IDs_sequence_covered=len(nonprimary_covered),alias_rule='idsinthesameexactsourceisoformIdsobject;allavailableIDsequencesmustagree;notprefixsuffixorcanonicalfeatureinheritance',global_form_IDs_with_multiple_sequence_hashes=len(conflicts),global_ID_merge_policy='perentry/sourcepointerbindingsauthoritative;neverflattenconflicts',feature_delta_views=features_total,feature_delta_status_counts=featurecounts,feature_supervision_rule='exactrequiredpredictorsequenceSHAonly;canonicalentryheadmustnotuseisoformcoordinates',binding_files=bindingfiles,feature_files=featurefiles,full_source_pointer_and_native_hashes=True,all_output_Arrow_roundtrips=True,task_admitted=0,canonical_labels_inherited=False,new_independent_experiments=0,External_and_Not_described='remainseparatelygated;notinvented',seconds=time.perf_counter()-start);(out/'manifest.json').open('x').write(json.dumps(result,indent=2)+'\n');print('COMPLETE',result['Described_IDs_sequence_covered'],'of',result['Described_distinct_IDs'],'nonprimary',len(nonprimary_covered),'features',features_total,'conflicts',len(conflicts),flush=True)
if __name__=='__main__':main()
