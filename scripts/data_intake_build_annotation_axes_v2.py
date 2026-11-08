"""Train-source annotation label/evidence registry; no model fitting or new examples."""
from pathlib import Path
import collections,datetime,gzip,hashlib,json,os,re,time
import pyarrow as pa
import pyarrow.parquet as pq
R=Path.cwd();B=Path('outputs/data-intake/20261101-annotation-target-axes-full-GOA-v2');B.mkdir(exist_ok=False);(B/'code').mkdir();(B/'code'/Path(__file__).name).write_bytes(Path(__file__).read_bytes());start=time.perf_counter();CAP=2*2**30
U=Path('outputs/data-intake/20261031-uniprot-full-annotations-v1');ap=U/'acquisition-manifest.json';a=json.loads(ap.read_text());gp=U/'GOA-typed-complete-v5/manifest.json';g=json.loads(gp.read_text());M=Path('outputs/data-intake/20261101-full-GOA-corpus-union-v2/master-corpus-manifest-v17-full-canonical-GOA-v1.json');assert json.loads(M.read_text())['additional_GOA_snapshot_admission_manifest']==str(gp);assert g['full107778coverage'];axes={};views=collections.Counter();seen=set();inputrefs=[];qc=collections.Counter()
EXP={'EXP','IDA','IPI','IMP','IGI','IEP','HTP','HDA','HMP','HGI','HEP'}
def ref(p,**kw):return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),**kw)
def checked(e):
 p=Path(e['path']);assert p.stat().st_size==e['bytes'] and ref(p)['sha256']==e['sha256'];return p
def live(stage,**kw):
 (B/'live-build.json').write_text(json.dumps(dict(pid=os.getpid(),stage=stage,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),seconds=time.perf_counter()-start,**kw),indent=2)+'\n');print(stage,kw,round(time.perf_counter()-start,2),flush=True)
def add(axis,symbol,parent,subject,origin,evidence='source_reference',polarity='source_assertion_not_boolean',scope='source_scope_preserved;registry_not_supervision_admission'):
 if symbol is None:return
 assert isinstance(symbol,str) and symbol
 key=(axis,symbol)
 r=axes.setdefault(key,dict(parents=set(),subjects=set(),origins=collections.Counter(),evidence=collections.Counter(),polarity=collections.Counter(),scope=collections.Counter(),assertions=0));r['parents'].add(parent);r['subjects'].add(subject);r['origins'][origin]+=1;r['evidence'][evidence]+=1;r['polarity'][polarity]+=1;r['scope'][scope]+=1;r['assertions']+=1;views[axis]+=1
(B/'build-plan.json').open('x').write(json.dumps(dict(schema='mora3.AnnotationTargetAxisPlan/v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),master=ref(M),fullKB_source=ref(ap),GOA_source=ref(gp),new_download_bytes=0,derived_cap_bytes=CAP,minimum_free_bytes=200*2**30,scope='Full107778qualifiedKBsource+admittedcurrentGOAsnapshot;targetclass/evidencefrequenciesonly;allcurrentproteinrows_trainonly',new_training_examples=0,new_model_heads=0,default_absence_negative_mask=False,not_a_fitted_neural_vocabulary=True,unprojected_fields_retained_in_full_source_refs=True),indent=2)+'\n')
try:
 for i,e in enumerate(a['page_receipts']):
  p=checked(e);inputrefs.append(ref(p));b=p.read_bytes();entries=json.loads(gzip.decompress(b) if e['compression']=='gzip' else b)['results'];assert len(entries)==e['entries']
  for entry in entries:
   acc=entry['primaryAccession'];assert acc not in seen and entry['annotationScore'] in [4,5] and entry['entryType']=='UniProtKB reviewed (Swiss-Prot)';seen.add(acc);origin='UniProtKB2026_03';seq=entry['sequence']['value'];assert len(seq)==entry['sequence']['length'];qc['qualified_entries']+=1
   for x in entry.get('uniProtKBCrossReferences',[]):
    database=x.get('database');symbol=x.get('id');scope='explicit_isoform_scope' if x.get('isoformId') else 'entry_annotation_scope'
    if database=='GO':
     props={p['key']:p.get('value') for p in x.get('properties',[]) if 'key' in p};ev=props.get('GoEvidenceType');code=ev.split(':',1)[0] if isinstance(ev,str) else 'unknown';neg='NOT' in str(props.get('Qualifier',props.get('GoAnnotationQualifier',''))).split('|');pol='unknown_ND_placeholder' if code=='ND' else 'explicit_NOT' if neg else 'source_positive_annotation';add('GO',symbol,acc,acc,origin,'experimental_GOcode' if code in EXP else 'ND_placeholder' if code=='ND' else 'inferred_or_other_GOcode',pol,scope);qc['KB_GO_assertions']+=1
    elif database in ['InterPro','Pfam','PROSITE','Gene3D','SUPFAM','PANTHER','CDD']:
     add('domain_family_reference:'+database,symbol,acc,acc,origin,'source_crossreference;experimental_status_not_inferred','source_reference_not_boolean',scope)
   for f in entry.get('features',[]):
    loc=f.get('location',{});st,en=loc.get('start',{}),loc.get('end',{});aa,bb=st.get('value'),en.get('value');scope='unmapped_sequence_scope' if loc.get('sequence') not in [None,acc] else 'exact_canonical' if type(aa) is int and type(bb) is int and st.get('modifier')=='EXACT' and en.get('modifier')=='EXACT' and 1<=aa<=bb<=len(seq) else 'fuzzy_or_unknown';add('feature_type',f.get('type'),acc,acc,origin,'local_ECO0000269' if any(x.get('evidenceCode')=='ECO:0000269' for x in f.get('evidences',[])) else 'other_or_missing_local_ECO','observed_feature_type;unannotated_positions_unknown',scope)
    lig=f.get('ligand')
    if isinstance(lig,dict) and isinstance(lig.get('id'),str):add('feature_ligand_source_ID',lig['id'],acc,acc,origin,scope=scope)
   def walk_ec(obj,path=''):
    if isinstance(obj,dict):
     for key,value in obj.items():
      if key=='ecNumbers' and isinstance(value,list):
       for x in value:
        if isinstance(x,dict) and isinstance(x.get('value'),str):
         symbol=x['value'];labeltype='partial_EC_symbol' if '-' in symbol else 'complete_numeric_EC_symbol' if re.fullmatch(r'[0-9]+(?:[.][0-9]+){3}',symbol) else 'other_native_EC_symbol';add('EC',symbol,acc,acc,origin,'local_ECO0000269' if any(e.get('evidenceCode')=='ECO:0000269' for e in x.get('evidences',[])) else 'other_or_missing_local_ECO',labeltype,'source_pointer:'+path+'/'+key)
      elif key=='ecNumber' and isinstance(value,dict) and isinstance(value.get('value'),str):add('EC',value['value'],acc,acc,origin,'local_ECO0000269' if any(e.get('evidenceCode')=='ECO:0000269' for e in value.get('evidences',[])) else 'other_or_missing_local_ECO','source_EC_symbol','source_pointer:'+path+'/'+key)
      else:walk_ec(value,path+'/'+key)
    elif isinstance(obj,list):
     for j,x in enumerate(obj):walk_ec(x,path+'/'+str(j))
   walk_ec(entry.get('proteinDescription',{}),'/proteinDescription')
   for j,c in enumerate(entry.get('comments',[])):
    walk_ec(c,'/comments/'+str(j));scope='explicit_molecule_scope' if c.get('molecule') else 'entry_annotation_scope'
    for x in c.get('cofactors',[]):
     cr=x.get('cofactorCrossReference',{})
     if cr.get('id'):add('cofactor_source_ID',cr['id'],acc,acc,origin,scope=scope)
    for x in c.get('subcellularLocations',[]):
     for k in ['location','topology','orientation']:
      if isinstance(x.get(k),dict) and x[k].get('id'):add('subcellular_'+k,x[k]['id'],acc,acc,origin,scope=scope)
   for x in entry.get('keywords',[]):
    if x.get('id'):add('keyword_source_ID',x['id'],acc,acc,origin)
  if i%20==0:live('full current qualified KB native target-class/evidence inventory',pages=i+1,entries=len(seen),axis_symbols=len(axes))
 assert len(seen)==107778
 q=json.loads(checked(g['quarantine_receipt']).read_text());bad={x for row in q for x in row['source_annotation_ids']};assert len(bad)==g['quarantined_annotations'];goa_total=0;goa_admitted=0
 for i,e in enumerate(g['assertion_files']):
  p=checked(e);inputrefs.append(ref(p))
  for batch in pq.ParquetFile(p).iter_batches(batch_size=4096):
   for row in batch.to_pylist():
    goa_total+=1
    if row['annotation_id'] in bad:continue
    t=json.loads(row['projection_json']);binding=json.loads(row['binding_json']);assert binding['parent_accession'] in seen and t['missing_is_negative'] is False;goa_admitted+=1;pol='explicit_NOT_target' if t['source_negative_GO_head_mask'] else 'source_positive_target' if t['source_positive_GO_head_mask'] else 'unknown_masked_annotation';add('GO',t['go_id'],binding['parent_accession'],row['subject'],'GOA2026-07-28',t['evidence_class'],pol,binding['subject_kind']+':'+binding['sequence_binding']);qc['GOA_explicit_NOT']+=t['negated'];qc['GOA_positive_mask']+=t['source_positive_GO_head_mask'];qc['GOA_negative_mask']+=t['source_negative_GO_head_mask'];qc['GOA_unknown_head_masks']+=not t['source_positive_GO_head_mask'] and not t['source_negative_GO_head_mask']
  if i%50==0:live('full admitted GOA target-ID/evidence/unknown-mask inventory',parts=i+1,available_views=goa_total,admitted_views=goa_admitted,axis_symbols=len(axes))
 assert goa_total==g['raw_available_annotations'] and goa_admitted==g['raw_available_annotations']-g['quarantined_annotations']
 byaxis=collections.defaultdict(list)
 for (axis,symbol),r in sorted(axes.items()):byaxis[axis].append(dict(axis_id=symbol,source_assertion_views=r['assertions'],distinct_existing_parent_entries=len(r['parents']),distinct_source_subjects=len(r['subjects']),source_origins_json=json.dumps(dict(r['origins']),sort_keys=True),evidence_classes_json=json.dumps(dict(r['evidence']),sort_keys=True),source_polarity_and_unknown_counts_json=json.dumps(dict(r['polarity']),sort_keys=True),source_scope_counts_json=json.dumps(dict(r['scope']),sort_keys=True),absence_is_negative=False,new_supervision_admitted=False))
 outputs=[]
 for axis,rs in sorted(byaxis.items()):
  p=B/(axis.replace(':','-')+'.parquet');pq.write_table(pa.Table.from_pylist(rs),p,compression='zstd');assert pq.read_table(p).to_pylist()==rs;outputs.append(ref(p,axis=axis,labels=len(rs),source_assertion_views=views[axis]))
 assert sum(p.stat().st_size for p in B.rglob('*') if p.is_file())<CAP;st=os.statvfs('.');assert st.f_bavail*st.f_frsize>200*2**30
 result=dict(schema='mora3.TrainSourceAnnotationAxes/v1',status='complete_fullKB_and_current_admitted_GOA_label_evidence_inventory_verified',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),master=ref(M),fullKB_source=ref(ap),GOA_source=ref(gp),qualified_current_KB_entries=len(seen),available_GOA_annotation_views=goa_total,admitted_GOA_annotation_views=goa_admitted,quarantined_GOA_annotation_views=g['quarantined_annotations'],category_source_view_counts=dict(views),labels_by_axis={k:len(v) for k,v in byaxis.items()},quality_counts=dict(qc),axis_files=outputs,all_axis_Parquet_roundtrips=True,all_source_payload_hashes_rechecked=True,source_payload_refs=inputrefs,unknown_negative_policy='Unlistedlabels/positionsareunknownbydefault;explicitNOTandsourceheadmasksretainedseparately;NDplaceholdersnotpositivebiology',EC_hierarchy='Sourceverbatimsymbolscomplete/partial/provisionalpreserved;derivedprefixesnotaddedasobservedchildlabels',source_scope_policy='Entry,isoform,processedformsandfuzzycoordinatesnotflattened;registrydoesnotgranttask-specificeligibility',GOA_KB_join='samecurrentqualifiedparentidentityonly;sourceversionsandrelationspreserved;historicalGOAsequenceversionunknown;notindependentexperimentaljoin',target_exclusion='AxisIDs/queryshapearepublictargetdefinition;actualcurrentlabels/names/citations/sourceIDsneverhidden-targetpredictorfeatures;desiredconditionsrequireseparateadmittedtask',neural_vocabulary_fitted=False,full_native_neural_objective_coverage=False,new_task_shards=0,new_training_examples=0,new_independent_experiments=0,weights=None,model_or_source_edits=False,optimizer_steps=0,unprojected_fields='allimmutablefullsourcerefsretained;registryisnota replacementforlosslesspayloads',code=ref(Path(__file__)),seconds=time.perf_counter()-start)
 (B/'manifest.json').open('x').write(json.dumps(result,indent=2)+'\n');live('annotation source-axis registry complete',labels_by_axis=result['labels_by_axis'])
except BaseException as exc:
 (B/'failure.json').open('x').write(json.dumps(dict(error=repr(exc),seconds=time.perf_counter()-start),indent=2)+'\n');raise
