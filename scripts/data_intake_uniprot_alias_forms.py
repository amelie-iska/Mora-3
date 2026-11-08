"""Bounded alias-form source acquisition and exact source-secondary accession audit."""
from pathlib import Path
import sys,json,gzip,hashlib,collections,urllib.request,urllib.parse,datetime,csv,time
import pyarrow as pa,pyarrow.parquet as pq
R=Path('/home/iska/Documents/amelie/bio/Mora-3');sys.path.insert(0,str(R/'scripts'));from data_intake_uniprot_isoforms import fasta_records
B=R/'outputs/data-intake/20261031-uniprot-full-annotations-v1';O=B/'accession-alias-and-form-source-v1'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.open('x').write(json.dumps(x,indent=2)+'\n')
def rec(p,**kw):return dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=sha(p),**kw)
def run():
 O.mkdir();(O/'raw').mkdir();start=time.monotonic();m=json.loads((B/'acquisition-manifest.json').read_text());aliases=[];aliasowners=collections.defaultdict(list);forms=collections.defaultdict(list);mismatches=[];allprimary=set();canonical={}
 for receipt in m['page_receipts']:
  p=R/receipt['path'];assert sha(p)==receipt['sha256'];entries=json.loads(gzip.decompress(p.read_bytes()))['results']
  for j,e in enumerate(entries):
   acc=e['primaryAccession'];allprimary.add(acc);seq=e['sequence']['value'];canonical[acc]=seq;digest=hashlib.sha256(seq.encode()).hexdigest()
   for secondary in e.get('secondaryAccessions',[]):
    row=dict(primary_accession=acc,source_secondary_accession=secondary,canonical_sequence_sha256=digest,annotation_score=e['annotationScore'],reviewed=True,source_release='2026_03',entry_version=e['entryAudit']['entryVersion'],source_ref_json=json.dumps(dict(raw_file=receipt['path'],raw_file_sha256=receipt['sha256'],results_index=j,source_pointer='/secondaryAccessions')));aliases.append(row);aliasowners[secondary].append(row)
   for i,c in enumerate(e.get('comments',[])):
    if c.get('commentType')=='ALTERNATIVE PRODUCTS':
     for k,f in enumerate(c.get('isoforms',[])):
      for fid in f.get('isoformIds',[]):
       r=dict(primary_accession=acc,form_id=fid,status=f.get('isoformSequenceStatus'),source_pointer=f'/comments/{i}/isoforms/{k}',prefix=fid.split('-',1)[0],prefix_is_primary=fid.split('-',1)[0]==acc,prefix_in_source_secondary=fid.split('-',1)[0] in e.get('secondaryAccessions',[]));forms[fid].append(r)
       if r['status']=='Described' and not r['prefix_is_primary']:mismatches.append(r)
 ap=O/'source-secondary-accession-map.parquet';pq.write_table(pa.Table.from_pylist(aliases),ap,compression='zstd');assert pq.read_table(ap).to_pylist()==aliases
 alias_collisions=[dict(secondary=k,source_primary_owners=[r['primary_accession'] for r in rs],distinct_canonical_sequences=len({r['canonical_sequence_sha256'] for r in rs})) for k,rs in aliasowners.items() if len(rs)>1];write(O/'source-secondary-accession-collisions.json',alias_collisions);write(O/'nonprimary-form-descriptors.json',mismatches)
 parents=sorted({r['primary_accession'] for r in mismatches});expected={r['form_id'] for r in mismatches};write(O/'acquisition-plan.json',dict(registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),one_connection=True,qualified_primary_parents=parents,expected_form_ids=sorted(expected),source_release='2026_03',rule='exactsourceformdescriptorownership;neverprefixalone;sourcealiasambiguityretained',raw_cap_bytes=16*2**20,task_admitted=0))
 native=[];receipts=[];actual=set();canonical_seen=set()
 for i in range(0,len(parents),100):
  batch=parents[i:i+100];query=' OR '.join('(accession:'+x+')' for x in batch);url='https://rest.uniprot.org/uniprotkb/stream?'+urllib.parse.urlencode(dict(query=query,format='fasta',includeIsoform='true',compressed='true'));p=O/'raw'/f'primary-parent-batch-{i//100:02d}.fasta.response'
  with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Mora3-public-source-intake/1'}),timeout=90) as response:headers=dict(response.headers);raw=response.read(8*2**20+1)
  assert len(raw)<=8*2**20 and headers.get('X-UniProt-Release')=='2026_03';p.open('xb').write(raw);rr=rec(p,url=url,headers=headers);write(O/'raw'/f'primary-parent-batch-{i//100:02d}.receipt.json',rr);receipts.append(rr)
  import io
  text=gzip.decompress(raw).decode() if raw[:2]==b'\x1f\x8b' else raw.decode()
  for index,(header,seq) in enumerate(fasta_records(io.StringIO(text))):
   sid=header.split('|')[1];assert seq and not(set(seq)-set('ACDEFGHIKLMNPQRSTVWYBXZJUO'));is_canon=sid in batch
   if is_canon:assert seq==canonical[sid];canonical_seen.add(sid)
   else:actual.add(sid)
   owners=[r for r in forms.get(sid,[]) if r['primary_accession'] in batch]
   native.append(dict(source_accession=sid,sequence=seq,sequence_length=len(seq),sequence_sha256=hashlib.sha256(seq.encode()).hexdigest(),source_header=header,is_canonical=is_canon,explicit_source_form_owners_json=json.dumps(owners),unique_owner=len(owners)==1,owner_ambiguity_preserved=len(owners)>1,full_source_ref=rr['path'],full_source_sha256=rr['sha256'],source_record_index=index,canonical_annotation_inheritance=False,task_admitted=False))
 assert canonical_seen==set(parents);np=O/'native-source-sequences.parquet';pq.write_table(pa.Table.from_pylist(native),np,compression='zstd');assert pq.read_table(np).to_pylist()==native
 result=dict(schema='mora3.UniProtAccessionAliasFormSource/v1',status='finite_fullKBalias_and_boundedformsource_verified',source_release='2026_03',source_license='CC-BY-4.0',fullKB_entries_scanned=len(allprimary),secondary_alias_rows=len(aliases),distinct_secondary_accessions=len(aliasowners),ambiguous_secondary_accessions=len(alias_collisions),secondary_alias_index=rec(ap,rows=len(aliases)),nonprimary_form_descriptor_rows=len(mismatches),nonprimary_form_ids=len(expected),qualified_primary_parents=len(parents),prefix_secondary_relation_counts=dict(collections.Counter(str(r['prefix_in_source_secondary']) for r in mismatches)),source_receipts=receipts,native_sequences=rec(np,rows=len(native)),native_source_records=len(native),returned_isoform_ids=len(actual),expected_form_ids_recovered=sorted(expected&actual),expected_form_ids_missing=sorted(expected-actual),source_ids_without_expected_descriptor=sorted(actual-set(forms)),owner_ambiguity='sourceassertionsretained;notforcedtooneprimary;exactseqneededforfuturejoins',canonical_annotation_inheritance=False,task_admitted=0,new_independent_experiments=0,seconds=time.monotonic()-start);write(O/'manifest.json',result);print('COMPLETE',len(aliases),len(alias_collisions),len(expected&actual),'missing',len(expected-actual),flush=True)
if __name__=='__main__':run()
