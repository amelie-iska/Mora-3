"""Bounded original column-range acquisition and old BIO metadata context crosswalk."""
from pathlib import Path
import sys,json,io,urllib.request,hashlib,collections,struct,math,time
import pyarrow as pa,pyarrow.parquet as pq
R=Path('/home/iska/Documents/amelie/bio/Mora-3');sys.path.insert(0,str(R/'scripts'))
from data_intake_rosetta_bounded_footers import SparseOriginalReader
B=R/'planning/research/data_intake/20261101-Rosetta-curation-code-v1';RAW=9*2**20;DERIVED=4*2**20;S=R/'outputs/data-intake/20261031-subagent/curated/cpg0003-rosetta-profile-links-v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p,**kw):return dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=sha(p),**kw)
def raw_used():return sum(p.stat().st_size for p in B.glob('*.payload'))
def derived_used():return sum(p.stat().st_size for p in B.iterdir() if p.is_file() and not p.name.endswith('.payload'))
class BoundedSourceReader(SparseOriginalReader):
 def __init__(self,asset,header,footer,index):super().__init__(asset['source_bytes'],header,footer);self.asset=asset;self.index=index;self.receipts=[];self.blocks=[]
 def read(self,size=-1):
  start=self.offset;stop=self.length if size<0 else min(self.length,start+size)
  if start==self.length:return b''
  if stop<=len(self.header) or start>=self.length-len(self.footer):return super().read(size)
  for lo,hi,data in self.blocks:
   if lo<=start and stop<=hi:self.offset=stop;return data[start-lo:stop-lo]
  assert stop-start<=RAW-raw_used(),'metadata-onlyrawcap;unselectedfeaturepagesmustnotbedownloaded';a=self.asset;request=urllib.request.Request(a['url'],headers={'Range':f'bytes={start}-{stop-1}','If-Match':a['etag'],'User-Agent':'Mora3-public-metadata-review'})
  with urllib.request.urlopen(request,timeout=30) as response:
   assert response.status==206 and response.headers['ETag']==a['etag'] and response.headers['Content-Range']==f'bytes {start}-{stop-1}/{self.length}';data=response.read(stop-start+1);assert len(data)==stop-start
  p=B/f'metadata-range-{self.index}-{len(self.receipts):02d}-{start}.payload';p.open('xb').write(data);self.receipts.append(dict(path=str(p.relative_to(R)),bytes=len(data),sha256=sha(p),source_url=a['url'],source_etag=a['etag'],source_range_start=start,source_range_stop=stop,original_source_file_size=self.length));self.blocks.append((start,stop,data));self.offset=stop;return data
def save(p,rows):
 sink=pa.BufferOutputStream();pq.write_table(pa.Table.from_pylist(rows),sink,compression='zstd');data=sink.getvalue().to_pybytes();assert derived_used()+len(data)<=DERIVED;p.open('xb').write(data);assert pq.read_table(p).to_pylist()==rows;return ref(p,rows=len(rows))
def text(v):return None if v is None or str(v).strip() in ['', 'nan','NA'] else str(v).strip()
def numberkey(v):
 try:
  x=float(v)
  if not math.isfinite(x):return None
  if x==0:x=0.0
  return struct.pack('<d',x).hex()
 except (ValueError,TypeError):return None
def main():
 start=time.perf_counter();tails=json.loads((B/'curated-footer-range-preflight.json').read_text());all_ranges=[];projected={};metadata_refs={};cross=[];summary={};native_manifest=json.loads((S/'manifest.json').read_text())
 for index,modality in [(0,'CellPainting'),(1,'L1000')]:
  asset=tails[index];parts=[]
  for suffix in ['header','footer']:
   p=B/f'curated-gene-{index}-{suffix}.payload'
   if not p.exists():p=B/f'curated-fullschema-{index}-{suffix}.payload'
   parts.append(p.read_bytes())
  source=BoundedSourceReader(asset,*parts,index);file=pq.ParquetFile(source,pre_buffer=False,buffer_size=0);columns=[n for n in file.schema_arrow.names if n.startswith('Metadata')];table=file.read(columns=columns,use_threads=False,use_pandas_metadata=False);rows=table.to_pylist();assert len(rows)==file.metadata.num_rows;projected[modality]=rows;all_ranges+=source.receipts;metadata_refs[modality]=save(B/f'curated-CDRP-{modality}-metadata-only.parquet',rows)
  p=S/f'CDRPBIO-BBBC036-Bray--{modality}--metadata.parquet';expected=native_manifest['artifacts'][p.name];assert p.stat().st_size==expected['bytes'] and sha(p)==expected['sha256'];old=pq.read_table(p).to_pylist();lookup=collections.defaultdict(list)
  def newkey(row):
   if modality=='CellPainting':return (text(row.get('Metadata_Plate')),text(row.get('Metadata_Well')),text(row.get('Metadata_pert_id')),numberkey(row.get('Metadata_pert_dose_micromolar')))
   return (text(row.get('Metadata_Plate')),text(row.get('Metadata_pert_id')),numberkey(row.get('Metadata_pert_dose_micromolar')),text(row.get('Metadata_cdrp_group')))
  for i,row in enumerate(rows):lookup[newkey(row)].append(i)
  stats=collections.Counter();missing=[]
  for local,row in enumerate(old):
   md=json.loads(row['source_metadata_json'])
   if modality=='CellPainting':key=(text(md.get('Metadata_Plate')),text(md.get('Metadata_Well')).upper() if text(md.get('Metadata_Well')) else None,text(md.get('Metadata_broad_sample')),numberkey(md.get('Metadata_mmoles_per_liter2')))
   else:key=(text(md.get('det_plate')),text(md.get('pert_id')),numberkey(md.get('pert_dose')),text(md.get('CPD_TYPE')))
   matched=lookup.get(key,[]);stats['old_rows']+=1;stats['matched_metadata_context_rows']+=bool(matched);stats['unique_metadata_row_matches']+=len(matched)==1;stats['nonunique_samecontext_matches']+=len(matched)>1
   if not matched:missing.append(dict(old_record_id=row['record_id'],old_metadata_key_json=json.dumps(key)))
   doses={rows[i]['Metadata_pert_dose_micromolar'] for i in matched};times={rows[i]['Metadata_pert_timepoint'] for i in matched};cells={rows[i]['Metadata_cell_id'] for i in matched};assert len(doses)<=1 and len(times)<=1 and len(cells)<=1
   value=next(iter(doses)) if doses else None;valid_dose=value is not None and math.isfinite(float(value)) and float(value)>=0;stats['matched_available_nominal_dose']+=bool(matched) and valid_dose;stats['matched_protocol_time']+=bool(matched) and bool(times);cross.append(dict(old_record_id=row['record_id'],modality=modality,old_native_metadata_ref_json=json.dumps(dict(ref(p),row_index=local)),curated_metadata_projection_ref_json=json.dumps(metadata_refs[modality]),curated_original_source_row_indices_json=json.dumps(matched),metadata_context_match=bool(matched),metadata_row_unique=len(matched)==1,source_curated_nominal_dose_micromolar=value if valid_dose else None,source_curated_dose_literal_retained=value,source_curated_protocol_hours=next(iter(times)) if times else None,source_curated_cell_line=next(iter(cells)) if cells else None,dose_available=bool(matched) and valid_dose,time_available=bool(matched) and bool(times),cell_line_available=bool(matched) and bool(cells),context_only_not_numeric_profile_identity=True,source_profile_bytes_equivalence_proved=False,old_task_payloads_unchanged=True,new_task_condition_admitted=False,same_cell_or_cross_assay_replicate_pair=False,source_nominal_units='authorcuratedschema_micromolar_and_protocol_hours;not_oldliteralCSVunits'))
  summary[modality]=dict(stats,remote_curated_metadata_rows=len(rows),unmatched_old_contexts=missing,curated_group_counts=dict(collections.Counter(text(r.get('Metadata_cdrp_group')) or 'missing' for r in rows)) if modality=='L1000' else None)
 crossref=save(B/'oldBIO-to-author-curated-metadata-context-crosswalk.parquet',cross);receipt=dict(schema='mora3.RosettaBIOAuthorMetadataContextCrosswalk/v1',status='complete_metadata_only;numericprofile_equivalence_and_newtask_admission_not_asserted',source_commit='f23205944e17f47d7e8959be71f4b7d25075b191',source_objects=[tails[0],tails[1]],metadata_projections=metadata_refs,all_requested_metadata_columns_fully_decoded=True,unselected_profile_feature_columns_not_read=True,source_header_footer_or_columnchunk_bytes_preserved=True,all_original_range_SHA_receipts=all_ranges,context_match_summary=summary,crosswalk=crossref,old_native_profile_bytes_not_modified=True,source_context_units='officialauthorMetadata_pert_dose_micromolar;Metadata_pert_timepointinhours',source_exposure_directions='CellPainting48h;L10006h;U2OS;not same-cellorreplicate',physical_dose_negative_sentinel_not_a_real_concentration=True,old_source_prefix_is_BIO_subset_not_wholeCDRP=True,all_same_cell_or_new_measurement_claims=False,new_training_tasks=0,raw_research_bytes=raw_used(),raw_cap=RAW,derived_note_bytes_before_this_receipt=derived_used(),derived_cap=DERIVED,seconds=time.perf_counter()-start);p=B/'BIO-context-crosswalk-receipt.json';blob=json.dumps(receipt,indent=2)+'\n';assert derived_used()+len(blob.encode())<=DERIVED;p.open('x').write(blob);print(json.dumps(dict(summary=summary,range_requests=len(all_ranges),raw_bytes=raw_used(),derived_bytes=derived_used(),seconds=time.perf_counter()-start),indent=2))
if __name__=='__main__':main()
