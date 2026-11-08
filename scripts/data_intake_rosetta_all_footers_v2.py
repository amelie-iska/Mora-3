"""Finish eight official footer/schema snapshots under the metadata-only extension."""
from pathlib import Path
import sys,json,urllib.request,hashlib
import pyarrow.parquet as pq
R=Path('/home/iska/Documents/amelie/bio/Mora-3');sys.path.insert(0,str(R/'scripts'))
from data_intake_rosetta_bounded_footers import SparseOriginalReader
B=R/'planning/research/data_intake/20261101-Rosetta-curation-code-v1';CAP=9*2**20
def main():
 ack=R/'outputs/monitoring/budget-20261101T065755Z-Rosetta-curated-release-metadata-v1.json';assert ack.exists();(B/'metadata-budget-extension-v1.json').open('x').write(json.dumps(dict(parent_receipt=str(ack.relative_to(R)),parent_receipt_sha256=hashlib.sha256(ack.read_bytes()).hexdigest(),prior_rawcap=2**20,additional_rawcap=8*2**20,total_rawcap=CAP,additional_derived_notes_cap=2*2**20,whole_file_download_authorized=False),indent=2)+'\n');tails=json.loads((B/'curated-footer-range-preflight.json').read_text());rows=[];receipts=[]
 for i,a in enumerate(tails):
  size=a['source_bytes'];length=a['metadata_length']+8;parts=[]
  for suffix,start,stop in [('header',0,4),('footer',size-length,size)]:
   old=B/f'curated-gene-{i}-{suffix}.payload'
   if old.exists():data=old.read_bytes();assert len(data)==stop-start
   else:
    remaining=CAP-sum(p.stat().st_size for p in B.glob('*.payload'));assert stop-start<=remaining;req=urllib.request.Request(a['url'],headers={'Range':f'bytes={start}-{stop-1}','If-Match':a['etag'],'User-Agent':'Mora3-public-source-review'})
    with urllib.request.urlopen(req,timeout=30) as response:
     assert response.status==206 and response.headers['ETag']==a['etag'] and response.headers['Content-Range']==f'bytes {start}-{stop-1}/{size}';data=response.read(stop-start+1);assert len(data)==stop-start
    p=B/f'curated-fullschema-{i}-{suffix}.payload';p.open('xb').write(data)
   parts.append(data);receipts.append(dict(source_url=a['url'],source_etag=a['etag'],source_total_bytes=size,range_start=start,range_stop=stop,range_sha256=hashlib.sha256(data).hexdigest(),range_bytes=len(data)))
  assert parts[0]==b'PAR1';meta=pq.read_metadata(SparseOriginalReader(size,*parts));names=meta.schema.to_arrow_schema().names;features=[name for name in names if name.endswith('_at')] if '/L1000/' in a['source_key'] else [name for name in names if name.startswith(('Cells_','Cytoplasm_','Nuclei_'))];rows.append(dict(source_key=a['source_key'],source_url=a['url'],remote_file_bytes=size,etag=a['etag'],last_modified=a['last_modified'],footer_rows=meta.num_rows,row_groups=meta.num_row_groups,feature_axis=len(features),metadata_columns=[n for n in names if n not in features],source_physical_types=sorted({meta.schema.column(j).physical_type for j in range(meta.num_columns)}),whole_file_downloaded=False,full_row_QC_performed=False,task_admitted=0,footer_only_evidence=True))
 p=B/'all-eight-curated-schema-rowcount-metadata-v2.json';p.open('x').write(json.dumps(dict(schema='mora3.RosettaAuthorReleaseFooterSchemaEvidence/v2',source_commit='f23205944e17f47d7e8959be71f4b7d25075b191',objects=rows,range_receipts=receipts,total_remote_file_bytes=sum(r['remote_file_bytes'] for r in rows),remote_footer_row_counts_not_whole_data_QC=True,raw_research_bytes=sum(p.stat().st_size for p in B.glob('*.payload')),full_download_or_new_tasks=0),indent=2)+'\n');print(json.dumps(rows,indent=2));print('RAW_USED',sum(p.stat().st_size for p in B.glob('*.payload')),'CAP',CAP)
if __name__=='__main__':main()
