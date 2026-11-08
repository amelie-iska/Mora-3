"""Remote footer evidence only; uncached original bytes are never fabricated."""
from pathlib import Path
import json,hashlib,urllib.request,io
import pyarrow.parquet as pq
R=Path('/home/iska/Documents/amelie/bio/Mora-3');B=R/'planning/research/data_intake/20261101-Rosetta-curation-code-v1'
class SparseOriginalReader(io.RawIOBase):
 def __init__(self,size,header,footer):super().__init__();self.length=size;self.header=header;self.footer=footer;self.offset=0
 def readable(self):return True
 def seekable(self):return True
 def tell(self):return self.offset
 def seek(self,offset,whence=0):
  self.offset=(0 if whence==0 else self.offset if whence==1 else self.length)+offset
  if not 0<=self.offset<=self.length:raise OSError('outside original source range')
  return self.offset
 def read(self,size=-1):
  if size<0:size=self.length-self.offset
  stop=min(self.length,self.offset+size)
  if self.offset==self.length:return b''
  if stop<=len(self.header):data=self.header[self.offset:stop]
  elif self.offset>=self.length-len(self.footer):data=self.footer[self.offset-(self.length-len(self.footer)):stop-(self.length-len(self.footer))]
  else:raise OSError('original column payload not acquired; no zero-filled or synthetic bytes')
  self.offset=stop;return data
def main():
 tails=json.loads((B/'curated-footer-range-preflight.json').read_text());out=[]
 for i,a in enumerate(tails[:4]):
  if '/L1000/' not in a['source_key']:continue
  size=a['source_bytes'];length=a['metadata_length']+8;parts=[]
  for suffix,start,stop in [('header',0,4),('footer',size-length,size)]:
   remaining=1048576-sum(p.stat().st_size for p in B.glob('*.payload'));assert stop-start<=remaining;req=urllib.request.Request(a['url'],headers={'Range':f'bytes={start}-{stop-1}','User-Agent':'Mora3-public-source-review'})
   with urllib.request.urlopen(req,timeout=30) as response:
    assert response.status==206 and response.headers['ETag']==a['etag'];assert response.headers['Content-Range']==f'bytes {start}-{stop-1}/{size}';data=response.read(stop-start+1);assert len(data)==stop-start
   p=B/f'curated-gene-{i}-{suffix}.payload';p.open('xb').write(data);parts.append(data)
  assert parts[0]==b'PAR1' and parts[1][-4:]==b'PAR1';reader=SparseOriginalReader(size,*parts);meta=pq.read_metadata(reader);names=meta.schema.to_arrow_schema().names;features=[n for n in names if n.endswith('_at')];row=dict(source_key=a['source_key'],remote_total_bytes=size,etag=a['etag'],remote_footer_listed_rows=meta.num_rows,remote_footer_columns=meta.num_columns,row_groups=meta.num_row_groups,feature_axis=len(features),metadata_columns=[n for n in names if n not in features],source_column_physical_types=sorted({meta.schema.column(j).physical_type for j in range(meta.num_columns)}),header_sha256=hashlib.sha256(parts[0]).hexdigest(),footer_sha256=hashlib.sha256(parts[1]).hexdigest(),footer_source_offset=size-length,whole_file_acquired=False,source_full_QC_pass=False,task_admitted=0,old_BIO_subset_alias_equivalence=False,scope='actualsourceheader/footer_only;no column values read,no missing bytes fabricated');out.append(row)
 p=B/'remote-gene-footer-schema-evidence.json';p.open('x').write(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2));print('TOTAL_RAW_RESEARCH_BYTES',sum(p.stat().st_size for p in B.glob('*.payload')))
if __name__=='__main__':main()
