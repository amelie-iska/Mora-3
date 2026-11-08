from pathlib import Path
import json,urllib.request,urllib.parse,hashlib,datetime,gzip,csv,sys
R=Path('/home/iska/Documents/amelie/bio/Mora-3');sys.path.insert(0,str(R/'scripts'));from data_intake_uniprot_isoforms import fasta_records
B=R/'outputs/data-intake/20261031-uniprot-full-annotations-v1';out=B/'isoform-exact-parent-probes-v1';out.mkdir()
with gzip.open(R/'outputs/data-intake/20261031-subagent/raw/uniprot2026_03/reviewed-annotation4-5-sequences.tsv.gz','rt') as f:baseline={r['Entry']:r['Sequence'] for r in csv.DictReader(f,delimiter='\t')}
rows=[]
for name,query in [('EGFR-parent','accession:P00533'),('EGFR-scorefiltered','(accession:P00533) AND ((annotation_score:4) OR (annotation_score:5))'),('LCK-parent','accession:P06239'),('TP53-parent','accession:P04637')]:
 url='https://rest.uniprot.org/uniprotkb/stream?'+urllib.parse.urlencode(dict(query=query,format='fasta',includeIsoform='true'))
 with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Mora3-public-source-intake/1'}),timeout=60) as response:headers=dict(response.headers);raw=response.read(2**20+1)
 assert len(raw)<=2**20 and headers.get('X-UniProt-Release')=='2026_03';p=out/(name+'.fasta');p.open('xb').write(raw)
 with p.open() as f:records=list(fasta_records(f))
 ids=[h.split('|')[1] for h,s in records]
 for h,s in records:
  acc=h.split('|')[1]
  if '-' not in acc:assert s==baseline[acc]
 rows.append(dict(probe=name,query=query,url=url,path=str(p.relative_to(R)),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),headers=headers,source_ids=ids,canonical_sequence_exact=True));print(name,ids,flush=True)
(out/'receipt.json').open('x').write(json.dumps(dict(schema='mora3.UniProtIsoformExportProbe/v1',retrieved_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_release='2026_03',one_connection=True,probes=rows,query_effect='compareparentonlyagainstscorefilter;donotassumebroadqueryexpandsallisoforms'),indent=2)+'\n')
