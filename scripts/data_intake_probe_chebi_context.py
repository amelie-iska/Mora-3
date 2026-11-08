import pathlib,json,hashlib,time,requests
R=pathlib.Path('/home/iska/Documents/amelie/bio/Mora-3');B=R/'outputs/data-intake/20261031-pathway-semantics-v2/chebi-context-v1';B.mkdir(exist_ok=False);(B/'raw').mkdir();session=requests.Session();session.trust_env=False;out=[]
for value in ['15377','10036','24431']:
 url='https://www.ebi.ac.uk/chebi/backend/api/public/compound/CHEBI:'+value+'/'
 r=session.get(url,timeout=(20,60));p=B/'raw'/('CHEBI-'+value+'.json');p.write_bytes(r.content);d=r.json() if r.status_code==200 else {};receipt=dict(id='CHEBI:'+value,url=r.url,http=r.status_code,path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),license='CC-BY-4.0 ChEBI',headers={k:r.headers.get(k) for k in ['ETag','Last-Modified','Content-Length']},keys=list(d));out.append(receipt);print(json.dumps(receipt),flush=True)
 print('structure_fields',json.dumps({k:v for k,v in d.items() if any(w in k.lower() for w in ['struct','smile','inchi','star','id','modified'])})[:4000],flush=True)
with (B/'probe-receipts.json').open('x') as f:json.dump(out,f,indent=2)
