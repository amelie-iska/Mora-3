import pathlib,json,hashlib,requests,zipfile,io,time
R=pathlib.Path('/home/iska/Documents/amelie/bio/Mora-3');B=R/'outputs/data-intake/20261031-sciplex-original-context-v1/research-gperturb';B.mkdir(exist_ok=False);s=requests.Session();s.trust_env=False;url='https://zenodo.org/api/records/15305114';r=s.get(url,timeout=(15,30));r.raise_for_status();meta=r.json();(B/'zenodo-15305114.json').write_bytes(r.content);files=meta.get('files',[]);print(json.dumps(dict(title=meta.get('metadata',{}).get('title'),license=meta.get('metadata',{}).get('license'),files=[dict(key=f['key'],size=f['size'],checksum=f.get('checksum'),url=f['links'].get('self')) for f in files])),flush=True)
receipts=[]
for f in files:
 if f['key'].lower().endswith('.zip') and f['size']<=20*2**20:
  resp=s.get(f['links']['self'],timeout=(15,60));resp.raise_for_status();data=resp.content;assert len(data)==f['size'];expected=f.get('checksum','');assert not expected.startswith('md5:') or hashlib.md5(data).hexdigest()==expected[4:];p=B/f['key'].replace('/','_');p.write_bytes(data);z=zipfile.ZipFile(io.BytesIO(data));candidates=[]
  for name in z.namelist():
   if any(term in name.lower() for term in ['sciplex','preprocess','readme']) and not name.endswith('/'):
    raw=z.read(name);target=B/(hashlib.sha256(name.encode()).hexdigest()[:12]+'-'+pathlib.Path(name).name);target.write_bytes(raw);candidates.append(dict(member=name,path=str(target.relative_to(R)),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
  receipts.append(dict(path=str(p.relative_to(R)),source_url=resp.url,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),upstream_checksum=expected,members=len(z.namelist()),review_candidates=candidates))
with (B/'publication-code-receipt.json').open('x') as f:json.dump(dict(metadata_url=url,metadata_sha256=hashlib.sha256(r.content).hexdigest(),source_release='Zenodo15305114',files=receipts,role='preprocessing provenance review only;not code execution or underlying dataset'),f,indent=2)
print(json.dumps(receipts),flush=True)
