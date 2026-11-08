from pathlib import Path
import urllib.request,urllib.parse,json,hashlib,datetime,shutil,time
R=Path('/home/iska/Documents/amelie/bio/Mora-3');B=R/'outputs/data-intake/20261031-chembl-logd-context-v1';raw=B/'raw';raw.mkdir(parents=True);assets=[];total=0;budget=20*2**20
assert shutil.disk_usage(R).free>200*2**30+budget

def get(name,url):
 global total
 req=urllib.request.Request(url,headers={'User-Agent':'Mora3-public-source-audit/1.0','Accept':'application/json,text/plain;q=0.9,*/*;q=0.8'})
 with urllib.request.urlopen(req,timeout=60) as response:
  data=response.read(budget-total+1);headers=dict(response.headers);assert len(data)<=budget-total
 path=raw/name
 with path.open('xb') as f:f.write(data)
 total+=len(data);assets.append(dict(path=str(path.relative_to(R)),url=url,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),retrieved_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),HTTP_headers={k:v for k,v in headers.items() if k.lower() in ['etag','last-modified','content-type','content-length']}))
 return data
try:
 status=json.loads(get('status-before.json','https://www.ebi.ac.uk/chembl/api/data/status.json'))
 assay=json.loads(get('assay-CHEMBL3301363.json','https://www.ebi.ac.uk/chembl/api/data/assay/CHEMBL3301363.json'))
 get('source-license-about.md','https://chembl.gitbook.io/chembl-interface-documentation/about.md')
 get('source-license-general.md','https://chembl.gitbook.io/chembl-interface-documentation/frequently-asked-questions/general-questions.md')
 first=json.loads(get('activities-offset-0.json','https://www.ebi.ac.uk/chembl/api/data/activity.json?assay_chembl_id=CHEMBL3301363&limit=1000&offset=0&order_by=activity_id'))
 count=first['page_meta']['total_count'];assert count<=5000
 rows=first['activities']
 for offset in range(1000,count,1000):
  part=json.loads(get(f'activities-offset-{offset}.json',f'https://www.ebi.ac.uk/chembl/api/data/activity.json?assay_chembl_id=CHEMBL3301363&limit=1000&offset={offset}&order_by=activity_id'))
  assert part['page_meta']['total_count']==count;rows+=part['activities']
 assert len(rows)==count and len({r['activity_id'] for r in rows})==count
 assert all(r['assay_chembl_id']=='CHEMBL3301363' for r in rows)
 status2=json.loads(get('status-after.json','https://www.ebi.ac.uk/chembl/api/data/status.json'));assert status==status2
 docs=sorted({r['document_chembl_id'] for r in rows if r['document_chembl_id']})
 for doc in docs:get(f'document-{doc}.json',f'https://www.ebi.ac.uk/chembl/api/data/document/{doc}.json')
 rec=dict(schema='mora3.public_ChEMBL_logD_source_acquisition.v1',status='complete_hash_pinned',budget_bytes=budget,bytes=total,assets=assets,release_status=status,source_activity_rows=count,source_moleculeIDs=len({r['molecule_chembl_id'] for r in rows}),source_assay='CHEMBL3301363',license='CC-BY-SA-3.0;attribution+data-adaptation sharealike retained;no commercial computed-property targets copied',source_documentIDs=docs,model_admission='pending full-native-row QC, relation/units, sourcevalue+structure crosswalk and global heldout groups')
 with (B/'acquisition-receipt.json').open('x') as f:json.dump(rec,f,indent=2)
 print(json.dumps(dict(status=rec['status'],rows=count,bytes=total,assets=len(assets),release=status,description=assay['description'],first_row={k:rows[0].get(k) for k in ['activity_id','type','standard_type','standard_relation','standard_units','standard_value','canonical_smiles','data_validity_comment','activity_comment']},fields=list(rows[0])),indent=2))
except Exception as e:
 with (B/'acquisition-failure.json').open('x') as f:json.dump(dict(error=repr(e),assets=assets,bytes=total),f,indent=2)
 raise
