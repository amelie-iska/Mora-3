from pathlib import Path
import urllib.request,json,hashlib,datetime,shutil,zipfile,io
R=Path('/home/iska/Documents/amelie/bio/Mora-3');B=R/'outputs/data-intake/20261031-tox21-original-and-evaluation-v1';raw=B/'raw';raw.mkdir(parents=True);assets=[];failures=[];total=0;budget=128*2**20;assert shutil.disk_usage(R).free>200*2**30+budget
items=[('challenge-data.html','https://tripod.nih.gov/tox21/challenge/data.jsp','source_protocol_review'),('training-original-sdf.zip','https://tripod.nih.gov/tox21/challenge/download?id=tox21_10k_data_allsdf&sec=','original_challenge_training_data'),('leaderboard-testing-labeled-sdf.zip','https://tripod.nih.gov/tox21/challenge/download?id=tox21_10k_challenge_testsdf&sec=','official_leaderboard_testing_data'),('final-evaluation-sdf.zip','https://tripod.nih.gov/tox21/challenge/download?id=tox21_10k_challenge_scoresdf&sec=','official_final_evaluation_data'),('final-evaluation-smiles.txt','https://tripod.nih.gov/tox21/challenge/download?id=tox21_10k_challenge_scoresmiles&sec=','official_final_evaluation_structure_reference'),('ncats-public-domain-2020.pdf','https://ntp.niehs.nih.gov/sites/default/files/iccvam/meetings/iccvam-forum-2020/ppt/13-ncats-simeonov-508.pdf','primary_provider_data_rights_review'),('nih-copyright-faq.html','https://www.nih.gov/about-nih/frequently-asked-questions','source_rights_review')]
for name,url,role in items:
 try:
  req=urllib.request.Request(url,headers={'User-Agent':'Mora3-public-data-source-review/1.0'})
  with urllib.request.urlopen(req,timeout=45) as response:
   data=response.read(budget-total+1);assert len(data)<=budget-total;headers=dict(response.headers);status=response.status
  p=raw/name
  with p.open('xb') as f:f.write(data)
  total+=len(data);entry=dict(path=str(p.relative_to(R)),url=url,role=role,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),retrieved_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),HTTP_status=status,source_headers={k:v for k,v in headers.items() if k.lower() in ['content-type','content-length','last-modified','etag','content-disposition']})
  if name.endswith('.zip'):
   z=zipfile.ZipFile(io.BytesIO(data));assert z.testzip() is None;entry['archive_members']=[dict(name=i.filename,bytes=i.file_size,CRC=i.CRC) for i in z.infolist()];assert sum(i.file_size for i in z.infolist())<=512*2**20
  assets.append(entry);print('ACQUIRED',name,len(data),flush=True)
 except Exception as exc:failures.append(dict(name=name,url=url,role=role,error=repr(exc)));print('SOURCE_FAILURE',name,repr(exc),flush=True)
receipt=dict(schema='mora3.original_Tox21_challenge_public_source_acquisition.v1',source_release='Tox21 Data Challenge2014 originalpublicdownload;bytesnapshot20261031',budget_bytes=budget,assets=assets,asset_bytes=total,failures=failures,underlying_data_count_not_assets=True,license_review='NCATSprimaryproviderevidence+NIHpolicy;validate source-specific exceptions beforeadmission',admitted_tasks=0,source_splits='training;leaderboardtesting;finalevaluation preserved;notyetcorpusassigned;newglobalchemicalauditneeded',source_geometry='SDFdepictioncoordinatesnotexperimentalconformers',predictions_not_acquired=True)
with (B/'acquisition-receipt.json').open('x') as f:json.dump(receipt,f,indent=2)
print('COMPLETE',json.dumps(dict(assets=len(assets),bytes=total,failures=failures)),flush=True)
