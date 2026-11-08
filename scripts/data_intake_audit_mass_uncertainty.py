"""Bounded same-release native-format check: JSON zero is not zero measured error."""
from pathlib import Path
import datetime,gzip,hashlib,json,re,urllib.request
import pyarrow.parquet as pq
B=Path('outputs/data-intake/20261101-uniprot-physical-units-v1');O=B/'uncertainty-source-format-audit-v1';O.mkdir(exist_ok=False);P=B/'unit-projection-v2/schema-bound-physical-assertions.parquet';rows=[r for r in pq.read_table(P).to_pylist() if r['source_parameter_type']=='mass_spectrometry_source_mass'];selected=[r for r in rows if r['source_numeric_error']==0][:2]+[r for r in rows if r['source_numeric_error']>0][:1]
def ref(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
checks=[]
for r in selected:
 sr=json.loads(r['source_ref_json']);b=Path(sr['raw_file']).read_bytes();assert hashlib.sha256(b).hexdigest()==sr['raw_file_sha256'];entry=json.loads(gzip.decompress(b) if b[:2]==b'\x1f\x8b' else b)['results'][sr['results_index']];url='https://rest.uniprot.org/uniprotkb/'+r['accession']+'.txt'
 with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Mora3-private-source-format-audit/1.0'}),timeout=30) as resp:body=resp.read(2**20);release=resp.headers.get('X-UniProt-Release');status=resp.status
 assert status==200 and release=='2026_03';p=O/(r['accession']+'.txt');p.open('xb').write(body);text=body.decode();seqtext=text.split('\nSQ ',1)[1];seqtext=seqtext.split('\n',1)[1].split('//',1)[0];seq=''.join(re.findall('[A-Z]+',seqtext));assert seq==entry['sequence']['value'];version=int(re.search(r'entry version ([0-9]+)',text).group(1));assert version==entry['entryAudit']['entryVersion']
 blocks=[];cur=None
 for line in text.splitlines():
  if line.startswith('CC   -!- '):
   if cur:blocks.append(' '.join(cur))
   cur=[line[5:].strip()] if line.startswith('CC   -!- MASS SPECTROMETRY:') else None
  elif line.startswith('CC') and cur is not None:cur.append(line[5:].strip())
  elif cur is not None:blocks.append(' '.join(cur));cur=None
 if cur:blocks.append(' '.join(cur))
 matching=[]
 for block in blocks:
  match=re.search(r'\bMass=([0-9.Ee+-]+)',block)
  if match and float(match[1])==r['source_numeric_value'] and ('Method='+r['source_method']+';') in block:matching.append(block)
 assert len(matching)==1;block=matching[0];err=re.search(r'\bMass_error=([0-9.Ee+-]+)',block);native_error=float(err[1]) if err else None
 if r['source_numeric_error']==0:assert native_error is None
 else:assert native_error==r['source_numeric_error']
 checks.append(dict(accession=r['accession'],original_source_ref=sr,txt_source_url=url,txt_source=ref(p),release=release,entry_version=version,canonical_sequence_exact=True,source_comment_value=r['source_numeric_value'],JSON_native_error=r['source_numeric_error'],flatfile_error_literal=native_error,flatfile_mass_error_present=err is not None,native_comment_matches=1))
result=dict(schema='mora3.UniProtMassErrorNativeFormatAudit/v1',status='passed',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),projection=ref(P),full_JSON_source_error_census=dict(assertions=len(rows),zero_literal_fields=sum(r['source_numeric_error']==0 for r in rows),positive_literal_fields=sum(r['source_numeric_error']>0 for r in rows)),independent_same_release_probes=checks,scope='Twozero-errorcasesandonepositivecontrol;notclaimingall3887flatfileschecked',finding='BothJSONzero0.0caseshaveomittedMass_errorinthenativesamereleaserecord;positivecontrolhasexactnativeMass_error. ZeroJSONfielddoesnotproveperfectexperimentalprecision.',admission_policy='Preserveeveryrawzeroexactly;maskzerofrommeasured-uncertaintyobjectiveunlessownsourceexplicituncertaintysemanticsproveit. Positivesareaccuracy/estimatederror,notautomaticallySD.',new_corpus_tasks=0,numeric_admission_changed=False,model_or_old_source_edits=False,code=ref(Path('scripts/data_intake_audit_mass_uncertainty.py')))
(O/'manifest.json').open('x').write(json.dumps(result,indent=2)+'\n');print('COMPLETE',json.dumps(result['full_JSON_source_error_census']))
