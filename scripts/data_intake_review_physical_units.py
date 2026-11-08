"""Bounded official manual provenance and source-specific unit gap review."""
from pathlib import Path
import collections,datetime,gzip,hashlib,json,re,urllib.request,urllib.error
from html.parser import HTMLParser
import pyarrow as pa
import pyarrow.parquet as pq
R=Path.cwd();B=Path('outputs/data-intake/20261101-uniprot-physical-units-v1');B.mkdir(exist_ok=False);(B/'research').mkdir();start=datetime.datetime.now(datetime.timezone.utc)
def ref(p,**kw):return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),**kw)
receipts=[]
for name,url in [('manual-2026_03.html','https://prosite.expasy.org/docs/userman.html'),('mass-spectrometry-help.response','https://rest.uniprot.org/help/mass_spectrometry'),('biophysicochemical-help.response','https://rest.uniprot.org/help/biophysicochemical_properties')]:
 status=None
 try:
  with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Mora3-private-research-source-review/1.0'}),timeout=25) as resp:body=resp.read(8*2**20+1);status=resp.status;headers=dict(resp.headers)
 except urllib.error.HTTPError as e:body=e.read(8*2**20+1);status=e.code;headers=dict(e.headers)
 except Exception as e:receipts.append(dict(url=url,status='failed',error=type(e).__name__));continue
 assert len(body)<=8*2**20;p=B/'research'/name;p.open('xb').write(body);receipts.append(dict(url=url,status=status,content=ref(p),headers={k:v for k,v in headers.items() if k.lower() in ['content-type','etag','last-modified','x-uniprot-release','x-uniprot-release-date']}))
(B/'acquisition-receipts.json').open('x').write(json.dumps(receipts,indent=2)+'\n')
manual=(B/'research/manual-2026_03.html').read_text()
class Text(HTMLParser):
 def __init__(self):super().__init__();self.parts=[]
 def handle_data(self,d):self.parts.append(d)
h=Text();h.feed(manual);text=' '.join(' '.join(h.parts).split());assert '2026_03' in text and 'Abs(max)=xx nm;' in text
# Manual explicitly defines absorption maxima in nm. Its mass-spectrometry section does not explicitly name a unit; SQ calculated mass is separate.
ix=text.index("Syntax of the topic 'MASS SPECTROMETRY'",text.index("Syntax of the topic 'MASS SPECTROMETRY'")+1) if text.count("Syntax of the topic 'MASS SPECTROMETRY'")>1 else text.index("Syntax of the topic 'MASS SPECTROMETRY'")
section=text[ix:ix+1400];mass_unit_explicit=bool(re.search(r'\b(Daltons?|Da|kDa)\b',section));assert not mass_unit_explicit
source_manifest=Path('outputs/data-intake/20261031-uniprot-full-annotations-v1/physical-form-supplement-v2/manifest.json');m=json.loads(source_manifest.read_text());selected=[];types=collections.Counter()
for e in m['files']:
 if e['kind']!='physical':continue
 p=Path(e['path']);assert ref(p)['sha256']==e['sha256']
 for row in pq.read_table(p,columns=['record_json']).to_pylist():
  d=json.loads(row['record_json']);types[d['parameter_type']]+=1
  if 'absorp' in d['parameter_type'].lower():selected.append(d)
cache={};rows=[];stats=collections.Counter()
def pointer(obj,p):
 for part in p.strip('/').split('/'):
  part=part.replace('~1','/').replace('~0','~');obj=obj[int(part)] if isinstance(obj,list) else obj[part]
 return obj
for d in selected:
 sr=json.loads(d['source_ref_json']);key=sr['raw_file']
 if key not in cache:
  p=Path(key);body=p.read_bytes();assert hashlib.sha256(body).hexdigest()==sr['raw_file_sha256'];cache[key]=json.loads(gzip.decompress(body) if body[:2]==b'\x1f\x8b' else body)['results']
 entry=cache[key][sr['results_index']];obj=pointer(entry,d['source_pointer']);assert obj==json.loads(d['source_value_json']);value=obj.get('max') if isinstance(obj,dict) else None;valid=type(value) in [int,float] and value>0;direct=any(e.get('evidenceCode')=='ECO:0000269' for e in obj.get('evidences',[])) if isinstance(obj,dict) else False;stats['assertion_views']+=1;stats['numeric_positive_max']+=valid;stats['direct_assertion_ECO0000269']+=direct;stats['canonical_scope_flag']+=d['canonical_property_supervision'];stats['approximate_true']+=bool(obj.get('approximate')) if isinstance(obj,dict) else False
 rows.append(dict(accession=d['accession'],source_pointer=d['source_pointer'],source_raw_json=json.dumps(obj,sort_keys=True),source_ref_json=d['source_ref_json'],source_instance_unit=d['unit'],schema_declared_unit='nm',unit_provenance='UniProt2026_03manual_CC_BIOPHYSICOCHEMICAL_PROPERTIES_Abs(max)',schema_source_sha256=ref(B/'research/manual-2026_03.html')['sha256'],numeric_source_max=value,original_approximate_flag=obj.get('approximate') if isinstance(obj,dict) else None,molecule_scope=d['molecule_scope'],canonical_property_scope=d['canonical_property_supervision'],assertion_direct_experimental_ECO=direct,all_local_evidence_json=d['evidence_bindings_json'],assay_pH=None,assay_temperature=None,new_numeric_task_admitted=False,task_gate='schemaunitresolved;canonical/form/chromophore_mutant_notes/context_and_numeric_objective_review_separate;originalfullannotationtargetalreadyretained'))
p=B/'absorption-schema-unit-projection.parquet';pq.write_table(pa.Table.from_pylist(rows),p,compression='zstd');assert pq.read_table(p).to_pylist()==rows
review=dict(schema='mora3.UniProtPhysicalUnitSourceReview/v1',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),official_manual=ref(B/'research/manual-2026_03.html'),manual_release='2026_03',manual_text_only_reference_not_training_text=True,source_original=ref(source_manifest),absorption_projection=ref(p,rows=len(rows)),counts=dict(stats),absorption_unit_status='schema_declared_nm;not_a_literal_instance_unit',mass_spectrometry_unit_status='unresolved_in_selected_official_definitions;measured_molWeight_not_given_SQcomputed_Da_unit_by_analogy',calculated_sequence_mass='Da_explicit_in_SQmanual_and_RDF;providercomputednotmeasuredprocessedproteinmass',Km_and_Vmax='record-specific_literal_units_retained;maximumVelocity_never_kcat;globalpH_temperature_comments_notassayconditions',redox_pH_temperature='nativefree_text_retained;noinventedpI_unit_or_globalassaycontext',new_corpus_tasks=0,new_numeric_model_objectives_admitted=0,new_experiments=0,source_raw_or_old_outputs_changed=False,code=ref(Path('scripts/data_intake_review_physical_units.py')))
(B/'manifest.json').open('x').write(json.dumps(review,indent=2)+'\n');print('COMPLETE',json.dumps(dict(stats)),[r.get('status') for r in receipts])
