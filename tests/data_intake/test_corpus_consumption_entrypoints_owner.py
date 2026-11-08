"""Synthetic corpus entrypoint policy checks, including clean standalone scripts."""
from pathlib import Path
import hashlib,importlib.util,json,os,subprocess,sys,types
import pytest
import pyarrow as pa
import pyarrow.parquet as pq
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'scripts/data_intake_corpus.py').is_file())
CANDIDATE=Path(os.environ.get('MORA3_CORPUS_OWNER_CANDIDATE',ROOT/'scripts/data_intake_corpus.py'))


def module():
 s=importlib.util.spec_from_file_location('owner_corpus_entrypoint_candidate',CANDIDATE);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def task_row(split='train',rgd=False,supported=True,opaque=False):
 raw=(b'\x81not-json-native-array' if opaque else json.dumps({'schema':'mora3.CorpusTask/v1','task':'RGD1_energy' if rgd else 'synthetic_property','conditions':{'molecule':'C'},'targets':{'source_value':1},'audit':{'source_family':'RGD1-computed-energy' if rgd else 'Synthetic-unrelated'},'missing':[]},sort_keys=True).encode())
 return dict(task_id='synthetic-task',record_id='synthetic-source-row',source_family='RGD1-computed-energy' if rgd else 'Synthetic-unrelated',split=split,model_adapter_supported=supported,native_schema='mora3.ArrayTask/v1' if opaque else 'mora3.CorpusTask/v1',native_payload=raw,native_payload_sha256=hashlib.sha256(raw).hexdigest())


def manifest(root,row):
 p=root/'row.parquet';pq.write_table(pa.Table.from_pylist([row]),p)
 f=root/'manifest.json';f.write_text(json.dumps({'shards':[{'path':'row.parquet','rows':1,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}]}));return f


def test_codec_preserves_RGD1_audit_bytes_but_conditioning_refuses():
 m=module();t=m.CorpusTask('RGD1_energy',{'reactant':'C'},{'computed_energy':1},{'source_family':'RGD1-computed-energy'},[]);raw=t.canonical_bytes()
 assert m.CorpusTask.from_bytes(raw).canonical_bytes()==raw
 with pytest.raises(ValueError,match='RGD1.*quarantine'):t.input_bytes()
 assert t.canonical_bytes()==raw


def test_unrelated_conditions_are_unchanged_and_targets_remain_hidden():
 m=module();t=m.CorpusTask('synthetic_property',{'molecule':'C'},{'source_value':1},{'source_family':'FreeSolv'},[])
 assert t.input_bytes()==m.canonical({'task':'synthetic_property','conditions':{'molecule':'C'}})

@pytest.mark.parametrize('split',['train','validation','test'])
@pytest.mark.parametrize('supported_only',[True,False])
def test_all_consumed_RGD1_splits_refuse_even_with_legacy_flags(tmp_path,split,supported_only):
 m=module();row=task_row(split,rgd=True,supported=True);row.update(loss_mask=0,audit_only=True,source_release='v9');f=manifest(tmp_path,row)
 with pytest.raises(ValueError,match='RGD1.*quarantine'):next(m.read_corpus_tasks(f,root=tmp_path,split=split,supported_only=supported_only))

@pytest.mark.parametrize('split,purpose',[('train','objective'),('validation','evaluation'),('test','evaluation')])
def test_consumption_purpose_is_explicit_and_precedes_yield(tmp_path,split,purpose):
 m=module();calls=[];m._corpus_consumption_guard=types.SimpleNamespace(require_corpus_consumption=lambda row,*,purpose:calls.append(purpose));f=manifest(tmp_path,task_row(split))
 assert len(list(m.read_corpus_tasks(f,root=tmp_path,split=split)))==1 and calls==[purpose]


def mirror(tmp_path,guard=True):
 r=tmp_path/'synthetic-repository';(r/'scripts').mkdir(parents=True);(r/'scripts/data_intake_corpus.py').write_bytes(CANDIDATE.read_bytes())
 if guard:
  d=r/'src/mora3/data';d.mkdir(parents=True);(r/'src/mora3/__init__.py').write_text('"""Synthetic package shell."""\n');(d/'__init__.py').write_text('"""Synthetic package shell."""\n');(d/'rgd1_consumption.py').write_bytes((ROOT/'src/mora3/data/rgd1_consumption.py').read_bytes());p=r/'planning';p.mkdir();(p/'rgd1-v6-energy-objective-quarantine-20261101T195010Z-v1.json').write_bytes((ROOT/'planning/rgd1-v6-energy-objective-quarantine-20261101T195010Z-v1.json').read_bytes())
 return r


def run_clean(argv,cwd):
 env=dict(os.environ);env.pop('PYTHONPATH',None);env['PYTHONDONTWRITEBYTECODE']='1';env['OMP_NUM_THREADS']='1';env['OPENBLAS_NUM_THREADS']='1';env['CUDA_VISIBLE_DEVICES']=''
 return subprocess.run([sys.executable,'-B',*argv],cwd=cwd,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=30)

@pytest.mark.parametrize('opaque',[False,True])
def test_standalone_CLI_without_PYTHONPATH_reads_unrelated_synthetic_family(tmp_path,opaque):
 r=mirror(tmp_path);f=manifest(r,task_row(opaque=opaque));result=run_clean([str(r/'scripts/data_intake_corpus.py'),'--manifest',str(f),'--limit','1'],tmp_path)
 assert result.returncode==0,result.stderr
 assert json.loads(result.stdout)['probe_rows']==1 and 'ModuleNotFoundError' not in result.stderr

@pytest.mark.parametrize('split',['train','validation','test'])
def test_standalone_CLI_without_PYTHONPATH_refuses_RGD1_with_policy_error(tmp_path,split):
 r=mirror(tmp_path);f=manifest(r,task_row(split,rgd=True,supported=False));result=run_clean([str(r/'scripts/data_intake_corpus.py'),'--manifest',str(f),'--split',split,'--include-unsupported','--limit','1'],tmp_path)
 assert result.returncode!=0 and 'RGD1 energy quarantine' in result.stderr and 'ModuleNotFoundError' not in result.stderr


def test_standalone_audit_serialization_needs_no_source_guard_import(tmp_path):
 r=mirror(tmp_path,guard=False);script=r/'scripts/data_intake_corpus.py';code=f"import importlib.util; s=importlib.util.spec_from_file_location('codec_only',{str(script)!r});m=importlib.util.module_from_spec(s);s.loader.exec_module(m); t=m.CorpusTask('RGD1_energy',{{}},{{'energy':1}},{{}},[]); raw=t.canonical_bytes(); assert m.CorpusTask.from_bytes(raw).canonical_bytes()==raw; print('AUDIT_CODEC_OK')"
 result=run_clean(['-c',code],tmp_path);assert result.returncode==0 and result.stdout.strip()=='AUDIT_CODEC_OK',result.stderr


def test_standalone_conditioning_import_bootstrap_works_without_PYTHONPATH(tmp_path):
 r=mirror(tmp_path);script=r/'scripts/data_intake_corpus.py';code=f"import importlib.util; s=importlib.util.spec_from_file_location('input_only',{str(script)!r});m=importlib.util.module_from_spec(s);s.loader.exec_module(m); t=m.CorpusTask('synthetic_property',{{'molecule':'C'}},{{'value':1}},{{}},[]); assert b'molecule' in t.input_bytes(); print('CONDITIONING_OK')"
 result=run_clean(['-c',code],tmp_path);assert result.returncode==0 and result.stdout.strip()=='CONDITIONING_OK',result.stderr


@pytest.mark.parametrize('split',['train','validation','test'])
def test_recognized_identity_scope_accepts_unrelated_RGD1_gene_content(tmp_path,split):
 m=module();t=m.CorpusTask('sequence_to_curated_function_annotations',{'amino_acid_sequence':'MPEPTIDE'},{'source_value':'RGD1 biological content'}, {'source_family':'UniProt','gene_primary':'RGD1','free_text':'RGD1 annotated gene'},[])
 assert m.CorpusTask.from_bytes(t.canonical_bytes()).canonical_bytes()==t.canonical_bytes()
 assert json.loads(t.input_bytes())=={'task':'sequence_to_curated_function_annotations','conditions':{'amino_acid_sequence':'MPEPTIDE'}}
 row=task_row(split);row['native_payload']=t.canonical_bytes();row['native_payload_sha256']=hashlib.sha256(row['native_payload']).hexdigest()
 f=manifest(tmp_path,row);assert len(list(m.read_corpus_tasks(f,root=tmp_path,split=split)))==1


def test_standalone_unrelated_RGD1_gene_content_without_PYTHONPATH(tmp_path):
 r=mirror(tmp_path);row=task_row();payload=json.loads(row['native_payload']);payload['audit']['gene_primary']='RGD1';payload['targets']['annotation']='RGD1 annotation text';row['native_payload']=json.dumps(payload,sort_keys=True).encode();row['native_payload_sha256']=hashlib.sha256(row['native_payload']).hexdigest();f=manifest(r,row)
 result=run_clean([str(r/'scripts/data_intake_corpus.py'),'--manifest',str(f),'--limit','1'],tmp_path)
 assert result.returncode==0,result.stderr
 assert json.loads(result.stdout)['probe_rows']==1
