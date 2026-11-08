"""New synthetic owner checks; no native sources, weights or training updates."""
from pathlib import Path
import hashlib,importlib.util,json,os,subprocess,sys
import pytest
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'scripts/data_intake_runtime.py').is_file())
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from mora3.data.array_tasks import ArrayTask,OrderedArray
from mora3.data import rgd1_consumption as gate
CANDIDATE=Path(os.environ.get('MORA3_RUNTIME_OWNER_CANDIDATE',ROOT/'scripts/data_intake_runtime.py'))

def runtime():
 s=importlib.util.spec_from_file_location('owner_direct_runtime',CANDIDATE);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def row(split='train',target=1.):
 t=ArrayTask((OrderedArray('known',('a','b'),(2.,3.),'synthetic','u','observed'),OrderedArray('hidden',('a','b'),(target,4.),'synthetic','u','observed')),(('known','condition'),('hidden','target')),'0'*64);raw=t.canonical_bytes()
 return dict(task_id='synthetic',record_id='synthetic-source-only',native_payload=raw,native_payload_sha256=hashlib.sha256(raw).hexdigest(),native_schema='mora3.ArrayTask/v1',admitted_for_task=True,split=split,source_family='Synthetic-unrelated',source_revision='v1',source_sha256='0'*64,group_ids_json='["synthetic"]',group_metadata_status='manifest_verified_effective_groups',model_adapter_supported=True)

@pytest.mark.parametrize('split,training',[('train',True),('validation',False),('test',False)])
@pytest.mark.parametrize('field',['source_record_id','source_reaction_family','task_name'])
def test_direct_projector_refuses_alternate_source_identity_before_any_native_decode(split,training,field):
 r=row(split);r['audit_identity']={'nested':[dict([(field,' RgD1:unqualified')])]};r['native_payload']=b'not-a-native-format';r['native_payload_sha256']='incorrect';r['loss_mask']=0;r['audit_only']=True;r['source_revision']='claimed-corrected-v9'
 with pytest.raises(ValueError,match='RGD1.*quarantine'):runtime().project_corpus_row(r,manifest_sha256='1'*64,for_training=training,allow_profile_inspection=True)

@pytest.mark.parametrize('split,training',[('train',True),('validation',False),('test',False)])
def test_unrelated_runtime_native_objective_and_evaluation_preserve_inputs_targets_masks(split,training):
 r=row(split);r['annotations']={'gene_primary':'RGD1','text':'RGD1 source biological content'};m=runtime();a=m.project_corpus_row(r,manifest_sha256='1'*64,for_training=training)
 assert a['model_inputs']['known'].tolist()==[2.,3.] and a['supervision_targets']['hidden'].tolist()==[1.,4.]
 assert a['supervision_masks']['hidden'].all() and a['context_masks']['physical_dose'] is False
 r2=row(split,99.);b=m.project_corpus_row(r2,manifest_sha256='1'*64,for_training=training)
 assert m.input_fingerprint(a)==m.input_fingerprint(b) and a['resume_binding']!=b['resume_binding']


def test_unrelated_heldout_still_cannot_be_requested_as_training():
 with pytest.raises(ValueError,match='heldout'):runtime().project_corpus_row(row('test'),manifest_sha256='1'*64,for_training=True)


def test_direct_runtime_guard_runs_before_admission_or_payload_access_and_selects_role(monkeypatch):
 m=runtime();seen=[]
 def stop(r,*,purpose):seen.append(purpose);raise RuntimeError('STOP_BEFORE_PROJECTION')
 monkeypatch.setattr(m,'require_corpus_consumption',stop)
 for train in (True,False):
  with pytest.raises(RuntimeError,match='STOP_BEFORE_PROJECTION'):m.project_corpus_row({},manifest_sha256='1'*64,for_training=train)
 assert seen==['objective','evaluation']


def test_fresh_policy_is_checked_again_after_a_warm_direct_refusal(tmp_path,monkeypatch):
 m=runtime();r=row();r['source_family']='RGD1-computed-energy';p=tmp_path/'synthetic-policy-copy.json';p.write_bytes(gate.POLICY_PATH.read_bytes());monkeypatch.setattr(gate,'POLICY_PATH',p)
 with pytest.raises(ValueError,match='RGD1.*quarantine'):m.project_corpus_row(r,manifest_sha256='1'*64,for_training=True)
 p.write_bytes(b'{}')
 with pytest.raises(ValueError,match='policy integrity'):m.project_corpus_row(r,manifest_sha256='1'*64,for_training=True)
 assert r['native_payload']==row()['native_payload']


def mirror(tmp_path):
 # Read-only references to existing CODE, no source/data alias changes. The
 # candidate is tested in its actual proposed scripts/ layout so its unchanged
 # ROOT/src+ROOT/scripts bootstrap is exercised without pytest/PYTHONPATH.
 r=tmp_path/'synthetic-runtime-repository';(r/'scripts').mkdir(parents=True);(r/'src').symlink_to(ROOT/'src',target_is_directory=True)
 (r/'scripts/data_intake_runtime.py').write_bytes(CANDIDATE.read_bytes());(r/'scripts/data_intake_corpus.py').write_bytes((ROOT/'scripts/data_intake_corpus.py').read_bytes());return r

def clean(argv,cwd):
 env=dict(os.environ);env.pop('PYTHONPATH',None);env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
 return subprocess.run([sys.executable,'-B',*argv],cwd=cwd,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=30)

@pytest.mark.parametrize('identified',[False,True])
def test_actual_standalone_scripts_layout_without_PYTHONPATH(tmp_path,identified):
 r=mirror(tmp_path);native=row();native['source_family']='RGD1-computed-energy' if identified else 'Synthetic-unrelated';native['annotations']={'gene_primary':'RGD1'}
 # Tiny manufactured array envelope only. No native source fixture is replayed.
 d={k:v for k,v in native.items() if k!='native_payload'};d['native_payload_hex']=native['native_payload'].hex();j=tmp_path/'synthetic-row.json';j.write_text(json.dumps(d));script=r/'scripts/data_intake_runtime.py'
 code=f"import importlib.util,json; s=importlib.util.spec_from_file_location('target_layout',{str(script)!r});m=importlib.util.module_from_spec(s);s.loader.exec_module(m);r=json.load(open({str(j)!r}));r['native_payload']=bytes.fromhex(r.pop('native_payload_hex'));p=m.project_corpus_row(r,manifest_sha256='1'*64,for_training=True);assert p['model_inputs']['known'].tolist()==[2.,3.];print('STANDALONE_DIRECT_OK')"
 got=clean(['-c',code],tmp_path)
 if identified:assert got.returncode!=0 and 'RGD1 energy quarantine' in got.stderr and 'ModuleNotFoundError' not in got.stderr
 else:assert got.returncode==0 and got.stdout.strip()=='STANDALONE_DIRECT_OK',got.stderr


def test_standalone_CLI_help_has_no_PYTHONPATH_or_install_requirement(tmp_path):
 r=mirror(tmp_path);got=clean([str(r/'scripts/data_intake_runtime.py'),'--help'],tmp_path)
 assert got.returncode==0 and '--manifest' in got.stdout and 'ModuleNotFoundError' not in got.stderr
