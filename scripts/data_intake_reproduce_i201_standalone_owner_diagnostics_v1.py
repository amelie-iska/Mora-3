from pathlib import Path
import datetime,hashlib,json,os,subprocess
R=Path('/home/iska/Documents/amelie/bio/Mora-3');O=R/'planning/research/data_intake/20261101-RGD1-corpus-consumption-owner-review-v1';O.mkdir(exist_ok=False)
env=dict(os.environ);env.pop('PYTHONPATH',None);env['PYTHONDONTWRITEBYTECODE']='1'
code="""import importlib.util
p='/home/iska/Documents/amelie/bio/Mora-3/outputs/implementation/i201-rgd1-corpus-guard/data_intake_corpus_candidate.py'
s=importlib.util.spec_from_file_location('i201_candidate_clean',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
t=m.CorpusTask('ordinary_property',{'molecule':'C'},{'source_value':1},{'source_family':'FreeSolv'},[])
print(t.input_bytes())
"""
r=subprocess.run(['/home/iska/miniconda3/envs/tokengt/bin/python','-B','-c',code],cwd='/tmp',env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=30)
(O/'original-candidate-standalone-no-PYTHONPATH.log').open('x').write(r.stdout+r.stderr)
code2="""import sys
sys.path.insert(0,'/home/iska/Documents/amelie/bio/Mora-3/src')
from mora3.data.rgd1_consumption import require_corpus_consumption
require_corpus_consumption({'source_family':'UniProt-full-record-annotations','native_payload':b'{"task":"protein_function","audit":{"gene_primary":"RGD1"}}'},purpose='objective')
"""
s=subprocess.run(['/home/iska/miniconda3/envs/tokengt/bin/python','-B','-c',code2],cwd='/tmp',env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=30)
(O/'existing-guard-unrelated-gene-name-scope-counterexample.log').open('x').write(s.stdout+s.stderr)
assert r.returncode!=0 and 'ModuleNotFoundError' in r.stderr
assert s.returncode!=0 and 'RGD1 energy quarantine' in s.stderr
p=O/'initial-owner-diagnostics.json';p.open('x').write(json.dumps(dict(schema='mora3.CorpusOwnerGuardInitialDiagnostics/v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),original_candidate_clean_standalone_exit=r.returncode,original_candidate_unrelated_family_ModuleNotFoundError_reproduced=True,existing_model_guard_unrelated_gene_name_counterexample_exit=s.returncode,model_guard_scope_counterexample='An unrelated UniProt-like synthetic audit gene_primary=RGD1 triggers recursive startswith identity detector; requires model-owner/root disposition, not an actual observed native row',all_records_synthetic=True,actual_source_rows_used=0,live_corpus_or_model_edits=0),indent=2)+'\n')
print(p);print('ORIGINAL_ERROR',r.stderr.splitlines()[-1]);print('SCOPE_ERROR',s.stderr.splitlines()[-1])
