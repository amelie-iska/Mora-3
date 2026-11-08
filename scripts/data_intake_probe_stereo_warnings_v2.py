from pathlib import Path
import sys,json,logging
from rdkit import Chem,rdBase
import pyarrow.parquet as pq
R=Path('/home/iska/Documents/amelie/bio/Mora-3');sys.path.insert(0,str(R/'scripts'))
rdBase.LogToPythonLogger();messages=[]
class Capture(logging.Handler):
 def emit(self,record):messages.append(record.getMessage())
logging.getLogger('rdkit').addHandler(Capture())
m=json.loads((R/'outputs/data-intake/20261101-PubChem-corpus-pilot-v1/independent-master-union-v20/master-corpus-manifest-v20-PubChem-native-codec-pilot-v1.json').read_text())
for sh in m['shards']:
 if sh.get('source_family') not in [None,'ReactZyme','RGD1-computed-energy']:continue
 for row in pq.read_table(R/sh['path'],columns=['source_family','native_payload','task_id']).to_pylist():
  if row['source_family'] not in ['ReactZyme','RGD1-computed-energy']:continue
  d=json.loads(row['native_payload'])
  for role,smi in [(k,v) for block in [d['conditions'],d['targets']] for k,v in block.items() if 'smiles' in k and isinstance(v,str)]:
   messages.clear();Chem.MolFromSmiles(smi)
   if any(any(word in msg.lower() for word in ['stereo','conflict','ambiguous']) for msg in messages):
    x=dict(source_task_id=row['task_id'],role=role,native_source_smiles=smi,RDKit_messages=list(messages),source_task_unchanged=True);o=R/'outputs/data-intake/20261101-PubChem-corpus-pilot-v1/staged-tests/real-source-stereo-warning-fixture-v2.json';o.open('x').write(json.dumps(x,indent=2)+'\n');print(json.dumps(x));sys.exit(0)
raise RuntimeError('No source parse warning in ReactZyme')
