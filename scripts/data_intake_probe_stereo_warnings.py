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
 if sh.get('source_family') not in [None,'ReactZyme']:continue
 for row in pq.read_table(R/sh['path'],columns=['source_family','native_payload','task_id']).to_pylist():
  if row['source_family']!='ReactZyme':continue
  d=json.loads(row['native_payload'])
  for role,smi in [('substrate',d['conditions']['substrate_smiles']),('product',d['targets']['product_smiles'])]:
   messages.clear();Chem.MolFromSmiles(smi)
   if messages:
    x=dict(source_task_id=row['task_id'],role=role,native_source_smiles=smi,RDKit_messages=list(messages),source_task_unchanged=True);o=R/'outputs/data-intake/20261101-PubChem-corpus-pilot-v1/staged-tests/real-source-stereo-warning-fixture.json';o.open('x').write(json.dumps(x,indent=2)+'\n');print(json.dumps(x));sys.exit(0)
raise RuntimeError('No source parse warning in ReactZyme')
