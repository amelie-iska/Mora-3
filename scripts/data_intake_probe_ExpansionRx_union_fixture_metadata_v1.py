from pathlib import Path
import json,sqlite3
R=Path('/home/iska/Documents/amelie/bio/Mora-3')
f=R/'outputs/data-intake/20261101-ADMET-corpus-v1/ExpansionRx-closed-native-reader-fixture-v1/manifest.json';d=json.loads(f.read_text());print('FIXTURE_KEYS',list(d));
for k,v in d.items():
 if isinstance(v,list): print(k,len(v),json.dumps(v[:2])[:3500])
 elif isinstance(v,dict):print(k,list(v))
p=R/'outputs/data-intake/20261101-RNAcentral-full-native-corpus-v1/materialized/task-identity-serialization-index.sqlite'
c=sqlite3.connect('file:'+str(p)+'?mode=ro&immutable=1',uri=True)
print('RNA_TASK_INDEX_SCHEMA',c.execute('PRAGMA table_info(tasks)').fetchall());c.close()
