from pathlib import Path
import json
p=Path('/home/iska/Documents/amelie/bio/Mora-3/planning/research/data_intake/20261101-DRAMP-offline-preparation-v1/actual-source-role-review-v1/native-modification-field-full-row-inventory.json')
d=json.loads(p.read_text())
for k,v in d['native_fields'].items():
 print('FIELD',k,'TYPE',type(v).__name__)
 if isinstance(v,list):print('FIRST_ROW_KEYS',list(v[0]) if v else [])
 elif isinstance(v,dict):
  print('LEVEL_KEYS',list(v)[:8])
  for x,y in v.items():
   print('VALUE_TYPE',type(y).__name__,'KEYS',list(y) if isinstance(y,dict) else None);break
