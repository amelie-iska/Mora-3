from pathlib import Path
import json
R=Path('/home/iska/Documents/amelie/bio/Mora-3');P=R/'planning/research/data_intake/20261101-DRAMP-offline-preparation-v1/actual-source-role-review-v1'
for name in ['all110-context-literal-source-row-review.json','native-modification-field-full-row-inventory.json']:
 d=json.loads((P/name).read_text());print(name,'KEYS',list(d))
 if 'source_field_roles' in d:
  for field,byliteral in d['source_field_roles'].items():
   print('FIELD',field)
   for literal,v in byliteral.items():print('REVIEW_ROLE',repr(literal),{k:x for k,x in v.items() if k!='rows'},'ROW_KEYS',list(v['rows'][0]) if v.get('rows') else [])
 else:
  for k,v in d.items():
   if isinstance(v,list):print('LIST',k,len(v),'FIRST_KEYS',list(v[0]) if v and isinstance(v[0],dict) else None)
   elif isinstance(v,dict):print('DICT',k,'KEYS',list(v)[:6])
