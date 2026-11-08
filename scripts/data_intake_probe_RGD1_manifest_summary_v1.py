from pathlib import Path
import json
R=Path('/home/iska/Documents/amelie/bio/Mora-3');q=R/'outputs/data-intake/20261031-final-corpus-build-v1/source-admission-v1'
print('FILES',[(x.name,x.stat().st_size) for x in q.iterdir() if x.is_file()])
d=json.loads((R/'outputs/data-sources/20261031-expansion-v1/curated/source-native-v1/manifest.json').read_text())
print('SOURCE_RAW',json.dumps([x for x in d['raw_assets'] if any(t in json.dumps(x) for t in ['39521815','38170374'])],indent=2))
for p in q.glob('*.json'):
 d=json.loads(p.read_text());print('META_KEYS',p.name,list(d))
 for k,v in d.items():
  if k in ['code','script','code_version','counts','source_counts','inputs','sources','task_kinds','native_schemas']:print(k,json.dumps(v)[:3000])
for p in (R/'scripts').glob('*.py'):
 if p.name.startswith('data_intake_probe'):continue
 s=p.read_text()
 if 'RGD1Energy' in s or 'activation_energy_forward' in s or 'rgd1_computed' in s:
  print('STATIC_NATIVE_SCRIPT',p.name)
  for i,line in enumerate(s.splitlines()):
   if any(t in line.lower() for t in ['rgd1','activation','dft','energy']):print(i+1,line[:700])
