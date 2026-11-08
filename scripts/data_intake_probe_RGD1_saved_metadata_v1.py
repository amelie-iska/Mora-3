from pathlib import Path
import json
R=Path('/home/iska/Documents/amelie/bio/Mora-3')
p=R/'outputs/data-sources/20261031-expansion-v1/curated/source-native-v1/manifest.json';d=json.loads(p.read_text());print('SOURCE_NATIVE_KEYS',list(d));
for k,v in d.items():
 if isinstance(v,list):
  for x in v:
   if any(t in json.dumps(x).lower() for t in ['rgd1','39521815','38170374']):print('SOURCE_EVIDENCE',k,json.dumps(x)[:10000])
 elif 'rgd1' in (str(k)+json.dumps(v)).lower():print('SOURCE_EVIDENCE',k,json.dumps(v)[:8000])
q=R/'outputs/data-intake/20261031-final-corpus-build-v1/source-admission-v1';print('ADMISSION_FILES',[(x.name,x.stat().st_size) for x in q.iterdir() if x.is_file()])
for p in sorted((R/'scripts').glob('data_intake*.py')):
 s=p.read_text()
 if 'RGD1-computed-energy' in s:
  lines=s.splitlines();found=[]
  for i,line in enumerate(lines):
   if 'RGD1' in line or 'activation_energy' in line or 'reaction_energy' in line:
    found.extend(f'{j+1}:{lines[j]}' for j in range(max(0,i-1),min(len(lines),i+2)))
  print('CODE_SELECTOR',p.name,'\n'.join(found[:35]))
