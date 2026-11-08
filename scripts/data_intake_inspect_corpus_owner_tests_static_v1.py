from pathlib import Path
R=Path('/home/iska/Documents/amelie/bio/Mora-3')
for p in (R/'tests/data_intake').glob('test*.py'):
 s=p.read_text()
 if 'RGD1' in s or 'rgd1' in s:
  print('TEST_FILE',p.name)
  lines=s.splitlines()
  for i,line in enumerate(lines):
   if 'input_bytes' in line or 'rgd1' in line.lower():print(i+1,line[:220])
for n in ['pytest.ini','setup.cfg','environment.yaml']:
 p=R/n
 if p.exists():print('CONFIG',n,p.read_text()[:2500])
