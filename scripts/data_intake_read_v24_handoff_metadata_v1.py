from pathlib import Path
import json,hashlib
R=Path('/home/iska/Documents/amelie/bio/Mora-3')
paths=['outputs/monitoring/attestation-20261101T203757Z-constructor-quiescence-v24-handoff-preparation-v1/attestation.json','outputs/monitoring/closure-20261101T202009Z-ExpansionRx-v24-exact-staging-v1/closure.json','outputs/data-intake/20261101-ADMET-corpus-v1/independent-master-union-v24-ExpansionRx-v1/receipt.json','outputs/data-intake/20261101-ADMET-corpus-v1/independent-master-union-v24-ExpansionRx-v1/staging-receipt.json']
for f in paths:
 p=R/f;print('FILE',f,p.stat().st_size,hashlib.sha256(p.read_bytes()).hexdigest());print(p.read_text()[:15000])
h=R/'outputs/data-intake/20261101-RNAcentral-full-native-corpus-v1/model-handoff-v23'
for name in ['reader-readiness.json','executable-model-handoff.json']:
 d=json.loads((h/name).read_text());print('OLD_HANDOFF_KEYS',name,list(d));print('COUNTS',d.get('counts'))
 if 'families' in d:print('EXISTING_FAMILY',json.dumps(d['families'][0],indent=2)[:4500])
 if 'tasks' in d:print('TASK_IDS',[x['id'] for x in d['tasks']])
print('AGENTS_LAST', (R/'AGENTS.md').read_text()[-6000:])
