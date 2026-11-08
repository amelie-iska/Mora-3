from pathlib import Path
import json
R=Path('/home/iska/Documents/amelie/bio/Mora-3')
p=R/'planning/rgd1-v6-energy-objective-quarantine-20261101T195010Z-v1.json';print('POLICY',p.read_text())
m=json.loads((R/'outputs/data-intake/20261101-RNAcentral-full-native-corpus-v1/independent-master-union-v23/master-corpus-manifest-v23-whole-RNAcentral-native-v1.json').read_text())
for key in ['serialized_distinct_record_counts_by_source','contexts','source_ledger','source_curation_manifests']:
 v=m[key];items=v.items() if isinstance(v,dict) else enumerate(v)
 for k,x in items:
  s=json.dumps(x)
  if 'rgd1' in (str(k)+s).lower():print('AFFECTED_METADATA',key,k,json.dumps(x)[:9000])
families={x.get('source_family') for x in m['shards'] if 'rgd1' in str(x.get('source_family','')).lower()};print('SHARD_FAMILIES',families)
for family in families:
 selected=[x for x in m['shards'] if x.get('source_family')==family];print('SHARD_ROLE',family,len(selected),sum(x['rows'] for x in selected));print('FIRST_SHARD',selected[0])
