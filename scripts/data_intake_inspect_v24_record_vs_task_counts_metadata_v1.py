from pathlib import Path
import json
R=Path('/home/iska/Documents/amelie/bio/Mora-3');m=json.loads((R/'outputs/data-intake/20261101-ADMET-corpus-v1/independent-master-union-v24-ExpansionRx-v1/master-corpus-manifest-v24-ExpansionRx-native-v1.json').read_text());rd=json.loads((R/'outputs/data-intake/20261101-RNAcentral-full-native-corpus-v1/model-handoff-v23/reader-readiness.json').read_text())
counts={f['source_family']:f['serialized_tasks'] for f in rd['families']};counts['ExpansionRx-source-native-ADMET-readouts']=61426
print('TASKS',m['serialized_tasks'],'MASTER_NAMED_RECORD_MAP_SUM',sum(m['serialized_distinct_record_counts_by_source'].values()),'INHERITED_READINESS_PLUS_NEW_VIEW_SUM',sum(counts.values()))
print('DIVERGENT_MAP_UNITS',[(k,v,counts.get(k)) for k,v in m['serialized_distinct_record_counts_by_source'].items() if v!=counts.get(k)])
print('MISSING_SHARD_FAMILY_METADATA',sum('source_family' not in s or s['source_family'] is None for s in m['shards']))
