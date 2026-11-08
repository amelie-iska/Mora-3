from pathlib import Path
import json
R=Path('/home/iska/Documents/amelie/bio/Mora-3')
p=R/'outputs/data-intake/20261101-ADMET-corpus-v1/source-endpoint-and-joint-benchmark-review-v3/endpoint-review-registry.json';d=json.loads(p.read_text());print('REGISTRY_KEYS',list(d));print('FIRST',json.dumps(d['endpoint_reviews'][0],indent=2));print('COORDINATE',json.dumps([v for v in d['endpoint_reviews'] if v.get('native_coordinate_only')],indent=2)[:4500])
q=json.loads((R/'planning/data_intake_progress.json').read_text());print('PROGRESS_ADMET_KEYS',list(q['scope_extensions']['ADMET-CORPUS-01']));print('LATEST_HARD_GATES',q.get('full_dataset_training_gate'))
