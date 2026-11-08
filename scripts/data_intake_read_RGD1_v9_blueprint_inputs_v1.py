from pathlib import Path
import json,hashlib
R=Path('/home/iska/Documents/amelie/bio/Mora-3');P=R/'outputs/data-sources/20261101-RGD1-original-native-geometry-v1/research/remaining-RPs-RitS-primary-release-rights-v1'
for name in ['complete-primary-release-rights-review-and-next-source-handoff-v1.json','eligible-v9-four-original-assets-acquisition-proposal-v1.json']:
 p=P/name;b=p.read_bytes();print('FILE',str(p.relative_to(R)),len(b),hashlib.sha256(b).hexdigest());print(b.decode())
I=R/'planning/research/data_intake/20261101-expanded-scope-v4/ExpansionRx-post-RNA-exact-union-preparation-v1/inherited-RGD1-v6-affected-scope-v1'
for name in ['receipt.json','affected-source-task-objective-inventory.json','current-readiness-addendum.json']:
 p=I/name;print('INHERITED',name,p.stat().st_size,hashlib.sha256(p.read_bytes()).hexdigest())
