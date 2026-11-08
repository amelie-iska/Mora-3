from pathlib import Path
import json,hashlib
import pyarrow.parquet as pq
R=Path('/home/iska/Documents/amelie/bio/Mora-3')
paths=['outputs/data-intake/20261101-ADMET-corpus-v1/DRAMP4-stability-source-symbolic-candidate-component-v1/manifest.json','outputs/monitoring/closure-20261101T201240Z-DRAMP110-source-candidate-component-v1/closure.json','outputs/data-intake/20261101-ADMET-corpus-v1/DRAMP4-stability-whole-native-preparation-v1/manifest.json','planning/research/data_intake/20261101-DRAMP-offline-preparation-v1/actual-source-role-review-v1/manifest.json']
for f in paths:
 p=R/f;print('META',f,p.stat().st_size,hashlib.sha256(p.read_bytes()).hexdigest());d=json.loads(p.read_text());print(json.dumps(d,indent=2)[:13000])
p=R/'outputs/data-sources/20261101-ADMET-cross-modality-v1/curated/DRAMP4-full-native-v1/stability_amps.xlsx.native-rows.parquet';pf=pq.ParquetFile(p);print('ORIGINAL_NATIVE_FOOTER_SCHEMA_ONLY',pf.metadata.num_rows,str(pf.schema_arrow))
for x in (R/'scripts').glob('*dramp*'):print('SOURCE_CODE_PATH',x.name)
for x in (R/'scripts').glob('*DRAMP*'):print('OUR_CODE_PATH',x.name)
