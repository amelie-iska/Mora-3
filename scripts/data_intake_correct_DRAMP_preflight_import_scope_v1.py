from pathlib import Path
import json,hashlib,datetime
R=Path('/home/iska/Documents/amelie/bio/Mora-3');O=R/'planning/research/data_intake/20261101-DRAMP-offline-preparation-v1/writer-execution-review-v2'
def ref(p):return dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def write(p,d):p.open('x').write(json.dumps(d,indent=2)+'\n')
old=O/'preflight-current-pins-AST-quiescence.json';d=json.loads(old.read_text());assert d.pop('candidate_builder_imported_or_called') is False
# The approval-contract import transitively imports logic and the builder module,
# without invoking a builder/projector or decoding source data rows.
d.update(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),supersedes_preflight=ref(old),scope_correction='Prior metadata boolean incorrectly combined module import and function call; pure writer import transitively loaded builder definitions. No source candidate builder/projector was called.',candidate_builder_module_imported_as_transitive_writer_dependency=True,candidate_builder_called=False,source_project_row_called=False,source_rows_projected=0,original_quiescence_evidence_time_scope='Inherited actualpreflight timestamp in superseded receipt; no repeated nativecheck or component execution')
new=O/'preflight-current-pins-AST-quiescence-corrected-import-scope-v1.json';write(new,d)
p=O/'execution-review-package.json';package=json.loads(p.read_text());package.update(preflight=ref(new),supersedes_review_package=ref(p),import_scope_correction='Pure approval-contract import loaded code definitions transitively; zero builder/projector/source-row invocations. Corrected preflight supersedes the erroneous combined import/call boolean; old artifacts retained.',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
for pin in package['current_pins']:
 pp=R/pin['path'];assert ref(pp)==pin,pin['path']
np=O/'execution-review-package-corrected-import-scope-v1.json';write(np,package);print(json.dumps(ref(np)));print('CORRECTED_PREFLIGHT',ref(new))
