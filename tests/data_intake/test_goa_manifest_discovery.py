import importlib.util
from pathlib import Path
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').exists())
s=importlib.util.spec_from_file_location('goa_discovery',ROOT/'scripts/data_intake_goa_manifest_discovery.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_discovery_keeps_legacy_and_new_parallel_completed_manifest_names(tmp_path):
 (tmp_path/'curated').mkdir();names=['manifest-full-part-a-v1-batch-0164.json','manifest-quickgo-parallel-v2-annotation-0-batch-0000.json']
 for n in names:(tmp_path/'curated'/n).write_text('{}')
 assert [p.name for p in m.candidate_source_manifests(tmp_path)]==names

def test_raw_ontology_and_partial_files_cannot_enter_annotation_manifest_candidates(tmp_path):
 (tmp_path/'curated').mkdir();(tmp_path/'raw').mkdir()
 for n in ['manifest-quickgo-parallel-v2-ontology-0-batch-0000.json','manifest-quickgo-parallel-v2-annotation-0-batch-0000.json.tmp']:(tmp_path/'curated'/n).write_text('{}')
 (tmp_path/'raw/manifest-full-part-a-v1-batch-0165.json').write_text('{}')
 assert m.candidate_source_manifests(tmp_path)==[]
