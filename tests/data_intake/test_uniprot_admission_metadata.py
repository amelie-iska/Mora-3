import importlib.util,json
from pathlib import Path
import pyarrow as pa,pyarrow.parquet as pq,pytest
root=next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').exists());spec=importlib.util.spec_from_file_location('admit',root/'scripts/data_intake_admit_uniprot_annotations.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def test_shard_without_summary_split_uses_actual_envelope_split(tmp_path):
 assert hasattr(m,'resolved_shard_rows'),'envelope-based source/split resolution missing'
 source=tmp_path/'source.parquet';groups=tmp_path/'groups.parquet';pq.write_table(pa.Table.from_pylist([dict(task_id='a',source_family='native',split='train')]),source);pq.write_table(pa.Table.from_pylist([dict(task_id='a',effective_groups_json='["accession:P00001"]')]),groups)
 assert m.resolved_shard_rows(source,groups,{})==[dict(task_id='a',source_family='native',split='train',effective_groups_json='["accession:P00001"]')]
def test_group_sidecar_cannot_reassign_or_omit_tasks(tmp_path):
 assert hasattr(m,'resolved_shard_rows'),'envelope-based source/split resolution missing'
 source=tmp_path/'source.parquet';groups=tmp_path/'groups.parquet';pq.write_table(pa.Table.from_pylist([dict(task_id='a',source_family='native',split='test')]),source);pq.write_table(pa.Table.from_pylist([dict(task_id='other',effective_groups_json='[]')]),groups)
 with pytest.raises(ValueError):m.resolved_shard_rows(source,groups,{})

def test_validation_name_is_preserved_from_existing_corpus(tmp_path):
 source=tmp_path/'source.parquet';groups=tmp_path/'groups.parquet';pq.write_table(pa.Table.from_pylist([dict(task_id='a',source_family='FreeSolv',split='validation')]),source);pq.write_table(pa.Table.from_pylist([dict(task_id='a',effective_groups_json='[]')]),groups)
 assert m.resolved_shard_rows(source,groups,{'split':'validation'})[0]['split']=='validation'
