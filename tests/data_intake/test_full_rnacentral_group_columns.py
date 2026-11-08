import json,sys
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
sys.path.insert(0,'/home/iska/Documents/amelie/bio/Mora-3/scripts')
from data_intake_full_rnacentral_proof_helpers import overlay_group_table_preserving_columns


def test_new_group_sidecar_keeps_rare_unknown_source_columns_and_schema(tmp_path):
 schema=pa.schema([('task_id',pa.string()),('effective_groups_json',pa.string()),('rare',pa.struct([('new_field',pa.list_(pa.binary()))]))])
 original=pa.Table.from_pylist([dict(task_id='a',effective_groups_json='["old-a"]',rare=None),dict(task_id='b',effective_groups_json='["old-b"]',rare={'new_field':[b'\x00\xff',None,b'unchanged']})],schema=schema)
 result,n=overlay_group_table_preserving_columns(original,{'a':['new-URS']})
 assert n==1 and result.schema==original.schema
 assert result.drop(['effective_groups_json']).equals(original.drop(['effective_groups_json']))
 assert original.column('effective_groups_json').to_pylist()==['["old-a"]','["old-b"]']
 assert json.loads(result.column('effective_groups_json')[0].as_py())==['new-URS','old-a']
 assert result.column('effective_groups_json')[1].as_py()=='["old-b"]'
 p=tmp_path/'new-derived.parquet';pq.write_table(result,p);assert pq.read_table(p).equals(result)


def test_empty_update_preserves_all_native_columns_and_values():
 table=pa.table({'task_id':['a'],'effective_groups_json':['[ "verbatim" ]'],'untouched':[b'unknown']})
 result,n=overlay_group_table_preserving_columns(table,{})
 assert result.equals(table) and n==0
