import importlib.util,pathlib
import pyarrow as pa,pyarrow.parquet as pq
ROOT=pathlib.Path.cwd()
def module():
    p=ROOT/'scripts/data_sources_string_v2.py';assert p.is_file(), 'native list roundtrip semantic validator not implemented'
    s=importlib.util.spec_from_file_location('data_sources_string_v2',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_parquet_list_child_field_name_does_not_change_qualified_accession_values(tmp_path):
    m=module();t=pa.Table.from_pylist([dict(id='9606.x',qualified=['P00001']),dict(id='1.y',qualified=[])],schema=pa.schema([('id',pa.string()),('qualified',pa.list_(pa.string()))]));p=tmp_path/'sample.parquet';pq.write_table(t,p);decoded=pq.read_table(p)
    assert m.native_table_equal(t,decoded)
    changed=pa.Table.from_pylist([dict(id='9606.x',qualified=['Q00002']),dict(id='1.y',qualified=[])],schema=t.schema)
    assert not m.native_table_equal(t,changed)
    assert not m.native_table_equal(t,t.rename_columns(['other','qualified']))
