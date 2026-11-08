import importlib.util
from pathlib import Path
import pytest
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').exists())
s=importlib.util.spec_from_file_location('sciplex_native_meta',ROOT/'scripts/data_intake_sciplex_native_metadata.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_R_implicit_rownames_are_retained_without_shifting_cell_metadata():
 text='Cell "sample" "n.umi" "top_oligo"\ncellA "cellA" "library" 1234 "Dex_0.1_BC07"\n'
 r=m.r_pdata_records(text);assert r==[{'native_row_name':'cellA','Cell':'cellA','sample':'library','n.umi':'1234','top_oligo':'Dex_0.1_BC07'}]

def test_native_width_and_rowname_cell_disagreement_fail_closed():
 for text in ['Cell "sample"\na "b" "lib"\n','Cell "sample"\na "a"\n']:
  with pytest.raises(ValueError):m.r_pdata_records(text)

def test_headerless_cell_and_sample_rows_are_not_dropped_or_truncated():
 assert m.headerless_tab('cellA\tlib\ncellB\tlib\n',2)==[['cellA','lib'],['cellB','lib']]
 assert m.headerless_tab('Nutlin_100_AA01\tACGT\t1\t\n',3)==[['Nutlin_100_AA01','ACGT','1','']]
