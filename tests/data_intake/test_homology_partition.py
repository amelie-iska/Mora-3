import importlib.util
from pathlib import Path
import pytest
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').exists())
s=importlib.util.spec_from_file_location('homology_partition',ROOT/'scripts/data_intake_homology_partition.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_native_partition_requires_full_unique_member_axis_and_self_representatives():
 assert m.parse_partition('a\ta\na\tb\nc\tc\n',{'a','b','c'})=={'a':'a','b':'a','c':'c'}

def test_missing_or_unknown_native_sequences_cannot_pass_full_coverage():
 for text in ['a\ta\n','a\ta\na\tz\n']:
  with pytest.raises(ValueError):m.parse_partition(text,{'a','b'})

def test_duplicate_members_and_nonrepresentative_representative_are_rejected():
 for text in ['a\ta\na\tb\na\tb\n','a\tb\nb\ta\n']:
  with pytest.raises(ValueError):m.parse_partition(text,{'a','b'})
