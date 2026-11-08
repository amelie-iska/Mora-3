from pathlib import Path
import sys,time
import pytest
R=next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').exists());sys.path.insert(0,str(R/'scripts'))
from data_intake_full_pubchem_guards_v2 import BuildGuard

def test_materialization_checks_completed_census_cap_as_well_as_total(tmp_path):
 (tmp_path/'census-data').write_bytes(b'x'*101);g=BuildGuard(tmp_path,census_cap=100,total_cap=120,deadline_epoch=time.time()+100,min_free=0)
 with pytest.raises(RuntimeError,match='census'):g.check('materialized')
 assert (tmp_path/'census-data').stat().st_size==101

def test_census_cannot_ignore_existing_materialized_cap_or_borrow_other_phase(tmp_path):
 (tmp_path/'materialized').mkdir();(tmp_path/'materialized/preserved').write_bytes(b'x'*21);g=BuildGuard(tmp_path,census_cap=100,total_cap=120,deadline_epoch=time.time()+100,min_free=0)
 with pytest.raises(RuntimeError,match='materialized'):g.check('census')
 assert (tmp_path/'materialized/preserved').stat().st_size==21
