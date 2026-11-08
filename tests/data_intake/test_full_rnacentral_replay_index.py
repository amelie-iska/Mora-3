from pathlib import Path
import hashlib,sys
import pytest
sys.path.insert(0,'/home/iska/Documents/amelie/bio/Mora-3/scripts')
from data_intake_full_rnacentral_state import NativeIdentityIndex
from data_intake_full_rnacentral_replay_index import NativeReplayIndex


def indices(tmp_path):
 census=NativeIdentityIndex.create(tmp_path/'census.sqlite');census.db.execute('CREATE TABLE holdout_blocked(native_sha BLOB,group_id TEXT,PRIMARY KEY(native_sha,group_id)) WITHOUT ROWID');proof=NativeReplayIndex(tmp_path/'proof.sqlite')
 for file,row,seq,id in [(0,1,'ACGT','URS1'),(0,2,'ACGT','URS2'),(1,0,'ACGA','URS1')]:
  state=census.observe(seq,id,file,row);proof.observe(state['native_sha'],id,file,row,[])
 census.commit();return census,proof


def test_exact_native_multiplicity_and_conflicting_URS_edges_match(tmp_path):
 c,p=indices(tmp_path);assert not any(p.compare_readonly_census(c.path).values());assert p.summary()==dict(native_sequences=2,quality_pass_source_occurrences=3,native_URS_edges=3,known_heldout_block_edges=0);c.close();p.close()

@pytest.mark.parametrize('corruption',['count','first_pointer','phantom_URS','phantom_block'])
def test_independent_source_replay_detects_phantom_or_inconsistent_census_state(tmp_path,corruption):
 c,p=indices(tmp_path);digest=hashlib.sha256(b'ACGT').digest()
 if corruption=='count':c.db.execute('UPDATE sequences SET occurrences=99 WHERE native_sha=?',(digest,))
 elif corruption=='first_pointer':c.db.execute('UPDATE sequences SET first_row=50 WHERE native_sha=?',(digest,))
 elif corruption=='phantom_URS':c.db.execute('INSERT INTO identifiers VALUES (?,?)',('PHANTOM',digest))
 elif corruption=='phantom_block':c.db.execute('INSERT INTO holdout_blocked VALUES (?,?)',(digest,'invented-heldout'))
 c.commit();assert any(p.compare_readonly_census(c.path).values());c.close();p.close()


def test_comparison_keeps_completed_census_bytes_immutable(tmp_path):
 c,p=indices(tmp_path);c.close();path=tmp_path/'census.sqlite';before=hashlib.sha256(path.read_bytes()).hexdigest();assert not any(p.compare_readonly_census(path).values());p.close();assert hashlib.sha256(path.read_bytes()).hexdigest()==before
