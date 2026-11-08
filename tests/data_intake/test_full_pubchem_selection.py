from pathlib import Path
import sys,sqlite3
import pytest
R=next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').exists());sys.path.insert(0,str(R/'scripts'))
from data_intake_full_pubchem_selection import create_selection_schema,queue_graph_views,expected_view_mask,mark_view_seen
ISO=b'a'*32
def db():
 c=sqlite3.connect(':memory:');create_selection_schema(c);return c
def test_existing_pilot_prior_and_codec_are_not_serialized_again():
 c=db();queue_graph_views(c,ISO,17,17,1,1);assert c.execute('SELECT COUNT(*) FROM selection').fetchone()[0]==0
def test_new_codec_for_old_unsupported_encoding_does_not_rewrite_old_prior():
 c=db();queue_graph_views(c,ISO,17,29,1,0);assert c.execute('SELECT source_row,emit_prior,emit_codec FROM selection').fetchall()==[(29,0,1)]
def test_new_prior_and_codec_share_original_source_row_once():
 c=db();queue_graph_views(c,ISO,3,3,0,0);assert c.execute('SELECT source_row,emit_prior,emit_codec FROM selection').fetchall()==[(3,1,1)]
def test_graph_level_codec_availability_does_not_transfer_to_failed_source_encoding():
 c=db();queue_graph_views(c,ISO,3,7,0,0);assert c.execute('SELECT source_row,emit_prior,emit_codec FROM selection ORDER BY source_row').fetchall()==[(3,1,0),(7,0,1)]
def test_source_row_cannot_be_rebound_to_different_graph():
 c=db();queue_graph_views(c,ISO,3,None,0,0)
 with pytest.raises(ValueError,match='graph'):queue_graph_views(c,b'b'*32,3,3,0,0)
 assert c.execute('SELECT iso FROM selection WHERE source_row=3').fetchone()[0]==ISO
def test_full_proof_masks_reject_repeated_direction_and_preserve_missing_codec():
 seen=bytearray(32);mark_view_seen(seen,17,'reference_prior');assert seen[17]==expected_view_mask(1,0)==1
 mark_view_seen(seen,17,'smiles_to_selfies');mark_view_seen(seen,17,'selfies_to_smiles');assert seen[17]==expected_view_mask(1,1)==7
 with pytest.raises(ValueError,match='duplicate'):mark_view_seen(seen,17,'selfies_to_smiles')
def test_foreign_row_or_unknown_mask_never_defaults_to_train_objective():
 c=db()
 with pytest.raises(ValueError):queue_graph_views(c,ISO,-1,None,0,0)
 with pytest.raises(ValueError):expected_view_mask(None,1)
 with pytest.raises(ValueError):mark_view_seen(bytearray(2),0,'unknown')
