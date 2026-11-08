from pathlib import Path
import sys
import pytest
R=next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').exists());sys.path.insert(0,str(R/'scripts'))
from data_intake_pubchem_native_units import NativeUnitLedger
def row(codec=True):return dict(native_smiles='C[C@H](O)Cl',source_canonical_isomeric_smiles='C[C@H](O)Cl',standard_selfies='[C][C@H1][Branch1][C][O][Cl]' if codec else None,standard_selfies_token_count=6 if codec else None,derived_atom_count=5,derived_bond_count=4)
def test_prior_has_no_molecule_input_and_native_characters_are_not_model_tokens():
 l=NativeUnitLedger();r=row();l.reference_graph(r);l.view('reference_prior',r);s=l.snapshot();p=s['per_view_direction']['reference_prior'];assert p['input_SMILES_characters']==0 and p['target_native_SMILES_characters']==len(r['native_smiles']);assert all(v is None for v in s['actual_model_token_totals'].values())
def test_codec_directions_repeat_views_but_not_unique_graph_units():
 l=NativeUnitLedger();r=row();l.codec_graph(r);l.view('smiles_to_selfies',r);l.view('selfies_to_smiles',r);s=l.snapshot();assert s['unique_codec_graphs']['graphs']==1 and s['unique_codec_graphs']['standard_SELFIES_symbols']==6;assert s['per_view_direction']['smiles_to_selfies']['target_standard_SELFIES_symbols']==6;assert s['per_view_direction']['selfies_to_smiles']['input_standard_SELFIES_symbols']==6
def test_unsupported_SELFIES_encoding_preserves_native_reference_metrics_only():
 l=NativeUnitLedger();r=row(False);l.reference_graph(r);l.view('reference_prior',r);assert l.snapshot()['unique_reference_graphs']['graphs']==1
 with pytest.raises(ValueError):l.codec_graph(r)
 assert l.snapshot()['unique_codec_graphs'].get('graphs',0)==0
def test_missing_derived_graph_size_is_missing_metadata_not_zero_or_data_rejection():
 l=NativeUnitLedger();r=row();r['derived_atom_count']=None;l.reference_graph(r);s=l.snapshot()['unique_reference_graphs'];assert s['graphs']==1 and s['source_atom_slots_missing']==1 and s.get('source_derived_atom_slots',0)==0

def test_preserved_pilot_snapshot_is_extended_once_without_mutating_old_receipt():
    import copy
    old=NativeUnitLedger();r=row();old.reference_graph(r);old.view("reference_prior",r);snapshot=old.snapshot();before=copy.deepcopy(snapshot);new=NativeUnitLedger(snapshot);new.reference_graph(r);new.view("reference_prior",r);assert snapshot==before;assert new.snapshot()["unique_reference_graphs"]["graphs"]==2;assert new.snapshot()["per_view_direction"]["reference_prior"]["views"]==2
