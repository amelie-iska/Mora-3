from pathlib import Path
import sys
import pytest
sys.path.insert(0,'/home/iska/Documents/amelie/bio/Mora-3/scripts')
from data_intake_admet_graph_scope import source_graph_scope


def test_enhanced_stereo_is_retained_and_not_silently_plain_SMILES_identity():
 source='F[C@H](Cl)Br |o1:1|';a=source_graph_scope(source);b=source_graph_scope('F[C@H](Cl)Br')
 assert a['native_representation']==source and a['source_original_unchanged']
 assert a['source_enhanced_stereo_group_count']==1 and 'o1:' in a['canonical_source_CXSMILES']
 assert not a['legacy_plain_SMILES_exact_comparison_eligible']
 assert b['legacy_plain_SMILES_exact_comparison_eligible']
 assert a['conservative_leakage_groups']==b['conservative_leakage_groups']
 assert a['native_predictor_identity_candidate'] and not a['complete_material_identity_proven']


def test_derived_map_normalization_preserves_original_graph_string():
 native='[CH3:9][OH:2]';a=source_graph_scope(native);b=source_graph_scope('CO')
 assert a['native_representation']==native and a['source_atom_maps_present']
 assert ':9' not in a['canonical_source_CXSMILES']
 assert a['canonical_CXSMILES_SHA256']==b['canonical_CXSMILES_SHA256']


def test_nonstructural_names_and_wildcards_do_not_become_known_chemical_identity():
 name=source_graph_scope('CCO SOURCE-IDENTIFIER')
 assert name['source_name_present'] and not name['native_predictor_identity_candidate']
 wild=source_graph_scope('C[*]')
 assert wild['source_wildcard_atom_count']==1 and not wild['native_predictor_identity_candidate']


def test_no_source_graph_authorizes_material_or_same_assay_join():
 a=source_graph_scope('CCO')
 assert not a['complete_material_identity_proven']
 assert not a['assay_or_source_observation_join_authorized']
 assert a['parser_messages']==[]
