"""Complete source-native UniProt kinetic training route; no training/heldout."""
from pathlib import Path
import json
import struct
import pytest
from mora3.research.common_source_uniprot_kinetics_complete import (
    audit_complete_uniprot_kinetics_train, independent_kinetic_reference,
    kinetic_declared_group_components,
)
ROOT=Path('/home/iska/Documents/amelie/bio/Mora-3')


@pytest.fixture(scope='module')
def actual():
    return audit_complete_uniprot_kinetics_train(ROOT)


def test_all_original_retained_kinetics_not_just_two_windows(actual):
    d=actual
    assert d['schema']=='Mora3.UniProtKineticsCompleteOriginalTrainReplay/v1'
    assert d['native_train_rows']==d['observed_source_coordinates']==618
    assert d['original_source_rows']==622
    assert d['excluded_source_ordinals']==4
    assert d['original_shards']==3
    assert d['both_original_train_iterators_exhausted'] is True
    assert d['candidate_common_all_rows_equal'] is True
    assert d['independent_native_scalar_references_equal'] is True
    assert d['frozen_per_trial_window_limit_unchanged']==2
    assert d['final_cursor_position']==[3,0]
    assert d['source_boundary_restarts_checked']==4
    assert len(d['rows'])==len(set((r['source_shard_sha256'],r['source_ordinal']) for r in d['rows']))==618


def test_original_km_and_vmax_keep_literal_units_and_study_groups(actual):
    d=actual
    counts=d['measurement_type_counts']
    assert sum(counts.values())==618
    assert counts['Km']>0 and counts['Vmax']>0
    assert any(k.startswith('Km:') for k in d['literal_quantity_unit_counts'])
    assert any(k.startswith('Vmax:') for k in d['literal_quantity_unit_counts'])
    assert d['effective_group_census']['source_rows']==618
    assert d['effective_group_census']['connected_components']>0
    assert d['effective_group_census']['global_split_admitted'] is False
    for r in d['rows']:
        assert r['original_float64_bits']==r['independent_float64_bits']
        assert r['original_numeric_lexeme']==r['independent_lexeme']
        assert r['measurement_type'] in ('Km','Vmax')
        assert r['literal_source_unit']
        assert r['physical_missingness']==[
            'assay_method','assay_pH','assay_temperature',
            'assayed_construct','parent_comment_molecule_scope']
        assert r['source_group_sha256'] and r['effective_group_sha256']
        assert r['candidate_cursor_sha256']!=r['common_cursor_sha256']
    assert d['no_heldout_targets_decoded'] is True
    for k in ('full_expanded_corpus_ready','all62_route_replay',
              'global_split_admitted','training_admission','matched_owner_validation'):
        assert d[k] is False
    assert d['optimizer_updates']==d['actual_proxy_trials']==d['actual_target_runs']==0


def _payload(kind='Km'):
    q='constant' if kind=='Km' else 'velocity'
    alias='substrate' if kind=='Km' else 'enzyme'
    return dict(schema='mora3.CorpusTask/v1',task='curated_kinetic_annotation_value',
        conditions=dict(amino_acid_sequence='MAG',measurement_type=kind,substrate_alias='ATP',unit='uM'),
        targets=dict(value=2.5),missing=['assay_protocol'],
        audit=dict(accession='Q12345',evidence=[],original_measurement={q:2.5,alias:'ATP', 'unit':'uM'}))


@pytest.mark.parametrize('kind',('Km','Vmax'))
def test_independent_reference_keeps_exact_bits_and_lexeme(kind):
    raw=json.dumps(_payload(kind),separators=(',',':')).encode().replace(b'2.5',b'2.500e0')
    d=independent_kinetic_reference(raw)
    assert d['measurement_type']==kind and d['literal_unit']=='uM'
    assert d['value_float64_bits']==struct.pack('>d',2.5).hex()
    assert d['numeric_lexeme']=='2.500e0'


@pytest.mark.parametrize('damage',('bool','nonfinite','unit','target_mismatch','alias','gold_in_conditions','invented_context','duplicate_key'))
def test_independent_reference_rejects_changed_original_semantics(damage):
    p=_payload()
    if damage=='bool':p['targets']['value']=p['audit']['original_measurement']['constant']=True
    elif damage=='nonfinite':p['targets']['value']=float('nan')
    elif damage=='unit':p['audit']['original_measurement']['unit']='mM'
    elif damage=='target_mismatch':p['audit']['original_measurement']['constant']=3.0
    elif damage=='alias':p['audit']['original_measurement']['substrate']='other'
    elif damage=='gold_in_conditions':p['conditions']['value']=2.5
    elif damage=='invented_context':p['audit']['original_measurement']['pH']=7
    raw=json.dumps(p).encode()
    if damage=='duplicate_key':raw=raw.replace(b'"task":',b'"task":"x","task":',1)
    with pytest.raises(ValueError):independent_kinetic_reference(raw)


def test_protein_group_transitive_closure_preserves_shared_paper_link():
    x=(dict(task_id='a'*64, groups=('accession:P1','pubmed:20')),
       dict(task_id='b'*64, groups=('accession:P2','pubmed:20')),
       dict(task_id='c'*64, groups=('accession:P3','pubmed:21')))
    d=kinetic_declared_group_components(x)
    assert d['source_rows']==3 and d['connected_components']==2
    assert d['largest_component_rows']==2
    assert d['global_split_admitted'] is False
    assert d==kinetic_declared_group_components(tuple(reversed(x)))


def test_refuses_wrong_original_checkout(tmp_path):
    with pytest.raises(ValueError):audit_complete_uniprot_kinetics_train(tmp_path)
