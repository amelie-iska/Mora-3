"""Cross-source qualification of real original AZ/FreeSolv TRAIN chemical exposures."""
from pathlib import Path
import pytest
from mora3.research.cross_source_az_freesolv import chemical_exposure_pairs, audit_original_az_freesolv_train

ROOT=Path('/home/iska/Documents/amelie/bio/Mora-3')

def _r(source,record,smiles,groups=()):
    return dict(source=source,record_id=record,task_id=(record+'0'*64)[:64],smiles=smiles,groups=tuple(groups))

def test_exact_stereochemistry_and_connectivity_only_separated():
    a=(_r('AZ','a1','C[C@H](O)F'),_r('AZ','a2','CO'))
    f=(_r('FreeSolv','f1','C[C@H](O)F'),_r('FreeSolv','f2','C[C@@H](O)F'))
    r=chemical_exposure_pairs(a,f)
    assert r['exact_isomeric_chemical_count']==1
    assert r['exact_cross_source_record_pair_count']==1
    assert r['connectivity_chemical_count']==1
    assert r['connectivity_only_record_pair_count']==1
    assert r['direct_exact_freesolv_record_count']==1
    assert r['direct_connectivity_freesolv_record_count']==2
    assert r['global_split_admitted'] is False
    assert r['same_experimental_assay_claimed'] is False

def test_scoped_source_group_transitivity_and_no_cross_namespace_alias():
    a=(_r('AZ','a1','CO',('assay:shared',)),_r('AZ','a2','CN',('assay:shared',)))
    f=(_r('FreeSolv','f1','CO',('scaffold:joined',)),
       _r('FreeSolv','f2','CC',('scaffold:joined',)),
       _r('FreeSolv','f3','CCC',('assay:shared',)))
    r=chemical_exposure_pairs(a,f)
    assert r['exact_isomeric_chemical_count']==1
    assert r['two_source_exact_group_closure']['freesolv_train_rows_in_cross_source_components']==2
    assert r['two_source_exact_group_closure']['cross_source_component_count']==1
    assert r['two_source_exact_group_closure']['global_split_admitted'] is False

def test_duplicate_identity_invalid_structure_and_source_refused():
    a=(_r('AZ','a1','CO'),)
    for incorrect in [(_r('FreeSolv','f1','not_SMILES'),),
                      (_r('AZ','f1','CO'),),
                      (_r('FreeSolv','f1','CO'),_r('FreeSolv','f1','CC'))]:
        with pytest.raises(ValueError):chemical_exposure_pairs(a,incorrect)
    with pytest.raises(ValueError):chemical_exposure_pairs(a+a,(_r('FreeSolv','f2','CO'),))

@pytest.fixture(scope='module')
def real_corpus():
    return audit_original_az_freesolv_train(ROOT)

def test_real_complete_train_corpora_with_immutable_source_references(real_corpus):
    assert real_corpus['train_rows']=={'AZ':4184,'FreeSolv':556}
    assert real_corpus['train_shards']=={'AZ':17,'FreeSolv':3}
    assert real_corpus['source_rows_fully_checked'] is True
    assert real_corpus['source_original_payloads_verified'] is True
    assert real_corpus['candidate_common_bindings_preserved'] is True
    assert real_corpus['heldout_source_files_read'] is False
    assert real_corpus['heldout_target_payload_decoded'] is False
    assert real_corpus['optimizer_updates']==real_corpus['actual_proxy_runs']==real_corpus['actual_target_runs']==0
    assert real_corpus['training_admission'] is False

def test_real_cross_source_train_overlap_not_global_split(real_corpus):
    x=real_corpus['chemical_exposure']
    assert x['exact_isomeric_chemical_count']==10
    assert x['connectivity_chemical_count']==11
    assert x['direct_exact_freesolv_record_count']==10
    assert x['direct_connectivity_freesolv_record_count']==11
    assert x['global_split_admitted'] is False
    assert x['same_experimental_assay_claimed'] is False
    assert len(x['exact_matches'])==10
    assert len(x['connectivity_only_matches'])==3
    assert x['connectivity_only_record_pair_count']==3
    closure=x['two_source_exact_group_closure']
    assert closure['original_source_local_component_count']==300
    assert closure['two_source_exact_component_count']==293
    assert closure['cross_source_component_count']==1
    assert closure['freesolv_train_rows_in_cross_source_components']==185
    assert closure['cross_source_component_record_counts']==[4369]
    assert real_corpus['global_chemical_grouping_completed'] is False
