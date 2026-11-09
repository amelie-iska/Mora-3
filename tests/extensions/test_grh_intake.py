"""Partitioned independent qualification checks; no training admission."""
import json
import pytest
from mora3.extensions.scri.grh_intake import project_candidate,native_ef_view


def row():
    condition={'atomic_numbers':[1,1],'coordinates_angstrom':[[0.,0.,0.],[1.,0.,0.]],'bonds':[],
        'known_chemical_topology':True,'method':'MMFF94','protonation':'explicit hydrogen'}
    targets={'energy_kcal_per_mol':1.,'forces_kcal_per_mol_angstrom':[[1.,0.,0.],[-1.,0.,0.]],
        'forcefield_partial_charges_e':[0.,0.],'heavy_atom_VR':{'H0':[],'H1':[]}}
    return dict(record_id='id',family='molecular',group_id='parent',split='train',origin_class='synthetic',
        label_origin='computed',condition_json=json.dumps(condition),target_json=json.dumps(targets),
        payload_json=json.dumps({'conditions':condition,'targets':targets,'parent':'p','units':{'energy':'kcal/mol'}}))


def test_grh_condition_only_projection_and_computed_evidence():
    a=project_candidate(row(),configuration='data',expected_split='train')
    b=row();b['record_id']='different';b['group_id']='other';b['target_json']=json.dumps({'hidden':999});b['payload_json']='{}'
    c=project_candidate(b,configuration='data',expected_split='train')
    assert a.condition_json==c.condition_json
    assert a.evidence_kind=='computed' and a.training_admitted is False
    with pytest.raises(ValueError):a.require_training_admission()
    ef=native_ef_view(a)
    assert ef['energy_unit']=='kcal/mol' and ef['force_convention']=='negative-gradient-in-same-native-frame'
    assert ef['training_admitted'] is False


def test_grh_rejects_bad_shape_nonfinite_split_and_method():
    bad=row();bad['split']='test'
    with pytest.raises(ValueError):project_candidate(bad,configuration='data',expected_split='train')
    bad=row();t=json.loads(bad['target_json']);t['forces_kcal_per_mol_angstrom'].pop();bad['target_json']=json.dumps(t)
    with pytest.raises(ValueError):native_ef_view(project_candidate(bad,configuration='data',expected_split='train'))
    bad=row();c=json.loads(bad['condition_json']);c['method']='unknown';bad['condition_json']=json.dumps(c)
    with pytest.raises(ValueError):native_ef_view(project_candidate(bad,configuration='data',expected_split='train'))
    bad=row();bad['condition_json']='{"x":NaN}'
    with pytest.raises(ValueError):project_candidate(bad,configuration='data',expected_split='train')


def test_all_four_grh_configurations_keep_only_source_conditions():
    for kind in ('source_anchors','electronic_dynamics','nuclear_md'):
        r=row()
        if kind!='source_anchors':
            r.pop('condition_json');r.pop('target_json')
        task=project_candidate(r,configuration=kind,expected_split='train')
        assert 'atomic_numbers' in json.loads(task.condition_json)
        assert task.training_admitted is False
