"""Native duplicate-column and atom-map conservation, independent of admission."""
import copy,json
import pytest
from mora3.extensions.scri.grh_conservation import verify_native_export,group_closure
from test_grh_intake import row


def full_row():
    r=row();c=json.loads(r['condition_json']);t=json.loads(r['target_json']);payload=json.loads(r['payload_json'])
    payload.update(canonical_smiles='[H][H]',reference_energy_kcal_per_mol=0.,
        inverse_task={'conditions':{},'targets':{'reference_coordinates_angstrom':c['coordinates_angstrom']}})
    r['payload_json']=json.dumps(payload)
    r.update(condition_atomic_numbers=c['atomic_numbers'],condition_coordinates_angstrom=c['coordinates_angstrom'],
        condition_bond_sources=[],condition_bond_targets=[],condition_bond_orders=[],
        target_forces_kcal_mol_angstrom=t['forces_kcal_per_mol_angstrom'],target_energy_kcal_mol=t['energy_kcal_per_mol'],
        target_partial_charges_e=t['forcefield_partial_charges_e'],target_VR_H0_angstrom=[],target_VR_H1_angstrom=[],
        target_reference_coordinates_angstrom=c['coordinates_angstrom'],evaluation_canonical_smiles='[H][H]',evaluation_reference_energy_kcal_mol=0.)
    return r


def test_native_columns_and_payload_must_agree_without_relabeling():
    r=full_row();assert verify_native_export(r,'data')['native_columns_checked']==13
    for field in ('condition_coordinates_angstrom','target_forces_kcal_mol_angstrom'):
        bad=copy.deepcopy(r);bad[field][0][0]+=1
        with pytest.raises(ValueError,match='conservation'):verify_native_export(bad,'data')
    bad=copy.deepcopy(r);bad['target_energy_kcal_mol']=True
    with pytest.raises(ValueError):verify_native_export(bad,'data')


def test_bond_endpoints_and_atom_order_must_remain_valid():
    bad=full_row();c=json.loads(bad['condition_json']);c['bonds']=[{'source':0,'target':9,'order':1.,'aromatic':False}]
    bad['condition_json']=json.dumps(c);d=json.loads(bad['payload_json']);d['conditions']=c;bad['payload_json']=json.dumps(d)
    with pytest.raises(ValueError,match='bond'):verify_native_export(bad,'data')


def test_transitive_parent_and_record_links_preserve_official_splits():
    records=[('data','one','g1','train','p',None),('data','two','g2','train','p',None),
             ('nuclear_md','traj','g3','test',None,'two')]
    result=group_closure(records)
    assert result['components']==1 and result['conflicting_components']==1
    assert result['resolved_parent_record_links']==1
    assert result['unresolved_parent_record_links']==0
    records.append(('electronic_dynamics','missing','g4','validation',None,'absent'))
    assert group_closure(records)['unresolved_parent_record_links']==1
