import importlib.util,pathlib,numpy as np
import pytest
ROOT=pathlib.Path.cwd()
def module():
    p=ROOT/'scripts/data_sources_transition1x_native.py';assert p.is_file(), 'original-native frame-preserving role exporter not implemented'
    s=importlib.util.spec_from_file_location('transition1x_native',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def source_roles():
    z=np.array([6,1,8],np.int32);p=np.array([[[4.,2.,1.],[5.,2.,1.],[4.,3.,1.]]]);f=np.array([[[.2,.4,.8],[.1,.3,.7],[.5,.6,.9]]]);return z,{k:dict(atomic_numbers=z,positions=p.copy(),forces=f.copy(),energy=np.array([-100.]),hash=12345) for k in ['reactant','product','transition_state']}

def test_original_frame_vectors_all_atoms_and_configuration_axis_are_preserved():
    m=module();z,roles=source_roles();q=m.native_record('CHO','rxn1','test',z,roles,5,'/data/CHO/rxn1','original.h5');assert q['atomic_numbers']==[6,1,8] and q['source_native_split']=='test' and q['native_path_shape']==[5,3,3]
    for role in roles:
        assert np.array_equal(q['roles'][role]['positions'],roles[role]['positions']) and np.array_equal(q['roles'][role]['forces'],roles[role]['forces'])
        assert q['roles'][role]['native_configuration_count']==1 and not q['roles'][role]['frame_transformed']
    assert q['electronic_charge'] is None and q['spin_multiplicity'] is None and not q['task_admitted'] and not q['conditional_TS_inference_admitted']
    assert q['source_hash_is_global_identity'] is False

def test_mismatched_axes_nonfinite_values_and_foreign_split_fail_closed():
    m=module()
    for kind in ['badshape','nan','atomorder','split']:
        z,roles=source_roles();split='train'
        if kind=='badshape':roles['product']['forces']=np.ones((3,3))
        if kind=='nan':roles['transition_state']['positions'][0,0,0]=np.nan
        if kind=='atomorder':roles['reactant']['atomic_numbers']=np.array([1,6,8])
        if kind=='split':split='guessed_validation'
        with pytest.raises(ValueError):m.native_record('CHO','rxn1',split,z,roles,5,'/data/CHO/rxn1','original.h5')
