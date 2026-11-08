"""Synthetic exact protocol/measure tests; no model, source or training grant."""
from dataclasses import replace
import hashlib
import math

import pytest

from mora3.teacher_protocol import (SystemSpec,TeacherSpec,TeacherRefusal,canonical,
    KB_EV_PER_K,Evaluation)
from mora3.boltzmann_target import BoxSupport,BoltzmannTarget


def system():
    atoms=[dict(key='H:'+str(i),entity='synthetic',ordinal=i,element='H',atomic_number=1,
                name='H'+str(i),component='H2',residue=None,isotope=None)for i in range(2)]
    topology=dict(system_id='synthetic',topology_id='synthetic-H2',atoms=atoms,
        bonds=[dict(first='H:0',second='H:1',kind='covalent',order=1.,
                    periodic_image=None,provenance='synthetic-unit-condition')],
        formal_charges=[None,None],total_charge=0,spin_multiplicity=1,
        protonation=None,cofactor_ids=[],chain_links=[],cell_angstrom=None,
        pbc=[False,False,False],chemistry_context_json='{}',source_refs=[],
        sequence_chains=[],cofactor_atom_map=[])
    return SystemSpec.from_export_mapping(topology)


def target():
    source=system()
    teacher=TeacherSpec(teacher_id='synthetic-no-model-loaded',family='UMA',
        repository_id='facebook/UMA',repository_revision='1f43bb7ff24084b72e8253f9e16fbac19320adae',
        fairchem_revision='ac3561498abfb5d869a9379b1afb44b065047e58',
        checkpoint_path='/unopened-synthetic-placeholder.pt',checkpoint_sha256='a'*64,
        task='omol',position_unit='angstrom',energy_unit='eV',force_unit='eV/angstrom',
        energy_reference_scheme='raw-local-no-reference-restoration',
        atom_reference_sha256=None,formation_reference_sha256=None,
        inference_settings_json='{}',execution_fingerprint_sha256='b'*64,seed=41)
    support=BoxSupport(source.sha256,(1,),((0.,0.,0.),None),(.65,-.1,-.1),(.85,.1,.1),
        'synthetic-caller-frame','c'*64)
    # Declared analytic unit-domain lower bound0, never a bound for actual UMA.
    return BoltzmannTarget(source,teacher,support,300.,0.,0.,'d'*64,'synthetic-unit-v1',True)


def test_system_preserves_order_and_unknown_formal_charges():
    value=system()
    assert value.atom_ids==('H:0','H:1')and value.formal_charges==(None,None)
    assert SystemSpec.from_mapping(__import__('dataclasses').asdict(value))==value


def test_electronic_state_and_topology_mismatch_refuse():
    value=system()
    with pytest.raises(TeacherRefusal,match='electron count'):replace(value,spin_multiplicity=2)
    with pytest.raises(TeacherRefusal,match='atom maps'):replace(value,atom_ids=value.atom_ids[::-1])
    with pytest.raises(TeacherRefusal,match='electronic'):replace(value,formal_charges=(0,0))


def test_box_is_normalized_product_uniform_and_explicit_dirac_coordinates():
    value=target().support
    assert value.dimension==3
    assert math.exp(value.log_volume)==pytest.approx(.2**3)
    full=value.complete_positions((.74,0.,0.))
    assert full==((0.,0.,0.),(.74,0.,0.))
    assert value.fixed_xyz_angstrom==((0.,0.,0.),None)


@pytest.mark.parametrize('coordinates',[(.65,0.,0.),(.85,0.,0.),(.74,0.),(.74,float('nan'),0.)])
def test_complete_open_support_has_no_clipping_or_truncation(coordinates):
    with pytest.raises(TeacherRefusal):target().support.complete_positions(coordinates)


def test_target_temperature_scaling_and_reference_gauge_are_explicit():
    value=target();energy=1.
    original=value.energy_log_reward(energy)
    assert original<0
    assert replace(value,temperature_kelvin=600.).energy_log_reward(energy)==pytest.approx(original/2)
    shifted=replace(value,energy_reference_ev=.5)
    assert shifted.energy_log_reward(energy)-original==pytest.approx(.5/(KB_EV_PER_K*300.))
    assert shifted.checkpoint_binding()!=value.checkpoint_binding()
    with pytest.raises(TeacherRefusal):shifted.validate_checkpoint_binding(value.checkpoint_binding())


@pytest.mark.parametrize('temperature',[0.,-1.,float('nan'),True,300])
def test_invalid_temperature_is_never_defaulted(temperature):
    with pytest.raises(TeacherRefusal):replace(target(),temperature_kelvin=temperature)


def test_declared_analytic_lower_bound_refuses_inconsistent_energy():
    value=target()
    with pytest.raises(TeacherRefusal,match='lower-bound'):value.energy_log_reward(-1.)
    with pytest.raises(TeacherRefusal):value.energy_log_reward(float('inf'))


def test_teacher_change_requires_new_target_and_normalizer_identity():
    value=target();other=replace(value,teacher=replace(value.teacher,execution_fingerprint_sha256='e'*64))
    assert value.sha256!=other.sha256
    with pytest.raises(TeacherRefusal):other.validate_checkpoint_binding(value.checkpoint_binding())


def test_periodic_or_variable_region_constraints_are_not_silently_relaxed():
    value=target()
    with pytest.raises(TeacherRefusal):replace(value.support,system_sha256='f'*64).bind_system(value.system)
    topology=__import__('json').loads(value.system.topology_json)
    topology['sequence_chains']=[dict(redesign_residue_ids=['synthetic-residue'])]
    source=SystemSpec.from_export_mapping(topology)
    with pytest.raises(TeacherRefusal,match='variable sequence'):
        replace(value,system=source,support=replace(value.support,system_sha256=source.sha256))


def test_numeric_force_evidence_does_not_become_conservative_by_default():
    fields=dict(frame_id='synthetic',xyz_sha256='a'*64,status='ok',energy_ev=1.,
        forces_ev_per_angstrom=((0.,0.,0.),(0.,0.,0.)),energy_unit='eV',
        force_unit='eV/angstrom',position_unit='angstrom',execution_fingerprint_sha256='b'*64,
        teacher_trainable_parameter_count=0,teacher_training_mode=False,
        force_relation='returned_numeric_unverified',conservative_evidence_sha256=None,refusal_reason=None)
    assert Evaluation(**fields).force_relation=='returned_numeric_unverified'
    fields['force_relation']='verified_conservative_same_energy'
    with pytest.raises(TeacherRefusal):Evaluation(**fields)
