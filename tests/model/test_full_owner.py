"""Real cooperative assembly paths; fixture assets never imply corpus admission."""
import copy
from dataclasses import replace

import pytest
import torch
from torch import nn

from mora3.model.full_owner import (PrimaryConfiguration,SourceAssemblyInputs,
    SourceBoundOwner,construct_primary_owner,unique_parameter_inventory)


@pytest.mark.parametrize('change',[{'width':True},{'width':7},{'base_width':0},
    {'width':4,'base_width':8},{'source_snapshot_sha256':'unbound'},
    {'readout_mult':float('nan')},{'attn_mult':0},{'encoder_var':True},
    {'flow_version':True},{'flow_version':3},{'sigma_min':1}])
def test_primary_configuration_refuses_ambiguous_values(change):
    values=dict(width=8,base_width=8,source_snapshot_sha256='a'*64)
    values.update(change)
    with pytest.raises(ValueError):PrimaryConfiguration(**values)


def test_parameter_inventory_counts_exact_ties_without_mutation():
    layer=nn.Linear(3,2)
    model=nn.ModuleDict({'left':layer,'right':layer})
    model.eval();before=copy.deepcopy(model.state_dict());rng=torch.get_rng_state().clone()
    result=unique_parameter_inventory(model)
    assert result['total_parameters']==result['trainable_parameters']==8
    assert result['unique_parameter_tensors']==2 and result['alias_names']==4
    assert result['parameters'][0]['aliases']==('left.weight','right.weight')
    assert result['parameters'][1]['aliases']==('left.bias','right.bias')
    assert not model.training and torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(before[key],value)for key,value in model.state_dict().items())
    assert not result['mup_operator_qualification']


def test_distinct_parameter_storage_views_refuse():
    model=nn.Module()
    model.a=nn.Parameter(torch.arange(6.))
    model.b=nn.Parameter(model.a[2:])
    with pytest.raises(ValueError,match='sharing storage'):unique_parameter_inventory(model)


def test_independent_empty_parameters_are_not_false_storage_aliases():
    model=nn.ParameterList([nn.Parameter(torch.empty(0)),nn.Parameter(torch.empty(0))])
    result=unique_parameter_inventory(model)
    assert result['unique_parameter_tensors']==2 and result['total_parameters']==0


@pytest.fixture
def actual_core_assets(tmp_path,request):
    from test_lincs_cooperative import build
    from mora3.model.annotation_vocabulary import AnnotationVocabulary
    from annotation_fixture import fixture_registry
    cases=request.getfixturevalue('cases')
    reference=build(tmp_path,cases).model
    physical=getattr(reference.annotation,'physical',None)
    return SourceAssemblyInputs(
        vocabulary=AnnotationVocabulary(reference.annotation.payload._vocabulary_json),
        annotation_registry=fixture_registry(),response_axis=reference.native.response.axis_ids,
        residue_feature_axis=reference.residue.head.feature_axis,
        physical_parameter_axis=physical.parameter_axis if physical else ('fixture-parameter',),
        physical_method_axis=physical.method_axis if physical else ('fixture-method',),
        profile_axes=reference.profile.axes,lincs_axes=reference.lincs.source_axes,
        codec=reference.decoder.codec,max_decoder_tokens=reference.decoder.max_tokens,
        rfam_family_axis=reference.rfam.family_axis,rfam_manifest_sha256='a'*64,
        goa_packets=tuple(reference.goa._fixture_packets)if hasattr(reference.goa,'_fixture_packets')else (),
        reconstruction_dimension=2,reconstruction_horizon=3,reconstruction_hidden=5,reconstruction_source={'pool':'fixture'},
        geometry_id_width=16,color_id_width=2,instance_slots=3,instance_max_output_bytes=1000000,
        payload_max_tokens=8192,color_input_policy='native-max512-v1',instance_slot_policy='bounded-max64-v1')


# Existing source fixtures supply the complete native lineage/axes and GOA
# packet shape. No production/final model counts are inferred from this test.
from test_lincs_cooperative import cases


def _construct(tmp_path,assets):
    from test_goa_objective import view
    assets=replace(assets,goa_packets=(view(tmp_path),))
    return construct_primary_owner(PrimaryConfiguration(8,8,'a'*64),assets)


def test_literal_primary_chain_aliases_and_exact_recovery(tmp_path,actual_core_assets):
    torch.manual_seed(20261106)
    model=_construct(tmp_path,actual_core_assets)
    anchor=model.core.parent.parent
    assert anchor.native.response.backbone is anchor.annotation.backbone
    assert anchor.residue.shared is anchor.annotation
    assert anchor.flow.native is anchor.native.response
    assert model.core.instance.context is model.core.parent.color.context
    inventory=unique_parameter_inventory(model)
    assert inventory['total_parameters']==sum(p.numel()for p in model.parameters())
    assert inventory['alias_names']>inventory['unique_parameter_tensors']
    state=copy.deepcopy(model.state_dict());clone=copy.deepcopy(model)
    clone.load_state_dict(state)
    assert model.get_extra_state()==clone.get_extra_state()
    with pytest.raises(ValueError,match='not full corpus'):model.require_training_admission()


@pytest.mark.parametrize('drift',['snapshot','late-sidecar','alias-value'])
def test_full_checkpoint_preflight_before_shared_tensor_copy(tmp_path,actual_core_assets,drift):
    model=_construct(tmp_path,actual_core_assets)
    state=copy.deepcopy(model.state_dict())
    before={name:p.detach().clone()for name,p in model.named_parameters()}
    if drift=='snapshot':state['_extra_state']['source_snapshot_sha256']='b'*64
    elif drift=='late-sidecar':
        key=next(key for key in reversed(state)if key.endswith('_extra_state')and key!='_extra_state')
        state[key]={'wrong':'source'}
    else:
        row=next(row for row in unique_parameter_inventory(model)['parameters']if len(row['aliases'])>1)
        key=row['aliases'][-1];state[key]=state[key].clone()+1
    with pytest.raises(ValueError):model.load_state_dict(state)
    assert all(torch.equal(before[name],parameter)for name,parameter in model.named_parameters())


def test_unknown_registered_module_cannot_enter_frozen_assembly(tmp_path,actual_core_assets):
    model=_construct(tmp_path,actual_core_assets)
    model.unregistered_capability=nn.Identity()
    with pytest.raises(ValueError,match='topology'):model.get_extra_state()


@pytest.mark.parametrize('drift',['buffer-shape','parameter-dtype','conditional-policy','unexpected-key'])
def test_complete_recovery_preflight_prevents_partial_shared_mutation(tmp_path,actual_core_assets,drift):
    first=_construct(tmp_path,actual_core_assets)
    # This buffer is registered before assembling the owner being tested.
    first.core.register_buffer('checkpoint_probe',torch.zeros(2,dtype=torch.float32))
    model=SourceBoundOwner(first.core,
        source_snapshot_sha256=first.get_extra_state()['source_snapshot_sha256'],
        configuration=first.get_extra_state()['primary_configuration'],
        admet=first.admet,mcsa=first.mcsa)
    state=copy.deepcopy(model.state_dict())
    before={name:value.detach().clone()for name,value in model.named_parameters()}
    # Every alias has the same changed value, so alias disagreement alone cannot
    # explain rejection and cannot accidentally hide the late-buffer bug.
    row=next(row for row in unique_parameter_inventory(model)['parameters']if row['numel'])
    for name in row['aliases']:state[name]=state[name].clone()+1
    if drift=='buffer-shape':state['core.checkpoint_probe']=torch.zeros(3)
    elif drift=='parameter-dtype':
        for name in row['aliases']:state[name]=state[name].double()
    elif drift=='conditional-policy':
        candidates=[key for key in state if key.endswith(('_payload_parameterization',
            '_pixel_parameterization','_native_initialization','_attention_initialization'))]
        assert candidates,'the actual native constructor must serialize its conditional policies'
        state[candidates[-1]]={'incorrect':'conditional source/operator policy'}
    else:state['core.unrecognized_persistent_state']=torch.tensor(1.)
    with pytest.raises(ValueError):model.load_state_dict(state)
    assert all(torch.equal(before[name],value)for name,value in model.named_parameters())
    assert torch.equal(model.core.checkpoint_probe,torch.zeros(2))


@pytest.mark.parametrize('options',[{'strict':False},{'assign':True}])
def test_source_bound_recovery_refuses_nonexact_modes_before_mutation(tmp_path,actual_core_assets,options):
    model=_construct(tmp_path,actual_core_assets)
    state=copy.deepcopy(model.state_dict())
    before={name:value.detach().clone()for name,value in model.named_parameters()}
    with pytest.raises(ValueError):model.load_state_dict(state,**options)
    assert all(torch.equal(before[name],value)for name,value in model.named_parameters())



def _native_extension_assets(assets):
    from test_pubchem_decoder import ROOT,MASTER,SHA
    from test_boltzmann_target import target
    from mora3.model.cartesian_gfn_policy import CartesianBudget
    from test_admet_model import setup
    contract=setup('cpu')[2]
    return replace(assets,admet_contracts=(contract,),pubchem_source_root=ROOT,
        pubchem_manifest=MASTER,pubchem_manifest_sha256=SHA,
        cartesian_budget=CartesianBudget(5,32,40,1000000,2,100000),
        cartesian_id_width=8,cartesian_horizon=3,
        cartesian_teacher_sha256=target().teacher.sha256,
        cartesian_input_contract_sha256='e'*64)


def test_actual_native_extensions_register_shared_owners_and_exact_checkpoint(tmp_path,actual_core_assets):
    assets=_native_extension_assets(actual_core_assets)
    model=_construct(tmp_path,assets);anchor=model.core.parent.parent
    assert model.pubchem.shared is anchor.annotation
    assert model.admet.shared is anchor.annotation
    assert model.admet.get_extra_state()['version']==2
    assert model.cartesian.backbone is anchor.annotation.backbone
    for name in ('feature','role','graph_token','layers','final_norm'):
        assert getattr(model.cartesian.graph,name)is getattr(anchor.annotation.backbone,name)
    state=copy.deepcopy(model.state_dict());clone=copy.deepcopy(model)
    clone.load_state_dict(state)
    assert model.get_extra_state()['schema']=='mora3-source-bound-owner-v2'
    assert clone.get_extra_state()==model.get_extra_state()
    assert unique_parameter_inventory(model)==unique_parameter_inventory(clone)
    with pytest.raises(ValueError,match='not full corpus'):model.require_training_admission()


@pytest.mark.parametrize('change',[
    {'pubchem_manifest_sha256':None},
    {'cartesian_teacher_sha256':None},
    {'cartesian_id_width':4},
    {'cartesian_backbone_identifier_axis':(['unhashable'],'observation')}])
def test_partial_or_invalid_native_extension_bindings_refuse_before_rng(tmp_path,actual_core_assets,change):
    assets=replace(_native_extension_assets(actual_core_assets),**change)
    rng=torch.get_rng_state().clone()
    with pytest.raises(ValueError):construct_primary_owner(PrimaryConfiguration(8,8,'a'*64),assets)
    assert torch.equal(rng,torch.get_rng_state())


def test_cartesian_teacher_change_is_preflighted_before_shared_parameter_copy(tmp_path,actual_core_assets):
    model=_construct(tmp_path,_native_extension_assets(actual_core_assets))
    state=copy.deepcopy(model.state_dict())
    before={name:p.detach().clone()for name,p in model.named_parameters()}
    state['cartesian._extra_state']['teacher_sha256']='f'*64
    with pytest.raises(ValueError):model.load_state_dict(state)
    assert all(torch.equal(before[name],p)for name,p in model.named_parameters())



def test_source_annotation_extension_reuses_full_native_owners_and_recovers(tmp_path,actual_core_assets):
    # Eight is a deliberately small fixture budget, not the actual source census.
    assets=replace(_native_extension_assets(actual_core_assets),residue_list_max_target_length=8)
    model=_construct(tmp_path,assets);anchor=model.core.parent.parent
    branch=model.source_annotations
    assert branch.annotation is anchor.annotation
    assert branch.rfam is anchor.rfam
    assert branch.residue is anchor.residue
    assert branch.residue_list.max_target_length==8
    assert model.get_extra_state()['schema']=='mora3-source-bound-owner-v3'
    state=copy.deepcopy(model.state_dict());clone=copy.deepcopy(model)
    clone.load_state_dict(state)
    assert clone.get_extra_state()==model.get_extra_state()
    before={name:p.detach().clone() for name,p in model.named_parameters()}
    state['source_annotations._extra_state']['residue_semantics']='wrong-source-order'
    with pytest.raises(ValueError):model.load_state_dict(state)
    assert all(torch.equal(before[name],p) for name,p in model.named_parameters())
    from mora3.model.full_owner_mup import full_owner_mup_parameter_roles
    roles=full_owner_mup_parameter_roles(model,
        geometry_identifier_axis=tuple('geometry-source-coordinate:'+str(i) for i in range(16)),
        color_identifier_axis=('color-context-coordinate:0','color-context-coordinate:1'))
    assert set(roles)==set(dict(model.named_parameters(remove_duplicate=False)))
    with pytest.raises(ValueError,match='not full corpus'):model.require_training_admission()


@pytest.mark.parametrize('budget',[True,0,-1,1.5])
def test_bad_source_annotation_budget_refuses_before_constructor_rng(tmp_path,actual_core_assets,budget):
    assets=replace(actual_core_assets,residue_list_max_target_length=budget)
    rng=torch.get_rng_state().clone()
    with pytest.raises(ValueError,match='residue source-list budget'):
        construct_primary_owner(PrimaryConfiguration(8,8,'a'*64),assets)
    assert torch.equal(rng,torch.get_rng_state())



from test_cell_native import source_assets as native_cell_assets


def test_cell_text_annotations_share_actual_owners_and_checkpoint(tmp_path,actual_core_assets,native_cell_assets):
    vocabulary,protocol=native_cell_assets
    assets=replace(_native_extension_assets(actual_core_assets),residue_list_max_target_length=47,
        cell_vocabulary=vocabulary,cell_protocol=protocol,source_text_max_target_bytes=1024)
    model=_construct(tmp_path,assets);anchor=model.core.parent.parent
    assert model.cell.shared is anchor.annotation
    assert model.cell.vocabulary is vocabulary and model.cell.protocol is protocol
    assert model.text.decoder is model.pubchem
    assert model.source_annotations.residue_list.max_target_length==47
    assert model.get_extra_state()['schema']=='mora3-source-bound-owner-v4'
    assert {id(p) for p in model.text.parameters()}=={id(p) for p in model.pubchem.parameters()}
    state=copy.deepcopy(model.state_dict());clone=copy.deepcopy(model)
    clone.load_state_dict(state)
    assert clone.text.decoder is clone.pubchem
    assert clone.cell.shared is clone.core.parent.parent.annotation
    assert clone.get_extra_state()==model.get_extra_state()
    before={name:p.detach().clone() for name,p in model.named_parameters()}
    state['cell._extra_state']['protocol']['coordinate']='wrong-native-transform'
    with pytest.raises(ValueError):model.load_state_dict(state)
    assert all(torch.equal(before[name],p) for name,p in model.named_parameters())


@pytest.mark.parametrize('change',[{'cell_protocol':None},{'cell_max_positions':True},
    {'source_text_max_target_bytes':True},{'pubchem_source_root':None}])
def test_incomplete_cell_text_source_contract_refuses_before_rng(tmp_path,actual_core_assets,native_cell_assets,change):
    vocabulary,protocol=native_cell_assets
    assets=replace(_native_extension_assets(actual_core_assets),cell_vocabulary=vocabulary,
        cell_protocol=protocol,source_text_max_target_bytes=1024)
    assets=replace(assets,**change);rng=torch.get_rng_state().clone()
    with pytest.raises(ValueError):construct_primary_owner(PrimaryConfiguration(8,8,'a'*64),assets)
    assert torch.equal(rng,torch.get_rng_state())



def test_response_source_columns_keep_duplicate_labels_and_exact_parameter_axis(tmp_path,actual_core_assets):
    source_axis=('gene-a','Y_RNA','Y_RNA','STPG4','STPG4','gene-z')
    model=_construct(tmp_path,replace(actual_core_assets,response_axis=source_axis))
    assert model.core.parent.parent.native.response.axis_ids==source_axis
    state=copy.deepcopy(model.state_dict());clone=copy.deepcopy(model);clone.load_state_dict(state)
    assert clone.core.parent.parent.native.response.axis_ids==source_axis
    assert clone.get_extra_state()==model.get_extra_state()



def test_native_lincs_inverse_and_root_text_are_registered_aliases(tmp_path,actual_core_assets):
    assets=replace(_native_extension_assets(actual_core_assets),source_text_max_target_bytes=1024,
        source_text_root_list_policy='observed-native-root-list-v1',lincs_inverse_max_target_bytes=1024)
    model=_construct(tmp_path,assets);anchor=model.core.parent.parent
    assert model.lincs_inverse.lincs is anchor.lincs
    assert model.lincs_inverse.decoder is model.pubchem is model.text.decoder
    assert model.text.root_list_policy=='observed-native-root-list-v1'
    assert model.get_extra_state()['schema']=='mora3-source-bound-owner-v5'
    clone=copy.deepcopy(model);clone.load_state_dict(copy.deepcopy(model.state_dict()))
    assert clone.lincs_inverse.decoder is clone.pubchem is clone.text.decoder
    assert clone.get_extra_state()==model.get_extra_state()
    from mora3.model.full_owner_mup import full_owner_mup_parameter_roles
    roles=full_owner_mup_parameter_roles(model,
        geometry_identifier_axis=tuple('geometry-source-coordinate:'+str(i) for i in range(16)),
        color_identifier_axis=('color-context-coordinate:0','color-context-coordinate:1'))
    assert set(roles)==set(dict(model.named_parameters(remove_duplicate=False)))
    assert any(name.startswith('lincs_inverse.')for name in roles)


@pytest.mark.parametrize('change',[{'lincs_inverse_max_target_bytes':True},
    {'source_text_root_list_policy':'unknown'},{'source_text_max_target_bytes':None}])
def test_new_inverse_root_policy_refuses_incomplete_binding_before_rng(tmp_path,actual_core_assets,change):
    assets=replace(_native_extension_assets(actual_core_assets),source_text_max_target_bytes=1024,
        source_text_root_list_policy='observed-native-root-list-v1',lincs_inverse_max_target_bytes=1024)
    assets=replace(assets,**change);rng=torch.get_rng_state().clone()
    with pytest.raises(ValueError):construct_primary_owner(PrimaryConfiguration(8,8,'a'*64),assets)
    assert torch.equal(rng,torch.get_rng_state())



def test_current_native_prior_wrapper_is_registered_shared_and_checkpointed(tmp_path,actual_core_assets):
    assets=replace(_native_extension_assets(actual_core_assets),source_text_max_target_bytes=1024,
        source_text_root_list_policy='observed-native-root-list-v1',lincs_inverse_max_target_bytes=1024,
        native_prior_max_target_bytes=1024)
    model=_construct(tmp_path,assets);anchor=model.core.parent.parent
    assert model.native_priors.nucleotide is anchor.nucleotide
    assert model.native_priors.decoder is model.pubchem is model.text.decoder is model.lincs_inverse.decoder
    assert model.get_extra_state()['schema']=='mora3-source-bound-owner-v6'
    clone=copy.deepcopy(model);clone.load_state_dict(copy.deepcopy(model.state_dict()))
    assert clone.native_priors.nucleotide is clone.core.parent.parent.nucleotide
    assert clone.native_priors.decoder is clone.pubchem
    from mora3.model.full_owner_mup import full_owner_mup_parameter_roles
    roles=full_owner_mup_parameter_roles(model,
        geometry_identifier_axis=tuple('geometry-source-coordinate:'+str(i) for i in range(16)),
        color_identifier_axis=('color-context-coordinate:0','color-context-coordinate:1'))
    assert set(roles)==set(dict(model.named_parameters(remove_duplicate=False)))
    assert any(name.startswith('native_priors.')for name in roles)
    with pytest.raises(ValueError,match='not full corpus'):model.require_training_admission()


@pytest.mark.parametrize('budget',[True,0])
def test_native_prior_complete_budget_refuses_before_constructor_rng(tmp_path,actual_core_assets,budget):
    assets=replace(_native_extension_assets(actual_core_assets),native_prior_max_target_bytes=budget)
    rng=torch.get_rng_state().clone()
    with pytest.raises(ValueError):construct_primary_owner(PrimaryConfiguration(8,8,'a'*64),assets)
    assert torch.equal(rng,torch.get_rng_state())


@pytest.mark.parametrize('policy',['source-envelope-v3','source-layer-list-v4'])
def test_complete_source_raster_envelope_reaches_real_registered_owner(tmp_path,actual_core_assets,policy):
    from mora3.model.raster_source_envelope import RasterSourceEnvelope
    envelope=RasterSourceEnvelope('b'*64,1024,1024,297)
    assets=replace(actual_core_assets,raster_source_envelope=envelope,
        color_input_policy='source-envelope-v2',instance_slot_policy=policy,instance_slots=297)
    model=_construct(tmp_path,assets)
    native=model.core.parent.color.color
    instance=model.core.instance.color
    assert native.source_envelope is envelope and instance.source_envelope is envelope
    assert instance.head.slots==297
    assert instance.pixels is native.pixels and instance.backbone is native.backbone
    if policy=='source-layer-list-v4':
        assert instance.count.out_features==298
        assert instance.count.weight.shape[-1]==8
    else:assert getattr(instance,'count',None)is None
    state=copy.deepcopy(model.state_dict());clone=copy.deepcopy(model)
    clone.load_state_dict(state)
    assert clone.get_extra_state()==model.get_extra_state()
    with pytest.raises(ValueError,match='not full corpus'):model.require_training_admission()


@pytest.mark.parametrize('changes',[
    {'color_input_policy':'source-envelope-v2'},
    {'instance_slot_policy':'source-layer-list-v4'},
    {'raster_source_envelope':'fake'},
    {'raster_source_envelope':'valid','color_input_policy':'native-max512-v1','instance_slot_policy':'source-layer-list-v4','instance_slots':297},
    {'raster_source_envelope':'valid','color_input_policy':'source-envelope-v2','instance_slot_policy':'bounded-max64-v1','instance_slots':297},
    {'raster_source_envelope':'valid','color_input_policy':'source-envelope-v2','instance_slot_policy':'source-layer-list-v4','instance_slots':296}])
def test_source_raster_policy_refuses_before_any_parameter_rng(tmp_path,actual_core_assets,changes):
    from mora3.model.raster_source_envelope import RasterSourceEnvelope
    changes=dict(changes)
    if changes.get('raster_source_envelope')=='valid':changes['raster_source_envelope']=RasterSourceEnvelope('b'*64,1024,1024,297)
    before=torch.get_rng_state().clone()
    with pytest.raises(ValueError,match='source'):_construct(tmp_path,replace(actual_core_assets,**changes))
    assert torch.equal(before,torch.get_rng_state())


def test_registered_scalar_pdb_and_gray_use_exact_shared_owner(tmp_path,actual_core_assets):
    from mora3.data.pdb_masked_coordinates import PDBCoordinateAxes,ATOM_FIELDS
    from mora3.model.raster_source_envelope import RasterSourceEnvelope
    from mora3.model.source_scalar_objectives import SourceScalarAxis
    key=('FreeSolv','experimental_hydration_free_energy','experimental_hydration_free_energy',
         'kcal/mol',None,None,'source_experimental_reported_point')
    scalar=SourceScalarAxis('a'*64,'b'*64,((key,1),))
    pdb=PDBCoordinateAxes(tuple(('fixture',) for _ in ATOM_FIELDS),3,(('fixture.parquet',0,'c'*64),),'d'*64)
    gray=RasterSourceEnvelope('e'*64,1024,1024,None)
    model=_construct(tmp_path,replace(actual_core_assets,scalar_source_axis=scalar,pdb_coordinate_axes=pdb,grayscale_source_envelope=gray))
    anchor=model.core.parent.parent
    assert model.scalar.shared is anchor.annotation and model.pdb.backbone is anchor.annotation.backbone
    assert anchor.native.image.source_envelope is gray
    assert model.get_extra_state()['schema']=='mora3-source-bound-owner-v7'
    state=copy.deepcopy(model.state_dict());clone=copy.deepcopy(model);clone.load_state_dict(state)
    assert clone.get_extra_state()==model.get_extra_state()
    before=model.scalar._partition(key).weight.detach().clone()
    state['_extra_state']['pdb']['source']['native_view_policy_sha256']='f'*64
    with pytest.raises(ValueError):model.load_state_dict(state)
    assert torch.equal(before,model.scalar._partition(key).weight)
    with pytest.raises(ValueError,match='not full corpus'):model.require_training_admission()


@pytest.mark.parametrize('field',['grayscale_source_envelope','scalar_source_axis','pdb_coordinate_axes'])
def test_bad_scalar_pdb_gray_configuration_refuses_before_rng(tmp_path,actual_core_assets,field):
    rng=torch.get_rng_state().clone()
    with pytest.raises(ValueError):_construct(tmp_path,replace(actual_core_assets,**{field:'not-bound'}))
    assert torch.equal(rng,torch.get_rng_state())


def test_source_rosetta_unconditional_owner_strict_sharing_and_recovery(tmp_path,actual_core_assets):
    from mora3.data.rosetta_unconditional_profiles import RosettaProfileAxis,_DIMENSIONS
    axis=RosettaProfileAxis(next(iter(_DIMENSIONS)), 'explicit-source-fixture', ('ordered-b','ordered-a'), 'literal-source-unit')
    model=_construct(tmp_path,replace(actual_core_assets,rosetta_flow_axes=(axis,),
        rosetta_flow_source_census_sha256='e'*64,rosetta_flow_native_view_policy_sha256='f'*64))
    assert model.rosetta_flow.backbone is model.core.parent.parent.annotation.backbone
    assert model.get_extra_state()['schema']=='mora3-source-bound-owner-v8'
    assert model.rosetta_flow.get_extra_state()['source_census_sha256']=='e'*64
    state=copy.deepcopy(model.state_dict());clone=copy.deepcopy(model);clone.load_state_dict(state)
    assert clone.get_extra_state()==model.get_extra_state()
    with pytest.raises(ValueError,match='not full corpus'):model.require_training_admission()


def test_source_rosetta_incomplete_binding_refuses_before_rng(tmp_path,actual_core_assets):
    rng=torch.get_rng_state().clone()
    with pytest.raises(ValueError,match='Rosetta'):_construct(tmp_path,replace(actual_core_assets,rosetta_flow_source_census_sha256='f'*64))
    assert torch.equal(rng,torch.get_rng_state())


def test_registered_gray_reverse_and_bioelectric_exact_shared_owners(tmp_path,actual_core_assets):
    from mora3.model.raster_source_envelope import RasterSourceEnvelope
    from mora3.model.bioelectric_outcomes import BioelectricAssayAxis
    envelope=RasterSourceEnvelope('b'*64,512,512,None)
    axis=BioelectricAssayAxis('fixture intensity','source-normalized fluorescence',None,'native_scalar')
    model=_construct(tmp_path,replace(actual_core_assets,grayscale_source_envelope=envelope,
        grayscale_reverse_detector_dtype='>u2',bioelectric_assay_axes=(axis,),bioelectric_source_contract_sha256='c'*64))
    anchor=model.core.parent.parent
    assert model.gray_reverse.geometry is anchor.geometry_fusion.geometry
    assert model.bioelectric.shared is anchor.annotation and model.gray_reverse.source_envelope is envelope
    assert model.get_extra_state()['schema']=='mora3-source-bound-owner-v9'
    clone=copy.deepcopy(model);clone.load_state_dict(copy.deepcopy(model.state_dict()))
    assert clone.get_extra_state()==model.get_extra_state()
    with pytest.raises(ValueError,match='not full corpus'):model.require_training_admission()


@pytest.mark.parametrize('changes',[{'grayscale_reverse_detector_dtype':'>u2'},
    {'bioelectric_source_contract_sha256':'c'*64}])
def test_incomplete_gray_reverse_or_bioelectric_refused_before_rng(tmp_path,actual_core_assets,changes):
    rng=torch.get_rng_state().clone()
    with pytest.raises(ValueError):_construct(tmp_path,replace(actual_core_assets,**changes))
    assert torch.equal(rng,torch.get_rng_state())
