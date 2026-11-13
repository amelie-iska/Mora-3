"""Exact complete native owner integration of the forest velocity head."""
import copy
from dataclasses import replace
import pytest
import torch
from test_full_owner import actual_core_assets,cases,_construct
from test_goa_objective import view as goa_view
from test_forest_flow import problem,skeleton,state,SOURCE,COORD
from mora3.model.forest_flow import ForestFlowSpec,NativeForestFlowHead,forest_flow_matching_loss,native_forest_flow_objective
from mora3.model.full_owner import construct_primary_owner,PrimaryConfiguration,unique_parameter_inventory
from mora3.model.full_owner_mup import full_owner_mup_parameter_roles,full_owner_mup_width_family

AXES=dict(geometry_identifier_axis=tuple('geometry-source-coordinate:'+str(i)for i in range(16)),
          color_identifier_axis=('color-context-coordinate:0','color-context-coordinate:1'))


def assets(tmp_path,source):
    spec=ForestFlowSpec(2,SOURCE,COORD,max_nodes=16,max_batch_size=8,max_attention_elements=200000)
    return replace(source,goa_packets=(goa_view(tmp_path),),forest_flow_spec=spec)


@pytest.mark.parametrize('width',(8,12,16))
def test_factory_registers_full_forest_head_with_literal_shared_context(tmp_path,actual_core_assets,width):
    src=assets(tmp_path,actual_core_assets);torch.manual_seed(9)
    owner=construct_primary_owner(PrimaryConfiguration(width,8,SOURCE),src)
    h=owner.forest_flow;assert type(h)is NativeForestFlowHead
    assert h.geometry is owner.core.parent.parent.geometry_fusion.geometry
    assert h.geometry._source_backbone_reference[0]is owner.core.parent.parent.annotation.backbone
    assert h.spec.source_snapshot_sha256==owner.get_extra_state()['source_snapshot_sha256']
    assert owner.get_extra_state()['forest_flow']==h.get_extra_state()
    r,t=problem();f=replace(skeleton(),states=skeleton().states.float());x=state().float();mask=torch.ones_like(x,dtype=torch.bool);mask[:,0]=False
    loss=forest_flow_matching_loss(h,r,t,f,x+.2,x,torch.tensor([.2,.8],dtype=x.dtype),mask)['loss']
    loss.backward();assert h.readout.weight.grad.abs().sum()>0
    assert owner.core.parent.parent.annotation.backbone.feature.weight.grad.abs().sum()>0
    assert unique_parameter_inventory(owner)['alias_names']>unique_parameter_inventory(owner)['unique_parameter_tensors']
    roles=full_owner_mup_parameter_roles(owner,**AXES)
    assert set(roles)==set(dict(owner.named_parameters(remove_duplicate=False)))
    assert any(name.startswith('forest_flow.readout')for name in roles)


def test_disabled_head_preserves_native_counts_and_serialized_configuration(tmp_path,actual_core_assets):
    original=_construct(tmp_path,actual_core_assets)
    src=assets(tmp_path,actual_core_assets)
    enabled=construct_primary_owner(PrimaryConfiguration(8,8,SOURCE),src)
    extra=unique_parameter_inventory(enabled)['total_parameters']-unique_parameter_inventory(original)['total_parameters']
    assert extra==154
    assert original.forest_flow is None and 'forest_flow'not in original.get_extra_state()
    assert all(not k.startswith('forest_flow.')for k in original.state_dict())
    duplicate=copy.deepcopy(original);duplicate.load_state_dict(copy.deepcopy(original.state_dict()))
    assert duplicate.get_extra_state()==original.get_extra_state()


def test_complete_width_family_keeps_source_coordinate_axis_and_aliases(tmp_path,actual_core_assets):
    src=assets(tmp_path,actual_core_assets)
    models=[construct_primary_owner(PrimaryConfiguration(w,8,SOURCE),src)for w in(8,12,16)]
    family=full_owner_mup_width_family(*models,**AXES)
    assert family.total_numel==tuple(sum(p.numel()for p in m.parameters())for m in models)
    assert family.total_numel[0]<family.total_numel[1]<family.total_numel[2]
    changed=construct_primary_owner(PrimaryConfiguration(16,8,SOURCE),replace(src,forest_flow_spec=replace(src.forest_flow_spec,coordinate_sha256='f'*64)))
    with pytest.raises(ValueError):full_owner_mup_width_family(models[0],models[1],changed,**AXES)


@pytest.mark.parametrize('change',[{'source_snapshot_sha256':'f'*64},{'max_nodes':17}])
def test_forest_source_preflight_happens_before_constructor_rng(tmp_path,actual_core_assets,change):
    src=assets(tmp_path,actual_core_assets);src=replace(src,forest_flow_spec=replace(src.forest_flow_spec,**change))
    rng=torch.get_rng_state().clone()
    with pytest.raises(ValueError):construct_primary_owner(PrimaryConfiguration(8,8,SOURCE),src)
    assert torch.equal(rng,torch.get_rng_state())


def test_native_production_objective_keeps_existing_admission_boundary(tmp_path,actual_core_assets):
    owner=construct_primary_owner(PrimaryConfiguration(8,8,SOURCE),assets(tmp_path,actual_core_assets))
    before={n:p.detach().clone()for n,p in owner.named_parameters()};rng=torch.get_rng_state().clone()
    with pytest.raises(ValueError,match='not full corpus'):
        native_forest_flow_objective(owner,'untrusted arguments deliberately not inspected')
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(before[n],p)and p.grad is None for n,p in owner.named_parameters())


def test_complete_owner_checkpoint_recovers_aliases_and_refuses_partial_copy(tmp_path,actual_core_assets):
    owner=construct_primary_owner(PrimaryConfiguration(8,8,SOURCE),assets(tmp_path,actual_core_assets))
    saved=copy.deepcopy(owner.state_dict());clone=copy.deepcopy(owner);clone.load_state_dict(saved)
    assert clone.forest_flow.geometry is clone.core.parent.parent.geometry_fusion.geometry
    r,t=problem();x=state().float();times=torch.tensor([.3,.6],dtype=x.dtype)
    f=replace(skeleton(),states=skeleton().states.float())
    assert torch.equal(owner.forest_flow(r,t,x,times,f),clone.forest_flow(r,t,x,times,f))
    before={n:p.detach().clone()for n,p in owner.named_parameters()}
    saved['forest_flow.readout.weight'].add_(10)
    saved['forest_flow._extra_state']['spec']['source_snapshot_sha256']='f'*64
    with pytest.raises(ValueError):owner.load_state_dict(saved)
    assert all(torch.equal(before[n],p)for n,p in owner.named_parameters())


def test_actual_owner_registers_learned_analogy_without_duplicate_native_parameters(tmp_path,actual_core_assets):
    from mora3.model.forest_analogy import ForestAnalogySpec,NativeForestAnalogyHead,native_forest_analogy_objective
    src=assets(tmp_path,actual_core_assets)
    both=replace(src,forest_analogy_spec=ForestAnalogySpec(4,'d'*64,max_memory=32,max_centers=8))
    models=[construct_primary_owner(PrimaryConfiguration(w,8,SOURCE),both)for w in(8,12,16)]
    owner=models[0]
    assert type(owner.forest_analogy)is NativeForestAnalogyHead and owner.forest_analogy.field is owner.forest_flow
    single=construct_primary_owner(PrimaryConfiguration(8,8,SOURCE),src)
    assert sum(p.numel()for p in owner.parameters())-sum(p.numel()for p in single.parameters())==72
    assert owner.get_extra_state()['forest_analogy']==owner.forest_analogy.get_extra_state()
    family=full_owner_mup_width_family(*models,**AXES)
    assert family.total_numel==tuple(sum(p.numel()for p in m.parameters())for m in models)
    clone=copy.deepcopy(owner);clone.load_state_dict(copy.deepcopy(owner.state_dict()))
    assert clone.forest_analogy.field is clone.forest_flow
    with pytest.raises(ValueError,match='not full corpus'):
        native_forest_analogy_objective(owner,'untrusted inputs must not be examined before admission')


def test_analogy_without_registered_flow_refuses_before_rng(tmp_path,actual_core_assets):
    from mora3.model.forest_analogy import ForestAnalogySpec
    src=replace(actual_core_assets,forest_analogy_spec=ForestAnalogySpec(4,'d'*64))
    rng=torch.get_rng_state().clone()
    with pytest.raises(ValueError,match='forest'):construct_primary_owner(PrimaryConfiguration(8,8,SOURCE),src)
    assert torch.equal(rng,torch.get_rng_state())


def decoder_assets(tmp_path,source):
    from mora3.tokenization.forest_tokens import ForestTokenAxis
    from mora3.model.forest_token_decoder import ForestTokenCodebook,ForestDecoderSpec
    src=assets(tmp_path,source);axis=ForestTokenAxis.from_annotation(src.vocabulary)
    values=torch.randn(axis.tokenizer().size,3,generator=torch.Generator().manual_seed(776))
    book=ForestTokenCodebook(axis,values,SOURCE,COORD,'c'*64,'d'*64,'e'*64)
    return replace(src,forest_decoder_codebook=book,forest_decoder_spec=ForestDecoderSpec(8,max_score_elements=1000000))


def test_native_factory_registers_decoder_with_shared_field_and_finite_axis(tmp_path,actual_core_assets):
    from mora3.model.forest_token_decoder import NativeForestTokenDecoder,native_forest_token_objective
    src=decoder_assets(tmp_path,actual_core_assets)
    models=[construct_primary_owner(PrimaryConfiguration(w,8,SOURCE),src)for w in(8,12,16)]
    owner=models[0];decoder=owner.forest_decoder
    assert type(decoder)is NativeForestTokenDecoder and decoder.field is owner.forest_flow
    assert decoder.axis.base_sha256==__import__('hashlib').sha256(src.vocabulary.specification_json.encode()).hexdigest()
    plain=construct_primary_owner(PrimaryConfiguration(8,8,SOURCE),assets(tmp_path,actual_core_assets))
    assert sum(p.numel()for p in owner.parameters())-sum(p.numel()for p in plain.parameters())==8*8+8*3+3
    family=full_owner_mup_width_family(*models,**AXES)
    assert family.total_numel==tuple(sum(p.numel()for p in m.parameters())for m in models)
    assert owner.get_extra_state()['forest_decoder']==decoder.get_extra_state()
    clone=copy.deepcopy(owner);clone.load_state_dict(copy.deepcopy(owner.state_dict()))
    assert clone.forest_decoder.field is clone.forest_flow
    assert torch.equal(clone.forest_decoder.codebook,decoder.codebook)
    with pytest.raises(ValueError,match='not full corpus'):native_forest_token_objective(owner,'untrusted source')


@pytest.mark.parametrize('fault',('missing-flow','missing-spec','wrong-source','wrong-axis','impossible-score-cap'))
def test_native_decoder_preflight_refuses_before_random_initialization(tmp_path,actual_core_assets,fault):
    from mora3.tokenization.forest_tokens import ForestTokenAxis
    from test_forest_tokens import base
    src=decoder_assets(tmp_path,actual_core_assets)
    if fault=='missing-flow':src=replace(src,forest_flow_spec=None)
    if fault=='missing-spec':src=replace(src,forest_decoder_spec=None)
    if fault=='impossible-score-cap':src=replace(src,forest_decoder_spec=replace(src.forest_decoder_spec,max_score_elements=1))
    if fault=='wrong-source':src=replace(src,forest_decoder_codebook=replace(src.forest_decoder_codebook,source_snapshot_sha256='f'*64))
    if fault=='wrong-axis':
        axis=ForestTokenAxis.from_annotation(base());book=replace(src.forest_decoder_codebook,axis=axis,
            values=torch.zeros(axis.tokenizer().size,3));src=replace(src,forest_decoder_codebook=book)
    rng=torch.get_rng_state().clone()
    with pytest.raises(ValueError):construct_primary_owner(PrimaryConfiguration(8,8,SOURCE),src)
    assert torch.equal(rng,torch.get_rng_state())


def test_full_owner_refuses_frozen_decoder_buffer_change_before_any_shared_copy(tmp_path,actual_core_assets):
    owner=construct_primary_owner(PrimaryConfiguration(8,8,SOURCE),decoder_assets(tmp_path,actual_core_assets))
    saved=copy.deepcopy(owner.state_dict());saved['forest_decoder.codebook'][0,0]+=1
    saved['forest_decoder.readout.weight'].add_(2)
    before={n:p.detach().clone()for n,p in owner.named_parameters()}
    with pytest.raises(ValueError):owner.load_state_dict(saved)
    assert all(torch.equal(before[n],p)for n,p in owner.named_parameters())


def test_all_four_native_heads_share_one_registered_field_and_complete_roles(tmp_path,actual_core_assets):
    from mora3.model.forest_quality import ForestQualitySpec,NativeForestQualityHead,native_forest_quality_objective
    from mora3.model.forest_analogy import ForestAnalogySpec
    src=replace(decoder_assets(tmp_path,actual_core_assets),forest_quality_spec=ForestQualitySpec(('constraint','order'),'f'*64,(1.,1.)),
        forest_analogy_spec=ForestAnalogySpec(4,'d'*64,max_memory=16))
    models=[construct_primary_owner(PrimaryConfiguration(w,8,SOURCE),src)for w in(8,12,16)]
    family=full_owner_mup_width_family(*models,**AXES)
    assert family.total_numel==tuple(sum(p.numel()for p in m.parameters())for m in models)
    owner=models[0]
    assert type(owner.forest_quality)is NativeForestQualityHead
    assert owner.forest_quality.field is owner.forest_analogy.field is owner.forest_decoder.field is owner.forest_flow
    before=sum(p.numel()for p in construct_primary_owner(PrimaryConfiguration(8,8,SOURCE),replace(src,forest_quality_spec=None)).parameters())
    assert sum(p.numel()for p in owner.parameters())-before==36
    clone=copy.deepcopy(owner);clone.load_state_dict(copy.deepcopy(owner.state_dict()))
    assert clone.forest_quality.field is clone.forest_flow
    with pytest.raises(ValueError,match='not full corpus'):native_forest_quality_objective(owner,'untrusted')
