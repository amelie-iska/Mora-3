"""New frozen minimal source-likelihood reward equations; synthetic scope only."""
import copy,math
from dataclasses import replace
import pytest
import torch
from test_annotation_mup import make as annotation
from test_molecule_profile_model import view
from mora3.model.molecule_profile import SharedMoleculeProfileModel
from mora3.model.profile_decoder import ProfileSelfiesDecoder,SelfiesCodec
from mora3.model.profile_gfn import ProfileGFNPolicy
from mora3.model.profile_latent_decoder import ProfileLatentSelfiesDecoder


def native():
    shared=annotation(16,max_tokens=8192).double();v=view('profile_to_molecule')
    axes={'L1000':v.axis,'CellPainting':tuple('CP:'+str(i)for i in range(1783))}
    profiles=SharedMoleculeProfileModel(shared,axes,parameterization_policy='mup-molecule-profile-v1').double()
    decoder=ProfileSelfiesDecoder(profiles,SelfiesCodec(('CCO','CCCCO')),max_tokens=64,parameterization_policy='mup-sequence-v1')
    policy=ProfileGFNPolicy(profiles,dimension=32,horizon=8,hidden=64,source_binding={'scope':'synthetic-unit-no-source-admission'},parameterization_policy='mup-profile-gfn-v1')
    return ProfileLatentSelfiesDecoder(decoder,policy,share_decoder=True).eval(),v


def test_frozen_minimal_all_observation_mean_likelihood_normal_prior_and_stop():
    from mora3.model.frozen_profile_reward import FrozenProfileLikelihoodReward,ProfileObservedMolecules
    m,v=native();rng=torch.get_rng_state().clone();original_modes=[x.training for x in m.modules()]
    r=FrozenProfileLikelihoodReward(m,stage_optimizer_update=0,source_group_manifest_sha256='a'*64)
    assert torch.equal(rng,torch.get_rng_state()) and original_modes==[x.training for x in m.modules()]
    original={p.untyped_storage().data_ptr()for p in m.parameters()}
    assert all(p.untyped_storage().data_ptr()not in original and not p.requires_grad for p in r.parameters())
    assert set(r._modules)=={'payload','backbone','profile_input','embedding','rnn','output','latent_projection'}
    assert not any('.go.'in n or '.ec.'in n for n,_ in r.named_parameters())
    assert sum(p.numel()for p in r.parameters())<sum(p.numel()for p in m.decoder.parameters())
    z=torch.randn(1,32,dtype=torch.float64);ys=('CCO','CCO','CCCCO');group=ProfileObservedMolecules(v.input_bytes,ys,'a'*64)
    likelihoods=[]
    for smiles in ys:
        ids=m.decoder.codec.encode(smiles,max_tokens=64);prefix=torch.tensor([ids[:-1]])
        logits=m((v,),prefix,z).double();likelihoods.append(logits.log_softmax(-1)[0].gather(-1,torch.tensor(ids[1:])[:,None]).sum())
    expected=-.5*(z.square().sum(-1)+32*math.log(2*math.pi))+torch.logsumexp(torch.stack(likelihoods),0)-math.log(len(ys))
    actual=r.log_reward((v,),z,(group,),optimizer_update=249)
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
    assert not actual.requires_grad
    torch.testing.assert_close(r.log_reward((v,),z,(group,),optimizer_update=0,optional_stop=True),expected-math.log(8),rtol=1e-12,atol=1e-12)
    # Live updates must never change the immutable block reference.
    with torch.no_grad():m.decoder.output.bias.add_(2*torch.arange(m.decoder.codec.size))
    torch.testing.assert_close(r.log_reward((v,),z,(group,),optimizer_update=1),actual,rtol=0,atol=0)


def test_frozen_stage_boundaries_and_source_condition_mismatch_refuse():
    from mora3.model.frozen_profile_reward import FrozenProfileLikelihoodReward,ProfileObservedMolecules
    m,v=native();r=FrozenProfileLikelihoodReward(m,stage_optimizer_update=250,source_group_manifest_sha256='a'*64)
    z=torch.zeros(1,32,dtype=torch.float64);g=ProfileObservedMolecules(v.input_bytes,('CCO',),'a'*64)
    for step in (249,500,True):
        with pytest.raises(ValueError):r.log_reward((v,),z,(g,),optimizer_update=step)
    with pytest.raises(ValueError):FrozenProfileLikelihoodReward(m,stage_optimizer_update=1,source_group_manifest_sha256='a'*64)
    for bad in (replace(g,condition_bytes=b'{}'),replace(g,source_group_manifest_sha256='b'*64),replace(g,smiles=())):
        with pytest.raises(ValueError):r.log_reward((v,),z,(bad,),optimizer_update=250)
    with pytest.raises(ValueError):r.train()
    with torch.no_grad():r.output.weight.add_(1)
    with pytest.raises(ValueError,match='frozen'):r.log_reward((v,),z,(g,),optimizer_update=250)


def test_exact_frozen_reference_checkpoint_restore_after_live_weights_change():
    from mora3.model.frozen_profile_reward import FrozenProfileLikelihoodReward,ProfileObservedMolecules
    m,v=native();r=FrozenProfileLikelihoodReward(m,stage_optimizer_update=0,source_group_manifest_sha256='a'*64)
    g=ProfileObservedMolecules(v.input_bytes,('CCO','CCCCO'),'a'*64);z=torch.ones(1,32,dtype=torch.float64)
    before=r.log_reward((v,),z,(g,),optimizer_update=7);saved=r.checkpoint()
    with torch.no_grad():m.latent_projection.weight.add_(1);m.decoder.rnn.weight_hh_l0.mul_(2)
    rng=torch.get_rng_state().clone();restored=FrozenProfileLikelihoodReward.from_checkpoint(m,saved,source_group_manifest_sha256='a'*64,stage_optimizer_update=0)
    assert torch.equal(rng,torch.get_rng_state());assert restored.get_extra_state()==r.get_extra_state()
    torch.testing.assert_close(restored.log_reward((v,),z,(g,),optimizer_update=8),before,rtol=0,atol=0)
    bad=copy.deepcopy(saved);bad['state']['output.weight'][0,0]+=1
    with pytest.raises(ValueError,match='digest'):FrozenProfileLikelihoodReward.from_checkpoint(m,bad,source_group_manifest_sha256='a'*64,stage_optimizer_update=0)
    torch.testing.assert_close(r.log_reward((v,),z,(g,),optimizer_update=9),before,rtol=0,atol=0)


@pytest.mark.parametrize('drift',['codec_index','stage','source','readout'])
def test_frozen_mutable_non_tensor_owner_drift_refuses(drift):
    from mora3.model.frozen_profile_reward import FrozenProfileLikelihoodReward,ProfileObservedMolecules
    m,v=native();r=FrozenProfileLikelihoodReward(m,stage_optimizer_update=0,source_group_manifest_sha256='a'*64)
    if drift=='codec_index':r.codec.index['[C]']+=1
    elif drift=='stage':r.stage_optimizer_update=250
    elif drift=='source':r.source_group_manifest_sha256='b'*64
    else:r._readout['output_mult']=9
    with pytest.raises(ValueError,match='frozen'):
        r.log_reward((v,),torch.zeros(1,32,dtype=torch.float64),(ProfileObservedMolecules(v.input_bytes,('CCO',),'a'*64),),optimizer_update=0)


def test_snapshot_recurrent_cache_references_only_copied_parameters_before_forward():
    from mora3.model.frozen_profile_reward import FrozenProfileLikelihoodReward
    m,v=native();r=FrozenProfileLikelihoodReward(m,stage_optimizer_update=0,source_group_manifest_sha256='a'*64)
    assert all(ref()is getattr(r.rnn,name)for ref,name in zip(r.rnn._flat_weight_refs,r.rnn._flat_weights_names))
    assert all(ref()is not getattr(m.decoder.rnn,name)for ref,name in zip(r.rnn._flat_weight_refs,r.rnn._flat_weights_names))
