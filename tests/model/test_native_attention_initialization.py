"""Explicit native/SciPlex initializer recovery; legacy schemas stay unchanged."""
import copy

import pytest
import torch

from mora3.model.tokengt import TokenGT
from mora3.model.sciplex import SciPlexSharedModel
from mora3.model.response_flow import ResponseFlowHead
from test_sciplex_shared import task
from test_sciplex_opt_in_owner import equation


def backbone(width=8):
    return TokenGT(width,2,width=width,heads=2,layers=2,scaling_policy='mup-v1',
        attn_mult=.5,initialization_policy='mup-transformer-v1',encoder_var=.7)


def owner(width=8):
    return SciPlexSharedModel(('A','B'),width=width,scaling_policy='mup-v1',attn_mult=.5,
        initialization_policy='mup-transformer-v1',encoder_var=.7)


def test_explicit_native_and_owner_versions_full_configuration_tensor_reconstruction():
    native=backbone();m=owner()
    assert native.configuration()['schema']=='tokengt-native-configuration-v2'
    assert native.configuration()['initialization_policy']=='mup-transformer-v1'
    assert native.configuration()['encoder_var']==.7
    assert m.config['version']==3 and m.config['native_backbone']==m.backbone.configuration()
    for original,factory,config in [(native,TokenGT.from_configuration,native.configuration()),
                                    (m,SciPlexSharedModel.from_configuration,m.get_extra_state())]:
        with torch.no_grad():
            for p in original.parameters():p.add_(.125)
        saved=copy.deepcopy(original.state_dict());new=factory(config)
        rng=torch.get_rng_state().clone();new.load_state_dict(saved,strict=True)
        assert torch.equal(rng,torch.get_rng_state())
        for p,q in zip(original.parameters(),new.parameters()):assert torch.equal(p,q)
        # Query is now learned/nonzero; loading never reapplies zero initialization.
        root=new.backbone if isinstance(new,SciPlexSharedModel) else new
        assert torch.count_nonzero(root.layers[0].attention.qkv.weight[:8])==64
        assert '_native_initialization' in (saved if root is new else {
            k.removeprefix('backbone.'):v for k,v in saved.items() if k.startswith('backbone.')})
    with pytest.raises(ValueError,match='legacy|version|schema'):ResponseFlowHead(m)


@pytest.mark.parametrize('scale',['legacy','mup-v1'])
def test_previous_native_owner_versions_and_state_keys_unchanged(scale):
    torch.manual_seed(2026110614)
    a=TokenGT(8,2,width=8,heads=2,layers=2,scaling_policy=scale)
    arng=torch.get_rng_state().clone()
    torch.manual_seed(2026110614)
    b=TokenGT(8,2,width=8,heads=2,layers=2,scaling_policy=scale,
              initialization_policy='legacy',encoder_var=1.)
    assert torch.equal(arng,torch.get_rng_state())
    assert a.configuration()==b.configuration() and a.configuration()['schema']=='tokengt-native-configuration-v1'
    assert set(a.state_dict())==set(b.state_dict())
    assert not any('initialization' in key for key in a.state_dict())
    for p,q in zip(a.parameters(),b.parameters()):assert torch.equal(p,q)
    m=SciPlexSharedModel(('A','B'),width=8,scaling_policy=scale)
    assert m.config['version']==(1 if scale=='legacy' else 2)
    assert not any('initialization' in key for key in m.state_dict())


@pytest.mark.parametrize('factory',['native','owner'])
@pytest.mark.parametrize('failure',['scale','tropical','variance','bool-var','legacy-var','unknown'])
def test_invalid_new_native_constructor_precedes_rng(factory,failure):
    kw=dict(scaling_policy='mup-v1',initialization_policy='mup-transformer-v1',encoder_var=.7)
    if failure=='scale':kw['scaling_policy']='legacy'
    if failure=='tropical':kw['attention']='tropical-code'
    if failure=='variance':kw['encoder_var']=-1.
    if failure=='bool-var':kw['encoder_var']=True
    if failure=='legacy-var':kw['initialization_policy']='legacy'
    if failure=='unknown':kw['initialization_policy']='unknown'
    rng=torch.get_rng_state().clone()
    with pytest.raises(ValueError):
        if factory=='native':TokenGT(8,2,width=8,heads=2,layers=2,**kw)
        else:SciPlexSharedModel(('A','B'),width=8,**kw)
    assert torch.equal(rng,torch.get_rng_state())


@pytest.mark.parametrize('factory',['native','owner'])
@pytest.mark.parametrize('failure',['old-version','legacy-policy','missing-var','bool-var','unknown'])
def test_new_configuration_drift_refuses_before_reconstruction_rng(factory,failure):
    m=backbone() if factory=='native' else owner()
    config=m.configuration() if factory=='native' else m.get_extra_state()
    native=config if factory=='native' else config['native_backbone']
    if failure=='old-version':
        if factory=='native':native['schema']='tokengt-native-configuration-v1'
        else:config['version']=2
    if failure=='legacy-policy':native['initialization_policy']='legacy'
    if failure=='missing-var':del native['encoder_var']
    if failure=='bool-var':native['encoder_var']=False
    if failure=='unknown':native['extra']=1
    rng=torch.get_rng_state().clone()
    with pytest.raises(ValueError):type(m).from_configuration(config)
    assert torch.equal(rng,torch.get_rng_state())


@pytest.mark.parametrize('factory',['native','owner'])
@pytest.mark.parametrize('failure',['root-metadata','layer-metadata','variance','legacy-source','legacy-target'])
def test_initialization_checkpoint_preflight_before_root_or_child_tensor_copy(factory,failure):
    m=backbone() if factory=='native' else owner()
    target=copy.deepcopy(m);state=copy.deepcopy(m.state_dict())
    prefix='' if factory=='native' else 'backbone.'
    if failure=='root-metadata':del state[prefix+'_native_initialization']
    if failure=='layer-metadata':del state[prefix+'layers.1.attention._attention_initialization']
    if failure=='variance':state[prefix+'layers.1.attention._attention_initialization']['encoder_var']=2.
    if failure=='legacy-source':
        old=(TokenGT(8,2,width=8,heads=2,layers=2,scaling_policy='mup-v1',attn_mult=.5)
             if factory=='native' else SciPlexSharedModel(('A','B'),width=8,
                                                       scaling_policy='mup-v1',attn_mult=.5))
        state=old.state_dict()
    if failure=='legacy-target':
        target=(TokenGT(8,2,width=8,heads=2,layers=2,scaling_policy='mup-v1',attn_mult=.5)
                if factory=='native' else SciPlexSharedModel(('A','B'),width=8,
                                                           scaling_policy='mup-v1',attn_mult=.5))
    first=next(k for k,v in state.items() if isinstance(v,torch.Tensor))
    state[first]=torch.full_like(state[first],19.)
    before=[p.detach().clone() for p in target.parameters()];rng=torch.get_rng_state().clone()
    with pytest.raises(ValueError):target.load_state_dict(state,strict=True)
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(a,b) for a,b in zip(before,target.parameters()))


def test_legacy_native_rejects_grafted_child_initializer_before_graph_token_copy():
    target=TokenGT(8,2,width=8,heads=2,layers=2,scaling_policy='mup-v1',attn_mult=.5)
    state=copy.deepcopy(target.state_dict())
    state['layers.1.attention._attention_initialization']=backbone().layers[1].attention.scaling_config()['initialization']
    state['graph_token']=torch.full_like(state['graph_token'],91.)
    before=[p.detach().clone() for p in target.parameters()]
    with pytest.raises(ValueError):target.load_state_dict(state)
    assert all(torch.equal(a,b) for a,b in zip(before,target.parameters()))


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('width',[8,12])
def test_native_initializer_alias_binding_independent_owner_equations_and_gradients(device,dtype,width):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.manual_seed(2026110615);m=owner(width).to(device=device,dtype=dtype)
    assert sum(p.numel() for p in m.parameters())<=2504
    for layer in m.backbone.layers:
        assert layer.attention.initialization_policy=='mup-transformer-v1'
        assert not torch.count_nonzero(layer.attention.qkv.weight[:width])
    t=task('response' if width==8 else 'intervention')
    actual=m([t]);expected=equation(m,t);tol=2e-5 if dtype==torch.float32 else 1e-11
    torch.testing.assert_close(actual,expected,rtol=tol,atol=tol)
    p=tuple(m.parameters())
    ga=torch.autograd.grad(actual.sum(),p,allow_unused=True)
    gb=torch.autograd.grad(expected.sum(),p,allow_unused=True)
    for a,b in zip(ga,gb):
        assert (a is None)==(b is None)
        if a is not None:torch.testing.assert_close(a,b,rtol=tol,atol=tol)
