"""Full native inverse source context and shared byte objective, no admission."""
import copy
from dataclasses import replace
import json
import pytest
import torch
from torch.nn import functional as F
from test_lincs_native import cases,project,api
from test_pubchem_admet_mup import pubchem
from mora3.model.lincs_model import SharedLincsModel


def make(cases,width=16):
    from mora3.model.lincs_inverse import SharedLincsInverseModel
    decoder=pubchem(width).eval();vs=tuple(project(api(),c)for c in cases.values())
    axes={v.modality:c[7]for v,c in zip(vs,cases.values())}
    lincs=SharedLincsModel(decoder.shared,axes,parameterization_policy='mup-lincs-v1').double().eval()
    return SharedLincsInverseModel(lincs,decoder).eval(),tuple(v for v in vs if v.direction=='profile_to_molecule')


@pytest.mark.parametrize('chunk,recompute',[(1,False),(3,True),(1000,False)])
def test_full_native_smiles_eos_and_all_gradients_match_direct_shared_decoder(cases,chunk,recompute):
    m,vs=make(cases);actual=m.objective(vs,chunk_size=chunk,recompute=recompute)
    context=m.lincs.condition(vs);d=m.decoder;total=context.sum()*0;count=0
    for row,v in enumerate(vs):
        raw=v.targets()['canonical_isomeric_smiles'].encode('utf8');ids=d.codec.encode(raw)
        tokens=torch.tensor(ids,dtype=torch.long);hidden,_=d.rnn(d.embedding(tokens[:-1][None]),context[row:row+1][None].contiguous())
        total+=F.cross_entropy(d._readout(hidden).reshape(-1,d.codec.size),tokens[1:],reduction='sum');count+=len(ids)-1
    expected=total/count
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
    ps=tuple(m.parameters());a=torch.autograd.grad(actual,ps,allow_unused=True);b=torch.autograd.grad(expected,ps,allow_unused=True)
    for x,y in zip(a,b):
        assert(x is None)==(y is None)
        if x is not None:torch.testing.assert_close(x,y,rtol=1e-10,atol=1e-11)
    grad={id(p):g for p,g in zip(ps,a)}
    for p in (d.embedding.weight,d.rnn.weight_hh_l0,d.output.weight,m.lincs.profile_input.L1000.weight,m.lincs.profile_mask.L1000.weight):
        assert grad[id(p)]is not None and bool(grad[id(p)].abs().sum()>0)
    assert len(m.lincs.axes['L1000'])==978
    assert {id(p)for p in m.parameters()}=={id(p)for p in m.lincs.parameters()}|{id(p)for p in d.parameters()}


def test_target_and_audit_mutation_never_changes_native_context(cases):
    from mora3.data.lincs_native import _binding
    m,vs=make(cases);v=vs[0];target=json.dumps({'canonical_isomeric_smiles':'C'}).encode();audit=b'{}'
    changed=replace(v,target_bytes=target,provenance_bytes=audit,binding_sha256=_binding(v.input_bytes,target,audit))
    torch.testing.assert_close(m.condition((v,)),m.condition((changed,)),rtol=0,atol=0)
    assert not torch.equal(m.objective((v,)),m.objective((changed,)))


def test_all_target_and_direction_refusals_precede_encoder_work(cases):
    from mora3.data.lincs_native import _binding
    m,vs=make(cases);v=vs[0];bad=b'{"canonical_isomeric_smiles":""}'
    changed=replace(v,target_bytes=bad,binding_sha256=_binding(v.input_bytes,bad,v.provenance_bytes))
    def fail(*args,**kw):raise AssertionError('encoder entered before target/resource validation')
    m.lincs.condition=fail;rng=torch.get_rng_state().clone()
    with pytest.raises(ValueError,match='complete'):m.objective((changed,))
    m.max_total_target_bytes=1
    with pytest.raises(ValueError):m.objective(vs)
    assert torch.equal(rng,torch.get_rng_state())


def test_exact_shared_owner_and_checkpoint_configuration(cases):
    from mora3.model.lincs_inverse import SharedLincsInverseModel
    m,vs=make(cases);before=copy.deepcopy(m.state_dict());clone=copy.deepcopy(m);clone.load_state_dict(before)
    assert clone.decoder.shared is clone.lincs.shared
    assert clone.get_extra_state()==m.get_extra_state()
    with pytest.raises(ValueError,match='shared'):SharedLincsInverseModel(m.lincs,pubchem(16))


def test_inverse_exact_alias_roles_across_full_width_family(cases):
    from mora3.model.lincs_inverse import lincs_inverse_parameter_roles
    from mora3.model.parameterization import parameter_ledger,width_family
    from mora3.model.mu_optimizer import MuAdamW
    ms=tuple(make(cases,w)[0]for w in(8,12,16));roles=[lincs_inverse_parameter_roles(m,identifier_axis=('protein-coordinate',))for m in ms]
    family=width_family(*(parameter_ledger(m,r)for m,r in zip(ms,roles)))
    m=ms[-1];r=roles[-1]
    assert r['lincs.shared.backbone.layers.0.attention.qkv.weight']==r['decoder.shared.backbone.layers.0.attention.qkv.weight']
    assert set(r)==set(dict(m.named_parameters(remove_duplicate=False)))
    assert family.total_numel[-1]==sum(p.numel()for p in m.parameters())
    opt=MuAdamW(m,family,r,base_lr=.001,weight_decay=0.,decay_aliases=());assert not opt.state
