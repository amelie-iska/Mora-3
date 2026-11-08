"""Independent generic μAdamW equations and strict recovery, not model qualification."""
import copy
import dataclasses

import pytest
import torch
from torch import nn

from mora3.model.parameterization import Axis, ParameterRole, parameter_ledger, width_family
from mora3.model.mu_optimizer import MuAdamW, build_mu_adamw_groups


def fixture(width=8, second=12):
    m = nn.Module()
    # Deliberately nonsquare and independently widening axes. No NN forward.
    m.input = nn.Parameter(torch.full((width, 3), 0.25, dtype=torch.float64))
    m.hidden = nn.Parameter(torch.full((second, width), 0.25, dtype=torch.float64))
    m.output = nn.Parameter(torch.full((5, second), 0.25, dtype=torch.float64))
    m.vector = nn.Parameter(torch.full((width,), 0.25, dtype=torch.float64))
    m.logZ = nn.Parameter(torch.tensor(0.25, dtype=torch.float64))
    m.fixed = nn.Parameter(torch.ones(2, dtype=torch.float64), requires_grad=False)
    m.tied = m.output
    h, f = Axis('width', 'hidden'), Axis('width', 'ffn')
    x, y = Axis('finite', 'a'*64), Axis('finite', 'b'*64)
    out = ParameterRole('embedding_readout', (y, f))
    roles = dict(input=ParameterRole('input', (h,x)), hidden=ParameterRole('hidden',(f,h)),
                 output=out, tied=out, vector=ParameterRole('vector',(h,)),
                 logZ=ParameterRole('scalar',()), fixed=ParameterRole('vector',(Axis('finite','c'*64),)))
    return m, roles


def family():
    rows = [fixture(4,6), fixture(6,12), fixture(8,18)]
    return width_family(*(parameter_ledger(m,r) for m,r in rows))


def target():
    return fixture(8,18)


def optimizer(m, r, **kw):
    return MuAdamW(m, family(), r, base_lr=0.02, weight_decay=0.1,
                   decay_aliases=('hidden','input','output','tied'), **kw)


def test_reference_rightmost_infinite_axis_lr_and_default_decay():
    m,r = target(); rng=torch.random.get_rng_state().clone()
    groups=build_mu_adamw_groups(m,family(),r,base_lr=.02,weight_decay=.1,
                               decay_aliases=('hidden','input','output','tied'))
    assert len(groups)==5 and len({id(p) for g in groups for p in g['params']})==5
    by={g['mup_aliases'][0]:g for g in groups}
    # Hidden shape is (18,8), base (6,4): upstream main is LAST infinite axis, 8/4=2.
    assert by['hidden']['lr']==.01 and by['hidden']['weight_decay']==.2
    for key in ('input','output','vector','logZ'):
        assert by[key]['lr']==.02
    assert by['output']['mup_aliases']==('output','tied')
    assert by['output']['weight_decay']==.1 and by['vector']['weight_decay']==0
    assert sum(p.numel() for g in groups for p in g['params'])==267
    assert torch.equal(rng,torch.random.get_rng_state())
    assert all(p.grad is None and torch.all(p==.25) for p in m.parameters() if p.requires_grad)


def test_reference_decoupled_flag_keeps_master_decay():
    m,r=target();opt=optimizer(m,r,decoupled_wd=True)
    g=next(g for g in opt.param_groups if g['mup_aliases']==('hidden',))
    assert g['lr']==.01 and g['weight_decay']==.1


@pytest.mark.parametrize('decoupled',[False,True])
def test_one_step_independent_adamw_equation(decoupled):
    m,r=target();opt=optimizer(m,r,decoupled_wd=decoupled)
    before={id(p):p.detach().clone() for p in m.parameters()}
    for p in m.parameters():
        if p.requires_grad:p.grad=torch.full_like(p,2)
    opt.step()
    for g in opt.param_groups:
        p=g['params'][0]
        expected=before[id(p)]*(1-g['lr']*g['weight_decay'])-g['lr']*2/(2+1e-8)
        torch.testing.assert_close(p,expected,rtol=0,atol=1e-16)
    assert torch.equal(m.fixed,before[id(m.fixed)])


def test_scheduler_ratios_and_fresh_optimizer_resume_exact():
    m,r=target();opt=optimizer(m,r)
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda n: .8**n)
    for _ in range(3):
        for p in m.parameters():
            if p.requires_grad:p.grad=torch.ones_like(p)
        opt.step();sched.step()
    saved=copy.deepcopy(opt.state_dict()); weights=copy.deepcopy(m.state_dict())
    saved_moments={i:s['exp_avg'].clone() for i,s in saved['state'].items()}
    resumed,rr=target();resumed.load_state_dict(weights);other=optimizer(resumed,rr)
    rng=torch.random.get_rng_state().clone();other.load_state_dict(saved)
    assert torch.equal(rng,torch.random.get_rng_state())
    for left,right in zip(opt.param_groups,other.param_groups):
        assert left['lr']==right['lr'] and left['weight_decay']==right['weight_decay']
    for model,o in ((m,opt),(resumed,other)):
        for p in model.parameters():
            if p.requires_grad:p.grad=torch.full_like(p,.3)
        o.step()
    for p,q in zip(m.parameters(),resumed.parameters()):assert torch.equal(p,q)
    # Loading does not alias the incoming checkpoint's moment tensors.
    for i,state in saved['state'].items():assert torch.equal(state['exp_avg'],saved_moments[i])


@pytest.mark.parametrize('failure',['missing-decay','partial-tie-decay','stale-shape','stale-finite',
                                  'forged-numel','forged-total','custom','three-widths',
                                  'nan-lr','bool-lr','negative-wd','bool-flag'])
def test_invalid_binding_refuses_before_tensor_rng_or_gradient_mutation(failure):
    m,r=target();f=family();kw=dict(base_lr=.02,weight_decay=.1,
                                 decay_aliases=('hidden','input','output','tied'))
    if failure=='missing-decay':kw['decay_aliases']=('absent',)
    if failure=='partial-tie-decay':kw['decay_aliases']=('output',)
    if failure=='stale-shape':m.hidden=nn.Parameter(torch.zeros(18,7,dtype=torch.float64))
    if failure=='stale-finite':r['input']=ParameterRole('input',(Axis('width','hidden'),Axis('finite','c'*64)))
    if failure in ('forged-numel','forged-total','custom','three-widths'):
        entry=f.target.parameters[0]
        if failure=='forged-numel':entry=dataclasses.replace(entry,numel=entry.numel+1)
        if failure=='custom':entry=dataclasses.replace(entry,role='custom')
        if failure=='three-widths':
            entry=dataclasses.replace(entry,shape=(2,3,4),numel=24,
                axes=tuple(Axis('width',n) for n in ('u','v','w')))
        ledger=dataclasses.replace(f.target,parameters=(entry,)+f.target.parameters[1:])
        if failure=='forged-total':ledger=dataclasses.replace(ledger,total_numel=1)
        f=dataclasses.replace(f,target=ledger)
    if failure=='nan-lr':kw['base_lr']=float('nan')
    if failure=='bool-lr':kw['base_lr']=True
    if failure=='negative-wd':kw['weight_decay']=-.1
    if failure=='bool-flag':kw['decoupled_wd']=1
    before=[p.detach().clone() for p in m.parameters()];rng=torch.random.get_rng_state().clone()
    with pytest.raises(ValueError):build_mu_adamw_groups(m,f,r,**kw)
    assert torch.equal(rng,torch.random.get_rng_state())
    assert all(torch.equal(a,b) and b.grad is None for a,b in zip(before,m.parameters()))


@pytest.mark.parametrize('failure',['ratio','decay','replacement','alias','role'])
def test_live_drift_precedes_adam_state_or_parameter_mutation(failure):
    m,r=target();opt=optimizer(m,r)
    if failure=='ratio':opt.param_groups[0]['lr']*=2
    if failure=='decay':opt.param_groups[0]['weight_decay']=.7
    if failure=='replacement':m.input=nn.Parameter(m.input.detach().clone())
    if failure=='alias':m.tied=nn.Parameter(m.tied.detach().clone())
    if failure=='role':m.input.requires_grad_(False)
    before=[p.detach().clone() for p in m.parameters()]
    for p in m.parameters():
        if p.requires_grad:p.grad=torch.ones_like(p)
    with pytest.raises(ValueError):opt.step()
    assert not opt.state
    assert all(torch.equal(a,b) for a,b in zip(before,m.parameters()))


@pytest.mark.parametrize('failure',['metadata','family','lr-ratio','decay','owner','extra-state',
                                  'shape','nan-moment','negative-square','fractional-step','legacy'])
def test_checkpoint_prevalidation_refuses_without_optimizer_or_rng_mutation(failure):
    m,r=target();opt=optimizer(m,r)
    for p in m.parameters():
        if p.requires_grad:p.grad=torch.ones_like(p)
    opt.step();state=copy.deepcopy(opt.state_dict())
    if failure=='metadata':state['mora3_mu_adamw']['schema']='wrong'
    if failure=='family':state['mora3_mu_adamw']['family_sha256']='f'*64
    if failure=='lr-ratio':state['param_groups'][0]['lr']*=2
    if failure=='decay':state['param_groups'][0]['weight_decay']=1
    if failure=='owner':state['param_groups'][0]['params']=[999]
    if failure=='extra-state':state['state'][999]={}
    if failure=='shape':state['state'][0]['exp_avg']=torch.zeros(1)
    if failure=='nan-moment':state['state'][0]['exp_avg'].fill_(float('nan'))
    if failure=='negative-square':state['state'][0]['exp_avg_sq'].fill_(-1)
    if failure=='fractional-step':state['state'][0]['step'].fill_(.5)
    if failure=='legacy':del state['mora3_mu_adamw']
    fresh,rr=target();other=optimizer(fresh,rr);rng=torch.random.get_rng_state().clone()
    before=copy.deepcopy(other.state_dict())
    with pytest.raises(ValueError):other.load_state_dict(state)
    assert other.state_dict()==before and torch.equal(rng,torch.random.get_rng_state())


def test_zero_lr_schedule_and_conflicting_alias_decay_refusal():
    m,r=target();opt=optimizer(m,r)
    for group in opt.param_groups:group['lr']=0.
    opt.step();state=opt.state_dict();assert state['mora3_mu_adamw']['scheduled_master_lr']==0.
    assert all(g['lr']==0 for g in state['param_groups'])


@pytest.mark.parametrize('failure',['shared-storage','complex','sparse'])
def test_unqualified_parameter_storage_and_numeric_domains_refuse(failure):
    m,r=target()
    if failure=='shared-storage':m.vector=nn.Parameter(m.input.reshape(-1)[:8])
    if failure=='complex':m.hidden=nn.Parameter(m.hidden.to(torch.complex128))
    if failure=='sparse':m.hidden=nn.Parameter(m.hidden.to_sparse())
    f=family(); live=parameter_ledger(m,r)
    f=dataclasses.replace(f,target=live)
    # Update declared metadata legitimately; storage/domain must still refuse.
    f=dataclasses.replace(f,total_numel=(f.base.total_numel,f.delta.total_numel,live.total_numel))
    with pytest.raises(ValueError):
        build_mu_adamw_groups(m,f,r,base_lr=.02,weight_decay=.1,
                             decay_aliases=('hidden','input','output','tied'))


def test_paired_deepcopy_exact_owner_aliases_state_and_rng():
    model,roles=target();opt=optimizer(model,roles)
    state=copy.deepcopy(opt.state_dict())
    rng=torch.get_rng_state().clone()
    staged_model,staged_opt=copy.deepcopy((model,opt))
    staged_opt.load_state_dict(state)
    assert staged_opt.state_dict()==state
    assert torch.equal(rng,torch.get_rng_state())
    assert staged_model.output is staged_model.tied
    a=dict(model.named_parameters());b=dict(staged_model.named_parameters())
    for left,right in zip(opt.param_groups,staged_opt.param_groups):
        name=left['mup_aliases'][0]
        assert left['params'][0] is a[name] and right['params'][0] is b[name]
        assert right['params'][0] is not left['params'][0]
    with torch.no_grad():staged_model.input.add_(2.)
    assert not torch.equal(staged_model.input,model.input)
    assert opt.state_dict()==state


def test_standalone_optimizer_deepcopy_requires_paired_explicit_model_identity():
    model,roles=target();opt=optimizer(model,roles)
    with pytest.raises(ValueError,match='paired'):copy.deepcopy(opt)


@pytest.mark.parametrize('member,width,second,mult',[('base',4,6,1.),('delta',6,12,1.5),('target',8,18,2.)])
def test_explicit_family_member_groups_and_checkpoint_policy(member,width,second,mult):
    model,roles=fixture(width,second)
    opt=MuAdamW(model,family(),roles,base_lr=.02,weight_decay=.1,
        decay_aliases=('hidden','input','output','tied'),member=member)
    group=next(g for g in opt.param_groups if g['mup_aliases']==('hidden',))
    assert group['lr']==pytest.approx(.02/mult,rel=1e-15,abs=0)
    assert group['weight_decay']==.1*mult
    meta=opt.state_dict()['mora3_mu_adamw']
    assert meta['schema']==('mora3.mu-adamw.v1' if member=='target' else 'mora3.mu-adamw.v2')
    assert ('member' not in meta) if member=='target' else meta['member']==member
    clone,copied=copy.deepcopy((model,opt));copied.load_state_dict(opt.state_dict())
    assert copied.state_dict()==opt.state_dict()


def test_member_mismatch_refuses_optimizer_checkpoint_before_state_mutation():
    a,ra=fixture(4,6);b,rb=fixture(8,18)
    oa=MuAdamW(a,family(),ra,base_lr=.02,weight_decay=.1,
        decay_aliases=('hidden','input','output','tied'),member='base')
    ob=optimizer(b,rb);before=copy.deepcopy(ob.state_dict())
    with pytest.raises(ValueError):ob.load_state_dict(oa.state_dict())
    assert ob.state_dict()==before
