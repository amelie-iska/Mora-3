"""Complete native instance owner inventory: exact aliases and nonmutation."""
import copy
import pytest
import torch
from mora3.model import parameterization as p
from mora3.model.instance_cooperative import InstanceCooperativeModel
from mora3.model.color_instance_task import ColorInstanceTask
from test_lincs_cooperative import cases
from test_color_cooperative_parameterization import make, inventory as parent_inventory
from test_reconstruction_owner_parameterization import inherited
from test_color_supervised_task import source


def inventory(m,owners=None,**kw):
    assert hasattr(p,'instance_cooperative_parameter_ledger'),'missing cooperative instance inventory'
    args=dict(owners=inherited(m.parent.parent) if owners is None else owners,
        identifier_axis=('context','observation'),latent_axis=('x','y'),
        configuration=m.parent.parent.reconstruction.get_extra_state(),
        geometry_identifier_axis=tuple(f'object:{i}' for i in range(16)),
        image_identifier_axis=('context','observation'),channel_axis=('R','G','B'),
        color_identifier_axis=('context','observation'))
    args.update(kw)
    return p.instance_cooperative_parameter_ledger(m,**args)


def model(tmp_path,cases,device='cpu',dtype=torch.float32,policy='native-max512-v1',slots=3,slot_policy='bounded-max64-v1'):
    parent=make(tmp_path,cases,device,policy).to(dtype=dtype)
    parent.parent.configuration['reconstruction']=parent.parent.reconstruction.get_extra_state()
    parent.configuration['parent']=copy.deepcopy(parent.parent.configuration)
    return InstanceCooperativeModel(parent,slots=slots,max_output_bytes=1000000,slot_policy=slot_policy).eval()


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('policy',['native-max512-v1','native-max1000-v1'])
def test_complete_counts_aliases_numerical_and_state_preservation(tmp_path,cases,device,dtype,policy):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.manual_seed(36501)
    m=model(tmp_path,cases,device,dtype,policy)
    s=source('ordered_binary_instances');task=ColorInstanceTask(s,s.binding_sha256)
    params=tuple(m.parameters());before=copy.deepcopy(m.state_dict());modes=[x.training for x in m.modules()]
    loss=m.instance_objective(task)['loss'];grads=torch.autograd.grad(loss,params,allow_unused=True)
    rng=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state().clone() if device=='cuda' else None
    ledger=inventory(m);parent=parent_inventory(m.parent)
    w=m.instance.color.width;k=m.instance.color.head.slots
    # Shared pixels, image backbone and entire conditioner add zero parameters.
    # Only slot 1x1 projection (2W inputs) and contextual offsets are new.
    assert ledger.total_numel==parent.total_numel+k*(2*w+1)+k*(w+1)
    assert ledger.total_numel==sum(t.numel() for t in params)
    rows={a:e for e in ledger.parameters for a in e.aliases}
    assert set(rows)==set(dict(m.named_parameters(remove_duplicate=False)))
    for a,b in [('instance.color.pixels.enc0.0.weight','parent.color.color.pixels.enc0.0.weight'),
                ('instance.color.backbone.incidence.weight','parent.color.color.backbone.incidence.weight'),
                ('instance.context.payload.embedding.weight','parent.color.context.payload.embedding.weight'),
                ('instance.context.graph.incidence.weight','parent.color.context.graph.incidence.weight')]:
        assert rows[a] is rows[b]
    assert torch.equal(rng,torch.get_rng_state())
    if gpu is not None:assert torch.equal(gpu,torch.cuda.get_rng_state())
    second=m.instance_objective(task)['loss'];torch.testing.assert_close(loss,second,rtol=0,atol=0)
    second_grads=torch.autograd.grad(second,params,allow_unused=True)
    rtol,atol=(0,0) if device=='cpu' else ((1e-5,1e-8) if dtype==torch.float32 else (1e-10,1e-12))
    for a,b in zip(grads,second_grads):
        assert (a is None)==(b is None)
        if a is not None:torch.testing.assert_close(a,b,rtol=rtol,atol=atol)
    for key,value in m.state_dict().items():
        if torch.is_tensor(value):assert torch.equal(value,before[key])
        else:assert value==before[key]
    assert modes==[x.training for x in m.modules()] and all(t.grad is None for t in params)
    with pytest.raises(ValueError,match='custom'):p.width_family(ledger,ledger,ledger)


@pytest.mark.parametrize('drift',['architecture','version','pairing','admission','parent-config',
 'instance-config','color-config','head-config','extra-config','extra-module','parameter','buffer',
 'subclass','pixels','backbone','context','context-incidence','context-payload','owner-count',
 'parent-source','parent-pairing','axis','rgb','owners-missing','owners-extra','owners-stale','tie','budget','slots','slot-policy'])
def test_configuration_ownership_and_axis_refusals(tmp_path,cases,drift):
    m=model(tmp_path,cases);o=inherited(m.parent.parent);kw={}
    if drift in ('architecture','version','pairing'):m.configuration[drift]='other'
    elif drift=='admission':m.configuration['training_admission']=True
    elif drift in ('parent-config','instance-config','color-config','head-config'):
        m.configuration[{'parent-config':'parent','instance-config':'instance','color-config':'instance_color','head-config':'instance_head'}[drift]]={}
    elif drift=='extra-config':m.configuration['extra']='unknown'
    elif drift=='extra-module':m.extra=torch.nn.Identity()
    elif drift=='parameter':m.extra=torch.nn.Parameter(torch.ones(1))
    elif drift=='buffer':m.register_buffer('extra',torch.ones(1))
    elif drift=='subclass':
        class Other(type(m)):pass
        m.__class__=Other
    elif drift in ('pixels','backbone'):setattr(m.instance.color,drift,copy.deepcopy(getattr(m.instance.color,drift)))
    elif drift=='context':m.instance.context=copy.deepcopy(m.instance.context)
    elif drift=='context-incidence':m.instance.context.graph.incidence=copy.deepcopy(m.instance.context.graph.incidence)
    elif drift=='context-payload':m.instance.context.payload=copy.deepcopy(m.instance.context.payload)
    elif drift=='owner-count':m._context_owners=m._context_owners[:1]
    elif drift=='parent-source':m.parent.parent.lincs.source_axes['L1000']['extra']='drift'
    elif drift=='parent-pairing':m.parent.configuration['pairing']='invented'
    elif drift=='axis':kw['color_identifier_axis']=('context',)
    elif drift=='rgb':kw['channel_axis']=('B','G','R')
    elif drift=='owners-missing':o.pop('goa')
    elif drift=='owners-extra':o['instance']=o['goa']
    elif drift=='owners-stale':m.parent.parent.residue.head.output.weight.requires_grad_(False)
    elif drift=='tie':m.instance.offset.weight=m.parent.color.offset.weight
    elif drift=='budget':m.instance.color.head.max_output_bytes=2
    elif drift=='slots':m.instance.color.head.slots=2
    elif drift=='slot-policy':m.instance.color.head.slot_policy='invented'
    with pytest.raises(ValueError):inventory(m,o,**kw)


def test_extended_slots_and_compatible_aliases(tmp_path,cases):
    m=model(tmp_path,cases,slots=65,slot_policy='extended-max512-v2')
    ledger=inventory(m)
    assert ledger.total_numel==sum(t.numel() for t in m.parameters())
    # Compatible output-coordinate declarations allow explicitly tied biases.
    m.instance.offset.bias=m.instance.color.head.projection.bias
    tied=inventory(m)
    assert tied.total_numel==ledger.total_numel-65
