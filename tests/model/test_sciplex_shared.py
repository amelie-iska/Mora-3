import copy
import hashlib
import pytest
import torch
from mora3.data.connected_sciplex import SciPlexTask
from mora3.data.gperturb_sciplex import array_task
from mora3.data.sciplex_admission import ObservationalSciPlexTask,POLICY
from mora3.model.sciplex import SciPlexSharedModel


def task(direction='response',counts=(1,2),identity='Dex',cell='a'):
    row=dict(cell_id=cell,source_index=0,intervention=identity,dose_coordinate=.1,
             dose_units='source_scaled_coordinate',counts=list(counts),n_genes=2,n_counts=sum(counts))
    t=SciPlexTask.from_sidecar(array_task(row,['A','B'],direction=direction),source_sha256='a'*64)
    return ObservationalSciPlexTask.from_receipt(t,dict(version=1,policy=POLICY,split='train_only',
        leakage_group='study:GSE139944:GSM4150377',audit_sha256='b'*64,
        source_sha256='a'*64,task_sha256=hashlib.sha256(t.canonical_bytes()).hexdigest()))


def test_hidden_targets_do_not_affect_graph_or_predictions():
    m=SciPlexSharedModel(('A','B'),width=8)
    for a,b in [(task(),task(counts=(91,72),cell='hidden')),
                (task('intervention'),task('intervention',identity='SAHA'))]:
        ga,gb=m.graph(a),m.graph(b)
        for k in ga: assert torch.equal(ga[k],gb[k])
        assert torch.equal(m([a]),m([b]))


def test_all_axes_used_and_shared_backbone_gradients_both_masks():
    m=SciPlexSharedModel(('A','B'),width=8).double()
    a=task('intervention');b=task('intervention',counts=(1,9))
    assert not torch.equal(m.graph(a)['features'],m.graph(b)['features'])
    for direction in ('response','intervention'):
        m.zero_grad();loss=m.loss([task(direction)]);loss.backward()
        assert torch.isfinite(loss)
        assert m.backbone.feature.weight.grad.abs().sum()>0
        if direction=='intervention':assert (m.expression.weight.grad.abs().sum(0)>0).all()


def test_config_axis_and_receipt_binding_fail_closed():
    m=SciPlexSharedModel(('A','B'),width=8)
    with pytest.raises(ValueError):m.graph(task().task)
    with pytest.raises(ValueError):SciPlexSharedModel(('B','A'),width=8).graph(task())
    with pytest.raises(ValueError):SciPlexSharedModel(('B','A'),width=8).load_state_dict(m.state_dict())
    binding=m.binding([task()]);assert binding!=m.binding([task(counts=(3,4))])
    assert binding!=m.binding([task('intervention')])


def test_cpu_cuda_forward_gradient_parity_and_loss_equation():
    torch.manual_seed(51);m=SciPlexSharedModel(('A','B'),width=8).double()
    for direction in ('response','intervention'):
        ts=[task(direction)];out=m(ts);p=ts[0].project()
        if direction=='response':expected=((out[0]-torch.log1p(p['targets']['expression']))**2).mean()
        else:expected=torch.nn.functional.cross_entropy(out[:,:5],p['targets']['intervention_index'].reshape(1))+((out[0,5]-p['targets']['dose_coordinate'])**2)
        assert torch.equal(m.loss(ts),expected)
        if torch.cuda.is_available():
            g=copy.deepcopy(m).cuda();a=m.loss(ts);b=g.loss(ts);a.backward();b.backward()
            assert torch.allclose(m(ts),g(ts).cpu(),atol=1e-10,rtol=1e-10)
            for x,y in zip(m.parameters(),g.parameters()):
                if x.grad is not None:assert torch.allclose(x.grad,y.grad.cpu(),atol=1e-10,rtol=1e-10)
        m.zero_grad()


def test_repeated_gene_symbols_are_distinct_source_positions():
    m=SciPlexSharedModel(('A','A'),width=8)
    assert m.expression.in_features==2
    assert m.config['axis_ids']==('A','A')


def test_exact_checkpoint_resume_and_receipt_refusal(tmp_path):
    from mora3.model.training import save_checkpoint,load_checkpoint
    torch.manual_seed(53);m=SciPlexSharedModel(('A','B'),width=8)
    opt=torch.optim.Adam(m.parameters(),lr=.001);ts=[task()]
    def step():
        opt.zero_grad();m.loss(ts).backward();opt.step()
    step();p=tmp_path/'checkpoint.pt';save_checkpoint(p,m,opt,step=1,binding=m.binding(ts))
    step();expected=copy.deepcopy(m.state_dict())
    assert load_checkpoint(p,m,opt,binding=m.binding(ts))==1
    step()
    for k,v in m.state_dict().items():
        if isinstance(v,torch.Tensor):assert torch.equal(v,expected[k])
        else:assert v==expected[k]
    with pytest.raises(ValueError):load_checkpoint(p,m,opt,binding=m.binding([task(counts=(7,8))]))
