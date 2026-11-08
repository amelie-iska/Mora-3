"""Shared native backbone identity, exact incidence and objective gradients."""
import copy
import pytest
import torch
from test_physical_trainer import view, batch
from test_annotation_vocabulary import admitted
from mora3.model.annotation_vocabulary import fit_annotation_vocabulary
from mora3.model.cooperative_training import CooperativeModel
from mora3.model.physical_trainer import SharedPhysicalModel


def make(device='cpu'):
    native=CooperativeModel(('gene-a','gene-b'),width=4).to(device)
    model=SharedPhysicalModel(fit_annotation_vocabulary([admitted()],learned_budget=2,max_length=2),
        ('GO:0000001',),('1.1.1.1',),parameter_axis=('mass_spectrometry_source_mass',),
        method_axis=('MALDI','ESI'),width=4,heads=2,layers=2,backbone=native.response.backbone).to(device)
    return native,model


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_shared_exact_identifier_and_native_physical_gradient(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.manual_seed(5001)
    native,m=make(device)
    assert m.backbone is native.image.backbone is native.response.backbone
    from mora3.model.uniprot_go import annotation_graph_task
    record,task=annotation_graph_task(view())
    weight=m.backbone.feature.weight
    graph=m.payload.forward_task(record,task=task,identifiers=torch.tensor([[1.,0.]],device=device))
    reference=m.backbone([graph])[0][:,0]
    z=m([view()]);torch.testing.assert_close(z['GO'],m.go(reference),rtol=0,atol=0)
    b=batch();loss=m.objective(z,b.views,applicable=b.applicability(device))['total']
    loss.backward()
    assert native.response.backbone.layers[0].attention.qkv.weight.grad.abs().sum()>0
    # A union optimizer deduplicates actual shared parameter objects.
    params=list(torch.nn.ModuleList([native,m]).parameters())
    assert len(params)==len({id(p) for p in params})
    before=weight.detach().clone();torch.optim.Adam(params,lr=.001).step()
    assert not torch.equal(before,weight)
    clone_native,clone=make(device);clone.load_state_dict(copy.deepcopy(m.state_dict()))
    for key,value in m([view()])['physical'].items():
        assert torch.equal(value,clone([view()])['physical'][key])


def test_incompatible_shared_backbone_rejected():
    from mora3.model.tokengt import TokenGT
    vocab=fit_annotation_vocabulary([admitted()],learned_budget=2,max_length=2)
    for backbone in (TokenGT(8,2,width=4,heads=2,layers=2),TokenGT(4,2,width=4,heads=1,layers=1)):
        with pytest.raises(ValueError,match='backbone'):
            SharedPhysicalModel(vocab,('GO:0000001',),('1.1.1.1',),parameter_axis=('mass_spectrometry_source_mass',),method_axis=('MALDI',),width=4,heads=2,layers=2,backbone=backbone)
