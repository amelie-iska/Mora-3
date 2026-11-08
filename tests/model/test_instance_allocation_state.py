import copy
import pytest
import torch
from test_instance_optimizer import boundary
from test_color_optimizer import fingerprint
from test_instance_schedule import make, cases

@pytest.mark.parametrize('device', ['cpu','cuda'])
def test_snapshot_detached_exact_and_read_only(tmp_path,cases,device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    b=boundary(make(tmp_path,cases,device)); before=fingerprint(b)
    assert hasattr(b,'allocation_state_dict'), 'missing strict allocation snapshot'
    state=b.allocation_state_dict(); b.validate_allocation_state(state)
    assert fingerprint(b)==before
    for name,value in b.model.state_dict().items():
        if isinstance(value,torch.Tensor):
            assert torch.equal(value,state['model'][name])
            assert value.data_ptr()!=state['model'][name].data_ptr()
    name=next(k for k,v in state['model'].items() if isinstance(v,torch.Tensor))
    state['model'][name].zero_()
    assert fingerprint(b)==before

@pytest.mark.parametrize('change',['historical','extra','steps','provider','optimizer','shape','dtype','nan','alias','configuration'])
def test_malformed_state_rejected_without_mutation(tmp_path,cases,change):
    b=boundary(make(tmp_path,cases));before=fingerprint(b)
    assert hasattr(b,'allocation_state_dict'), 'missing strict allocation snapshot'
    s=b.allocation_state_dict()
    if change=='historical':s={'model':s['model']}
    elif change=='extra':s['unknown']=True
    elif change=='steps':s['steps']=True
    elif change=='provider':s['provider']['weight']=99.
    elif change=='optimizer':s['optimizer']['state'][0]={'step':torch.tensor(1.)}
    elif change=='alias':s['parameter_aliases']=[]
    elif change=='configuration':s['configuration']['version']=99
    else:
        key=next(k for k,v in s['model'].items() if isinstance(v,torch.Tensor) and v.is_floating_point())
        if change=='shape':s['model'][key]=torch.zeros(1)
        if change=='dtype':s['model'][key]=s['model'][key].float()
        if change=='nan':s['model'][key].fill_(float('nan'))
    with pytest.raises(ValueError):b.validate_allocation_state(s)
    assert fingerprint(b)==before


@pytest.mark.parametrize('change',['alias_values','requires_grad','model_extra','device','keys'])
def test_tensor_and_alias_integrity(tmp_path,cases,change):
    b=boundary(make(tmp_path,cases));state=b.allocation_state_dict();before=fingerprint(b)
    tensors=[k for k,v in state['model'].items() if isinstance(v,torch.Tensor) and v.is_floating_point()]
    key=tensors[0]
    if change=='alias_values':
        aliases=next(a for a in state['parameter_aliases'] if len(a)>1)
        state['model'][aliases[-1]]=state['model'][aliases[-1]].clone()+1
    elif change=='requires_grad':state['model'][key].requires_grad_(True)
    elif change=='model_extra':
        key=next(k for k,v in state['model'].items() if not isinstance(v,torch.Tensor))
        state['model'][key]={'invalid':True}
    elif change=='device':state['model'][key]=state['model'][key].to('meta')
    elif change=='keys':state['model'].pop(key)
    with pytest.raises(ValueError):b.validate_allocation_state(state)
    assert fingerprint(b)==before
