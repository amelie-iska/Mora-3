"""Exact deepcopy semantics with fresh reuse of immutable string-axis tuples."""
import copy
import importlib.util
import pytest
import torch


def api():
    assert importlib.util.find_spec('mora3.model.configuration_snapshot') is not None, 'missing exact configuration snapshot'
    from mora3.model.configuration_snapshot import configuration_snapshot
    return configuration_snapshot


def test_repeated_complete_axes_avoid_recursive_atomic_copy(monkeypatch):
    snapshot=api()
    axis=tuple(f'GO:{i:07}' for i in range(32768))
    shared={'axis':axis,'mutable':[1,2,3]}
    state={'left':shared,'right':shared,'other':{'axis':axis}}
    expected=copy.deepcopy(state)
    visits=[];original=copy._deepcopy_dispatch[str]
    def traced(x,memo):
        visits.append(x);return original(x,memo)
    monkeypatch.setitem(copy._deepcopy_dispatch,str,traced)
    actual=snapshot(state)
    assert actual==expected
    assert actual['left'] is actual['right'] and actual['left'] is not shared
    assert actual['left']['axis'] is axis # stdlib deepcopy also retains this tuple
    assert len(visits)<128, 'full immutable axis repeatedly traversed by deepcopy'
    actual['left']['mutable'][0]=99
    assert state['left']['mutable']==[1,2,3]
    state['left']['mutable'][0]=7
    assert snapshot(state)['left']['mutable'][0]==7 # no cross-call stale snapshot
    state['left']['axis']=axis[::-1]
    assert snapshot(state)['left']['axis']==axis[::-1]


@pytest.mark.parametrize('length',[0,1,15,16,128])
def test_standard_tuple_identity_aliases_and_cycles(length):
    snapshot=api();axis=tuple(str(i) for i in range(length))
    loop=[];node={'axis':axis,'loop':loop};loop.extend([node,loop])
    value={'a':node,'b':node,'tuple':(loop,axis),'numbers':(None,True,1,1.0,complex(2,3),b'x')}
    expected=copy.deepcopy(value);actual=snapshot(value)
    assert actual is not value and actual['a'] is actual['b']
    assert actual['a']['loop'][0] is actual['a']
    assert actual['a']['loop'][1] is actual['a']['loop']
    assert actual['tuple'][0] is actual['a']['loop']
    assert actual['a']['axis'] is expected['a']['axis'] is axis
    assert actual['numbers']==expected['numbers']


@pytest.mark.parametrize('kind',['str-subclass','tuple-subclass','dict-subclass','custom','tuple-key','set'])
def test_unrecognized_values_keep_stdlib_copy_behavior(kind):
    snapshot=api()
    class Text(str):pass
    class Tuple(tuple):pass
    class Dict(dict):pass
    class Custom:
        def __deepcopy__(self,memo):return {'memo_was_empty':not memo}
    value={'str-subclass':tuple(Text(str(i)) for i in range(20)),
           'tuple-subclass':Tuple(['a','b']), 'dict-subclass':Dict(a=[1]),
           'custom':Custom(), 'tuple-key':{('a','b'):[1]},'set':{'a','b'}}[kind]
    expected=copy.deepcopy(value);actual=snapshot(value)
    assert actual==expected and type(actual) is type(expected)
    if kind in ('str-subclass','tuple-subclass'):assert type(actual[0]) is type(expected[0])
    if kind=='dict-subclass':actual['a'].append(2);assert value['a']==[1]


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_tensor_fallback_preserves_fresh_storage_alias_and_bytes(device):
    snapshot=api()
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    x=torch.arange(12,device=device,dtype=torch.float64).reshape(3,4).t()
    state={'a':x,'b':x,'axis':tuple(str(i) for i in range(32))}
    expected=copy.deepcopy(state);actual=snapshot(state)
    assert actual['a'] is actual['b'] and actual['a'] is not x
    assert actual['a'].data_ptr()!=x.data_ptr()
    assert actual['a'].stride()==expected['a'].stride()
    assert actual['a'].dtype==x.dtype and actual['a'].device==x.device
    assert torch.equal(actual['a'],expected['a'])


def test_mutable_axis_values_are_not_reused():
    snapshot=api();axis=tuple([str(i)] for i in range(32))
    state={'axis':axis};actual=snapshot(state);expected=copy.deepcopy(state)
    assert actual==expected and actual['axis'] is not axis
    actual['axis'][0].append('changed')
    assert state['axis'][0]==['0']


def test_custom_metaclass_falls_back_without_type_equality_hooks():
    snapshot=api()
    class Meta(type):
        __hash__=type.__hash__
        def __eq__(cls,other):raise RuntimeError('unexpected metaclass comparison')
    class Custom(metaclass=Meta):
        def __deepcopy__(self,memo):return {'fallback':True,'memo_empty':not memo}
    value=Custom()
    assert snapshot(value)==copy.deepcopy(value)=={'fallback':True,'memo_empty':True}
