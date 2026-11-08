"""Legacy byte-stream parity and call-local immutable-axis serialization reuse."""
import hashlib
import json
import pytest
import torch
from mora3.model.profile_reconstruction import _digest


def reference(value):
    h=hashlib.sha256()
    def visit(x):
        if isinstance(x,torch.Tensor):
            visit((str(x.dtype),tuple(x.shape)))
            data=x.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        elif isinstance(x,dict):
            for k in sorted(x):visit(k);visit(x[k])
            return
        elif isinstance(x,(tuple,list)):
            visit(len(x))
            for v in x:visit(v)
            return
        else:data=json.dumps(x,sort_keys=True,allow_nan=False).encode()
        h.update(len(data).to_bytes(8,'big'));h.update(data)
    visit(value)
    return h.hexdigest()


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64,torch.int64,torch.bfloat16])
def test_exact_legacy_bytes_and_mutation_detection(device,dtype):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('no CUDA')
    axes=tuple('GO:'+str(i).zfill(7) for i in range(128))
    tensor=torch.arange(12,device=device).to(dtype).reshape(3,4).t()
    value={'a':axes,'b':[axes,tuple(list(axes)),(),[],None,True,1,1.0,-0.0,'é\\\"'],
           'c':tensor,'d':torch.empty(0,device=device)}
    assert _digest(value)==reference(value)
    before=_digest(value);tensor[0,0]+=1
    assert _digest(value)==reference(value) and _digest(value)!=before
    value['a']=axes[::-1]
    assert _digest(value)==reference(value)


def test_repeated_axis_serialization_is_call_local_and_reused(monkeypatch):
    axes=tuple(f'gene{i}' for i in range(1024));value=[axes]*20
    expected=reference(value);original=json.dumps;calls=[]
    def traced(x,*args,**kwargs):
        calls.append(x);return original(x,*args,**kwargs)
    monkeypatch.setattr(json,'dumps',traced)
    assert _digest(value)==expected
    assert len(calls)<=1050, 'immutable axis serialized repeatedly within one call'
    first=len(calls);assert _digest(value)==expected
    assert len(calls)>first, 'serialization cache must not survive a digest call'


def test_mutable_lists_and_str_subclasses_are_not_reused():
    class Text(str):pass
    axes=[str(i) for i in range(64)];value=[axes,axes,tuple(Text(v) for v in axes)]
    assert _digest(value)==reference(value)
    previous=_digest(value);axes[0]='changed'
    assert _digest(value)==reference(value) and _digest(value)!=previous


def test_nonfinite_refusal_is_unchanged():
    with pytest.raises(ValueError):_digest([float('nan')])
    with pytest.raises(ValueError):_digest([float('inf')])


def test_astral_unicode_respects_eight_mib_cache_budget(monkeypatch):
    # ensure_ascii JSON uses twelve bytes per astral character, not six.
    a=tuple(('a'+'😀'*20000) for _ in range(18))
    b=tuple(('b'+'😀'*20000) for _ in range(18))
    value=[a,b,a,b];expected=reference(value);original=json.dumps;calls=[]
    def traced(x,*args,**kwargs):
        calls.append(x);return original(x,*args,**kwargs)
    monkeypatch.setattr(json,'dumps',traced)
    assert _digest(value)==expected
    # First tuple fits, the second exceeds remaining budget and is read twice.
    assert len(calls)==58, 'encoded cache exceeded8MiB for astral Unicode'
