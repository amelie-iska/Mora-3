from pathlib import Path
import torch,pytest
from test_cartesian_gfn_policy import policy,target
from test_cartesian_teacher_trajectory import _client,_exchange
from mora3.teacher_protocol import SocketEndpoint
from mora3.model.cartesian_mup import cartesian_mup_parameter_roles,cartesian_mup_width_family
from mora3.model.mu_optimizer import MuAdamW
from mora3.native_annealing import NativeAnnealing
from mora3.boltzmann_annealing import AnnealingPlan

def make():
 models=tuple(policy(w) for w in (8,12,16));family=cartesian_mup_width_family(*models);model=models[-1]
 opt=MuAdamW(model,family,cartesian_mup_parameter_roles(model),base_lr=.0001,weight_decay=0.,decay_aliases=())
 return NativeAnnealing(model,opt,AnnealingPlan(target(),(2,2,2,2,2)),owner_binding_sha256='e'*64,generator=torch.Generator().manual_seed(41),checkpoint_every=2)

def test_native_updates_cadence_and_exact_resume(tmp_path,monkeypatch):
 calls,exchange=_exchange(target());monkeypatch.setattr(SocketEndpoint,'exchange',exchange)
 a=make();client=_client(target());before=next(a.engine.model.parameters()).detach().clone()
 # No filesystem reads/writes allowed anywhere on the normal update path.
 with monkeypatch.context() as m:
  m.setattr(Path,'read_bytes',lambda *a:(_ for _ in ()).throw(AssertionError('history scan')))
  m.setattr(Path,'open',lambda *a,**kw:(_ for _ in ()).throw(AssertionError('per-step write')))
  first=a.update(client,batch_size=1,request_id='fixture-1');second=a.update(client,batch_size=1,request_id='fixture-2')
 assert first['stage']==second['stage']==0 and second['checkpoint_due']
 assert not torch.equal(before,next(a.engine.model.parameters()))
 with pytest.raises(ValueError,match='barrier'):a.update(client,batch_size=1,request_id='blocked')
 path=tmp_path/'boundary.pt';pin=a.save(path)
 assert a.stage()==1
 expected=a.update(client,batch_size=1,request_id='next')
 expected_weights={k:v.detach().clone() for k,v in a.engine.model.named_parameters()}
 b=make();b.restore(path,expected_sha256=pin)
 actual=b.update(client,batch_size=1,request_id='next')
 assert expected['loss']==actual['loss'] and expected['target_sha256']==actual['target_sha256']
 assert all(torch.equal(v,dict(b.engine.model.named_parameters())[k]) for k,v in expected_weights.items())
 assert len(list(tmp_path.iterdir()))==1 and len(calls)==4

def test_restore_mismatch_and_interruption(tmp_path,monkeypatch):
 a=make();p=tmp_path/'initial.pt';pin=a.save(p)
 with pytest.raises(ValueError,match='pin'):a.restore(p,expected_sha256='0'*64)
 def fail(*args,**kw):raise RuntimeError('interrupted RPC')
 monkeypatch.setattr(SocketEndpoint,'exchange',fail)
 with pytest.raises(RuntimeError):a.update(_client(target()),batch_size=1,request_id='fail')
 with pytest.raises(ValueError,match='failed'):a.update(_client(target()),batch_size=1,request_id='retry')
 a.restore(p,expected_sha256=pin);assert not a.failed and a.engine.steps==0
 with pytest.raises(FileExistsError):a.save(p)

def test_stage_only_from_count_and_schedule():
 a=make();assert [a.stage(n) for n in range(11)]==[0,0,1,1,2,2,3,3,4,4,None]
 with pytest.raises(ValueError):a.stage(11)
