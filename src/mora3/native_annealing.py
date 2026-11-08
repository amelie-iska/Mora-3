"""Native MuAdamW annealing; bounded checkpoint cadence, no per-step file IO.

Owner admission remains separate. Crashes discard the uncheckpointed suffix;
no claim of exactly-once external metric/teacher calls for that suffix.
"""
import hashlib,io,os
from pathlib import Path
import torch
from mora3.research.primary_training import NativeMuUpdateEngine
from mora3.research.mup_campaign_publication import _sync_directory
from mora3.model.cartesian_mup import cartesian_mup_parameter_roles
from mora3.model.cartesian_teacher_trajectory import cartesian_teacher_trajectory_loss
from mora3.teacher_protocol import canonical,sha
from mora3.boltzmann_annealing import AnnealingPlan

TASK='cartesian-annealing'
class NativeAnnealing:
 def __init__(self,model,optimizer,plan,*,owner_binding_sha256,generator,checkpoint_every):
  if type(plan)is not AnnealingPlan or type(checkpoint_every)is not int or checkpoint_every<1:raise ValueError('exact plan and positive cadence required')
  if type(owner_binding_sha256)is not str or len(owner_binding_sha256)!=64 or any(c not in '0123456789abcdef' for c in owner_binding_sha256):raise ValueError('exact owner identity required; identity is not admission')
  roles=cartesian_mup_parameter_roles(model)
  if optimizer._mup_declarations!=roles:raise ValueError('exact native Cartesian muP roles required')
  if not isinstance(generator,torch.Generator) or generator.device!=next(model.parameters()).device:raise ValueError('private generator device must match native model')
  self.plan=plan;self.generator=generator;self._generator=generator;self.cadence=checkpoint_every;self.failed=False;self.saved_step=0
  self.binding=dict(schema='Mora3.NativeAnnealing/v1',owner=owner_binding_sha256,plan=plan.binding(),checkpoint_every=checkpoint_every,generator_device=str(generator.device))
  self.engine=NativeMuUpdateEngine(model,optimizer,binding_sha256=sha(canonical(self.binding)),task_cycle=(TASK,),source_cursors={TASK:self.cursor(0)})
 def _check(self):
  if self.generator is not self._generator or self.cadence!=self.binding['checkpoint_every'] or canonical(self.plan.binding())!=canonical(self.binding['plan']) or str(self.generator.device)!=self.binding['generator_device'] or self.engine.binding_sha256!=sha(canonical(self.binding)):raise ValueError('annealing runtime ownership/configuration drift')
 def cursor(self,step):return canonical(dict(schema='Mora3.AnnealingCursor/v1',plan=self.plan.sha256,completed_updates=step))
 def stage(self,steps=None):
  n=self.engine.steps if steps is None else steps
  if type(n)is not int or not 0<=n<=sum(self.plan.updates_per_stage):raise ValueError('bounded completed update count required')
  end=0
  for i,b in enumerate(self.plan.updates_per_stage):
   end+=b
   if n<end:return i
  return None
 @property
 def checkpoint_due(self):
  n=self.engine.steps;end=0;boundaries=[]
  for b in self.plan.updates_per_stage:end+=b;boundaries.append(end)
  return n>self.saved_step and (n-self.saved_step>=self.cadence or n in boundaries)
 def update(self,teacher,*,batch_size,request_id):
  self._check()
  if self.failed or self.checkpoint_due or self.stage()is None:raise ValueError('failed, complete, or checkpoint barrier active')
  n=self.engine.steps
  if self.engine.source_cursors[TASK]!=self.cursor(n):raise ValueError('exact annealing cursor required')
  target=self.plan.target(self.stage())
  try:
   result=cartesian_teacher_trajectory_loss(self.engine.model,target,teacher,batch_size=batch_size,generator=self.generator,request_id=request_id)
   if result['target_sha256']!=target.sha256:raise ValueError('temperature-bound objective mismatch')
   update=self.engine.update(TASK,result['loss'],cursor_before=self.cursor(n),cursor_after=self.cursor(n+1))
  except BaseException:
   self.failed=True # native optimizer may have changed; only verified restore recovers
   raise
  return dict(update,stage=self.stage(n),target_sha256=target.sha256,checkpoint_due=self.checkpoint_due)
 def save(self,path):
  self._check()
  if self.failed:raise ValueError('failed update cannot checkpoint')
  path=Path(path).absolute()
  if any(p.is_symlink() for p in (path,*path.parents)):raise ValueError('symlink checkpoint refused')
  # One full state copy and serialization at explicit cadence/boundary only.
  payload=dict(schema='Mora3.NativeAnnealingCheckpoint/v1',binding=self.binding,native=self.engine.checkpoint_state(),private_rng=self.generator.get_state(),next_stage=self.stage())
  with path.open('xb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
  _sync_directory(path.parent)
  digest=hashlib.sha256(path.read_bytes()).hexdigest()
  self.saved_step=self.engine.steps
  return digest # external owner must retain exact pin; no old history read
 def restore(self,path,*,expected_sha256):
  self._check()
  path=Path(path).absolute()
  if any(p.is_symlink() for p in (path,*path.parents)):raise ValueError('symlink checkpoint refused')
  raw=path.read_bytes()
  if hashlib.sha256(raw).hexdigest()!=expected_sha256:raise ValueError('checkpoint pin mismatch')
  p=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
  if type(p)is not dict or set(p)!={'schema','binding','native','private_rng','next_stage'} or p['schema']!='Mora3.NativeAnnealingCheckpoint/v1' or canonical(p['binding'])!=canonical(self.binding):raise ValueError('exact schedule/owner/device/cadence binding required')
  n=p['native']['steps']
  if p['next_stage']!=self.stage(n) or p['native']['source_cursors']!={TASK:self.cursor(n)}:raise ValueError('stage/cursor/count mismatch')
  check=torch.Generator(device=self.generator.device);check.set_state(p['private_rng'])
  # Native implementation preflights paired model/optimizer plus complete
  # Python/NumPy/Torch CPU/CUDA RNG state before restoring live owners.
  self.engine.restore_state(p['native']);self.generator.set_state(p['private_rng'])
  self.saved_step=n;self.failed=False
