"""Bounded full-image morphology comparison, not full Mora-3 training.

Round-robin immutable source-train tasks; fixed evaluation thresholds. Baseline
objects use four-neighbour foreground components (all affinity edges enabled).
Checkpoints are trusted local artifacts; device, ordered data and config must match.
Exact continuation requires deterministic Torch algorithms and compatible CUDA
workspace settings in the caller; this module does not change runtime settings.
"""
import copy
import hashlib
import torch
from .morphology import MorphologyForeground, MorphologyInstance, foreground_objective
from .instance_affinity import affinity_objective, balanced_affinity_objective
from .maximin import raster_maximin_objective, CONVENTION as MAXIMIN_CONVENTION
from .instance_decoding import decode_instances, match_instances
from mora3.data.raster_morphology import RasterMorphologyTask
from mora3.data.livecell_raster import rasterize_instances


def targets(task, device):
    if isinstance(task,RasterMorphologyTask):
        return task.instances(device=device)
    return rasterize_instances(task.example).to(device)


class MorphologyTrainer:
    def __init__(self,tasks,*,variant='instance',width=16,learning_rate=.001,
                 normalization='image-minmax',device='cpu',affinity_mode='local-bce'):
        if not tasks or variant not in ('instance','foreground') or learning_rate<=0:
            raise ValueError('Nonempty tasks and valid training config required')
        if affinity_mode not in ('local-bce','balanced-strata','maximin-square') or (variant=='foreground' and affinity_mode!='local-bce'):
            raise ValueError('Invalid affinity objective configuration')
        for task in tasks:
            if isinstance(task,RasterMorphologyTask): task.require_training()
            else: task.example.require_training()
            if task.direction!='image_to_mask': raise ValueError('image_to_mask required')
        self.tasks=tuple(tasks)
        self.config=dict(version=1,variant=variant,width=width,learning_rate=learning_rate,
                         normalization=normalization,device=str(torch.device(device)),
                         objective='union-bce+affinity-bce' if variant=='instance' else 'union-bce',
                         schedule='ordered-round-robin-v1')
        if affinity_mode=='balanced-strata':
            self.config['objective']='union-bce+affinity-balanced-strata-v1'
        if affinity_mode=='maximin-square':
            self.config['objective']='union-bce+'+MAXIMIN_CONVENTION
        self.hashes=[hashlib.sha256(t.canonical_bytes()).hexdigest() for t in tasks]
        cls=MorphologyInstance if variant=='instance' else MorphologyForeground
        self.model=cls(width,normalization=normalization).to(device)
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=learning_rate)
        self.labels=[targets(t,device) for t in tasks]
        self.steps=0

    def step(self):
        self.model.train()
        i=self.steps%len(self.tasks)
        self.optimizer.zero_grad(set_to_none=True)
        z=self.model(self.tasks[i],for_training=True)
        loss=foreground_objective(z[0] if self.config['variant']=='instance' else z,self.labels[i])
        if self.config['variant']=='instance':
            objective = (balanced_affinity_objective if self.config['objective']=='union-bce+affinity-balanced-strata-v1'
                         else affinity_objective)
            if self.config['objective']=='union-bce+'+MAXIMIN_CONVENTION:
                objective = raster_maximin_objective
            loss=loss+objective(z[1:],self.labels[i])
        if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
        loss.backward(); self.optimizer.step(); self.steps+=1
        return float(loss.detach())

    def snapshot(self):
        return copy.deepcopy(dict(config=self.config,hashes=self.hashes,steps=self.steps,
             model=self.model.state_dict(),optimizer=self.optimizer.state_dict(),
             rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(self.config['device'])
             if self.config['device'].startswith('cuda') else None))

    def restore(self,state):
        if state['config']!=self.config or state['hashes']!=self.hashes:
            raise ValueError('Resume data/config mismatch')
        if type(state['steps']) is not int or state['steps']<0: raise ValueError('Invalid step')
        self.model.load_state_dict(state['model']); self.optimizer.load_state_dict(copy.deepcopy(state['optimizer']))
        self.steps=state['steps']; torch.set_rng_state(state['rng'].cpu())
        if state['cuda_rng'] is not None: torch.cuda.set_rng_state(state['cuda_rng'].cpu(),self.config['device'])

    @torch.no_grad()
    def evaluate(self,task):
        was_training=self.model.training
        self.model.eval()
        try:
            z=self.model(task)
            if self.config['variant']=='foreground':
                z=torch.stack((z,torch.ones_like(z),torch.ones_like(z)))
            truth=targets(task,'cpu'); union=truth.any(0); predicted=z[0].detach().cpu()>=0
            intersection=int((predicted & union).sum()); area=int((predicted | union).sum())
            return dict(foreground_iou=intersection/area if area else 1.,
                        foreground_bce=float(foreground_objective(z[0],truth.to(z.device))),
                        objects=match_instances(decode_instances(z),truth),
                        thresholds=dict(foreground=.5,affinity=.5,iou=.5))
        finally:
            self.model.train(was_training)
