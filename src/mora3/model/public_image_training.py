"""Native supervised public-image binding; no inferred cross-source pairing.

Requires externally admitted immutable tasks. Uses existing shared image losses,
with image-only inputs and explicit ordered source/admission/objective binding.
This component grants no rights or scientific admission to source manifests.
"""
import copy
import math
import torch
from mora3.data.public_image_admission import PublicImageAdmission
from .cooperative_training import CooperativeModel


class PublicImageTrainer:
    def __init__(self, model, tasks, *, learning_rate=.001):
        if (not isinstance(model, CooperativeModel) or not tasks
                or not math.isfinite(learning_rate) or learning_rate <= 0
                or any(not isinstance(t, PublicImageAdmission) for t in tasks)):
            raise ValueError('native shared model, admitted public tasks and finite positive rate required')
        self.tasks=tuple(tasks)
        for task in self.tasks: task.require_training()
        self.model=model
        parameter=next(model.parameters())
        self.binding=copy.deepcopy(dict(
            trainer='public-image-native-supervised-v1',
            native=model.binding({'image':self.tasks},batch_size=1),
            device=str(parameter.device),dtype=str(parameter.dtype),
            deterministic=torch.are_deterministic_algorithms_enabled(),
            optimizer='Adam',learning_rate=learning_rate))
        self.optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
        self.steps=0

    def step(self):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        loss=self.model.objective('image',[self.tasks[self.steps%len(self.tasks)]])
        if not torch.isfinite(loss): raise ValueError('nonfinite public image objective')
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.model.parameters()):
            raise ValueError('nonfinite public image gradient')
        self.optimizer.step(); self.steps+=1
        return float(loss.detach())

    def snapshot(self):
        device=next(self.model.parameters()).device
        return copy.deepcopy(dict(binding=self.binding,steps=self.steps,
            model=self.model.state_dict(),optimizer=self.optimizer.state_dict(),
            cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(device)
            if device.type=='cuda' else None))

    def restore(self,state):
        if state['binding']!=self.binding:
            raise ValueError('public image trainer binding mismatch')
        if type(state['steps']) is not int or state['steps']<0:
            raise ValueError('invalid public image trainer step')
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(copy.deepcopy(state['optimizer']))
        self.steps=state['steps']; torch.set_rng_state(state['cpu_rng'].cpu())
        if state['cuda_rng'] is not None:
            torch.cuda.set_rng_state(state['cuda_rng'].cpu(),next(self.model.parameters()).device)
