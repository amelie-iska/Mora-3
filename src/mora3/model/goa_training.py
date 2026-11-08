"""Opt-in fixed-pool native GOA training; source verification belongs to reader.

Sequence-only byte payload adaptation is unchanged from i109. No generation,
form inheritance, heldout eligibility or cellular joins are conferred here.
"""
import math
import torch
from .annotation_trainer import AnnotationTrainer, _digest
from .rfam_family_training import validate_adam_state
from .uniprot_go import NativeGOAHead, goa_graph_task
from .payload import PayloadEncoder
from .tokengt import TokenGT
from mora3.tokenization.shortest_path import ByteTokenizer


class NativeGOAModel(torch.nn.ModuleDict):
    def __init__(self,packets,*,width=16,heads=2,layers=2):
        super().__init__(dict(
            payload=PayloadEncoder(ByteTokenizer((),controls=('pad','node','edge')),width),
            backbone=TokenGT(width,1,width=width,heads=heads,layers=layers),
            head=NativeGOAHead(width,packets)))
        self.configuration=dict(version=1,width=width,heads=heads,layers=layers,
            attention='vanilla',identifier='single-node-eye1',
            payload='byte-positional-v1',max_tokens=8192)

    def forward(self,packets):
        p=next(self.parameters());graphs=[]
        for v in packets:
            record,task=goa_graph_task(v)
            graphs.append(self['payload'].forward_task(record,task=task,
                identifiers=torch.eye(1,device=p.device,dtype=p.dtype)))
        return self['head'](self['backbone'](graphs)[0][:,0])


class NativeGOATrainer(AnnotationTrainer):
    def __init__(self,model,packets,*,lr=.001):
        if not isinstance(model,NativeGOAModel) or not math.isfinite(lr) or lr<=0:
            raise ValueError('native model and finite positive lr required')
        self.model=model;self.packets=tuple(packets);self.steps=0
        if not self.packets:raise ValueError('nonempty source-verified pool required')
        self.optimizer=torch.optim.Adam(model.parameters(),lr=lr)
        self.binding=self._current_binding()

    def _current_binding(self):
        p=next(self.model.parameters())
        return _digest(dict(configuration=self.model.configuration,
            pool=[self.model['head']._receipt(v) for v in self.packets],
            head=self.model['head'].get_extra_state(),
            optimizer=self.optimizer.state_dict()['param_groups'],
            device=str(p.device),dtype=str(p.dtype),threads=torch.get_num_threads(),
            deterministic=torch.are_deterministic_algorithms_enabled(),
            cudnn_tf32=torch.backends.cudnn.allow_tf32,
            matmul_tf32=torch.backends.cuda.matmul.allow_tf32))

    def _check(self):
        payload=self.model['payload'];tokenizer=payload.codec.tokenizer
        if (payload.pooling!='positional' or payload.max_tokens!=8192
                or tokenizer.vocabulary!=tuple(bytes([i]) for i in range(256))
                or tokenizer.size!=259
                or payload.codec.roles!={'pad':256,'node':257,'edge':258}
                or any(tokenizer.control_id(k)!=v for k,v in payload.codec.roles.items())):
            raise ValueError('native GOA payload policy mismatch')
        if self.binding!=self._current_binding():
            raise ValueError('native GOA runtime/source policy mismatch')

    def step(self):
        self._check();self.model.train();self.optimizer.zero_grad(set_to_none=True)
        loss=self.model['head'].objective(self.model(self.packets),self.packets)
        if not torch.isfinite(loss):raise ValueError('nonfinite native GOA loss')
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.model.parameters()):
            raise ValueError('nonfinite native GOA gradient')
        self.optimizer.step();self.steps+=1
        return dict(step=self.steps,loss=loss.item())

    def state_dict(self):
        self._check()
        return super().state_dict()

    def load_state_dict(self,state):
        self._check();validate_adam_state(self.model,state)
        # Every parameter participates in this fixed full-pool objective. Missing
        # Adam history after an update would silently restart that parameter.
        ids=[i for g in state['optimizer']['param_groups'] for i in g['params']]
        slots=state['optimizer']['state']
        if state['steps']>0 and (set(slots)!=set(ids) or any(
                slot['step'].item()!=state['steps'] for slot in slots.values())):
            raise ValueError('native GOA incomplete Adam history')
        super().load_state_dict(state)

    def import_i109_state(self,state):
        """Explicit diagnostic migration, with independently pinned file hash.

        Caller verifies exact i109 file against its saved real-training receipt.
        The old script/runtime receipt must also be checked by that caller; the
        legacy checkpoint itself does not contain a complete runtime binding.
        """
        if (state.get('version')!=1 or state.get('step')!=16 or state.get('seed')!=10901
                or state.get('diagnostic_only') is not True
                or state.get('recovery_verified') is not False
                or self.model.configuration!=dict(version=1,width=16,heads=2,layers=2,
                    attention='vanilla',identifier='single-node-eye1',payload='byte-positional-v1',max_tokens=8192)):
            raise ValueError('i109 diagnostic configuration mismatch')
        cuda=state['cuda_rng'] if next(self.model.parameters()).is_cuda else None
        self.load_state_dict(dict(version=1,binding=self.binding,steps=16,
            model=state['model'],optimizer=state['optimizer'],
            cpu_rng=state['cpu_rng'],cuda_rng=cuda))
