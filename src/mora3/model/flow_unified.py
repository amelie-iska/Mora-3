"""Unpaired native annotation/image/response plus Gaussian response flow.

One optimizer owns deduplicated shared parameters. Equal objective updates are
an explicit experimental mixture, not source pairing or a quality-selected default.
Prior unified and standalone flow checkpoints retain their original formats.
"""
import copy
import torch
from .mixed_payload_projection import exact_wire
from .annotation_trainer import _digest
from .unified_trainer import UnifiedModel, UnifiedTrainer
from .response_flow import ResponseFlowHead, ResponseFlowDenoiserV2
from .response_flow_trainer import ResponseFlowBatch, ResponseFlowTrainer
from .cooperative_training import objective_schedule
from .flow_matching import gaussian_path, velocity_loss


class FlowUnifiedModel(UnifiedModel):
    def __init__(self, native, annotation, *, flow_version=1, sigma_min=.01,
                 parameterization_policy='legacy',base_width=None,readout_mult=None,attn_mult=None,encoder_var=None):
        if type(flow_version) is not int or flow_version not in (1, 2):
            raise ValueError("flow version must be 1 or 2")
        if parameterization_policy not in ('legacy','mup-cooperative-v1'):
            raise ValueError('known explicit cooperative flow parameterization required')
        supplied=(base_width,readout_mult,attn_mult,encoder_var)
        if parameterization_policy=='legacy' and any(x is not None for x in supplied):
            raise ValueError('legacy flow owner has no muP declaration')
        if parameterization_policy!='legacy':
            native.validate_parameterization_configuration();annotation.validate_parameterization_configuration()
            n=native.response.get_extra_state();a=annotation.get_extra_state()
            actual=(n['mup']['base_width'],n['mup']['output_mult'],
                    n['native_backbone']['attn_mult'],n['native_backbone']['encoder_var'])
            if any(x is not None and (type(x) not in (int,float) or x!=y) for x,y in zip(supplied,actual)):
                raise ValueError('explicit cooperative flow numerical policy mismatch')
        super().__init__(native,annotation,parameterization_policy=parameterization_policy)
        options={} if parameterization_policy=='legacy' else dict(parameterization_policy='mup-response-flow-v1')
        self.flow = (ResponseFlowHead(native.response,**options) if flow_version == 1 else
                     ResponseFlowDenoiserV2(native.response,sigma_min=sigma_min,**options))
        self.configuration = dict(self.configuration,
            architecture=f'unpaired-native-annotation-image-response-flow-v{flow_version}',
            flow=self.flow.get_extra_state())


    def _load_from_state_dict(self,state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs):
        if self.parameterization_policy!='legacy':
            current=self.state_dict()
            for key,value in current.items():
                if not isinstance(value,torch.Tensor) and not exact_wire(state_dict.get(prefix+key),value):
                    raise ValueError('cooperative flow checkpoint configuration/policy mismatch')
        super()._load_from_state_dict(state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs)

    def validate_configuration(self):
        """Validate the flow-owned declaration without freezing derived schemas.

        Descendants append their own architecture/heads. This boundary binds only
        the response flow and its native declaration plus exact shared aliases.
        Descendant configuration and objective policies remain their owners' work.
        """
        if self._parameterization_declaration!='legacy':
            self.validate_parameterization_configuration()
            self.native.validate_parameterization_configuration()
            self.annotation.validate_parameterization_configuration()
        self.flow.validate_configuration()
        if (self.configuration.get('flow') != self.flow.get_extra_state()
                or self.configuration.get('native') != self.native.get_extra_state()):
            raise ValueError('cooperative flow configuration mismatch')
        if (self.flow.native is not self.native.response
                or self.native.image.backbone is not self.native.response.backbone
                or self.annotation.backbone is not self.native.response.backbone):
            raise ValueError('cooperative flow shared configuration mismatch')

    def get_extra_state(self):
        self.validate_configuration()
        return super().get_extra_state()

    def set_extra_state(self, state):
        self.validate_configuration()
        return super().set_extra_state(state)


class FlowUnifiedTrainer(UnifiedTrainer):
    """Cyclic native objectives then flow, with independent per-pool cursors.

    Flow reuses the exact admitted response pool, preserving gene order and all
    coordinates. Source targets bind restore only; they never enter clean context.
    Checkpoint file identity for inference remains a caller-verified file hash.
    """
    def __init__(self, model, batch, pools, *, batch_size, flow_batch_size,
                 sigma_min=.01, lr=.001, prepare_flow_tasks=False):
        if type(prepare_flow_tasks) is not bool:
            raise ValueError('prepare_flow_tasks must be an explicit bool')
        if not isinstance(model, FlowUnifiedModel):
            raise ValueError('shared flow unified model required')
        model.validate_configuration()
        super().__init__(model, batch, pools, batch_size=batch_size, lr=lr)
        self.flow_batch = ResponseFlowBatch(self.pools.get('response', ()))
        # Reuse standalone admission/path/axis/batch validation and binding.
        # Its optimizer is never stepped or retained; the union optimizer owns all.
        validation = ResponseFlowTrainer(model.flow, self.flow_batch,
            batch_size=flow_batch_size, sigma_min=sigma_min, lr=lr,
            prepare_tasks=prepare_flow_tasks)
        # Immutable raw projections only; recompute learned context and the
        # autograd graph on every objective. Original pools still bind restore
        # and all native schedules. Runtime preparation is never serialized.
        self.flow_training_tasks = validation.training_tasks
        self.flow_batch_size = flow_batch_size
        self.sigma_min = sigma_min
        self.kinds = ('annotation',) + self.native_kinds + ('flow',)
        self.binding = _digest(dict(version=1, native_binding=self.binding,
            flow_binding=validation.binding, schedule=self.kinds,
            policy='equal-active-objective-cyclic-with-flow-v1'))

    def state_dict(self):
        self.model.validate_configuration()
        return super().state_dict()

    def load_state_dict(self, state):
        self.model.validate_configuration()
        # Adam restore can alias caller tensors; keep the supplied snapshot immutable.
        super().load_state_dict(copy.deepcopy(state))

    def step(self):
        self.model.validate_configuration()
        round_index, slot = divmod(self.steps, len(self.kinds))
        kind = self.kinds[slot]
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        indices = None
        if kind == 'annotation':
            ann = self.model.annotation
            loss = ann.objective(ann(self.batch.views), self.batch.views,
                applicable=self.batch.applicability(next(self.model.parameters()).device))['total']
            count = len(self.batch.views)
        elif kind == 'flow':
            indices = tuple((round_index*self.flow_batch_size+i) % len(self.flow_batch.tasks)
                            for i in range(self.flow_batch_size))
            tasks = tuple(self.flow_training_tasks[i] for i in indices)
            p = next(self.model.parameters())
            projected = [t.project(device=p.device, dtype=p.dtype, for_training=True) for t in tasks]
            y = torch.stack([torch.log1p(v['targets']['expression']) for v in projected])
            available = torch.stack([v['target_masks']['expression'] for v in projected])
            if not available.all():
                raise ValueError('partial flow states unsupported')
            noise = torch.randn_like(y)
            time = torch.rand(len(tasks), device=p.device, dtype=p.dtype)
            state, velocity = gaussian_path(noise, y, time, sigma_min=self.sigma_min)
            loss = velocity_loss(self.model.flow(tasks, state, time), velocity, available)
            count = len(tasks)
        elif kind in self.native_kinds:
            selected, tasks = objective_schedule(self.pools,
                step=round_index*len(self.native_kinds)+slot-1, batch_size=self.batch_size)
            assert selected == kind
            loss = self.model.native.objective(kind, tasks)
            count = len(tasks)
        else:
            loss, count = self.additional_objective(kind)
        if loss is None or not torch.isfinite(loss):
            raise ValueError('nonfinite unified flow loss')
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.model.parameters()):
            raise ValueError('nonfinite unified flow gradient')
        self.optimizer.step()
        self.steps += 1
        result = dict(step=self.steps, kind=kind, examples=count, loss=float(loss.detach()))
        if indices is not None:
            result['indices'] = indices
        return result

    def additional_objective(self, kind):
        raise ValueError(f'unsupported cooperative objective: {kind}')
