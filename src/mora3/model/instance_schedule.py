"""Bound native overlapping-instance objective provider, without optimizer or task admission.

Explicit record-mean weighting is a caller scale choice, not cross-assay
calibration. Evaluation is differentiable; optimizer-facing access requires
native corpus admission, which is currently unavailable.
"""
import math
import torch
from .annotation_trainer import _digest
from .instance_cooperative import InstanceCooperativeModel
from .color_instance_task import ColorInstanceTask
from .color_semantic_inference import image_artifact_digest, selected_image_condition_graph


class InstanceSchedule:
    def __init__(self, model, tasks, *, weight, batch_size, context_fields=None,
                 pixel_block_size=None):
        if type(model) is not InstanceCooperativeModel:
            raise ValueError('native instance cooperative model required')
        if type(weight) not in (int, float) or not math.isfinite(weight) or weight <= 0:
            raise ValueError('explicit finite positive native instance weight required')
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError('positive exact batch size required')
        tasks=tuple(tasks)
        if not tasks or any(type(t) is not ColorInstanceTask for t in tasks):
            raise ValueError('nonempty source-bound native instance tasks required')
        if len({t.expected_source_sha256 for t in tasks}) != len(tasks):
            raise ValueError('duplicate native source bindings')
        if pixel_block_size is not None and (type(pixel_block_size) is not int or pixel_block_size < 1):
            raise ValueError('pixel block size must be a positive integer')
        self.pixel_block_size=pixel_block_size
        self.model=model
        self.tasks=tasks
        self.weight=float(weight)
        self.batch_size=batch_size
        self.context_fields=None if context_fields is None else tuple(context_fields)
        if self.context_fields is not None:
            for t in tasks:
                selected_image_condition_graph(t.source.image,pixel_sha256='0'*64,
                    context_fields=self.context_fields,
                    expected_image_sha256=image_artifact_digest(t.source.image))
        self._binding=_digest(self._policy())

    def _policy(self):
        return dict(version=1,model=self.model.get_extra_state(),
            sources=[t.source.binding_sha256 for t in self.tasks],
            expected_sources=[t.expected_source_sha256 for t in self.tasks],
            pixel_block_size=self.pixel_block_size,
            weight=self.weight,batch_size=self.batch_size,context_fields=self.context_fields,
            schedule='cyclic-source-records-v1',objective='weight-times-record-mean-native-ordered-instance-set-BCE',
            training_admission=False,pairing='independent-source-pools-no-response-join')

    def state_dict(self):
        if _digest(self._policy()) != self._binding:
            raise ValueError('native instance provider binding changed')
        return dict(self._policy(),binding=self._binding)

    def validate_state(self,state):
        if _digest(state) != _digest(self.state_dict()):
            raise ValueError('native instance provider binding mismatch')

    def select(self,step):
        self.state_dict()
        if type(step) is not int or step < 0:
            raise ValueError('nonnegative exact native instance step required')
        size=min(self.batch_size,len(self.tasks))
        return tuple(self.tasks[(step*size+i)%len(self.tasks)] for i in range(size))

    def evaluate(self,step):
        tasks=self.select(step)
        terms=[]
        for task in tasks:
            result=self.model.instance_objective(task,context_fields=self.context_fields,
                pixel_block_size=self.pixel_block_size,
                expected_image_sha256=None if self.context_fields is None else image_artifact_digest(task.source.image))
            terms.append(result['loss'])
        loss=torch.stack(terms).mean()*self.weight
        if not torch.isfinite(loss.detach()).all():
            raise ValueError('nonfinite native instance objective')
        return loss,len(tasks)

    def objective(self,step):
        for task in self.select(step):
            task.require_training()
        return self.evaluate(step)
