"""Explicit diagnostic desired-response objective on the native shared model.

No optimizer/scheduler/admission is supplied. Externally requested intervals plus
an observed response do not establish eligible conditional supervision. Historical
models/checkpoints keep their existing architecture and trainer contracts.
"""
import copy
from torch import nn
from .annotation_trainer import _digest
from .flow_unified import FlowUnifiedModel
from .response_desired_flow import DesiredResponseFlow
from .response_desired_objective import desired_response_cfm_loss


class DesiredResponseCooperative(nn.Module):
    """Own the parent and new desired head once, retaining one native backbone."""
    def __init__(self, parent, *, sigma_min=.01):
        super().__init__()
        if not isinstance(parent, FlowUnifiedModel):
            raise ValueError('native shared flow cooperative parent required')
        native = parent.native.response
        if not (native.backbone is parent.native.image.backbone
                is parent.annotation.backbone) or parent.flow.native is not native:
            raise ValueError('one shared native response/image/annotation backbone required')
        desired = DesiredResponseFlow(native, sigma_min=sigma_min)
        p = native.edge
        desired.to(device=p.device, dtype=p.dtype)
        self.parent = parent
        self.desired = desired
        # Memo-preserved identity references, not duplicate module registrations.
        self._owners = (native, native.backbone)
        self.configuration = dict(version=1,
            architecture='native-shared-desired-response-objective-v1',
            parent=copy.deepcopy(parent.get_extra_state()),
            desired=desired.get_extra_state(),
            path=copy.deepcopy(desired.flow.config),
            training_admission=False,
            request_role='externally-supplied-desired-not-observed',
            supervision_role='source-observed-response-only',
            pairing='no-automatic-desired-response-association')

    def get_extra_state(self):
        return copy.deepcopy(self.configuration)

    def set_extra_state(self, state):
        if state != self.configuration:
            raise ValueError('desired cooperative configuration mismatch')

    def desired_objective(self, batch, *, requests, noise, time):
        """Expose differentiable loss only; caller owns mode/noise/time policy.

        This checks live ownership/configuration, not external source authenticity
        or independence/appropriateness of the caller's desired requests. Returned
        hashes bind configuration, not weights, admission or checkpoint validity.
        """
        native, backbone = self._owners
        if (self.parent.native.response is not native
                or self.desired.native is not native
                or self.parent.flow.native is not native
                or native.backbone is not backbone
                or self.parent.native.image.backbone is not backbone
                or self.parent.annotation.backbone is not backbone):
            raise ValueError('desired cooperative shared ownership changed')
        if (self.parent.get_extra_state() != self.configuration['parent']
                or self.desired.get_extra_state() != self.configuration['desired']
                or self.desired.flow.get_extra_state() != self.configuration['path']):
            raise ValueError('desired cooperative live configuration changed')
        result = desired_response_cfm_loss(self.desired, batch, requests=requests,
            noise=noise, time=time, sigma_min=self.configuration['path']['sigma_min'])
        configuration = dict(version=1,
            objective='native-shared-desired-response-cfm-v1',
            model=self.get_extra_state(), desired_objective=result['configuration'],
            training_admission=False, optimizer='none',
            supervision_eligibility='not-established-by-this-interface')
        return dict(loss=result['loss'], configuration=configuration,
                    binding=_digest(configuration))
