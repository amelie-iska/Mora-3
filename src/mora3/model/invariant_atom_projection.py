"""Split invariant atom projection; no geometry compiler or source admission.

Local correction to the reviewed Mora-3 geometry design: finite F channels and
hidden W sequence channels have distinct axes. Exactly one bias belongs to the
finite projection, giving W*F + W*W + W parameters. Caller-supplied scalar
features must already be invariant and target-free. This leaf cannot establish
those semantics, register reference frames, or qualify muP/geometry generation.
"""
import hashlib
import json
import torch
from torch import nn
from .parameterization import Axis, ParameterRole, parameter_ledger


class InvariantAtomProjection(nn.Module):
    def __init__(self, finite_axis: tuple[str, ...], width: int, *, width_identity: str):
        super().__init__()
        if (type(finite_axis) is not tuple or not finite_axis
                or any(type(x) is not str or not x for x in finite_axis)
                or len(set(finite_axis)) != len(finite_axis)):
            raise ValueError('nonempty unique ordered finite channel tuple required')
        if type(width) is not int or width <= 0:
            raise ValueError('positive integer hidden width required')
        if type(width_identity) is not str or not width_identity:
            raise ValueError('explicit hidden width identity required')
        self._finite_axis = finite_axis
        self._width = width
        self._width_identity = width_identity
        self.finite = nn.Linear(len(finite_axis), width, bias=True)
        self.sequence = nn.Linear(width, width, bias=False)

    def get_extra_state(self):
        return dict(version=1, architecture='native-invariant-atom-split-v1',
                    finite_axis=self._finite_axis, width=self._width,
                    width_identity=self._width_identity, bias_owner='finite',
                    feature_semantics='caller-declared-invariants-only')

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError('invariant atom projection configuration mismatch')

    def forward(self, finite_features, sequence_features):
        """Project complete caller-supplied [atoms,F] and [atoms,W] arrays.

        No target/reference coordinates, masks, or automatic missing-value fills.
        Validity checks precede both learned operations; no clipping/truncation.
        """
        x, s = finite_features, sequence_features
        weight = self.finite.weight
        if (not isinstance(x, torch.Tensor) or not isinstance(s, torch.Tensor)
                or x.ndim != 2 or s.ndim != 2 or x.shape[0] == 0
                or x.shape != (s.shape[0], len(self._finite_axis))
                or s.shape[1] != self._width
                or x.dtype != weight.dtype or s.dtype != weight.dtype
                or x.device != weight.device or s.device != weight.device
                or not torch.isfinite(x).all() or not torch.isfinite(s).all()):
            raise ValueError('complete finite aligned atom arrays matching model dtype/device required')
        return self.finite(x) + self.sequence(s)

    def parameter_declarations(self):
        """Strict local axes, complete aliases; declarations do not qualify scaling."""
        if (type(self) is not InvariantAtomProjection
                or set(self._modules) != {'finite', 'sequence'}
                or type(self.finite) is not nn.Linear or type(self.sequence) is not nn.Linear
                or self.finite.in_features != len(self._finite_axis)
                or self.finite.out_features != self._width
                or self.sequence.in_features != self._width
                or self.sequence.out_features != self._width
                or self.finite.bias is None or self.sequence.bias is not None):
            raise ValueError('exact native split projection operators required')
        hidden = Axis('width', self._width_identity)
        raw = json.dumps(['geometry.invariant-atom.finite.v1', self._finite_axis],
                         separators=(',', ':')).encode()
        finite = Axis('finite', hashlib.sha256(raw).hexdigest())
        declarations = {
            'finite.weight': ParameterRole('input', (hidden, finite)),
            'finite.bias': ParameterRole('vector', (hidden,)),
            'sequence.weight': ParameterRole('hidden', (hidden, hidden)),
        }
        expected = {'finite.weight': (self._width, len(self._finite_axis)),
                    'finite.bias': (self._width,),
                    'sequence.weight': (self._width, self._width)}
        named = dict(self.named_parameters(remove_duplicate=False))
        if set(named) != set(expected) or any(tuple(named[k].shape) != shape for k, shape in expected.items()):
            raise ValueError('split projection parameter ownership/shape mismatch')
        parameter_ledger(self, declarations)
        return declarations
