"""Current input declarations for bounded diagnostics, never source admission.

Atom identity/order and whole-XYZ availability are caller declarations. This
module does not authenticate coordinates, invariant features or free-region
provenance. It neither reads references nor derives masks from gold geometry.
"""
from dataclasses import dataclass
from . import spatial_design_constraints as q
from . import geometry_frame_policy as p


@dataclass(frozen=True)
class CurrentFrameRequestBinding:
    request: q.DesignRequest
    compiled: q.CompiledDesignPlan
    current: p.CurrentFrame
    role: q.Role
    native_axis: tuple[q.AtomKey, ...]
    native_axis_sha256: str
    xyz_available: tuple[bool, ...]
    free_atoms: tuple[bool, ...]

    @property
    def request_sha256(self):
        return self.compiled.request_sha256

    @property
    def synthetic_diagnostic(self):
        return self.current.source.origin is p.Origin.SYNTHETIC_LITERAL

    @property
    def native_model_authorized(self):
        return False

    def validate(self):
        fresh=q.compile_design_request(self.request)
        if type(self.compiled) is not q.CompiledDesignPlan or self.compiled.canonical_bytes()!=fresh.canonical_bytes():
            raise ValueError('compiled request mismatch')
        if type(self.current) is not p.CurrentFrame:
            raise ValueError('exact current-frame declaration required')
        self.current.__post_init__()
        if self.current.source.purpose is not p.Purpose.CONDITION_INPUT:
            raise ValueError('evaluation or withheld current geometry forbidden')
        if type(self.role) is not q.Role or self.role not in (
                q.Role.USER_SUPPLIED,q.Role.SOURCE_OBSERVED,q.Role.GENERATED_INTERMEDIATE):
            raise ValueError('explicit current-input role required; desired is not current state')
        item=next((x for x in self.request.inputs if x.key==self.current.input_key),None)
        if (item is None or not item.available or item.role is not self.role
                or item.kind not in (q.InputKind.TS_GEOMETRY,q.InputKind.ENZYME_STRUCTURE,q.InputKind.CONTEXT)):
            raise ValueError('current role/key/kind must bind available predictor input')
        digest=p.axis_digest(self.native_axis)
        if (type(self.native_axis_sha256) is not str or digest!=self.native_axis_sha256
                or not 1<=len(self.native_axis)<=min(self.request.budget.max_atoms,p.MAX_AXIS_ENTRIES)
                or any(a.entity!=item.key for a in self.native_axis)):
            raise ValueError('complete ordered current-input atom axis/hash/budget required')
        for mask in (self.xyz_available,self.free_atoms):
            if type(mask) is not tuple or len(mask)!=len(self.native_axis) or any(type(v) is not bool for v in mask):
                raise ValueError('whole-XYZ and free masks require exact per-atom booleans')
        if not all(self.xyz_available):
            raise ValueError('complete current XYZ required; no cropping or missing-value fill')

    def require_synthetic_diagnostic(self):
        self.validate()
        if not self.synthetic_diagnostic:
            raise ValueError('only synthetic literal declarations may drive diagnostics')

    def require_source_use(self):
        raise ValueError('no actual-source authority')

    def require_model_use(self):
        raise ValueError('no native model authority')

    def require_training_use(self):
        raise ValueError('no training authority')


def bind_current_frame_request(*,request,compiled,current,role,native_axis,
                               native_axis_sha256,xyz_available,free_atoms):
    result=CurrentFrameRequestBinding(request,compiled,current,role,native_axis,
        native_axis_sha256,xyz_available,free_atoms)
    result.validate()
    return result
