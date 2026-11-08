"""Target-independent current-input metadata preflight, without source authority.

This validates declarations only. Hash strings do not authenticate coordinates,
noise independence, source rights, or admission. Gold endpoint/TS coordinates
and target-dependent noising are deliberately unsupported in this interface.
"""
from dataclasses import dataclass
from .current_frame_request_binding import CurrentFrameRequestBinding
from .geometry_frame_policy import Origin


def _sha(value):
    if type(value) is not str or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('exact SHA256 declaration required')


@dataclass(frozen=True)
class NativeCurrentMetadataCandidate:
    binding: CurrentFrameRequestBinding
    context_sha256: str
    current_xyz_sha256: str
    recipe_sha256: str
    seed: int
    native_split: str
    status: str = 'unqualified_metadata_candidate'

    def require_source_use(self):
        raise ValueError('source authentication and rights admission unavailable')

    def require_model_use(self):
        raise ValueError('native model admission unavailable')

    def require_training_use(self):
        raise ValueError('native training admission unavailable')


def preflight_native_current(*,binding,request_sha256,atom_axis_sha256,
        source_identity_sha256,group_sha256,heldout_sha256,context_sha256,
        expected_context_sha256,current_xyz_sha256,provenance,parent_target_sha256,
        recipe_sha256,seed,native_split,use,source_role,expected_native_split):
    """Validate a bounded independent-noise declaration for held-out audit only.

    No tensor, reference, neural call or data read occurs. Independent-noise
    provenance remains a caller declaration pending an authenticated constructor.
    Native observed currents and training require separate reviewed interfaces.
    """
    if type(binding) is not CurrentFrameRequestBinding:
        raise ValueError('exact current request binding required')
    binding.validate()
    source=binding.current.source
    source.__post_init__()
    if source.origin is not Origin.UNQUALIFIED_SOURCE:
        raise ValueError('native metadata requires explicit unqualified source')
    for value in (request_sha256,atom_axis_sha256,source_identity_sha256,
            group_sha256,heldout_sha256,context_sha256,expected_context_sha256,
            current_xyz_sha256,recipe_sha256):
        _sha(value)
    if (request_sha256!=binding.request_sha256
            or atom_axis_sha256!=binding.native_axis_sha256
            or source_identity_sha256!=source.original_identity_sha256
            or group_sha256!=source.group_sha256
            or heldout_sha256!=source.heldout_sha256
            or context_sha256!=expected_context_sha256):
        raise ValueError('exact request/axis/source/group/heldout/context binding required')
    if type(seed) is not int or not 0<=seed<2**63:
        raise ValueError('bounded exact noise seed required')
    if (type(native_split) is not str or type(expected_native_split) is not str
            or native_split not in ('train','val','test') or native_split!=expected_native_split):
        raise ValueError('exact source split binding required')
    if type(use) is not str or use!='evaluation':
        raise ValueError('training/source-use admission unavailable')
    if (type(provenance) is not str or provenance!='independent_noise'
            or type(source_role) is not str or source_role!='independent_noise'
            or parent_target_sha256 is not None):
        raise ValueError('gold endpoints/TS or target-dependent noising are not current predictors')
    return NativeCurrentMetadataCandidate(binding,context_sha256,current_xyz_sha256,
        recipe_sha256,seed,native_split)
