"""Replayable target-independent Gaussian current, without native admission.

Recipe is pinned to the installed torch CPU generator implementation. Byte
replay across other torch versions/platforms is not assumed. Context/source
hashes remain declarations; this does not authenticate their biological roles.
"""
from dataclasses import dataclass, asdict
import hashlib
import json
import torch
from .current_frame_request_binding import CurrentFrameRequestBinding
from .native_current_preflight import NativeCurrentMetadataCandidate, preflight_native_current, _sha

RECIPE_BYTES = json.dumps(dict(version='independent-current-normal-v1',
    torch_version=str(torch.__version__),distribution='normal(0,1)',
    generator='torch.Generator(cpu)',operation='torch.randn((N,3))',
    dtype='float64',serialization='C-order little-endian float64',
    units='angstrom',max_atoms=256,all_atoms_free=True),
    sort_keys=True,separators=(',',':')).encode('ascii')


@dataclass(frozen=True)
class IndependentCurrentNoise:
    metadata: NativeCurrentMetadataCandidate
    xyz_bytes: bytes
    expected_context_sha256: str
    expected_native_split: str
    metadata_sha256: str


def _metadata_digest(metadata):
    # Snapshot the whole declaration independently of its mutable object graph.
    raw=json.dumps(asdict(metadata),sort_keys=True,separators=(',',':'),
        default=lambda value:value.value).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def construct_independent_current(*,binding,context_sha256,expected_context_sha256,
        seed,native_split,expected_native_split):
    """Generate all current XYZ without reading any reference/target coordinates.

    No masks or coordinate values are inferred from gold structures. Fixed atoms
    cannot be represented by this recipe. Only bounded metadata audit is allowed.
    """
    if type(binding) is not CurrentFrameRequestBinding:
        raise ValueError('exact request binding required')
    binding.validate()
    if not all(binding.free_atoms) or len(binding.native_axis)>256:
        raise ValueError('independent recipe requires <=256 complete free atoms')
    if type(seed) is not int or not 0<=seed<2**63:
        raise ValueError('bounded exact seed required')
    _sha(context_sha256);_sha(expected_context_sha256)
    # Preflight before allocation/RNG, then bind the computed whole-XYZ digest.
    source=binding.current.source
    declarations=dict(binding=binding,request_sha256=binding.request_sha256,
        atom_axis_sha256=binding.native_axis_sha256,
        source_identity_sha256=source.original_identity_sha256,
        group_sha256=source.group_sha256,heldout_sha256=source.heldout_sha256,
        context_sha256=context_sha256,expected_context_sha256=expected_context_sha256,
        provenance='independent_noise',parent_target_sha256=None,
        recipe_sha256=hashlib.sha256(RECIPE_BYTES).hexdigest(),seed=seed,
        native_split=native_split,expected_native_split=expected_native_split,
        use='evaluation',source_role='independent_noise')
    preflight_native_current(**declarations,current_xyz_sha256='0'*64)
    generator=torch.Generator(device='cpu').manual_seed(seed)
    xyz=torch.randn((len(binding.native_axis),3),dtype=torch.float64,
        device='cpu',generator=generator)
    raw=xyz.numpy().astype('<f8',copy=False).tobytes(order='C')
    metadata=preflight_native_current(**declarations,
        current_xyz_sha256=hashlib.sha256(raw).hexdigest())
    return IndependentCurrentNoise(metadata,raw,expected_context_sha256,expected_native_split,
        _metadata_digest(metadata))


def replay_independent_current(result,*,device='cpu',dtype=torch.float64):
    """Reconstruct and compare exact recipe bytes before returning a fresh copy."""
    if type(result) is not IndependentCurrentNoise or type(result.metadata) is not NativeCurrentMetadataCandidate:
        raise ValueError('exact noise artifact required')
    if dtype not in (torch.float32,torch.float64):
        raise ValueError('only FP32/FP64 diagnostic copies supported')
    c=result.metadata
    if result.metadata_sha256!=_metadata_digest(c):
        raise ValueError('ordered request/axis/context metadata receipt mismatch')
    replay=construct_independent_current(binding=c.binding,
        context_sha256=c.context_sha256,expected_context_sha256=result.expected_context_sha256,
        seed=c.seed,native_split=c.native_split,expected_native_split=result.expected_native_split)
    if (type(result.xyz_bytes) is not bytes or result.xyz_bytes!=replay.xyz_bytes
            or c!=replay.metadata):
        raise ValueError('noise bytes/recipe/metadata replay mismatch')
    # Reconstruct via the pinned CPU generator to avoid native-endian byte casts.
    generator=torch.Generator(device='cpu').manual_seed(c.seed)
    return torch.randn((len(c.binding.native_axis),3),dtype=torch.float64,
        device='cpu',generator=generator).to(device=device,dtype=dtype)
