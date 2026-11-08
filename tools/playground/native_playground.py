"""Load the locally pinned native foreground diagnostic for the playground."""
import base64
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import stat

import torch
from PIL import Image

import mora3
from mora3.data.cooperative_segmentation import SegmentationArtifact
from mora3.data.segmentation_image import SegmentationImageInput
from mora3.model.morphology import MorphologyForeground

MODEL = 'Mora3-native-foreground-diagnostic-v1'
PROTOCOL = 'mora3-playground-v1'
_MANIFEST_LIMIT = 65536
_WEIGHTS_LIMIT = 32 * 1024**2
_ARTIFACT_ROLES = ('synthetic-weight-fixture', 'research-diagnostic')


def digest(data):
    return sha256(data).hexdigest()


def source_identity():
    root = Path(mora3.__file__).parent
    sources = {str(path.relative_to(root)): digest(path.read_bytes())
               for path in sorted(root.rglob('*.py'))}
    return digest(json.dumps(sources, sort_keys=True, separators=(',', ':')).encode())


def _read_artifact(path, limit):
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError('Checkpoint paths must be regular and free of symbolic links.')
    if not hasattr(os, 'O_NOFOLLOW'):
        raise ValueError('The checkpoint adapter requires POSIX no-follow file access.')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= limit:
            raise ValueError('Checkpoint artifact type or size exceeds the local contract.')
        data = stream.read(limit + 1)
    if len(data) != info.st_size:
        raise ValueError('Checkpoint artifact changed while reading.')
    return data


def _configuration(manifest):
    expected = {'schema', 'model', 'config', 'checkpoint_sha256', 'source_sha256',
                'artifact_role', 'training_provenance'}
    if (type(manifest) is not dict or set(manifest) != expected
            or manifest['schema'] != 'Mora3.LocalForegroundRelease/v1'
            or manifest['model'] != MODEL):
        raise ValueError('Exact supported foreground checkpoint manifest required.')
    provenance = manifest['training_provenance']
    if (manifest['artifact_role'] not in _ARTIFACT_ROLES
            or type(provenance) is not str or not 0 < len(provenance) <= 500):
        raise ValueError('An explicit bounded diagnostic provenance is required.')
    config = manifest['config']
    if (type(config) is not dict or set(config) != {'width', 'normalization'}
            or type(config['width']) is not int or not 2 <= config['width'] <= 32
            or config['width'] % 2
            or config['normalization'] not in ('unsigned-dtype-max', 'image-minmax')):
        raise ValueError('Invalid foreground diagnostic configuration.')
    return config


def factory(checkpoint_dir):
    root = Path(checkpoint_dir).absolute()
    raw = _read_artifact(root / 'manifest.json', _MANIFEST_LIMIT)
    pin = os.environ.get('MORA3_PLAYGROUND_MANIFEST_SHA256')
    if pin is None or digest(raw) != pin:
        raise ValueError('An external exact manifest SHA256 pin is required.')
    manifest = json.loads(raw)
    config = _configuration(manifest)
    if manifest['source_sha256'] != source_identity():
        raise ValueError('Native source snapshot mismatch.')
    weights = _read_artifact(root / 'checkpoint.pt', _WEIGHTS_LIMIT)
    if digest(weights) != manifest['checkpoint_sha256']:
        raise ValueError('Checkpoint digest mismatch.')
    payload = torch.load(io.BytesIO(weights), map_location='cpu', weights_only=True)
    expected = {'schema', 'model', 'config', 'source_sha256', 'state_dict'}
    if (type(payload) is not dict or set(payload) != expected
            or payload['schema'] != 'Mora3.LocalForegroundWeights/v1'
            or payload['model'] != MODEL or payload['config'] != config
            or payload['source_sha256'] != manifest['source_sha256']):
        raise ValueError('Checkpoint payload identity mismatch.')
    with torch.random.fork_rng(devices=[]):
        model = MorphologyForeground(**config)
    reference = model.state_dict()
    state = payload['state_dict']
    if not isinstance(state, dict) or state.keys() != reference.keys():
        raise ValueError('Exact foreground state tensor keys are required.')
    for name, value in state.items():
        target = reference[name]
        if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
                or value.dtype != target.dtype or value.shape != target.shape
                or not torch.isfinite(value).all()):
            raise ValueError('Exact finite state tensors are required.')
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False)
    return Runtime(model, manifest, pin)


class Runtime:
    def __init__(self, model, manifest, pin):
        self.model = model
        self.manifest = manifest
        self.pin = pin

    def capabilities(self):
        return dict(model=MODEL, checkpoint=self.manifest['checkpoint_sha256'],
                    outputs=[dict(id='image', tasks=['predict'],
                                  notes='Single grayscale 8-bit PNG at most 64 × 64. This diagnostic has no text conditioning.')])

    def infer(self, request):
        expected = {'protocol', 'task', 'output', 'messages', 'inputs'}
        if (type(request) is not dict or set(request) != expected
                or (request['protocol'], request['task'], request['output'])
                != (PROTOCOL, 'predict', 'image')):
            raise ValueError('Only native foreground image prediction is supported.')
        if request['messages'] != []:
            raise ValueError('Native foreground prediction requires an empty conversation.')
        inputs = request['inputs']
        if type(inputs) is not list or len(inputs) != 1:
            raise ValueError('One grayscale image is required.')
        entry = inputs[0]
        if (type(entry) is not dict or set(entry) != {'modality', 'name', 'bytes', 'encoding', 'data'}
                or (entry['modality'], entry['encoding']) != ('image', 'base64')
                or type(entry['bytes']) is not int or not 1 <= entry['bytes'] <= 65536
                or type(entry['data']) is not str or len(entry['data']) > 90000):
            raise ValueError('A bounded, explicitly typed image is required.')
        raw = base64.b64decode(entry['data'], validate=True)
        if len(raw) != entry['bytes']:
            raise ValueError('Image byte count mismatch.')
        with Image.open(io.BytesIO(raw)) as image:
            if (image.format != 'PNG' or image.mode != 'L' or getattr(image, 'n_frames', 1) != 1
                    or not 1 <= image.width <= 64 or not 1 <= image.height <= 64):
                raise ValueError('Use one grayscale uint8 PNG of at most 64 by 64 pixels.')
            image.load()
            pixels, shape = image.tobytes(), (image.height, image.width)
        identity = digest(raw)
        artifact = SegmentationArtifact(
            'upload', 'image', 'source_observed', 'condition', 'caller-supplied-unverified',
            identity, (), identity, 'source-yx-no-transform-v1', channels=('brightfield',),
            spatial_shape=shape, quality_flags=('caller-content-unverified-no-training-authority',))
        task = SegmentationImageInput(artifact, 'uint8', pixels)
        with torch.inference_mode():
            logits = self.model(task)
        if logits.shape != shape or not torch.isfinite(logits).all():
            raise ValueError('Invalid native foreground prediction.')
        values = (torch.sigmoid(logits) * 255).round().to(torch.uint8).tolist()
        info = self.manifest
        provenance = (f"model={MODEL}; checkpoint={info['checkpoint_sha256']}; manifest={self.pin}; "
                      f"source={info['source_sha256']}; role={info['artifact_role']}; input={identity}; "
                      f"supplier provenance: {info['training_provenance']}")
        return dict(protocol=PROTOCOL, output='image', format='raster', pixels=values,
                    summary='Native foreground probability ×255 in the original pixel frame. Research diagnostic.',
                    provenance=provenance)
