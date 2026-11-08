"""Immutable, complete grayscale pixel input for the source-envelope image mode."""
from dataclasses import dataclass
import hashlib

import numpy as np
import torch

from .provenance import Provenance
from .schema import GraphRecord, Node
from mora3.model.raster_source_envelope import RasterSourceEnvelope


@dataclass(frozen=True)
class FullSourceRasterInput:
    height: int
    width: int
    image_dtype: str
    pixels: bytes
    pixel_sha256: str
    source_envelope: RasterSourceEnvelope
    provenance: Provenance

    def __post_init__(self):
        if type(self.source_envelope) is not RasterSourceEnvelope:
            raise ValueError('Explicit grayscale source envelope required')
        self.source_envelope.binding()
        if (
            type(self.height) is not int or type(self.width) is not int
            or not 1 <= self.height <= self.source_envelope.max_height
            or not 1 <= self.width <= self.source_envelope.max_width
        ):
            raise ValueError('Complete grayscale frame exceeds its source envelope')
        if type(self.image_dtype) is not str:
            raise ValueError('Native unsigned grayscale dtype required')
        dtype = np.dtype(self.image_dtype)
        if dtype.kind != 'u' or dtype.itemsize not in (1, 2):
            raise ValueError('Native uint8 or uint16 grayscale dtype required')
        if (
            type(self.pixels) is not bytes
            or len(self.pixels) != self.height * self.width * dtype.itemsize
            or hashlib.sha256(self.pixels).hexdigest() != self.pixel_sha256
            or type(self.provenance) is not Provenance
        ):
            raise ValueError('Exact complete grayscale pixels and provenance required')

    @property
    def direction(self):
        return 'image_to_mask'

    def input_image(self, *, device='cpu'):
        source = np.frombuffer(self.pixels, dtype=self.image_dtype).reshape(self.height, self.width)
        native = source.astype(source.dtype.newbyteorder('='), copy=True)
        return torch.from_numpy(native).to(device)

    def input_graph(self):
        return GraphRecord('source-envelope-grayscale', (Node('microscopy_image', ()),), (), (self.provenance,))

    def require_training(self):
        raise ValueError('Grayscale pixel input requires external source task admission')
