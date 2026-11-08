"""Explicit full-source raster/plane architecture limits; no source admission."""
from dataclasses import dataclass,asdict
import re


@dataclass(frozen=True)
class RasterSourceEnvelope:
    source_manifest_sha256:str
    max_height:int
    max_width:int
    max_instances:int | None

    def __post_init__(self):
        if (type(self.source_manifest_sha256)is not str or re.fullmatch('[0-9a-f]{64}',self.source_manifest_sha256)is None
                or any(type(x) is not int or x < 1 for x in (self.max_height, self.max_width))
                or (self.max_instances is not None and (type(self.max_instances) is not int or self.max_instances < 1))):
            raise ValueError('explicit source manifest and complete positive native raster/plane axes required')

    def binding(self):
        self.__post_init__()
        return dict(schema='Mora3.RasterSourceEnvelope/v1',**asdict(self),
            mask_axis='source-supplied-mask-layers-can-be-empty;not-biological-object-count',
            output_slots='exchangeable-prediction-planes;no-source-order-or-cardinality-output',training_admission=False)

    def binding_for_slot_policy(self,policy):
        if self.max_instances is None:
            raise ValueError('Complete source mask-layer capacity required for slot outputs')
        if policy=='source-envelope-v3':return self.binding()
        if policy!='source-layer-list-v4':raise ValueError('explicit source mask output policy required')
        value=self.binding();value['output_slots']='ordered-source-layer-prefix;count0..S-output;empty-supplied-layers-preserved'
        return value
