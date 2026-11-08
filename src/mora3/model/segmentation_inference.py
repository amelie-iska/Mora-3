"""Checkpoint-bound native inference for source-observed DeepBacs raster tasks.

No target projection, training admission, output writes or external model loading.
Caller persists predictions to the declared immutable asset. Image-only inputs
support observed and model-derived origins with explicit artifact/pixel binding.
"""
import hashlib
import io
from pathlib import Path
import torch
from mora3.data.raster_morphology import RasterMorphologyTask
from mora3.data.segmentation_measurements import measure_predicted_instances
from mora3.data.segmentation_image import SegmentationImageInput
from .morphology import MorphologyInstance
from .instance_decoding import decode_instances


class NativeRasterSegmentation:
    def __init__(self, checkpoint, *, expected_sha256, device='cpu'):
        raw=Path(checkpoint).read_bytes()
        digest=hashlib.sha256(raw).hexdigest()
        if digest!=expected_sha256: raise ValueError('checkpoint hash mismatch')
        # Trusted project checkpoint only; hash is checked before deserialization.
        state=torch.load(io.BytesIO(raw),map_location=device,weights_only=True)
        config=state['config']
        if config['variant']!='instance': raise ValueError('native instance head required')
        self.model=MorphologyInstance(config['width'],normalization=config['normalization']).to(device)
        self.model.load_state_dict(state['model'])
        self.model.eval()
        self.identity=str(checkpoint)+'#sha256='+digest

    def infer(self, request, tasks, *, prediction_assets, expected_pixel_sha256):
        if request.head_checkpoint not in (None,self.identity):
            raise ValueError('request checkpoint does not match loaded native head')
        result={}
        images={a.artifact_id:a for a in request.artifacts}
        for key,route in request.route():
            if route!='native_segmentation':
                result[key]={'status':route}
                continue
            image=images[key]
            if request.object_type!='whole_cell' or image.channels!=('brightfield',):
                raise ValueError('unsupported native raster object/channel signature')
            t=tasks.get(key)
            if isinstance(t,SegmentationImageInput):
                if (t.artifact!=image or hashlib.sha256(t.pixels).hexdigest()!=expected_pixel_sha256.get(key)):
                    raise ValueError('image artifact/pixel binding mismatch')
                result[key]=self._predict(t,image,prediction_assets.get(key))
                continue
            if image.evidence!='source_observed':
                result[key]={'status':'unavailable:source_neutral_image_adapter'}
                continue
            t=tasks.get(key)
            if not isinstance(t,RasterMorphologyTask) or t.direction!='image_to_mask':
                raise ValueError('aligned image_to_mask raster task required')
            context=dict(image.context)
            if (image.source_id!=t.source+':'+t.revision or image.parent_id!=t.parent_group
                or image.asset.rsplit('#',1)[-1]!=t.source_id
                or image.spatial_shape!=(t.height,t.width)
                or image.pixel_size_um!=t.pixel_size_um
                or image.reference_frame!='source_pixel_yx'
                or image.representation!='source-yx-no-transform-v1'
                or context!={'species':t.species,'strain':t.strain,'perturbation':None,'expression':None}
                or hashlib.sha256(bytes.fromhex(t.image_hex)).hexdigest()!=expected_pixel_sha256.get(key)):
                raise ValueError('image source/context/pixel binding mismatch')
            asset=prediction_assets.get(key)
            if not isinstance(asset,str) or not asset.strip():
                raise ValueError('prediction asset required')
            result[key]=self._predict(t,image,asset)
        return result

    def _predict(self, task, image, asset):
        if not isinstance(asset,str) or not asset.strip():
            raise ValueError('prediction asset required')
        with torch.inference_mode():
            labels=decode_instances(self.model(task))
        measurement=measure_predicted_instances(labels,image,checkpoint=self.identity,mask_asset=asset)
        return {'status':'predicted','labels':labels,'measurement':measurement}

    def infer_and_assess(self, request, tasks, *, constraints, prediction_assets,
                         expected_pixel_sha256, quality_accepted=True):
        """Execute native inference before discrete desired-geometry assessment.

        No caller-supplied labels/measurements are accepted. Unavailable routes
        remain unevaluated without resolving pixels or desired target payloads.
        Hard geometry is prediction consistency, never biological validation.
        """
        from mora3.data.segmentation_constraints import GeometryConstraint, assess_geometry
        images={a.artifact_id:a for a in request.artifacts if a.kind=='image'}
        if set(constraints)-set(images):
            raise ValueError('unknown constraint image')
        if type(quality_accepted) is not bool:
            raise ValueError('explicit boolean quality acceptance required')
        bound={key:tuple(values) for key,values in constraints.items()}
        for key,values in bound.items():
            image=images[key]
            for constraint in values:
                if not isinstance(constraint,GeometryConstraint):
                    raise ValueError('validated geometry constraint required')
                desired=constraint.desired
                if (constraint.object_type!=request.object_type
                    or desired.links!=(('objects_of',key),)
                    or any(getattr(desired,k)!=getattr(image,k) for k in
                           ('source_id','parent_id','context','spatial_shape','reference_frame','pixel_size_um'))):
                    raise ValueError('constraint/image binding mismatch')
        result=self.infer(request,tasks,prediction_assets=prediction_assets,
                          expected_pixel_sha256=expected_pixel_sha256)
        for key,values in bound.items():
            output=result[key]
            if output['status']!='predicted':
                reason=output['status'].removeprefix('unavailable:')
                output['assessments']=tuple(dict(status='unevaluated:'+reason,
                    satisfied=None,biological_validation=False) for _ in values)
            else:
                output['assessments']=tuple(assess_geometry(output['labels'],images[key],c,
                    object_type=request.object_type,checkpoint=self.identity,
                    mask_asset=prediction_assets[key],quality_accepted=quality_accepted) for c in values)
        return result
