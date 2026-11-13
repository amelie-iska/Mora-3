"""Five actual-source head bindings; construction never grants role/compute approval."""
from pathlib import Path
from hashlib import sha256
import json
from dataclasses import dataclass
from mora3.research.uniprot_full_record_route_factory import load_full_record_assets
from mora3.model.uniprot_full_record_model import full_record_stream_bound
from mora3.model.uniprot_native_objectives import _GUARD,_TARGET_BYTES_BOUND
from mora3.model.source_assembly_inputs import PINS,_pin
from mora3.data.lincx_fetch_native_rows import LinCxNativeFetchSource,M,SOURCE_SHA

@dataclass(frozen=True)
class AdditionalBudgets:
 batch_size:int
 payload_max_tokens:int
 annotation_target_bytes:int
 full_record_target_bytes:int
 def __post_init__(self):
  if any(type(x)is not int or x<1 for x in self.__dict__.values()):raise ValueError("explicit positive complete-source budgets required")

def load_additional(root,assets):
 root=Path(root).resolve()
 if assets.root!=root or assets.current_route_axes is None:raise ValueError("current actual source assets required")
 view,source,codec,availability,refs,raws=load_full_record_assets(root)
 guard=view.effective_filter["source_inverse_literal_guard_proposal"]
 if guard["sha256"]!=_GUARD or sha256((root/guard["path"]).read_bytes()).hexdigest()!=_GUARD:raise ValueError("exact inverse-literal guard bytes required")
 lincx=LinCxNativeFetchSource(root)
 image=json.loads(_pin(root,PINS["grayscale_envelope"]))
 if not image["rows"] or any(r["native_image_dtype"]!=">u2" for r in image["rows"]):raise ValueError("literal native uncropped detector dtype required")
 env=assets.current_route_axes.grayscale_source_envelope
 if (env.max_height,env.max_width)!=(image["max_height"],image["max_width"]):raise ValueError("current grayscale envelope/image proof mismatch")
 proof=dict(uniprot_source_refs={k:dict(path=v.path,sha256=v.sha256) for k,v in refs.items()},inverse_guard=guard,
  uniprot_roles=availability,full_record_axes=source.axes.binding(),grayscale_envelope=env.binding(),grayscale_detector_dtype=">u2",grayscale_original_proof=dict(path=PINS["grayscale_envelope"][0],sha256=PINS["grayscale_envelope"][1]),
  lincx=dict(path=M,sha256=SOURCE_SHA,exact_original_groups=len(lincx.rows),source_roles=None,training_authorized=False))
 return source.axes,proof

def minima(axes):return dict(payload_max_tokens=max(37000,full_record_stream_bound(axes)),annotation_target_bytes=_TARGET_BYTES_BOUND,full_record_target_bytes=max(f.max_wire_bytes for f in axes.fields))

def bind_additional(axes,budgets):
 if type(budgets)is not AdditionalBudgets:raise ValueError("typed complete-source resource bounds required")
 for key,value in minima(axes).items():
  if getattr(budgets,key)<value:raise ValueError("complete native axis exceeds "+key)
 return dict(grayscale_forward_enabled=True,grayscale_reverse_detector_dtype=">u2",lincx_fetch_source_sha256=SOURCE_SHA,
  uniprot_full_record_axes=axes,uniprot_full_record_max_batch_size=budgets.batch_size,uniprot_full_record_max_target_bytes=budgets.full_record_target_bytes,
  uniprot_native_annotation_guard_sha256=_GUARD,uniprot_native_annotation_max_batch_size=budgets.batch_size,uniprot_native_annotation_max_target_bytes=budgets.annotation_target_bytes,payload_max_tokens=budgets.payload_max_tokens)
