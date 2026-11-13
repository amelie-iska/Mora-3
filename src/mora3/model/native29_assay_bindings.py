"""Composable source-verified constructor inputs. No source role/admission change."""
from dataclasses import dataclass
from mora3.research.finite_assay_registry import load_original_assay_assets
from mora3.research.native_annotation_registry import load_original_annotation_assets
from mora3.research.reactzyme_route_factory import load_original_reactzyme_assets
from mora3.model.reactzyme_product import _stream_bound as rx_stream
from mora3.model.reactome_static_model import _stream_bound as pathway_stream
from mora3.model.biogrid_chemical_model import _entity_stream_bound as biogrid_stream
from mora3.model.string_score_model import _stream_bound as string_stream

@dataclass(frozen=True)
class HeadBudgets:
 batch_size:int
 nci_target_bytes:int
 expansion_target_bytes:int
 tox21_target_bytes:int
 reactzyme_target_bytes:int
 string_sequence_bytes:int
 reactome_attention_elements:int
 payload_max_tokens:int
 def __post_init__(self):
  if any(type(v)is not int or v<1 for v in self.__dict__.values()):raise ValueError("explicit positive integer constructor budgets required")

def load_axes(root):
 view,assays,vocab,arefs,_=load_original_assay_assets(root)
 av,annotations,availability,nrefs,_=load_original_annotation_assets(root)
 rx,codec,rrefs,_=load_original_reactzyme_assets(root)
 if view.effective_filter!=av.effective_filter or rx._view.effective_filter!=view.effective_filter:raise ValueError("exact same original guarded view required")
 axes={k:s.axes for k,s in assays.items()};axes.update({k:s.axes for k,s in annotations.items()});axes["reactzyme"]=rx.axes
 refs={prefix+":"+k:dict(path=v.path,sha256=v.sha256) for prefix,group in (("assay",arefs),("annotation",nrefs),("reactzyme",rrefs)) for k,v in group.items()}
 return axes,refs,availability

def minimum_budgets(axes):
 return dict(nci_target_bytes=axes["nci"].max_target_utf8_bytes,expansion_target_bytes=axes["expansionrx"].max_connectivity_bytes,tox21_target_bytes=axes["tox21"].max_full_smiles_bytes,reactzyme_target_bytes=axes["reactzyme"].max_source_bytes[2],string_sequence_bytes=2*axes["string"].max_sequence_utf8_bytes,reactome_attention_elements=4*(axes["reactome"].max_entities+1)**2,payload_max_tokens=max(rx_stream(axes["reactzyme"]),pathway_stream(axes["reactome"]),biogrid_stream(axes["biogrid_chemical"]),string_stream(axes["string"])))

def bind_axes(axes,budgets):
 if type(budgets)is not HeadBudgets:raise ValueError("typed explicit source-compatible budgets required")
 for k,v in minimum_budgets(axes).items():
  if getattr(budgets,k)<v:raise ValueError("complete native axis exceeds budget: "+k)
 b=budgets.batch_size
 return dict(nci_assay_source_axes=axes["nci"],nci_max_batch_size=b,nci_max_total_target_bytes=budgets.nci_target_bytes,expansionrx_source_axes=axes["expansionrx"],expansionrx_max_batch_size=b,expansionrx_max_total_target_bytes=budgets.expansion_target_bytes,tox21_current_axes=axes["tox21"],tox21_max_batch_size=b,tox21_max_total_target_bytes=budgets.tox21_target_bytes,reactzyme_source_axes=axes["reactzyme"],reactzyme_max_batch_size=b,reactzyme_max_total_target_bytes=budgets.reactzyme_target_bytes,reactome_static_axes=axes["reactome"],reactome_max_attention_elements=budgets.reactome_attention_elements,biogrid_chemical_axes=axes["biogrid_chemical"],biogrid_chemical_max_batch_size=b,string_score_axes=axes["string"],string_score_max_batch_size=b,string_score_max_batch_sequence_bytes=budgets.string_sequence_bytes,payload_max_tokens=budgets.payload_max_tokens)
