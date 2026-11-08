"""Source-bound ExpansionRx TASK candidates; no model/master admission.

Reuses existing native ADMET types and candidate reader. Two observational
directions share one original axis cell, never two independent experiments.
"""
from dataclasses import dataclass
from pathlib import Path
import json
import re
from .admet import (AdmetObservation,NativeEntity,NativeCell,AssayContext,EndpointDefinition,
 EndpointRegistry,DesiredProperty,desired_admet_inputs,read_admet_native_candidates,CONTEXT_KEYS,canonical,digest)

BASE="outputs/data-intake/20261101-ADMET-corpus-v1"
PLAN_SHA256="49c3e4321b804e2e3f915d1428ee65e1d31baace13463fbd5b8c45494e7115d1"
REGISTRY_REF=dict(path=BASE+"/source-endpoint-and-joint-benchmark-review-v3/endpoint-review-registry.json",bytes=21161,
 sha256="29ed338797b0b15269cd4036f43fe51b74118d76eb740f04c3e2ee9aae167a62")
GROUP_REVIEW_REF=dict(path=BASE+"/chemical-and-protocol-group-audit-v2/manifest.json",bytes=6371712,
 sha256="15352f4c58aacd413995474751b542623c87bd5cccf88068bbea530a331035f1")
DIRECTIONS=("entity_to_measurement","measurement_to_entity")
REVISION="openadmet-expansionrx-challenge-data@6b898ccc43d10d25b230fb09e22a6e30c30022b5"
EVIDENCE="source_curated_assay_readout;notindependentexperimentalreplicateorclinicalvalidation"
def _json(raw):
 if type(raw) not in (bytes,str):raise ValueError("serialized native JSON required")
 def pairs(items):
  d={}
  for k,v in items:
   if k in d:raise ValueError("duplicate native JSON key")
   d[k]=v
  return d
 def invalid(x):raise ValueError("nonfinite native JSON")
 return json.loads(raw,object_pairs_hook=pairs,parse_constant=invalid)
def _same(a,b,message):
 if canonical(a)!=canonical(b):raise ValueError(message)
def _sha(x):return type(x) is str and re.fullmatch("[0-9a-f]{64}",x) is not None
def _integer(x):return type(x) is int and x>=0
def _copy(x):return _json(canonical(x))
def _list(raw):
 x=_json(raw)
 if type(x) is not list:raise ValueError("exact native group/protocol list required")
 return x
def _gate():raise ValueError("independent component/master proof and native model endpoint qualification required")

@dataclass(frozen=True)
class ExpansionRxDisposition:
 observation_id: str
 source_family: str
 status: str
 reason: str
 allocation: str|None
 serialized_views: int
 provenance_bytes: bytes
 def __post_init__(self):
  if (not _sha(self.observation_id) or self.source_family not in ("ExpansionRx","ASAP")
      or self.status not in ("exact","censored_bound","missing","unparsed")
      or self.allocation not in (None,"train","test") or type(self.serialized_views) is not int
      or self.serialized_views!=(0 if self.allocation is None else 2) or type(self.provenance_bytes) is not bytes):
   raise ValueError("typed native axis-cell disposition/allocation required")
 def provenance(self):return _json(self.provenance_bytes)
 def require_model_use(self,purpose,*,proof=None):_gate()

def _source_identity(source,graph,joint):
 if type(source) is not AdmetObservation:raise ValueError("existing typed native candidate required")
 d=source.native();ref=d["source_ref"];family=ref["native_entity_namespace"]
 for k in ("source_family","native_entity_id","native_row_index","native_parent_protocol"):
  _same(graph[k],joint[k],"source graph/joint review identities do not match")
 if (family!=graph["source_family"] or ref["native_entity_id"]!=graph["native_entity_id"]
      or not _integer(ref["native_row_index"]) or not _integer(graph["native_row_index"])
      or ref["native_row_index"]!=graph["native_row_index"]):
  raise ValueError("exact source native entity/row/family required")
 parent=graph["native_parent_protocol"]
 expected={"source_train":"native_train","source_test":"native_test","source_unallocated":"raw_only_unallocated"}.get(parent)
 if family=="ASAP":expected={"source_train":"Train","source_test":"Test"}.get(parent)
 if expected is None or d["source_split"]["parent_protocol"]!=expected:raise ValueError("original source parent cannot be reallocated")
 if not _sha(ref["source"]["sha256"]):raise ValueError("native source hash required")
 scope=_json(graph["source_graph_scope_json"])
 if type(scope) is not dict or scope.get("native_representation")!=source.entity.as_mapping()["representation"]:
  raise ValueError("complete native enhanced stereo/source graph representation mismatch")
 for k in ("native_predictor_identity_candidate","legacy_plain_SMILES_exact_comparison_eligible"):
  if type(scope.get(k)) is not bool:raise ValueError("typed graph-scope gates required")
 if type(joint.get("candidate_train_after_joint_known_source_guards")) is not bool:
  raise ValueError("typed native source-train guard required")
 oid=digest(canonical([ref["source"]["sha256"],ref["native_row_index"],d["endpoint"]["source_endpoint"]]))
 return d,scope,oid

def _allocation(d,scope,graph,joint):
 if graph["source_family"]!="ExpansionRx":return None,"ASAP_source_units_and_tabular_transform_unresolved"
 if not scope["native_predictor_identity_candidate"] or not scope["legacy_plain_SMILES_exact_comparison_eligible"]:
  return None,"native_graph_representation_scope"
 if _list(graph["known_current_heldout_group_matches_json"]) or _list(graph["frozen_pilot_reservation_matches_json"]):
  return None,"existing_protected_group_or_frozen_reservation"
 if graph["native_parent_protocol"]=="source_train":
  if _list(joint["any_benchmark_heldout_parent_source_families_json"]):return None,"known_official_source_heldout_parent_graph_group"
  if joint["candidate_train_after_joint_known_source_guards"] is not True:return None,"joint_source_train_guard_not_qualified"
  return "train","guarded_original_source_train"
 if graph["native_parent_protocol"]=="source_test":
  if _list(graph["oldtrain_group_matches_json"]):return None,"source_test_known_prior_train_graph_group_exposure"
  return "test","known_group_screened_original_source_test_subset_not_global_blind_evaluation"
 return None,"raw_only_original_source_parent_unallocated"

def _qualified_recipe(source,graph,joint,review):
 d,scope,oid=_source_identity(source,graph,joint);allocation,reason=_allocation(d,scope,graph,joint)
 if allocation is None:return None,reason,oid
 if d["evidence_role"]!="source_curated_measurement" or d["admission"]["source_rights"] is not True:
  return None,"source_curated_measurement_or_rights_gate",oid
 if not source.cell.observation_present or not source.cell.numeric_loss_eligible:
  return None,"missing_or_unparsed_source_native_axis_cell",oid
 if review["source_family"]!="ExpansionRx" or review["source_endpoint"]!=d["endpoint"]["source_endpoint"]:
  raise ValueError("source-specific endpoint review required")
 for k in ("schema_unit_defined","native_coordinate_only","source_native_coordinate_objective_supported",
  "endpoint_definition_gate","task_units_or_declared_native_coordinate_gate"):
  if type(review.get(k)) is not bool:raise ValueError("typed endpoint coordinate/unit qualification metadata required")
 if review["conversions_performed"]!=[]:raise ValueError("no native unit conversions or transforms")
 out=_copy(d);out["original_endpoint_projection_audit"]=_copy(d["endpoint"]);out["endpoint_source_review_audit"]=_copy(review)
 out["endpoint"]=dict(endpoint_key=d["endpoint"]["endpoint_key"],source_endpoint=review["source_endpoint"],
  readout=review["source_readout"],unit=review["unit_from_official_schema"],
  unit_basis="official_source_schema" if review["schema_unit_defined"] else "unresolved",
  measurement_type="source-curated assay readout; not clinical PK or independent validation")
 for k,v in review["source_context"].items():
  if k in out["known_context"]:_same(out["known_context"][k],v,"contradictory source assay context cannot be overwritten")
  out["known_context"][k]=v
 if review["native_coordinate_only"] and review["source_native_coordinate_objective_supported"]:
  out["known_context"]["source_measurement_coordinate"]=review["source_readout"]
 out["admission"]["endpoint_definition"]=review["endpoint_definition_gate"]
 out["admission"]["task_units_or_declared_native_coordinate"]=review["task_units_or_declared_native_coordinate_gate"]
 if not out["admission"]["endpoint_definition"] or not out["admission"]["task_units_or_declared_native_coordinate"]:
  return None,"source_endpoint_or_native_coordinate_gate",oid
 out["admission"].update(exact_identity=True,split_group_review=True,
  exact_identity_scope="exact released native chemical reference graph; full material/batch/protonation context not established",
  allocation=allocation,allocation_reason=reason,global_novelty_or_study_disjointness_proven=False,
  original_source_protocol_parent_preserved=True,full_dataset_readiness=False,model_use_authorized=False)
 out["graph_scope_review_audit"]=_copy(scope);out["source_protocol_group_review_audit"]=_copy(graph)
 out["joint_source_benchmark_guard_audit"]=_copy(joint)
 out["inverse_design_scope_audit"]="source-supported observational property-to-reference-graph task; not clinical efficacy/safety or causal optimized-design validation"
 AdmetObservation.from_native(canonical(out))
 return out,reason,oid

def expansionrx_disposition(source,graph,joint,endpoint_review):
 q,reason,oid=_qualified_recipe(source,graph,joint,endpoint_review)
 d=source.native()
 return ExpansionRxDisposition(oid,d["source_ref"]["native_entity_namespace"],source.cell.status,reason,
  None if q is None else q["admission"]["allocation"],0 if q is None else 2,
  canonical(dict(source_native_observation_sha256=digest(source.native_bytes),source_native_axis_cell_multiplicity=1,
   independent_experimental_measurement_count=None,source_parent=d["source_split"],master_admitted=False,model_use_allowed=False)))

@dataclass(frozen=True)
class ExpansionRxTaskCandidate:
 observation: AdmetObservation
 direction: str
 native_payload: bytes
 provenance_bytes: bytes
 def __post_init__(self):
  if type(self.observation) is not AdmetObservation or self.direction not in DIRECTIONS or type(self.native_payload) is not bytes or type(self.provenance_bytes) is not bytes:
   raise ValueError("immutable native task direction/observation bytes required")
  p=_json(self.provenance_bytes)
  if (p.get("master_admitted") is not False or p.get("model_endpoint_qualified") is not False
      or p.get("native_observation_sha256")!=digest(self.observation.native_bytes)
      or p.get("native_payload_sha256")!=digest(self.native_payload)):
   raise ValueError("source native task pending-master/model binding mismatch")
 def provenance(self):return _json(self.provenance_bytes)
 def native_task(self):return _json(self.native_payload)
 def definition(self):return EndpointDefinition.from_native(self.observation.native()["endpoint"])
 def prediction_inputs(self):
  endpoint=self.definition().query();context=self.observation.context.as_mapping()
  if self.direction=="entity_to_measurement":
   return dict(role="source_native_ADMET_prediction_query",native_entity=self.observation.entity.as_mapping(),endpoint=endpoint,assay_context=context)
  return dict(role="source_native_ADMET_observational_property_to_entity_query",
   molecule_class=self.observation.entity.molecule_class,observed_native_measurement=self.observation.cell.constraint_fields(),
   endpoint=endpoint,assay_context=context)
 def desired_inputs(self,request):
  if (type(request) is not DesiredProperty or request.endpoint_key!=self.definition().key
      or request.assay_context.native_bytes!=self.observation.context.native_bytes):
   raise ValueError("independent explicit desired endpoint/context request required; no observed-cell autofill")
  return desired_admet_inputs(self.observation.entity.molecule_class,(request,),EndpointRegistry((self.definition(),)))
 def require_model_use(self,purpose,*,proof=None):_gate()
 def positive_loss_projection(self):_gate()
 def as_model_view(self):_gate()

@dataclass(frozen=True)
class ExpansionRxSelector:
 kind: str
 indices: tuple
 def __post_init__(self):
  if (self.kind not in ("tasks","dispositions") or type(self.indices) is not tuple or not 1<=len(self.indices)<=16
      or any(not _integer(i) for i in self.indices) or len(set(self.indices))!=len(self.indices)):
   raise ValueError("exact unique bounded task/disposition selectors required")

def expansionrx_task_candidate(row,source,graph,joint,endpoint_review,*,effective_groups,binding):
 """Pure typed recipe replay; file/source proof verification belongs to reader."""
 try:return _project(row,source,graph,joint,endpoint_review,effective_groups,binding)
 except (KeyError,TypeError,UnicodeError,OverflowError,AttributeError) as exc:
  raise ValueError("malformed ExpansionRx native task binding") from exc

def _project(row,source,graph,joint,review,effective_groups,binding):
 q,reason,oid=_qualified_recipe(source,graph,joint,review)
 if q is None:raise ValueError("source cell has no eligible task disposition: "+reason)
 direction=row["direction"]
 if direction not in DIRECTIONS:raise ValueError("explicit observational native task direction required")
 literal={k:q["cell"][k] for k in ("native_literal","status","decimal_literal","relation")}
 common=dict(endpoint=q["endpoint"],assay_context=q["known_context"],evidence_role=q["evidence_role"])
 conditions=dict(native_entity=q["entity"],**common) if direction=="entity_to_measurement" else dict(
  desired_native_measurement=literal,molecule_class=q["entity"]["molecule_class"],**common)
 targets=dict(native_measurement=literal) if direction=="entity_to_measurement" else dict(native_entity=q["entity"])
 missing=[k for k in sorted(CONTEXT_KEYS) if k not in q["known_context"]]+(["physical_unit"] if q["endpoint"]["unit_basis"]=="unresolved" else [])
 audit=dict(source_observation=q,native_model_adapter_supported=False,directions_reuse_one_source_observation=True,
  evidence_is_not_clinical_or_causal_validation=True,excluded_predictors=["source/entity/study/assay IDs","names/citations/audit",
   "hidden target native values/identity/length/hashes","forward target availability mask and unrequested endpoints"])
 payload=canonical(dict(schema="mora3.CorpusTask/v1",task="source_native_ADMET_"+direction,conditions=conditions,targets=targets,audit=audit,missing=missing))
 if type(row["native_payload"]) is not bytes or row["native_payload"]!=payload:raise ValueError("exact native payload/source/target/condition/audit byte mismatch")
 coarse=_list(graph["coarse_graph_groups_json"])
 if any(type(g) is not str or not g for g in coarse) or len(set(coarse))!=len(coarse):raise ValueError("exact unique native/coarse guards required")
 groups=sorted(set(coarse)|{"source_compound:ExpansionRx:"+graph["native_entity_id"],"source_native_axis_cell:"+oid})
 _same(effective_groups,groups,"closed effective groups differ from source-native graph/axis guards")
 _same(_json(row["group_ids_json"]),groups,"task groups differ from native graph/axis guards")
 expected=dict(task_id=digest(canonical(["ExpansionRx-source-native-ADMET-v1",oid,direction])),record_id=oid,
  source_family="ExpansionRx-source-native-ADMET-readouts",source_revision=REVISION,source_sha256=q["source_ref"]["source"]["sha256"],
  task_name="source_native_ADMET_"+direction,native_schema="mora3.CorpusTask/v1",native_payload_sha256=digest(payload),
  split=q["admission"]["allocation"],split_policy=reason,evidence_class=EVIDENCE,admitted_for_task=True,model_adapter_supported=False,direction=direction,full_axis_values=1)
 exclude={"native_payload","group_ids_json","admission_receipt_json","missing_context_json","condition_context_json"}
 _same({k:v for k,v in row.items() if k not in exclude},expected,"exact task/native source/direction/split/type envelope mismatch")
 receipt=dict(component_task_scoped=True,global_master_admission=False,source_rights=q["admission"]["source_rights"],
  native_schema_units_or_coordinate_review=REGISTRY_REF,known_group_guards=GROUP_REVIEW_REF,
  global_novelty_or_study_disjointness_proven=False,original_source_parent_protocol=source.native()["source_split"],independent_experimental_measurement_count=None)
 _same(_json(row["admission_receipt_json"]),receipt,"native component/endpoint/protocol receipt types or scope mismatch")
 _same(_json(row["missing_context_json"]),missing,"native missing assay context mismatch")
 context=dict(physical_unit_known=q["endpoint"]["unit_basis"]=="official_source_schema",native_coordinate_only=review["native_coordinate_only"],
  no_unit_conversion=True,source_missing_is_unknown_not_negative=True,original_official_test_not_train=True,model_training=False)
 _same(_json(row["condition_context_json"]),context,"native unit/coordinate/missing/split context mismatch")
 if (type(binding) is not dict or set(binding)!={"plan_sha256","task_shard_sha256","groups_sha256"}
      or binding["plan_sha256"]!=PLAN_SHA256 or any(not _sha(x) for x in binding.values())):
  raise ValueError("exact frozen plan/closed native shard/groups binding required")
 obs=AdmetObservation.from_native(canonical(q))
 provenance=dict(schema="mora3.ExpansionRxNativeTaskCandidate/v1",native_observation_sha256=digest(obs.native_bytes),
  native_payload_sha256=digest(payload),source_native_observation_sha256=digest(source.native_bytes),binding=binding,
  envelope={k:v for k,v in row.items() if k!="native_payload"},effective_groups=groups,master_admitted=False,
  model_endpoint_qualified=False,positive_loss_supported=False,causal_inverse_design_validated=False,
  independent_experimental_measurement_count=None,neural_model_token_count=None,actual_optimizer_exposure=None,
  native_source_bytes_verified=False)
 return ExpansionRxTaskCandidate(obs,direction,payload,canonical(provenance))

FIXTURE_REF=dict(path=BASE+"/ExpansionRx-closed-native-reader-fixture-v1/manifest.json",bytes=33872,
 sha256="3be3313d247f5fa21fcabaa26faee1afb89c775875c190eb7675166c3e45f3fd")
COMPONENT_REF=dict(path=BASE+"/ExpansionRx-native-readout-component-v1/manifest.json",bytes=95935,
 sha256="36df848050985bcbc06f1064574499bc8c24ee681d256400e99835ec93cc41fb")
def _typed(value):
 if type(value) is bytes:return {"native_bytes_hex":value.hex()}
 if type(value) is dict:return {k:_typed(v) for k,v in value.items()}
 if type(value) in (list,tuple):return [_typed(x) for x in value]
 return value

class ExpansionRxNativeTaskSource:
 """Twelve closed task copies/eleven cell witnesses, with model gates closed."""
 def __init__(self,*,root,manifest=FIXTURE_REF["path"],expected_manifest_sha256=FIXTURE_REF["sha256"]):
  self.root=Path(root).resolve();self.manifest=Path(manifest)
  if expected_manifest_sha256!=FIXTURE_REF["sha256"]:raise ValueError("exact closed task fixture required")
  self._snapshot()
 def _verified(self,ref,suffix):
  if (type(ref) is not dict or type(ref.get("path")) is not str or not ref["path"] or not _sha(ref.get("sha256"))
      or not _integer(ref.get("bytes")) or ref["bytes"]<1):
   raise ValueError("exact native source/reference types required")
  path=(self.root/ref["path"]).resolve()
  if not path.is_relative_to(self.root) or path.suffix.lower()!=suffix or suffix not in (".json",".parquet"):
   raise ValueError("safe task/native metadata JSON/Parquet only; no opaque/index/weight execution")
  raw=path.read_bytes()
  if len(raw)!=ref["bytes"] or digest(raw)!=ref["sha256"]:raise ValueError("source/task/component/protocol integrity mismatch")
  return raw
 def _table(self,ref):
  import pyarrow as pa
  import pyarrow.parquet as pq
  t=pq.read_table(pa.BufferReader(self._verified(ref,".parquet")),use_threads=False)
  if "rows" in ref and (not _integer(ref["rows"]) or t.num_rows!=ref["rows"]):raise ValueError("native file row allocation mismatch")
  return t
 def _row(self,ref,index):
  if not _integer(index):raise ValueError("exact closed native row ordinal required")
  table=self._table(ref)
  if index>=table.num_rows:raise ValueError("closed row outside allocation")
  return table.slice(index,1).to_pylist()[0]
 def _selected_metadata(self,ref,cases):
  import pyarrow as pa
  import pyarrow.compute as pc
  table=self._table(ref);wanted={c["source_disposition"]["source_entity"] for c in cases}
  table=table.filter(pc.is_in(table["native_entity_id"],value_set=pa.array(sorted(wanted))))
  out={}
  for row in table.to_pylist():
   key=(row["source_family"],row["native_entity_id"])
   if key in out:raise ValueError("duplicate source metadata entity allocation")
   out[key]=row
  return out
 def _snapshot(self):
  m=_json(self._verified(dict(FIXTURE_REF,path=str(self.manifest)),".json"))
  if (m["schema"]!="mora3.ClosedExpansionRxNativeTaskReaderFixture/v1"
      or m["current_master"]!="v22" or any(m[k] is not False for k in
       ("component_master_union_complete","all_model_supervision_and_optimizer_use_authorized","substantive_training_authorized","full_dataset_ready"))):
   raise ValueError("closed component candidate is not master/model readiness")
  _same(m["component"],COMPONENT_REF,"exact completed component version required")
  component=_json(self._verified(m["component"],".json"))
  if (component["status"]!="complete_task_scoped_component_pending_independent_master_union_and_native_model_handoff"
      or type(component["global_master_admitted_tasks"]) is not int or component["global_master_admitted_tasks"]!=0
      or component["model_adapter_verified"] is not False or component["model_tokens"] is not None):
   raise ValueError("component has no independently published master/native model qualification")
  _same(component["plan"],m["plan"],"completed component plan mismatch")
  _same(component["approval"],m["approval"],"exact reviewed finite execution approval required")
  plan=_json(self._verified(m["plan"],".json"))
  if m["plan"]["sha256"]!=PLAN_SHA256:raise ValueError("frozen plan mismatch")
  approval=_json(self._verified(m["approval"],".json"))
  if (approval["approved"] is not True or approval["plan_sha256"]!=PLAN_SHA256
      or approval["script_sha256"]!=plan["script"]["sha256"] or approval["master_publication"] is not False
      or approval["model_training"] is not False):
   raise ValueError("component execution approval cannot authorize master or model use")
  for mk,ck in (("native_projection","native_observation_projection"),("source_known_group_review","chemical_group_review"),("endpoint_and_joint_source_review","joint_endpoint_review")):
   _same(m[mk],plan["inputs"][ck],"exact source/protocol review input mismatch")
   _same(component["inputs"][ck],m[mk],"completed component source/protocol input mismatch")
  projection=_json(self._verified(m["native_projection"],".json"))
  group_review=_json(self._verified(m["source_known_group_review"],".json"))
  endpoint_review=_json(self._verified(m["endpoint_and_joint_source_review"],".json"))
  _same(endpoint_review["endpoint_registry"],REGISTRY_REF,"exact native unit/coordinate registry required")
  registry=_json(self._verified(endpoint_review["endpoint_registry"],".json"))
  _same(component["source_cell_and_task_index"],m["source_cell_task_index"],"all-source disposition index receipt mismatch")
  copies=self._table(m["native_source_observation_records"]).to_pylist()
  cases=m["cases"]
  if len(cases)!=11 or len(copies)!=11 or m["unique_copied_task_views"]!=12:
   raise ValueError("exact bounded native fixture allocation required")
  metadata=self._selected_metadata(group_review["source_identity_group_review"],cases)
  joint=self._selected_metadata(endpoint_review["joint_known_source_guard_rows"],cases)
  reviews={(x["source_family"],x["source_endpoint"]):x for x in registry["endpoint_reviews"]}
  shards={x["path"]:x for x in component["shards"]};source_shards={x["path"]:x for x in projection["shards"]}
  flattened=[];task_ids=set();obs_ids=set()
  for i,case in enumerate(cases):
   oid=case["source_disposition"]["observation_id"]
   if oid in obs_ids:raise ValueError("duplicate native source-axis disposition")
   obs_ids.add(oid)
   sr=case["native_source_projection_shard"]
   _same(sr,source_shards.get(sr["path"]),"fixture source shard absent from existing qualified v2 projection")
   for task in case["tasks"]:
    sh=task["source_task_shard"];gp=task["source_effective_group"]
    _same(sh,shards.get(sh["path"]),"closed task shard absent from exact completed component")
    _same(gp,component["effective_group_lookup"].get(sh["path"]),"closed task group absent from component")
    if task["component_task_scoped"] is not True or task["global_master_admission"] is not False or task["model_use"] is not False:
     raise ValueError("component eligibility is not model/master use")
    tid=task["selected_group_row"]["task_id"]
    if tid in task_ids:raise ValueError("duplicate task direction allocation")
    task_ids.add(tid);flattened.append((i,task))
  if len(flattened)!=12:raise ValueError("bounded task direction allocation mismatch")
  return dict(manifest=m,component=component,plan=plan,projection=projection,group_review=group_review,
   endpoint_review=endpoint_review,copies=copies,graphs=metadata,joint=joint,reviews=reviews,tasks=flattened,source_cache={})
 def _source_case(self,index,snapshot):
  m=snapshot["manifest"];case=m["cases"][index];disp=case["source_disposition"]
  oid=disp["observation_id"]
  if oid not in snapshot["source_cache"]:
   original=self._row(case["native_source_projection_shard"],case["native_source_projection_row"])
   copied_index=case["native_source_observation_fixture_row"]
   if not _integer(copied_index) or copied_index>=len(snapshot["copies"]):raise ValueError("exact copied native source ordinal required")
   copied=snapshot["copies"][copied_index]
   _same(_typed(copied),_typed(original),"copied full native observation differs from original v2 projection row/types")
   source,=read_admet_native_candidates(m["native_projection"]["path"],root=self.root,
    expected_manifest_sha256=m["native_projection"]["sha256"],shard_paths=(case["native_source_projection_shard"]["path"],),
    limit=1,observation_ids=(oid,))
   if source.native_bytes!=copied["native_observation"]:raise ValueError("existing candidate reader source replay differs from copy")
   snapshot["source_cache"][oid]=source
  source=snapshot["source_cache"][oid];key=(disp["source_family"],disp["source_entity"])
  graph=snapshot["graphs"][key];joint=snapshot["joint"][key];review=snapshot["reviews"][(disp["source_family"],disp["endpoint"])]
  result=expansionrx_disposition(source,graph,joint,review)
  expected=dict(observation_id=result.observation_id,source_family=result.source_family,source_entity=source.native()["source_ref"]["native_entity_id"],
   endpoint=source.native()["endpoint"]["source_endpoint"],source_parent=source.native()["source_split"]["parent_protocol"],status=result.status,
   disposition=result.reason,serialized_views=result.serialized_views)
  _same(disp,expected,"native source/protocol/status/disposition index witness mismatch")
  return source,graph,joint,review,result,case
 def _task(self,index,snapshot):
  if index>=len(snapshot["tasks"]):raise ValueError("task outside closed fixture")
  case_index,task=snapshot["tasks"][index]
  source,graph,joint,review,disposition,case=self._source_case(case_index,snapshot)
  copied=self._row(task["task_fixture"],0)
  original=self._row(task["source_task_shard"],task["source_task_shard_row"])
  _same(_typed(copied),_typed(original),"copied original task envelope bytes/types mismatch")
  if copied["direction"]!=task["direction"]:raise ValueError("source task direction mismatch")
  group=self._row(task["source_effective_group"],task["source_task_shard_row"])
  _same(group,task["selected_group_row"],"closed native effective groups mismatch")
  if group["task_id"]!=copied["task_id"]:raise ValueError("native task/group identity mismatch")
  binding=dict(plan_sha256=PLAN_SHA256,task_shard_sha256=task["source_task_shard"]["sha256"],groups_sha256=task["source_effective_group"]["sha256"])
  value=expansionrx_task_candidate(copied,source,graph,joint,review,effective_groups=_json(group["effective_groups_json"]),binding=binding)
  p=value.provenance();p.update(reader_binding=dict(fixture_sha256=FIXTURE_REF["sha256"],kind="tasks",case_index=index),
   closed_fixture_source_case=_copy(case),component_ref=snapshot["manifest"]["component"],
   task_fixture_ref=task["task_fixture"],closed_original_shard_ref=task["source_task_shard"],
   source_native_bytes_verified_by_existing_candidate_reader=True,native_source_bytes_verified=True,
   all_78980_dispositions_inherited_from_component_index=True,full_component_rescanned_by_reader=False,
   current_master_scope="v22; ExpansionRx component not in master",endpoint_registry_ref=REGISTRY_REF)
  return ExpansionRxTaskCandidate(value.observation,value.direction,value.native_payload,canonical(p))
 def _disposition(self,index,snapshot):
  if index>=len(snapshot["manifest"]["cases"]):raise ValueError("source disposition outside bounded fixture")
  source,graph,joint,review,value,case=self._source_case(index,snapshot)
  p=value.provenance();p.update(reader_binding=dict(fixture_sha256=FIXTURE_REF["sha256"],kind="dispositions",case_index=index),
   closed_fixture_source_case=_copy(case),component_ref=snapshot["manifest"]["component"],native_source_bytes_verified=True)
  return ExpansionRxDisposition(value.observation_id,value.source_family,value.status,value.reason,value.allocation,value.serialized_views,canonical(p))
 def read(self,selector):
  if type(selector) is not ExpansionRxSelector:raise ValueError("typed bounded task/disposition selector required")
  snapshot=self._snapshot();method=self._task if selector.kind=="tasks" else self._disposition
  try:return tuple(method(i,snapshot) for i in selector.indices)
  except (KeyError,TypeError,UnicodeError,OverflowError,IndexError) as exc:raise ValueError("malformed native task/source fixture") from exc
 def verify(self,value):
  if type(value) not in (ExpansionRxTaskCandidate,ExpansionRxDisposition):raise ValueError("typed native task/disposition required")
  b=value.provenance().get("reader_binding",{})
  if b.get("fixture_sha256")!=FIXTURE_REF["sha256"] or b.get("kind") not in ("tasks","dispositions") or not _integer(b.get("case_index")):
   raise ValueError("exact closed source/task selector required")
  checked,=self.read(ExpansionRxSelector(b["kind"],(b["case_index"],)))
  if checked!=value:raise ValueError("fresh source/task/censor/endpoint/type/protocol replay mismatch")
  return checked
 def source_references(self):
  s=self._snapshot();m=s["manifest"];tasks=s["tasks"]
  return dict(fixture=dict(FIXTURE_REF,path=str(self.manifest)),component=m["component"],plan=m["plan"],approval=m["approval"],
   native_projection=m["native_projection"],group_review=m["source_known_group_review"],endpoint_review=m["endpoint_and_joint_source_review"],
   endpoint_registry=s["endpoint_review"]["endpoint_registry"],group_rows=s["group_review"]["source_identity_group_review"],
   joint_rows=s["endpoint_review"]["joint_known_source_guard_rows"],native_copies=m["native_source_observation_records"],
   task0=tasks[0][1]["task_fixture"],task11=tasks[11][1]["task_fixture"],shard0=tasks[0][1]["source_task_shard"],
   groups0=tasks[0][1]["source_effective_group"])
 def gates(self):
  s=self._snapshot()
  return dict(master_admitted=False,model_endpoint_qualified=False,positive_loss_supported=False,
   whole_component_dispositions_inherited=s["component"]["source_native_axis_cell_counts"]["source_native_axis_slots_seen"],
   component_task_views_inherited=s["component"]["serialized_tasks"],actual_optimizer_exposure=None,whole_model_token_totals=None,
   full_component_rescanned=False,source_task_views_are_not_independent_experiments=True,causal_inverse_design_validated=False,
   ASAP_closed_endpoints=sorted(endpoint for (family,endpoint),review in s["reviews"].items() if family=="ASAP"
    and review["endpoint_definition_gate"] is False and review["task_units_or_declared_native_coordinate_gate"] is False),
   no_new_validation_split=True,full_dataset_ready=False)
