"""Native chemical-reaction/AND-OR route CANDIDATES, never task admission.

Raw native serializations and source roles are retained. Pure lexical map
exclusion is not chemical validation; extended syntax refuses rather than loses
stereo. Runtime source fixture consumption is JSON only, no protobuf execution.
"""
from dataclasses import asdict,dataclass
import hashlib
import json
from pathlib import Path
import re

BASE="outputs/data-sources/20261101-retrosynthesis-source-v1"
FIXTURE_REF=dict(path=BASE+"/curated/bounded-retrosynthesis-native-fixture-v1/handoff-bounded-retrosynthesis-native-fixture-v1.json",
 bytes=4348,sha256="5ce3309b4e2c1885ba778bf9a5d2740eb2c53f1ff3735fe06d6c5795875499d4")
FILE_PINS={"ORD-three-original-semantic-records.json":"20c1a68fcfd5a33a33915fe4b8ae854ff2a56f9d2c5f44804cc48e88ed448278",
 "PaRoutes-two-original-reference-route-trees-EVAL-ONLY.json":"4b9ae9872beb7292732bfb08c818c6b77aee2345051498493838cbd85d904d22",
 "bounded-original-chemical-string-QC.json":"c5c854c5b82822d8c2ba93f89ce64fddf1d8d090d74b88c6b68481b021173dc6",
 "PaRoutes15-originals-available-manifest-v1.json":"4fdcc21887e5574cc98395ff05b17005d7dc9a8ebda00a43dda774ea126433d3"}
def _canonical(x):return json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
def _sha(raw):return hashlib.sha256(raw).hexdigest()
def _json(raw):
 def pairs(items):
  d={}
  for k,v in items:
   if k in d:raise ValueError("duplicate source JSON key")
   d[k]=v
  return d
 def invalid(x):raise ValueError("nonfinite source JSON literal")
 return json.loads(raw,object_pairs_hook=pairs,parse_constant=invalid)
def _json_bytes(raw):
 if type(raw) is not bytes:raise ValueError("immutable native JSON bytes required")
 value=_json(raw)
 if _canonical(value)!=raw:raise ValueError("canonical typed native JSON required")
 return value
def _hash_dataclass(value):
 def encode(v):
  if type(v) is bytes:return {"bytes_hex":v.hex()}
  if type(v) is dict:return {k:encode(x) for k,x in v.items()}
  if type(v) in (tuple,list):return [encode(x) for x in v]
  return v
 return _sha(_canonical(encode(asdict(value))))
def _model_gate(purpose,*,role,admission=None):
 if purpose=="training" and role.startswith("PaRoutes_"):
  raise ValueError("official PaRoutes evaluation/exposure reference is never training")
 raise ValueError("native candidate has no constructor model task admission or cross-corpus heldout proof")
def _exact_text(x):return type(x) is str and bool(x)


@dataclass(frozen=True)
class ChemicalIdentity:
 native_text: str
 syntax: str = "SMILES"
 molecule_class: str = "small_molecule"
 map_origin: str = "source_reaction_derived_or_unknown"
 derived_reference_text: str|None = None
 derived_evidence_bytes: bytes|None = None
 def __post_init__(self):
  if (not _exact_text(self.native_text) or self.syntax not in ("SMILES","CXSMILES","MOLBLOCK","INCHI")
      or self.molecule_class not in ("small_molecule","peptide")
      or self.map_origin not in ("source_reaction_derived_or_unknown","absent","source_independent_known")):
   raise ValueError("literal native chemical representation/role required; amino-acid sequence is not chemical synthesis")
  if (self.derived_reference_text is None)!=(self.derived_evidence_bytes is None):
   raise ValueError("derived reference graph requires its separate source evidence")
  if self.derived_evidence_bytes is not None:
   if not _exact_text(self.derived_reference_text) or type(_json_bytes(self.derived_evidence_bytes)) is not dict:
    raise ValueError("explicit source-computed reference identity evidence required")
 def native_bytes(self):return self.native_text.encode("utf8")
 def product_input_text(self):
  if self.syntax!="SMILES" or "|" in self.native_text or self.native_text!=self.native_text.strip():
   raise ValueError("extended/unsupported syntax requires a separately verified stereo-safe product projection")
  if self.native_text.count("[")!=self.native_text.count("]") or ">" in self.native_text:
   raise ValueError("unsupported native molecular serialization; no repair")
  def bracket(match):
   atom=match.group(1)
   return "["+re.sub(r":[0-9]+$","",atom)+"]"
  return re.sub(r"\[([^\[\]]+)\]",bracket,self.native_text)
 def native_multiset_identity(self):
  return dict(scope="native_literal_map_excluded_serialization_not_general_chemical_equivalence",
    syntax=self.syntax,value=self.product_input_text(),molecule_class=self.molecule_class)


@dataclass(frozen=True)
class NativeAmount:
 value: str|None
 unit: str|None
 presence: str
 native_json_bytes: bytes|None = None
 def __post_init__(self):
  if (self.presence not in ("present","absent","unresolved")
      or (self.value is not None and not _exact_text(self.value))
      or (self.unit is not None and not _exact_text(self.unit))
      or (self.presence=="present" and self.value is None)
      or (self.presence=="absent" and (self.value is not None or self.unit is not None))):
   raise ValueError("exact native amount literal/unit/presence required; no float coercion/default")
  if self.native_json_bytes is not None:_json_bytes(self.native_json_bytes)


@dataclass(frozen=True)
class ReactionComponent:
 identity: ChemicalIdentity
 role: str
 amount: NativeAmount
 source_role: str|None = None
 native_component_bytes: bytes|None = None
 def __post_init__(self):
  if (type(self.identity) is not ChemicalIdentity or type(self.amount) is not NativeAmount
      or self.role not in ("product","precursor","reagent","solvent","catalyst","unknown","auxiliary")):
   raise ValueError("explicit native product/precursor/auxiliary role required")
  if self.source_role is not None and not _exact_text(self.source_role):raise ValueError("literal source role required")
  if self.native_component_bytes is not None:_json_bytes(self.native_component_bytes)


@dataclass(frozen=True)
class ReactionCondition:
 name: str
 value: str|None
 unit: str|None
 role: str
 availability: str
 def __post_init__(self):
  if (self.name not in ("temperature","time","pressure","solvent","catalyst","pH","light","electrochemical")
      or self.role not in ("known_input","source_audit","desired","outcome")
      or (self.value is not None and type(self.value) is not str)
      or (self.unit is not None and type(self.unit) is not str)
      or self.availability not in ("explicit_caller_known","not_inference_known","caller_desired","unavailable")):
   raise ValueError("typed explicit context/role required; no gold/precursor/outcome inputs")
  if self.role=="known_input" and self.availability!="explicit_caller_known":
   raise ValueError("known reaction context must be independently explicit")
  if self.role=="desired" and self.availability!="caller_desired":raise ValueError("explicit desired role required")
 def as_mapping(self):return asdict(self)
def desired_reaction_inputs(conditions):
 if type(conditions) is not tuple or any(type(c) is not ReactionCondition or c.role!="desired" for c in conditions):
  raise ValueError("explicit typed desired conditions only")
 return dict(evidence_role="caller_desired_conditions",conditions=[c.as_mapping() for c in conditions])


@dataclass(frozen=True)
class PeptideSynthesisContext:
 assembly_mode: str
 protecting_groups: tuple
 modifications: tuple
 evidence_role: str
 def __post_init__(self):
  if (self.assembly_mode not in ("SPPS","solution_phase","fragment_coupling","ligation","cyclization","disulfide_formation","unknown")
      or type(self.protecting_groups) is not tuple or type(self.modifications) is not tuple
      or any(not _exact_text(v) for v in self.protecting_groups+self.modifications)
      or self.evidence_role not in ("synthetic_test","source_reported")):
   raise ValueError("source-supported chemical peptide assembly/protection roles required, not inverse folding")
 @property
 def is_inverse_folding(self):return False
def peptide_native_coverage():
 return dict(eligible_native_records=0,ordered_native_trace_available=False,
  gate="no eligible ordered chemical peptide route supplied; accounting workbooks are not route chronology")


@dataclass(frozen=True)
class ReactionCandidate:
 products: tuple
 precursors: tuple
 native_serialization: bytes
 audit_bytes: bytes
 auxiliaries: tuple = ()
 source_role: str = "synthetic_test"
 source_locator_bytes: bytes = b"{}"
 def __post_init__(self):
  if (any(type(x) is not tuple for x in (self.products,self.precursors,self.auxiliaries))
      or not self.products or any(type(x) is not ReactionComponent or x.role!="product" for x in self.products)
      or any(type(x) is not ReactionComponent or x.role!="precursor" for x in self.precursors)
      or any(type(x) is not ReactionComponent or x.role in ("product","precursor") for x in self.auxiliaries)
      or type(self.native_serialization) is not bytes or not self.native_serialization
      or self.source_role not in ("synthetic_test","ORD_native_candidate")):
   raise ValueError("immutable exact native reaction product/precursor/auxiliary roles required")
  _json_bytes(self.audit_bytes);_json_bytes(self.source_locator_bytes)
 def prediction_inputs(self,*,known_conditions=(),product_indices=None):
  if product_indices is None:
   if len(self.products)!=1:raise ValueError("explicit known product selection required for multiple products")
   product_indices=(0,)
  if (type(product_indices) is not tuple or not product_indices or len(set(product_indices))!=len(product_indices)
      or any(type(i) is not int or not 0<=i<len(self.products) for i in product_indices)
      or type(known_conditions) is not tuple
      or any(type(c) is not ReactionCondition or c.role!="known_input" or c.availability!="explicit_caller_known" for c in known_conditions)):
   raise ValueError("exact known product/context allowlist required")
  out=dict(query_role="product_conditioned_chemical_retrosynthesis",products=[
   dict(syntax="SMILES",value=self.products[i].identity.product_input_text()) for i in product_indices])
  if known_conditions:out["known_conditions"]=[c.as_mapping() for c in known_conditions]
  return out
 def precursor_multiset_key(self):
  entries=[dict(identity=c.identity.native_multiset_identity(),amount=asdict(c.amount),role=c.role) for c in self.precursors]
  # Native component metadata is not a chemical quantity normalization.
  for e in entries:
   raw=e["amount"]["native_json_bytes"]
   e["amount"]["native_json_bytes"]=None if raw is None else raw.hex()
  return _sha(_canonical(sorted(entries,key=lambda x:_canonical(x))))
 def source_reference_multiset_key(self):
  """Fingerprint supplied source-reference metadata; source replay is separate."""
  if not self.precursors or any(c.identity.derived_reference_text is None for c in self.precursors):
   raise ValueError("canonical chemical semantic evidence unavailable; native multiset proxy is separate")
  return _sha(_canonical(sorted([dict(reference=c.identity.derived_reference_text,
   role=c.role,value=c.amount.value,unit=c.amount.unit,presence=c.amount.presence) for c in self.precursors],key=_canonical)))
 @property
 def binding_sha256(self):return _hash_dataclass(self)
 def require_model_use(self,purpose,*,admission=None):return _model_gate(purpose,role=self.source_role,admission=admission)


@dataclass(frozen=True)
class RouteMolecule:
 node_id: str
 identity: ChemicalIdentity
 def __post_init__(self):
  if not _exact_text(self.node_id) or type(self.identity) is not ChemicalIdentity:raise ValueError("typed route molecule node required")
@dataclass(frozen=True)
class RouteStep:
 step_id: str
 product_id: str
 precursor_ids: tuple
 native_step_bytes: bytes|None = None
 def __post_init__(self):
  if (not _exact_text(self.step_id) or not _exact_text(self.product_id) or type(self.precursor_ids) is not tuple
      or not self.precursor_ids or any(not _exact_text(x) for x in self.precursor_ids)):
   raise ValueError("explicit nonempty reaction AND dependency list required")
  if self.native_step_bytes is not None:_json_bytes(self.native_step_bytes)
@dataclass(frozen=True)
class StockSnapshot:
 role: str
 entries: tuple
 source_ref_bytes: bytes
 identity_scope: str
 coverage: str = "declared_entries_complete"
 def __post_init__(self):
  if (self.role not in ("benchmark_snapshot","source_stated_inventory") or type(self.entries) is not tuple
      or any(not _exact_text(x) for x in self.entries) or len(set(self.entries))!=len(self.entries)
      or self.identity_scope!="native_literal_SMILES_membership"
      or self.coverage not in ("declared_entries_complete","bounded_positive_membership_subset")):
   raise ValueError("exact source stock snapshot semantics required; no purchasability claim")
  ref=_json_bytes(self.source_ref_bytes)
  if (type(ref) is not dict or type(ref.get("path")) is not str or type(ref.get("sha256")) is not str
      or len(ref["sha256"])!=64):raise ValueError("exact stock snapshot reference required")
 def contains(self,native_text):
  if native_text in self.entries:return True
  return False if self.coverage=="declared_entries_complete" else None

@dataclass(frozen=True)
class RouteCandidate:
 root_id: str
 molecules: tuple
 steps: tuple
 stock_snapshot: StockSnapshot|None
 native_serialization: bytes
 source_role: str
 source_locator_bytes: bytes = b"{}"
 def __post_init__(self):
  if (not _exact_text(self.root_id) or type(self.molecules) is not tuple or type(self.steps) is not tuple
      or any(type(n) is not RouteMolecule for n in self.molecules) or any(type(s) is not RouteStep for s in self.steps)
      or (self.stock_snapshot is not None and type(self.stock_snapshot) is not StockSnapshot)
      or type(self.native_serialization) is not bytes or not self.native_serialization
      or self.source_role not in ("synthetic_test","PaRoutes_n1_evaluation_reference","PaRoutes_n5_evaluation_reference")):
   raise ValueError("explicit native route graph/source role required")
  nodes={n.node_id:n for n in self.molecules};steps={s.step_id:s for s in self.steps}
  if len(nodes)!=len(self.molecules) or len(steps)!=len(self.steps) or self.root_id not in nodes:
   raise ValueError("duplicate or missing route molecule/reaction allocation")
  edges={k:[] for k in nodes}
  for s in self.steps:
   if s.product_id not in nodes or any(p not in nodes for p in s.precursor_ids):raise ValueError("unknown route dependency")
   edges[s.product_id].extend(s.precursor_ids)
  active=set();seen=set()
  def visit(n):
   if n in active:raise ValueError("cyclic route dependency")
   if n in seen:return
   active.add(n)
   for child in edges[n]:visit(child)
   active.remove(n);seen.add(n)
  visit(self.root_id)
  if seen!=set(nodes):raise ValueError("unreachable route nodes/dependencies")
  _json_bytes(self.source_locator_bytes)
 def alternatives(self,node_id):return tuple(s.step_id for s in self.steps if s.product_id==node_id)
 def precursor_dependencies(self,step_id):
  return next(s.precursor_ids for s in self.steps if s.step_id==step_id)
 def stock_termination_status(self):
  if self.stock_snapshot is None:return "stock_snapshot_unavailable"
  internal={s.product_id for s in self.steps}
  leaves=[n for n in self.molecules if n.node_id not in internal]
  found=[self.stock_snapshot.contains(n.identity.native_text) for n in leaves]
  if all(x is True for x in found):return "all_leaves_in_declared_snapshot"
  if any(x is None for x in found):return "stock_membership_unresolved"
  return "leaf_outside_declared_snapshot"
 def structurally_stock_solvable(self):
  """AND/OR graph membership only; never chemical feasibility or purchase."""
  if self.stock_snapshot is None:return None
  nodes={n.node_id:n for n in self.molecules};cache={}
  def solve(node_id):
   if node_id in cache:return cache[node_id]
   membership=self.stock_snapshot.contains(nodes[node_id].identity.native_text)
   if membership is True:cache[node_id]=True;return True
   alternatives=[s for s in self.steps if s.product_id==node_id]
   if not alternatives:cache[node_id]=membership;return membership
   outcomes=[]
   for step in alternatives:
    children=[solve(p) for p in step.precursor_ids]
    outcomes.append(False if any(x is False for x in children) else True if all(x is True for x in children) else None)
   result=True if any(x is True for x in outcomes) else None if any(x is None for x in outcomes) else False
   cache[node_id]=result;return result
  return solve(self.root_id)
 @property
 def actual_purchasability_verified(self):return False
 def prediction_inputs(self):
  root=next(n for n in self.molecules if n.node_id==self.root_id)
  return dict(query_role="product_conditioned_route_query",product=dict(syntax="SMILES",value=root.identity.product_input_text()))
 @property
 def binding_sha256(self):return _hash_dataclass(self)
 def require_model_use(self,purpose,*,admission=None):return _model_gate(purpose,role=self.source_role,admission=admission)

@dataclass(frozen=True)
class RetrosynthesisSelector:
 source: str
 native_indices: tuple
 def __post_init__(self):
  if (self.source not in ("ORD","PaRoutes_n1","PaRoutes_n5") or type(self.native_indices) is not tuple
      or not 1<=len(self.native_indices)<=16 or any(type(i) is not int or i<0 for i in self.native_indices)
      or len(set(self.native_indices))!=len(self.native_indices)):
   raise ValueError("exact bounded unique native source indices required")


class RetrosynthesisNativeSource:
 """Pinned three-ORD/two-PaRoutes safe JSON fixture; no full-release decoder."""
 def __init__(self,*,root,manifest=FIXTURE_REF["path"],expected_manifest_sha256=FIXTURE_REF["sha256"]):
  self.root=Path(root).resolve();self.manifest=Path(manifest)
  if expected_manifest_sha256!=FIXTURE_REF["sha256"]:raise ValueError("exact pinned safe fixture required")
  self._snapshot()
 def _verified(self,ref):
  if (type(ref) is not dict or not _exact_text(ref.get("path")) or not _exact_text(ref.get("sha256"))
      or len(ref["sha256"])!=64 or ("bytes" in ref and (type(ref["bytes"]) is not int or ref["bytes"]<=0))):
   raise ValueError("typed exact native source reference required")
  p=(self.root/ref["path"]).resolve()
  if not p.is_relative_to(self.root) or p.suffix.lower()!=".json":
   raise ValueError("safe JSON only; no raw protobuf/Parquet/pickle/generated-code execution")
  raw=p.read_bytes()
  if _sha(raw)!=ref["sha256"] or ("bytes" in ref and len(raw)!=ref["bytes"]):
   raise ValueError("native source/fixture byte integrity mismatch")
  return raw
 def _snapshot(self):
  path=(self.root/self.manifest).resolve()
  if not path.is_relative_to(self.root) or path.suffix.lower()!=".json":raise ValueError("safe fixture JSON required")
  raw=path.read_bytes()
  if _sha(raw)!=FIXTURE_REF["sha256"]:raise ValueError("fixture integrity mismatch")
  m=_json(raw)
  if m["model_use_allowed"] is not False or type(m["task_admitted_records"]) is not int or m["task_admitted_records"]!=0:
   raise ValueError("bounded candidate fixture cannot grant model task admission")
  payloads={}
  for ref in m["files"]:
   name=Path(ref["path"]).name
   if name not in FILE_PINS or ref["sha256"]!=FILE_PINS[name] or name in payloads:
    raise ValueError("exact versioned native fixture file allocation required")
   payloads[name]=_json(self._verified(ref))
  if set(payloads)!=set(FILE_PINS):raise ValueError("complete bounded native source fixture required")
  available=_json(self._verified(m["ORD_available_manifest"]))
  schema=_json(self._verified(m["ORD_schema_primary_source_receipt"]))
  if (m["ORD_available_manifest"]["sha256"]!="f934d654f7439b0e991381c7c196cc7e2dac6d355cdd1430b5baa3646dfba542"
      or m["ORD_schema_primary_source_receipt"]["sha256"]!="966f6544cedb7972459cb5497894a5b7e5d7621d080990c02f8f37013b931f81"
      or available["revision"]!=m["ORD_source_revision"] or schema["source_revision"]!=m["ORD_schema_revision"]
      or available["whole_original_LFS_SHA_size_and_GitHub_pointer_matches"] is not True
      or payloads["bounded-original-chemical-string-QC.json"]["all_parsed"] is not True):
   raise ValueError("source/schema/reference graph provenance binding mismatch")
  return m,payloads,available,schema
 @staticmethod
 def _amount(component):
  if "amount" not in component:return NativeAmount(None,None,"absent")
  amount=component["amount"];raw=_canonical(amount)
  if type(amount) is not dict:return NativeAmount(None,None,"unresolved",raw)
  branches=[v for v in amount.values() if type(v) is dict and "value" in v]
  if len(branches)!=1:return NativeAmount(None,None,"unresolved",raw)
  branch=branches[0];value=branch["value"];unit=branch.get("units")
  if type(value) not in (int,float,str) or (unit is not None and type(unit) is not str):
   return NativeAmount(None,None,"unresolved",raw)
  # This lexical display is derived from the losslessly retained semantic JSON;
  # original protobuf byte SHA/length remains the source serialization identity.
  return NativeAmount(str(value),unit,"present",raw)
 @staticmethod
 def _identity(component,*,pointer,chemical_qc,chemical_ref):
  ids=component.get("identifiers",[])
  matching=[(i,x) for i,x in enumerate(ids) if x.get("type")=="SMILES"]
  if len(matching)!=1:raise ValueError("unambiguous source SMILES semantics unavailable; no identifier guessing")
  i,ident=matching[0];text=ident["value"];ptr=pointer+"/identifiers/"+str(i)
  proofs=[p for p in chemical_qc["native_SMILES_occurrences"] if p["pointer"]==ptr]
  if len(proofs)!=1 or proofs[0]["source_smiles"]!=text or proofs[0]["RDKit_parse"] is not True:
   raise ValueError("exact source/reference chemical-string evidence unavailable")
  proof=proofs[0]
  evidence=_canonical(dict(source_QC_ref=chemical_ref,pointer=ptr,source_smiles=text,
   role="source_computed_isomeric_reference_not_material_identity_or_chemical_feasibility"))
  return ChemicalIdentity(text,derived_reference_text=proof["reference_isomeric_smiles"],derived_evidence_bytes=evidence)
 def read_reactions(self,selector):
  if type(selector) is not RetrosynthesisSelector or selector.source!="ORD":raise ValueError("bounded native ORD selector required")
  m,payloads,available,schema=self._snapshot()
  records=payloads["ORD-three-original-semantic-records.json"]
  by_row={r["source_row"]:r for r in records}
  if len(by_row)!=len(records) or not set(selector.native_indices)<=set(by_row):
   raise ValueError("source row not available in bounded fixture; no raw protobuf fallback")
  chemical=payloads["bounded-original-chemical-string-QC.json"]
  chemical_ref=next(r for r in m["files"] if Path(r["path"]).name=="bounded-original-chemical-string-QC.json")
  assets={a["path"]:a for a in available["assets"]};out=[]
  roles={"REACTANT":"precursor","REAGENT":"reagent","SOLVENT":"solvent","CATALYST":"catalyst"}
  for row_index in selector.native_indices:
   record=by_row[row_index];native_index=records.index(record);semantic=record["semantic"]
   asset=assets.get(record["source_file"])
   if (asset is None or type(record["source_row"]) is not int
       or record["source_file_sha256"]!=asset["sha256"] or record["source_file_bytes"]!=asset["bytes"]
       or record["task_admitted"] is not False or record["adapter_supported"] is not False
       or record["unknown_wire_fields"] is not False or record["semantic_json_roundtrip_equal"] is not True
       or type(record["source_proto_bytes"]) is not int or record["source_proto_bytes"]<=0
       or type(record["source_proto_sha256"]) is not str or len(record["source_proto_sha256"])!=64
       or semantic["reaction_id"]!=record["native_column_reaction_id"]):
    raise ValueError("native source file/row/protobuf/presence/unknown-field gate failed")
   products=[];precursors=[];aux=[]
   for name,group in semantic["inputs"].items():
    for j,component in enumerate(group["components"]):
     source_role=component.get("reaction_role")
     role=roles.get(source_role,"unknown")
     ident=self._identity(component,pointer=f"/ORD/{native_index}/inputs/{name}/components/{j}",
      chemical_qc=chemical,chemical_ref=chemical_ref)
     value=ReactionComponent(ident,role,self._amount(component),source_role,_canonical(component))
     (precursors if role=="precursor" else aux).append(value)
   for k,outcome in enumerate(semantic.get("outcomes",[])):
    for j,component in enumerate(outcome.get("products",[])):
     if component.get("reaction_role")!="PRODUCT":raise ValueError("explicit native product role unavailable")
     ident=self._identity(component,pointer=f"/ORD/{native_index}/outcomes/{k}/products/{j}",
      chemical_qc=chemical,chemical_ref=chemical_ref)
     products.append(ReactionComponent(ident,"product",self._amount(component),"PRODUCT",_canonical(component)))
   if not products:raise ValueError("native product semantics unavailable")
   locator=_canonical(dict(schema="mora3.RetrosynthesisNativeSourceBinding/v1",kind="ORD",
    fixture_sha256=FIXTURE_REF["sha256"],source_file=asset,source_row=row_index,
    source_proto_sha256=record["source_proto_sha256"],source_proto_bytes=record["source_proto_bytes"],
    source_schema_revision=m["ORD_schema_revision"],source_revision=m["ORD_source_revision"],
    serialization_role="source_owner_semantic_JSON_of_native_protobuf",
    record_sha256=_sha(_canonical(record)),unknown_wire_fields=False,task_admitted=False))
   out.append(ReactionCandidate(tuple(products),tuple(precursors),_canonical(semantic),_canonical(record),
    tuple(aux),"ORD_native_candidate",locator))
  return tuple(out)
 @staticmethod
 def _pointer(value,pointer):
  if pointer=="":return value
  if type(pointer) is not str or not pointer.startswith("/"):raise ValueError("native route JSON pointer required")
  for token in pointer.split("/")[1:]:
   if type(value) is list:
    if re.fullmatch(r"0|[1-9][0-9]*",token) is None:raise ValueError("exact native child ordinal required")
    value=value[int(token)]
   elif type(value) is dict:value=value[token]
   else:raise ValueError("native route pointer outside tree")
  return value
 def _route(self,record,m,payloads):
  protocol=record["protocol"]
  if (protocol not in ("n1","n5") or record["role"]!="EVALUATION_AND_EXPOSURE_CONTROL_ONLY"
      or record["not_training"] is not True or record["task_admitted"] is not False
      or type(record["native_JSON_array_index"]) is not int or record["native_JSON_array_index"]<0):
   raise ValueError("official reference route must retain evaluation/exposure-only role")
  assets={f["path"]:f for f in payloads["PaRoutes15-originals-available-manifest-v1.json"]["files"]}
  for name in ("source_file","stock_source","target_source"):
   ref=record[name]
   if ref["path"] not in assets or _canonical(ref)!=_canonical(assets[ref["path"]]):
    raise ValueError("exact route/stock/target original snapshot refs required")
  tree=record["original_route"];molecules=[];steps=[];leaf_pointers=[]
  def visit(node,pointer):
   if type(node) is not dict or node.get("type")!="mol" or not _exact_text(node.get("smiles")):
    raise ValueError("explicit native molecule node required")
   node_id="mol:"+pointer
   molecules.append(RouteMolecule(node_id,ChemicalIdentity(node["smiles"])))
   children=node.get("children",[])
   if type(children) is not list:raise ValueError("native route alternatives must remain ordered list")
   if not children:leaf_pointers.append(pointer)
   for i,reaction in enumerate(children):
    rp=pointer+"/children/"+str(i)
    if type(reaction) is not dict or reaction.get("type")!="reaction" or type(reaction.get("children")) is not list:
     raise ValueError("explicit native reaction AND node required")
    precursors=tuple(visit(child,rp+"/children/"+str(j)) for j,child in enumerate(reaction["children"]))
    steps.append(RouteStep("rxn:"+rp,node_id,precursors,_canonical(reaction)))
   return node_id
  root=visit(tree,"")
  locator_rows=record["leaf_stock_locator_audit"]
  seen=set();members=[]
  for row in locator_rows:
   pointer=row["source_node_pointer"]
   if pointer in seen or pointer not in leaf_pointers:raise ValueError("stock locator must bind one original native leaf")
   seen.add(pointer);node=self._pointer(tree,pointer)
   indices=row["exact_literal_stock_line_zero_based"]
   if (row["native_smiles"]!=node["smiles"] or type(indices) is not list
       or any(type(i) is not int or i<0 for i in indices)):
    raise ValueError("native stock leaf/string/line binding mismatch")
   # Gold in_stock booleans are audit-only. Exact positive source-line locators
   # establish this bounded membership subset; absence stays unknown.
   if indices:members.append(row["native_smiles"])
  if seen!=set(leaf_pointers):raise ValueError("all native leaf stock dispositions required")
  target_lines=record["exact_target_line_zero_based"]
  if type(target_lines) is not list or not target_lines or any(type(i) is not int or i<0 for i in target_lines):
   raise ValueError("exact native evaluation target line references required")
  stock=StockSnapshot("benchmark_snapshot",tuple(dict.fromkeys(members)),_canonical(record["stock_source"]),
   "native_literal_SMILES_membership","bounded_positive_membership_subset")
  locator=_canonical(dict(schema="mora3.RetrosynthesisNativeSourceBinding/v1",kind="PaRoutes",protocol=protocol,
   fixture_sha256=FIXTURE_REF["sha256"],native_JSON_array_index=record["native_JSON_array_index"],
   source_file=record["source_file"],stock_source=record["stock_source"],target_source=record["target_source"],
   exact_target_lines=target_lines,leaf_stock_locator_audit=locator_rows,evaluation_only=True,
   task_admitted=False,native_record_sha256=_sha(_canonical(record)),
   serialization_role="source_owner_selected_original_JSON_tree;source_file_bytes_bound_separately"))
  return RouteCandidate(root,tuple(molecules),tuple(steps),stock,_canonical(tree),
   "PaRoutes_"+protocol+"_evaluation_reference",locator)
 def read_routes(self,selector):
  if type(selector) is not RetrosynthesisSelector or not selector.source.startswith("PaRoutes_"):
   raise ValueError("bounded native evaluation route selector required")
  m,payloads,*_=self._snapshot()
  protocol=selector.source.split("_",1)[1]
  records=[r for r in payloads["PaRoutes-two-original-reference-route-trees-EVAL-ONLY.json"] if r["protocol"]==protocol]
  by_index={r["native_JSON_array_index"]:r for r in records}
  if len(by_index)!=len(records) or not set(selector.native_indices)<=set(by_index):
   raise ValueError("reference route not available in bounded fixture; no whole-route fallback")
  return tuple(self._route(by_index[i],m,payloads) for i in selector.native_indices)
 def verify(self,candidate):
  if type(candidate) is ReactionCandidate:
   loc=_json_bytes(candidate.source_locator_bytes)
   if loc.get("kind")!="ORD" or type(loc.get("source_row")) is not int:raise ValueError("native source binding required")
   checked,=self.read_reactions(RetrosynthesisSelector("ORD",(loc["source_row"],)))
  elif type(candidate) is RouteCandidate:
   loc=_json_bytes(candidate.source_locator_bytes)
   if loc.get("kind")!="PaRoutes" or type(loc.get("native_JSON_array_index")) is not int:raise ValueError("native route binding required")
   checked,=self.read_routes(RetrosynthesisSelector("PaRoutes_"+loc["protocol"],(loc["native_JSON_array_index"],)))
  else:raise ValueError("typed native reaction/route candidate required")
  if candidate!=checked:raise ValueError("fresh native source/target/role/serialization binding mismatch")
  return checked
 def verified_reference_multiset_key(self,candidate):
  if type(candidate) is not ReactionCandidate:raise ValueError("native reaction candidate required")
  self.verify(candidate)
  return candidate.source_reference_multiset_key()
 def source_references(self):
  m,*_=self._snapshot();refs={Path(r["path"]).name:r for r in m["files"]}
  path=(self.root/self.manifest).resolve()
  return dict(fixture=dict(path=str(path),bytes=path.stat().st_size,sha256=FIXTURE_REF["sha256"]),
   ORD=refs["ORD-three-original-semantic-records.json"],
   PaRoutes=refs["PaRoutes-two-original-reference-route-trees-EVAL-ONLY.json"],
   chemistry=refs["bounded-original-chemical-string-QC.json"],
   PaRoutes_available=refs["PaRoutes15-originals-available-manifest-v1.json"],
   ORD_available=m["ORD_available_manifest"],schema=m["ORD_schema_primary_source_receipt"])
 def gates(self):
  return dict(peptide_eligible_records=0,whole_release_semantic_QC_claimed=False,task_admitted_records=0,
   full_dataset_ready=False,runtime_raw_protobuf_decoding=False,
   canonical_identity_scope="source QC reference graphs only; not material identity or chemical feasibility",
   route_stock_scope="bounded exact literal source-line membership, not current stock/purchase/cost",
   official_route_protocols="n1/n5 evaluation and exposure only; never training",
   cross_corpus_reaction_patent_product_scaffold_route_peptide_guards="constructor admission pending")
