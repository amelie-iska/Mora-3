"""Full-release PaRoutes route/stock native candidates; never model admission.

Canonical typed JSON is a source-preparation projection, not original raw JSON
serialization. Raw archive authority is inherited under the pinned source guard.
"""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass,field
from datetime import datetime,timezone
import hashlib,json,os,re
from pathlib import Path
BATCH="outputs/data-sources/20261101-retrosynthesis-source-v1"
HANDOFF_PATH=BATCH+"/handoff-complete-ORD-PaRoutes-native-reaction-sources-v1.json"
HANDOFF_SHA256="62ea114eeab722e8363884b4f3704143b7f0d38928791b01d2fa39694dad72d7"
MANIFEST_PATH=BATCH+"/curated/PaRoutes-whole15-native-QC-v1/manifest.json"
MANIFEST_SHA256="695f7ff874585f5c5f28053f7a5ed3ba36c5530466f101e0d5b966ef76b7099a"
PLAN_SHA256="5695a0edc2e72712d965b87b3f74d211f8b91d57078fb089eec69f0283870e30"
AVAILABLE_PATH=BATCH+"/PaRoutesV2-complete-data-release-v1/whole-15-original-data-available-manifest-v1.json"
AVAILABLE_SHA256="86181d60475a958fc779462bd5204bfb4347db16894fd34dead783f474aacb2e"
PROOF_SHA256="c4f0bfe52bab987929db580a5b8b57e33e92d2c7edd21abe08a32c456edfb119"
IDENTITY_PATH=BATCH+"/curated/bounded-PaRoutes-full-native-reader-fixture-v2/completed-QC-original-and-typed-current-file-identity-supplement-v1.json"
IDENTITY_SHA256="b983b4aca4452d9258dcf51d0474a9aeac1cfb8284131419c20b8223c462d9b2"
FIXTURE_READER_SHA256="2e33707c13cfccbbfa06fcfbe324dd61e3bd389eb1bf3d9fee90249b5714c7eb"
EVAL="EVALUATION_EXPOSURE_CONTROL_ONLY"
CANDIDATE="CANDIDATE_REQUIRES_HELDOUT_AND_CHEMICAL_GROUP_EXCLUSION"
ROUTE_NAMES=("all_loaded_routes.json.gz","ref_routes_n1.json","ref_routes_n5.json")
STOCK_NAMES=("stock_n1.txt","stock_n5.txt")
FIELDS=("source_file_sha256","source_row","source_role","payload_json","quality_reasons_json",
 "source_reference_hash_exposure","task_admitted","adapter_supported")
MAX_ROW_GROUP_BYTES=256*1024**2
MAX_ROUTE_JSON_BYTES=32*1024**2
MAX_ROUTE_NODES=100000
def _sha(raw):return hashlib.sha256(raw).hexdigest()
def _canonical(value):return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def _strict_json(raw):
 def pairs(items):
  d={}
  for k,v in items:
   if k in d:raise ValueError("duplicate native JSON key")
   d[k]=v
  return d
 def invalid(value):raise ValueError("nonfinite native JSON literal")
 return json.loads(raw,object_pairs_hook=pairs,parse_constant=invalid)
def _json_bytes(raw):
 if type(raw) is not bytes:raise ValueError("immutable canonical native JSON bytes required")
 value=_strict_json(raw)
 if _canonical(value)!=raw:raise ValueError("exact source canonical JSON bytes required")
 return value
def _hex(value):return type(value) is str and re.fullmatch("[0-9a-f]{64}",value) is not None
def _int(value):return type(value) is int and value>=0
def _role(name):return CANDIDATE if name=="all_loaded_routes.json.gz" else EVAL
def _identity(stat):return dict(dev=stat.st_dev,ino=stat.st_ino,size=stat.st_size,mtime_ns=stat.st_mtime_ns,ctime_ns=stat.st_ctime_ns)
def _valid_identity(value):
 return type(value) is dict and set(value)=={"dev","ino","size","mtime_ns","ctime_ns"} and all(_int(x) for x in value.values())
def _utc_ns(value):
 dt=datetime.fromisoformat(value)-datetime(1970,1,1,tzinfo=timezone.utc)
 return ((dt.days*86400+dt.seconds)*10**6+dt.microseconds)*1000
def _stream_sha(handle):
 h=hashlib.sha256()
 for block in iter(lambda:handle.read(8*1024**2),b""):h.update(block)
 return h.hexdigest()
def _model_gate(purpose,role,quarantined=False,exposed=None):
 if type(purpose) is not str or not purpose:raise ValueError("explicit model-use purpose required")
 if quarantined:raise ValueError("source parsing/query-chemistry quarantine cannot grant model supervision")
 if role==EVAL:raise ValueError("official PaRoutes evaluation/exposure control has no model/task admission; never training")
 raise ValueError("PaRoutes candidate has no constructor/master/model or cross-corpus heldout admission; exposure nonmembership is not safety proof")

@dataclass(frozen=True)
class PaRouteSelector:
 source_name:str
 source_row:int
 source_file_sha256:str
 typed_file_sha256:str
 def __post_init__(self):
  if type(self.source_name) is not str or self.source_name not in ROUTE_NAMES or not _int(self.source_row) or not _hex(self.source_file_sha256) or not _hex(self.typed_file_sha256):
   raise ValueError("exact bounded native route source/row and original/typed hashes required")

@dataclass(frozen=True)
class RouteNodeOccurrence:
 pointer:str
 parent_pointer:str|None
 kind:str
 smiles:str|None
 children:tuple
 children_field_present:bool
 native_node_bytes:bytes=field(repr=False)
def _node_occurrences(tree):
 nodes=[];active=set()
 def visit(node,pointer,parent,parent_kind,depth):
  if depth>256 or len(nodes)>=MAX_ROUTE_NODES:raise ValueError("native topology exceeds bounded runtime; no crop")
  if type(node) is not dict or node.get("type") not in ("mol","reaction"):raise ValueError("explicit native molecule/reaction node required")
  if id(node) in active:raise ValueError("cyclic native source graph")
  kind=node["type"]
  if parent_kind==kind or parent is None and kind!="mol":raise ValueError("native molecule/reaction alternation required")
  children=node.get("children",[])
  if type(children) is not list:raise ValueError("native children must remain ordered list")
  smiles=node.get("smiles")
  if smiles is not None and type(smiles) is not str:raise ValueError("literal native molecular/route text required")
  child_pointers=tuple(pointer+"/children/"+str(i) for i in range(len(children)))
  local={k:v for k,v in node.items() if k!="children"}
  # Full tree remains in candidate; local attributes avoid quadratic subtree copies.
  nodes.append(RouteNodeOccurrence(pointer,parent,kind,smiles,child_pointers,"children" in node,_canonical(local)))
  active.add(id(node))
  for child,cp in zip(children,child_pointers):visit(child,cp,pointer,kind,depth+1)
  active.remove(id(node))
 visit(tree,"",None,None,0)
 return tuple(nodes)

@dataclass(frozen=True)
class PaRouteCandidate:
 selector:PaRouteSelector
 typed_record_bytes:bytes=field(repr=False)
 binding_bytes:bytes=field(repr=False)
 def __post_init__(self):
  if type(self.selector) is not PaRouteSelector:raise ValueError("typed native route selector required")
  row=_json_bytes(self.typed_record_bytes);binding=_json_bytes(self.binding_bytes)
  if (type(row) is not dict or set(row)!=set(FIELDS) or type(binding) is not dict
      or binding.get("schema")!="mora3.PaRoutesFullCandidateBinding/v1"
      or binding.get("handoff_sha256")!=HANDOFF_SHA256 or binding.get("manifest_sha256")!=MANIFEST_SHA256
      or row["source_file_sha256"]!=self.selector.source_file_sha256 or not _int(row["source_row"])
      or row["source_row"]!=self.selector.source_row or row["source_role"]!=_role(self.selector.source_name)
      or type(row["payload_json"]) is not str or len(row["payload_json"].encode())>MAX_ROUTE_JSON_BYTES
      or type(row["quality_reasons_json"]) is not str or type(row["source_reference_hash_exposure"]) is not bool
      or row["task_admitted"] is not False or row["adapter_supported"] is not False):
   raise ValueError("exact native route/source/value/type/quality-mask binding required")
  _json_bytes(row["payload_json"].encode());reasons=_json_bytes(row["quality_reasons_json"].encode())
  if type(reasons) is not list or any(type(x) is not str or not x for x in reasons):raise ValueError("literal native quality reasons required")
 def native_record(self):return _json_bytes(self.typed_record_bytes)
 def native_tree(self):return _json_bytes(self.native_record()["payload_json"].encode())
 def binding(self):return _json_bytes(self.binding_bytes)
 @property
 def source_role(self):return self.native_record()["source_role"]
 @property
 def source_reference_hash_exposure(self):return self.native_record()["source_reference_hash_exposure"]
 @property
 def quality_reasons(self):return tuple(_strict_json(self.native_record()["quality_reasons_json"]))
 @property
 def status(self):return "native_quarantined_unadmitted" if self.quality_reasons else "native_candidate_unadmitted"
 @property
 def neural_token_count(self):return None
 @property
 def actual_purchasability_verified(self):return False
 def nodes(self):return _node_occurrences(self.native_tree())
 def alternatives(self,pointer):
  node=next((n for n in self.nodes() if n.pointer==pointer),None)
  if node is None or node.kind!="mol":raise ValueError("native molecule occurrence pointer required")
  return node.children
 def precursor_dependencies(self,pointer):
  node=next((n for n in self.nodes() if n.pointer==pointer),None)
  if node is None or node.kind!="reaction":raise ValueError("native reaction occurrence pointer required")
  return node.children
 @property
 def binding_sha256(self):return _sha(_canonical(dict(selector=self.selector.__dict__,typed_hex=self.typed_record_bytes.hex(),binding_hex=self.binding_bytes.hex())))
 def require_model_use(self,purpose,*,admission=None):return _model_gate(purpose,self.source_role,bool(self.quality_reasons),self.source_reference_hash_exposure)

@dataclass(frozen=True)
class PaRoutesStockSnapshot:
 source_name:str
 raw_bytes:bytes=field(repr=False)
 binding_bytes:bytes=field(repr=False)
 def __post_init__(self):
  binding=_json_bytes(self.binding_bytes)
  if (type(self.source_name) is not str or self.source_name not in STOCK_NAMES or type(self.raw_bytes) is not bytes or not self.raw_bytes
      or type(binding) is not dict or binding.get("schema")!="mora3.PaRoutesFullStockBinding/v1"
      or binding.get("handoff_sha256")!=HANDOFF_SHA256 or binding.get("manifest_sha256")!=MANIFEST_SHA256
      or binding.get("source_role")!=EVAL or not _int(binding.get("source_rows")) or binding["source_rows"]<=0):
   raise ValueError("exact native historical evaluation-stock source binding required")
  values=self.entries();indices=binding.get("selected_rows",list(range(len(values))))
  if (not values or type(indices) is not list or len(indices)!=len(values) or any(not _int(i) or i>=binding["source_rows"] for i in indices)
      or len(set(indices))!=len(indices) or any(not x for x in values)):
   raise ValueError("selected native stock occurrences/coverage required")
 def binding(self):return _json_bytes(self.binding_bytes)
 def entries(self):return tuple(line.rstrip("\r\n") for line in self.raw_bytes.decode("utf8").splitlines(keepends=True))
 @property
 def selected_rows(self):return tuple(self.binding().get("selected_rows",list(range(len(self.entries())))))
 @property
 def coverage(self):
  return "declared_entries_complete" if len(self.selected_rows)==self.binding()["source_rows"] and set(self.selected_rows)==set(range(self.binding()["source_rows"])) else "bounded_positive_subset"
 def contains_literal(self,value):
  if type(value) is not str:raise ValueError("literal native stock membership only")
  if value in self.entries():return True
  return False if self.coverage=="declared_entries_complete" else None
 @property
 def actual_purchasability_verified(self):return False
 @property
 def binding_sha256(self):return _sha(_canonical(dict(source=self.source_name,native_selected_line_hex=self.raw_bytes.hex(),binding_hex=self.binding_bytes.hex())))
 def require_model_use(self,purpose,*,admission=None):return _model_gate(purpose,EVAL)

@dataclass(frozen=True)
class RequestedStockContext:
 snapshot:PaRoutesStockSnapshot
 role:str="explicit_caller_requested_benchmark_stock"
 def __post_init__(self):
  if type(self.snapshot) is not PaRoutesStockSnapshot or self.role!="explicit_caller_requested_benchmark_stock":raise ValueError("explicit requested stock context only")
def _product_query(candidate,*,requested_stock=None):
 from mora3.data.retrosynthesis_native import ChemicalIdentity
 if type(candidate) is not PaRouteCandidate or candidate.quality_reasons:raise ValueError("nonquarantined source route query required")
 root=candidate.native_tree()
 if type(root) is not dict or root.get("type")!="mol" or type(root.get("smiles")) is not str:raise ValueError("literal native root molecule required")
 out=dict(query_role="product_conditioned_route_query",product=dict(syntax="SMILES",value=ChemicalIdentity(root["smiles"]).product_input_text()))
 if requested_stock is not None:
  if type(requested_stock) is not RequestedStockContext:raise ValueError("explicit requested stock context only; no source goldleaf inputs")
  out["requested_stock"]=dict(role=requested_stock.role,coverage=requested_stock.snapshot.coverage,
   entries=[dict(syntax="SMILES",value=ChemicalIdentity(v).product_input_text()) for v in requested_stock.snapshot.entries()])
 return out

class PaRoutesFullNativeSource:
 """Three native route sources and two historical stock sources; all unadmitted."""
 def __init__(self,*,root,handoff=HANDOFF_PATH,expected_manifest_sha256=MANIFEST_SHA256):
  self.root=Path(root).resolve()
  if handoff!=HANDOFF_PATH or expected_manifest_sha256!=MANIFEST_SHA256:raise ValueError("exact pinned full PaRoutes release required")
  self._snapshot()
 def _path(self,path,suffixes):
  if type(path) is not str or not path:raise ValueError("exact source path required")
  p=(self.root/path).resolve()
  if not p.is_relative_to(self.root) or p.suffix.lower() not in suffixes:raise ValueError("unsafe source format/path; no pickle/generated execution")
  return p
 def _json(self,path,sha,size=None):
  p=self._path(path,{".json"});raw=p.read_bytes()
  if len(raw)>8*1024**2 or not _hex(sha) or _sha(raw)!=sha or size is not None and len(raw)!=size:raise ValueError("source proof JSON integrity mismatch")
  return _strict_json(raw),dict(path=str(p),sha256=sha,bytes=len(raw))
 def _snapshot(self):
  h,hr=self._json(HANDOFF_PATH,HANDOFF_SHA256,13107);m,mr=self._json(MANIFEST_PATH,MANIFEST_SHA256,12293)
  plan,pr=self._json(m["plan_path"],PLAN_SHA256);available,ar=self._json(AVAILABLE_PATH,AVAILABLE_SHA256)
  proof,fr=self._json(h["handoff_proof"]["path"],PROOF_SHA256,97338);identity,ir=self._json(IDENTITY_PATH,IDENTITY_SHA256,16641)
  if (h["complete_native_manifests"]["PaRoutes"]["sha256"]!=MANIFEST_SHA256 or m["plan_sha256"]!=PLAN_SHA256
      or plan["code_and_input_bindings"].get(str((self.root/AVAILABLE_PATH).resolve()))!=AVAILABLE_SHA256
      or _canonical(h["source_worker_task_admitted"])!=b"0" or h["model_use_allowed"] is not False or h["adapter_supported"] is not False
      or _canonical(m["task_admitted_rows"])!=b"0" or m["adapter_supported"] is not False
      or proof["source_manifests"]["PaRoutes"]["sha256"]!=MANIFEST_SHA256
      or proof["no_missing_or_inprogress_payloads"] is not True or proof["no_new_task_admission"] is not True
      or _canonical(m["source_files"])!=b"15" or len(m["assets"])!=15 or len(plan["assets"])!=15
      or _canonical(m["source_rows_nonadditive"])!=b"6032740"
      or identity["QC_manifest"]["sha256"]!=MANIFEST_SHA256 or identity["no_file_mutations"] is not True):
   raise ValueError("exact full-release/QC/rights/closed-admission binding required")
  originals={Path(x["path"]).name:x for x in plan["assets"]};catalog={}
  for a in m["assets"]:
   name=Path(a["typed_path"]).name.removesuffix(".typed.parquet")
   if (name in catalog or name not in originals or not _int(a["source_rows"]) or a["quality_pass_rows"]+a["quarantined_rows"]!=a["source_rows"]):
    raise ValueError("duplicate or inconsistent native file allocation")
   catalog[name]=dict(asset=a,original=originals[name])
  if len(catalog)!=15:raise ValueError("complete15source catalog required")
  guards={}
  for g in identity["guards"]:
   key=(g["source_name"],g["kind"])
   if key in guards or g["source_name"] not in (*ROUTE_NAMES,*STOCK_NAMES) or g["kind"] not in ("original","typed"):raise ValueError("exact native identity guard allocation required")
   a=catalog[g["source_name"]]["asset"];o=catalog[g["source_name"]]["original"];typed=g["kind"]=="typed"
   expected=(a["typed_path"],a["typed_sha256"],a["typed_bytes"]) if typed else (o["path"],o["sha256"],o["bytes"])
   info=g["file_identity"];receipt_info=g["source_QC_receipt"]["file_identity"];lineage=g["lineage"]
   if (not _valid_identity(info) or not _valid_identity(receipt_info) or (g["path"],g["expected_whole_sha256"],info["size"])!=expected
       or g["source_QC_receipt"]["sha256"]!=a["receipt_sha256"] or g["source_QC_receipt"]["path"]!=a["receipt_path"]
       or g["fresh_whole_sha256"] is not None and g["fresh_whole_sha256"]!=expected[1]
       or lineage["current_mtime_and_ctime_predate_individual_QC_receipt"] is not True
       or lineage["current_mtime_and_ctime_predate_final_QC_completion"] is not True or lineage["QC_completed_utc"]!=m["completed_utc"]
       or max(info["mtime_ns"],info["ctime_ns"])>min(receipt_info["mtime_ns"],receipt_info["ctime_ns"],_utc_ns(m["completed_utc"]))):
    raise ValueError("unsupported/inconsistent inherited raw/typed identity lineage")
   guards[key]=g
  if set(guards)!={(n,k) for n in (*ROUTE_NAMES,*STOCK_NAMES) for k in ("original","typed")}:raise ValueError("complete supported source guard required")
  return catalog,guards,dict(handoff=hr,manifest=mr,plan=pr,available=ar,proof=fr,identity_supplement=ir)
 def catalog(self):
  c,_,_=self._snapshot()
  return tuple(dict(source_name=n,source_rows=v["asset"]["source_rows"],source_role=v["asset"]["source_role"],
   source_file_sha256=v["original"]["sha256"],typed_file_sha256=v["asset"]["typed_sha256"],
   supported_view="routes" if n in ROUTE_NAMES else "stock" if n in STOCK_NAMES else "unimplemented_closed") for n,v in c.items())
 def _guard(self,g):
  p=self._path(g["path"],{".json",".gz",".txt",".parquet"})
  if _canonical(_identity(p.stat()))!=_canonical(g["file_identity"]):raise ValueError("source identity changed; raw inherited proof cannot be reused")
  return p
 def _receipt(self,name,entry):
  a=entry["asset"];o=entry["original"];receipt,ref=self._json(a["receipt_path"],a["receipt_sha256"])
  expected=dict(source_path=o["path"],source_sha256=o["sha256"],source_bytes=o["bytes"],source_rows=a["source_rows"],
   typed_path=a["typed_path"],typed_sha256=a["typed_sha256"],typed_bytes=a["typed_bytes"],source_role=a["source_role"],
   quality_pass_rows=a["quality_pass_rows"],quarantined_rows=a["quarantined_rows"],full_typed_readback_rows=a["source_rows"],
   task_admitted_rows=0,adapter_supported=False,source_native_format="JSON_ARRAY" if name in ROUTE_NAMES else "TEXT_LINES")
  if any(_canonical(receipt.get(k))!=_canonical(v) for k,v in expected.items()):raise ValueError("selected native-QC receipt binding mismatch")
  return receipt,ref
 @contextmanager
 def _typed_file(self,g):
  p=self._guard(g)
  with p.open("rb") as handle:
   before=_identity(os.fstat(handle.fileno()))
   if _canonical(before)!=_canonical(g["file_identity"]) or _stream_sha(handle)!=g["expected_whole_sha256"]:raise ValueError("fresh selected typed integrity mismatch")
   handle.seek(0);yield handle
   if _canonical(_identity(os.fstat(handle.fileno())))!=_canonical(before):raise ValueError("typed source changed during read")
   self._guard(g)
 def _rows(self,g,indices,total):
  import pyarrow as pa,pyarrow.parquet as pq
  with self._typed_file(g) as handle:
   p=pq.ParquetFile(handle)
   schema=pa.schema(list(zip(FIELDS,[pa.string(),pa.int64(),pa.string(),pa.string(),pa.string(),pa.bool_(),pa.bool_(),pa.bool_()])))
   if not p.schema_arrow.equals(schema,check_metadata=True) or p.metadata.num_rows!=total:raise ValueError("exact native typed schema/axis required")
   wanted=set(indices);offset=0;out={}
   for group in range(p.num_row_groups):
    meta=p.metadata.row_group(group);end=offset+meta.num_rows;local=[i for i in wanted if offset<=i<end]
    if local:
     if meta.total_byte_size>MAX_ROW_GROUP_BYTES:raise ValueError("native rowgroup exceeds bounded runtime; no crop")
     table=p.read_row_group(group,use_threads=False)
     for i in local:out[i]=table.slice(i-offset,1).to_pylist()[0]
    offset=end
   if set(out)!=wanted:raise ValueError("selected native source row missing")
   return out
 def read_routes(self,selectors):
  if (type(selectors) is not tuple or not 1<=len(selectors)<=8 or any(type(s) is not PaRouteSelector for s in selectors)
      or len({(s.source_name,s.source_row) for s in selectors})!=len(selectors) or len({s.source_name for s in selectors})>2):
   raise ValueError("1..8unique native route occurrences across atmost2typed shards required")
  catalog,guards,refs=self._snapshot();out={}
  for name in dict.fromkeys(s.source_name for s in selectors):
   entry=catalog[name];a=entry["asset"];o=entry["original"];subset=[s for s in selectors if s.source_name==name]
   if any(s.source_file_sha256!=o["sha256"] or s.typed_file_sha256!=a["typed_sha256"] or s.source_row>=a["source_rows"] for s in subset):
    raise ValueError("wrong exact native source/typed/occurrence binding")
   self._guard(guards[(name,"original")]);receipt,rref=self._receipt(name,entry)
   rows=self._rows(guards[(name,"typed")],tuple(s.source_row for s in subset),a["source_rows"])
   self._guard(guards[(name,"original")])
   for s in subset:
    binding=dict(schema="mora3.PaRoutesFullCandidateBinding/v1",handoff_sha256=HANDOFF_SHA256,manifest_sha256=MANIFEST_SHA256,
     serialization_role="source_prepared_canonical_JSON_not_original_raw_serialization",
     references={**refs,"receipt":rref,"original":dict(path=o["path"],bytes=o["bytes"],sha256=o["sha256"]),
      "typed":dict(path=a["typed_path"],bytes=a["typed_bytes"],sha256=a["typed_sha256"])},
     raw_identity_guard=guards[(name,"original")],typed_identity_guard=guards[(name,"typed")],source_QC_digest=receipt["semantic_rows_digest"],
     raw_authority="prior complete QC original wholeSHA plus pinned current identity/OS timestamp/immutable history guard; no raw route read",
     model_use_allowed=False,task_admitted=False,chemistry_scope="source parser/QC flags only; query representation failure is not invalid chemistry or synthesis infeasibility")
    out[(name,s.source_row)]=PaRouteCandidate(s,_canonical(rows[s.source_row]),_canonical(binding))
  return tuple(out[(s.source_name,s.source_row)] for s in selectors)
 def read_stock_snapshot(self,name,indices):
  if (type(name) is not str or name not in STOCK_NAMES or type(indices) is not tuple or not 1<=len(indices)<=32
      or any(not _int(i) for i in indices) or len(set(indices))!=len(indices)):raise ValueError("bounded explicit one-stock occurrence selection required")
  catalog,guards,refs=self._snapshot();entry=catalog[name];a=entry["asset"];o=entry["original"]
  if max(indices)>=a["source_rows"]:raise ValueError("stock occurrence outside source axis")
  receipt,rref=self._receipt(name,entry);path=self._guard(guards[(name,"original")])
  typed=self._rows(guards[(name,"typed")],indices,a["source_rows"]);lines={};offsets={};offset=0
  with path.open("rb") as handle:
   if _canonical(_identity(os.fstat(handle.fileno())))!=_canonical(guards[(name,"original")]["file_identity"]):raise ValueError("stock source identity mismatch")
   for i,line in enumerate(handle):
    if offset+len(line)>1024**2:raise ValueError("stock prefix exceeds bounded runtime; no crop")
    if i in indices:lines[i]=line;offsets[i]=offset
    offset+=len(line)
    if i==max(indices):break
  self._guard(guards[(name,"original")])
  if set(lines)!=set(indices):raise ValueError("missing native stock line")
  for i in indices:
   row=typed[i];expected=dict(source_file_sha256=o["sha256"],source_row=i,source_role=EVAL,payload_json=json.dumps(lines[i].decode("utf8").rstrip("\r\n"),sort_keys=True,separators=(",",":"),allow_nan=False),
    quality_reasons_json="[]",source_reference_hash_exposure=None,task_admitted=False,adapter_supported=False)
   if _canonical(row)!=_canonical(expected):raise ValueError("native stock line/typed value/mask/source binding mismatch")
  binding=dict(schema="mora3.PaRoutesFullStockBinding/v1",handoff_sha256=HANDOFF_SHA256,manifest_sha256=MANIFEST_SHA256,
   source_role=EVAL,source_rows=a["source_rows"],selected_rows=list(indices),original_byte_offsets=[offsets[i] for i in indices],
   physical_prefix_bytes_read=offset,serialization_role="selected original TEXT_LINES physical bytes in requested order; not whole inventory",
   references={**refs,"receipt":rref,"original":dict(path=o["path"],bytes=o["bytes"],sha256=o["sha256"]),
    "typed":dict(path=a["typed_path"],bytes=a["typed_bytes"],sha256=a["typed_sha256"])},
   typed_row_fingerprints=[_sha(_canonical(typed[i])) for i in indices],raw_identity_guard=guards[(name,"original")],
   typed_identity_guard=guards[(name,"typed")],model_use_allowed=False,task_admitted=False)
  return PaRoutesStockSnapshot(name,b"".join(lines[i] for i in indices),_canonical(binding))
 def verify(self,candidate):
  if type(candidate) is not PaRouteCandidate:raise ValueError("typed native route candidate required")
  checked,=self.read_routes((candidate.selector,))
  if checked.binding_sha256!=candidate.binding_sha256:raise ValueError("exact native route replay mismatch")
  return checked
 def verify_stock(self,snapshot):
  if type(snapshot) is not PaRoutesStockSnapshot:raise ValueError("typed native stock snapshot required")
  checked=self.read_stock_snapshot(snapshot.source_name,snapshot.selected_rows)
  if checked.binding_sha256!=snapshot.binding_sha256:raise ValueError("exact native stock replay mismatch")
  return checked
 def project_product_inputs(self,candidate,*,requested_stock=None):
  checked=self.verify(candidate)
  from mora3.data import retrosynthesis_native as dependency
  if _sha(Path(dependency.__file__).read_bytes())!=FIXTURE_READER_SHA256:raise ValueError("reviewed literal projection dependency changed; compatibility review required")
  if requested_stock is not None:
   if type(requested_stock) is not RequestedStockContext:raise ValueError("explicit requested stock context required")
   requested_stock=RequestedStockContext(self.verify_stock(requested_stock.snapshot))
  return _product_query(checked,requested_stock=requested_stock)
 def leaf_stock_membership(self,candidate,snapshot):
  route=self.verify(candidate);stock=self.verify_stock(snapshot);result=[]
  for node in route.nodes():
   if node.kind=="mol" and not node.children:
    native=_json_bytes(node.native_node_bytes)
    result.append(dict(pointer=node.pointer,literal_membership=stock.contains_literal(node.smiles),
     source_reported_in_stock=native.get("in_stock"),scope="native literal historical membership only",current_purchasability_verified=False))
  return tuple(result)
