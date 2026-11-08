"""Pinned official PaRoutes evaluation/exposure text references; never training.

Complete reference snapshots establish literal membership only, not chemical
equivalence, benchmark qualification or global novelty/leakage safety.
"""
from __future__ import annotations
from dataclasses import dataclass,field
from pathlib import Path
import hashlib,json,os,re
BATCH="outputs/data-sources/20261101-retrosynthesis-source-v1"
HANDOFF_PATH=BATCH+"/handoff-complete-ORD-PaRoutes-native-reaction-sources-v1.json"
HANDOFF_SHA256="62ea114eeab722e8363884b4f3704143b7f0d38928791b01d2fa39694dad72d7"
MANIFEST_PATH=BATCH+"/curated/PaRoutes-whole15-native-QC-v1/manifest.json"
MANIFEST_SHA256="695f7ff874585f5c5f28053f7a5ed3ba36c5530466f101e0d5b966ef76b7099a"
PLAN_SHA256="5695a0edc2e72712d965b87b3f74d211f8b91d57078fb089eec69f0283870e30"
AVAILABLE_PATH=BATCH+"/PaRoutesV2-complete-data-release-v1/whole-15-original-data-available-manifest-v1.json"
AVAILABLE_SHA256="86181d60475a958fc779462bd5204bfb4347db16894fd34dead783f474aacb2e"
PROOF_SHA256="c4f0bfe52bab987929db580a5b8b57e33e92d2c7edd21abe08a32c456edfb119"
EVALUATION="EVALUATION_EXPOSURE_CONTROL_ONLY"
SPECS={
 "targets_n1.txt":dict(source_rows=10000,source_bytes=459136,
  source_sha256="14661af6052a8c2e6dad4c142b8d8693ed87692c5eee0d280110357a5af5e1ab",
  typed_bytes=295208,typed_sha256="ba729ee90aee833a2dfedf701c1404bc61d1ff6b0c03368181c87db64182a6db"),
 "targets_n5.txt":dict(source_rows=10000,source_bytes=487792,
  source_sha256="dd5d37e63ec05c3d72fee14607e4bd90d0ccc4eedf38129cd69dab3e5bdb0e56",
  typed_bytes=305774,typed_sha256="9c0947dc4c786cdcbfdc6b62ee986662ecb218cffd10cd69d1f06fa7fdc8b802"),
 "ref_routes_reaction_hashes.txt":dict(source_rows=56657,source_bytes=4403965,
  source_sha256="87d7e0013f97da27e4fd878b0598466ac07202fe131ac0f7dbdd2b423482751f",
  typed_bytes=2241229,typed_sha256="74a574d1963d9b9e00df872304bbe9c794845c87894de0a2ef62c2f281e8cb49")}
FIELDS=("source_file_sha256","source_row","source_role","payload_json","quality_reasons_json",
 "source_reference_hash_exposure","task_admitted","adapter_supported")
MAX_TOTAL_RAW_BYTES=16*1024**2
MAX_NATIVE_LINE_BYTES=1024**2
MAX_ROW_GROUP_BYTES=64*1024**2
def _sha(raw):return hashlib.sha256(raw).hexdigest()
def _canonical(value):return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def _int(value):return type(value) is int and value>=0
def _hex(value):return type(value) is str and re.fullmatch("[0-9a-f]{64}",value) is not None
def _strict_json(raw):
 def pairs(items):
  result={}
  for key,value in items:
   if key in result:raise ValueError("duplicate proof/typed JSON key")
   result[key]=value
  return result
 def invalid(value):raise ValueError("nonfinite JSON literal")
 try:return json.loads(raw,object_pairs_hook=pairs,parse_constant=invalid)
 except (json.JSONDecodeError,UnicodeError,RecursionError) as error:raise ValueError("invalid proof/typed JSON") from error
def _json_bytes(raw):
 if type(raw) is not bytes:raise ValueError("immutable canonical source-binding bytes required")
 value=_strict_json(raw)
 if _canonical(value)!=raw:raise ValueError("exact canonical source-binding bytes required")
 return value
def _identity(stat):return dict(dev=stat.st_dev,ino=stat.st_ino,size=stat.st_size,mtime_ns=stat.st_mtime_ns,ctime_ns=stat.st_ctime_ns)
def _closed_binding(raw,schema):
 value=_json_bytes(raw)
 if (type(value) is not dict or value.get("schema")!=schema or value.get("manifest_sha256")!=MANIFEST_SHA256
  or value.get("handoff_sha256")!=HANDOFF_SHA256 or value.get("model_use_allowed") is not False
  or value.get("task_admitted") is not False):
  raise ValueError("exact evaluation/source closed-admission binding required")
 return value
def _literal(value):
 if type(value) is not str:raise ValueError("exact native literal string required; missing values are not negative membership")
 return value
def _refuse(purpose):
 if type(purpose) is not str or not purpose:raise ValueError("explicit use purpose required")
 raise ValueError("official evaluation/exposure references have no constructor/model/protocol admission for task/model/positive-loss use")

@dataclass(frozen=True)
class NativeTextLine:
 source_row:int
 byte_start:int
 native_bytes:bytes=field(repr=False)
 def __post_init__(self):
  if (not _int(self.source_row) or not _int(self.byte_start) or type(self.native_bytes) is not bytes
    or not self.native_bytes or len(self.native_bytes)>MAX_NATIVE_LINE_BYTES):
   raise ValueError("complete bounded native line/ordinal/byte offset required")
  try:self.body.decode("utf8")
  except UnicodeError as error:raise ValueError("unsupported native UTF8 line") from error
  if b"\r" in self.body or b"\n" in self.body:raise ValueError("exact one native text occurrence required")
 @property
 def terminator(self):
  return b"\r\n" if self.native_bytes.endswith(b"\r\n") else self.native_bytes[-1:] if self.native_bytes[-1:] in (b"\r",b"\n") else b""
 @property
 def body(self):
  n=len(self.terminator);return self.native_bytes[:-n] if n else self.native_bytes
 @property
 def literal(self):return self.body.decode("utf8")
 @property
 def byte_stop(self):return self.byte_start+len(self.native_bytes)
 @property
 def status(self):return "native_empty_literal" if not self.body else "native_present_literal"
def parse_native_lines(raw):
 """Python newline='' text-iteration boundaries, retaining every native byte."""
 if type(raw) is not bytes or len(raw)>MAX_TOTAL_RAW_BYTES:raise ValueError("bounded immutable complete native text bytes required")
 out=[];start=0;i=0
 while i<len(raw):
  if raw[i] in (10,13):
   end=i+2 if raw[i]==13 and i+1<len(raw) and raw[i+1]==10 else i+1
   out.append(NativeTextLine(len(out),start,raw[start:end]));start=end;i=end
  else:i+=1
 if start<len(raw):out.append(NativeTextLine(len(out),start,raw[start:]))
 return tuple(out)

@dataclass(frozen=True)
class PaEvaluationSelector:
 source_name:str
 source_row:int
 source_file_sha256:str
 typed_file_sha256:str
 def __post_init__(self):
  if (type(self.source_name) is not str or self.source_name not in SPECS or not _int(self.source_row)
    or not _hex(self.source_file_sha256) or not _hex(self.typed_file_sha256)):
   raise ValueError("exact official reference source/occurrence and original/typed SHA required")

@dataclass(frozen=True)
class CompleteEvaluationSet:
 source_name:str
 native_file_bytes:bytes=field(repr=False)
 binding_bytes:bytes=field(repr=False)
 _lines:tuple=field(init=False,repr=False,compare=False)
 _literals:frozenset=field(init=False,repr=False,compare=False)
 def __post_init__(self):
  if type(self.source_name) is not str or self.source_name not in SPECS or type(self.native_file_bytes) is not bytes:
   raise ValueError("exact official complete source required")
  spec=SPECS[self.source_name];binding=_closed_binding(self.binding_bytes,"mora3.PaRoutesCompleteEvaluationSet/v1")
  if (len(self.native_file_bytes)!=spec["source_bytes"] or _sha(self.native_file_bytes)!=spec["source_sha256"]
    or binding.get("source_name")!=self.source_name or binding.get("source_sha256")!=spec["source_sha256"]
    or binding.get("coverage")!="COMPLETE_PINNED_LITERAL_REFERENCE_SET"):
   raise ValueError("complete original source bytes/hash/coverage proof required; no boolean admission")
  lines=parse_native_lines(self.native_file_bytes)
  if len(lines)!=spec["source_rows"]:raise ValueError("complete native occurrence axis required")
  object.__setattr__(self,"_lines",lines);object.__setattr__(self,"_literals",frozenset(l.literal for l in lines))
 @property
 def raw_lines(self):return self._lines
 @property
 def row_count(self):return len(self._lines)
 @property
 def unique_literal_count(self):return len(self._literals)
 @property
 def neural_token_count(self):return None
 @property
 def source_role(self):return EVALUATION
 @property
 def binding_sha256(self):return _sha(self.binding_bytes)
 def binding(self):return _json_bytes(self.binding_bytes)
 def membership(self,value):
  """Exact literal membership in this immutable complete pinned snapshot only."""
  return _literal(value) in self._literals
 def occurrences(self,value):
  _literal(value);return tuple(l.source_row for l in self._lines if l.literal==value)
 def require_model_use(self,purpose,*,admission=None):_refuse(purpose)

@dataclass(frozen=True)
class PaEvaluationCandidate:
 selector:PaEvaluationSelector
 byte_start:int
 native_line_bytes:bytes=field(repr=False)
 typed_record_bytes:bytes=field(repr=False)
 binding_bytes:bytes=field(repr=False)
 def __post_init__(self):
  if type(self.selector) is not PaEvaluationSelector:raise ValueError("typed official reference selector required")
  line=NativeTextLine(self.selector.source_row,self.byte_start,self.native_line_bytes)
  row=_json_bytes(self.typed_record_bytes);binding=_closed_binding(self.binding_bytes,"mora3.PaRoutesEvaluationOccurrence/v1")
  if (type(row) is not dict or set(row)!=set(FIELDS) or row["source_file_sha256"]!=self.selector.source_file_sha256
    or not _int(row["source_row"]) or row["source_row"]!=self.selector.source_row or row["source_role"]!=EVALUATION
    or type(row["payload_json"]) is not str or row["payload_json"].encode()!=_canonical(line.literal)
    or row["source_reference_hash_exposure"] is not None or row["task_admitted"] is not False
    or row["adapter_supported"] is not False or type(row["quality_reasons_json"]) is not str
    or binding.get("source_name")!=self.selector.source_name):
   raise ValueError("native reference bytes/target/ordinal/type/role/mask binding mismatch")
  reasons=_json_bytes(row["quality_reasons_json"].encode())
  if type(reasons) is not list or any(type(r) is not str or not r for r in reasons):raise ValueError("literal source quality reasons required")
 @property
 def literal(self):return NativeTextLine(self.selector.source_row,self.byte_start,self.native_line_bytes).literal
 @property
 def status(self):
  if self.quality_reasons:return "native_quarantined_evaluation_only"
  return "native_empty_literal_evaluation_only" if self.literal=="" else "native_literal_evaluation_only"
 @property
 def quality_reasons(self):return tuple(_strict_json(self.native_record()["quality_reasons_json"]))
 @property
 def neural_token_count(self):return None
 @property
 def source_role(self):return EVALUATION
 @property
 def binding_sha256(self):
  return _sha(_canonical(dict(selector=self.selector.__dict__,byte_start=self.byte_start,native_hex=self.native_line_bytes.hex(),
    typed_hex=self.typed_record_bytes.hex(),binding_hex=self.binding_bytes.hex())))
 def native_record(self):return _json_bytes(self.typed_record_bytes)
 def binding(self):return _json_bytes(self.binding_bytes)
 def require_model_use(self,purpose,*,admission=None):_refuse(purpose)
def partial_membership(candidates,value):
 _literal(value)
 if (type(candidates) is not tuple or not 1<=len(candidates)<=8 or any(type(c) is not PaEvaluationCandidate for c in candidates)
   or len({(c.selector.source_name,c.selector.source_file_sha256) for c in candidates})!=1):
  raise ValueError("bounded same-reference native occurrences required")
 return True if any(c.literal==value for c in candidates) else None

@dataclass(frozen=True)
class PaEvaluationRead:
 reference_sets:tuple
 candidates:tuple

class PaRoutesEvaluationSource:
 """Fresh complete small text sets + bounded exact typed occurrences; eval only."""
 def __init__(self,*,root,handoff=HANDOFF_PATH,expected_manifest_sha256=MANIFEST_SHA256):
  self.root=Path(root).resolve()
  if handoff!=HANDOFF_PATH or expected_manifest_sha256!=MANIFEST_SHA256:raise ValueError("exact finalized PaRoutes release required")
  self._snapshot()
 def _path(self,path,suffixes):
  if type(path) is not str or not path:raise ValueError("exact pinned source path required")
  p=(self.root/path).resolve()
  if not p.is_relative_to(self.root) or p.suffix.lower() not in suffixes:raise ValueError("unsafe source path/format")
  return p
 def _json(self,path,sha,size=None):
  p=self._path(path,{".json"})
  with p.open("rb") as h:raw=h.read(8*1024**2+1)
  if not _hex(sha) or _sha(raw)!=sha or len(raw)>8*1024**2 or size is not None and len(raw)!=size:
   raise ValueError("exact source proof integrity required")
  return _strict_json(raw),dict(path=str(p),sha256=sha,bytes=len(raw))
 def _snapshot(self):
  h,hr=self._json(HANDOFF_PATH,HANDOFF_SHA256,13107);m,mr=self._json(MANIFEST_PATH,MANIFEST_SHA256,12293)
  plan,pr=self._json(m["plan_path"],PLAN_SHA256);_,ar=self._json(AVAILABLE_PATH,AVAILABLE_SHA256)
  proof,fr=self._json(h["handoff_proof"]["path"],PROOF_SHA256,97338)
  if (h["complete_native_manifests"]["PaRoutes"]["sha256"]!=MANIFEST_SHA256 or m["plan_sha256"]!=PLAN_SHA256
   or plan["code_and_input_bindings"].get(str((self.root/AVAILABLE_PATH).resolve()))!=AVAILABLE_SHA256
   or _canonical(h["source_worker_task_admitted"])!=b"0" or h["model_use_allowed"] is not False or h["adapter_supported"] is not False
   or _canonical(m["task_admitted_rows"])!=b"0" or m["adapter_supported"] is not False
   or _canonical(m["source_files"])!=b"15" or _canonical(m["source_rows_nonadditive"])!=b"6032740"
   or len(m["assets"])!=15 or len(plan["assets"])!=15 or proof["source_manifests"]["PaRoutes"]["sha256"]!=MANIFEST_SHA256
   or proof["no_new_task_admission"] is not True or proof["no_missing_or_inprogress_payloads"] is not True):
   raise ValueError("exact whole15/source/protocol/QC/evaluation-only closed proof required")
  originals={Path(a["path"]).name:a for a in plan["assets"]}
  assets={Path(a["typed_path"]).name.removesuffix(".typed.parquet"):a for a in m["assets"]}
  if len(originals)!=15 or len(assets)!=15 or set(originals)!=set(assets):raise ValueError("complete unique source allocation required")
  result={}
  for name,spec in SPECS.items():
   a=assets[name];o=originals[name]
   expected=dict(source_rows=spec["source_rows"],quality_pass_rows=spec["source_rows"],quarantined_rows=0,
     source_role=EVALUATION,typed_bytes=spec["typed_bytes"],typed_sha256=spec["typed_sha256"])
   if (any(_canonical(a.get(k))!=_canonical(v) for k,v in expected.items())
      or _canonical(o["bytes"])!=_canonical(spec["source_bytes"]) or o["sha256"]!=spec["source_sha256"]):
    raise ValueError("exact official reference source/typed occurrence binding required")
   self._path(o["path"],{".txt"});self._path(a["typed_path"],{".parquet"})
   result[name]=dict(asset=a,original=o)
  return result,dict(handoff=hr,manifest=mr,plan=pr,available=ar,proof=fr)
 def _receipt(self,name,entry):
  a=entry["asset"];o=entry["original"];receipt,ref=self._json(a["receipt_path"],a["receipt_sha256"])
  expected=dict(source_path=o["path"],source_sha256=o["sha256"],source_bytes=o["bytes"],source_rows=a["source_rows"],
   typed_path=a["typed_path"],typed_sha256=a["typed_sha256"],typed_bytes=a["typed_bytes"],source_role=EVALUATION,
   quality_pass_rows=a["source_rows"],quarantined_rows=0,full_typed_readback_rows=a["source_rows"],
   native_TSV_ordered_columns=None,source_native_format="TEXT_LINES",source_reference_hash_exposure_rows=0,
   task_admitted_rows=0,adapter_supported=False)
  if any(_canonical(receipt.get(k))!=_canonical(v) for k,v in expected.items()):raise ValueError("exact native text/source-QC/evaluation receipt required")
  return receipt,ref
 def _read_original(self,o):
  p=self._path(o["path"],{".txt"});before=_identity(p.stat())
  if before["size"]!=o["bytes"] or o["bytes"]>MAX_TOTAL_RAW_BYTES:raise ValueError("native source extent/budget mismatch")
  with p.open("rb") as handle:
   if _canonical(_identity(os.fstat(handle.fileno())))!=_canonical(before):raise ValueError("source identity changed before read")
   raw=handle.read(o["bytes"]+1)
   if len(raw)!=o["bytes"] or _sha(raw)!=o["sha256"]:raise ValueError("fresh complete original integrity mismatch")
   lines=parse_native_lines(raw)
   if _canonical(_identity(os.fstat(handle.fileno())))!=_canonical(before):raise ValueError("source identity changed during parse/read")
  if _canonical(_identity(p.stat()))!=_canonical(before):raise ValueError("source path replaced/changed during read")
  return raw,lines,before
 def _rows(self,a,indices):
  import pyarrow as pa,pyarrow.parquet as pq
  p=self._path(a["typed_path"],{".parquet"});before=_identity(p.stat())
  if before["size"]!=a["typed_bytes"]:raise ValueError("typed native extent/integrity mismatch")
  with p.open("rb") as handle:
   if _canonical(_identity(os.fstat(handle.fileno())))!=_canonical(before):raise ValueError("typed identity changed before read")
   digest=hashlib.sha256()
   for block in iter(lambda:handle.read(8*1024**2),b""):digest.update(block)
   if digest.hexdigest()!=a["typed_sha256"]:raise ValueError("fresh selected typed integrity mismatch")
   handle.seek(0);reader=pq.ParquetFile(handle)
   expected=pa.schema(list(zip(FIELDS,[pa.string(),pa.int64(),pa.string(),pa.string(),pa.string(),pa.bool_(),pa.bool_(),pa.bool_()])))
   if not reader.schema_arrow.equals(expected,check_metadata=True) or reader.metadata.num_rows!=a["source_rows"]:
    raise ValueError("exact complete typed schema/occurrence axis required")
   offset=0;out={};wanted=set(indices)
   for group in range(reader.num_row_groups):
    info=reader.metadata.row_group(group);end=offset+info.num_rows;selected=[i for i in wanted if offset<=i<end]
    if selected:
     if info.total_byte_size>MAX_ROW_GROUP_BYTES:raise ValueError("typed rowgroup exceeds bounded runtime; no crop")
     table=reader.read_row_group(group,use_threads=False)
     for i in selected:out[i]=table.slice(i-offset,1).to_pylist()[0]
    offset=end
   if set(out)!=wanted:raise ValueError("native selected occurrence absent")
   if _canonical(_identity(os.fstat(handle.fileno())))!=_canonical(before):raise ValueError("typed changed during bounded read")
  if _canonical(_identity(p.stat()))!=_canonical(before):raise ValueError("typed replaced/changed during read")
  return out,before
 def read(self,selectors,*,reference_names=()):
  if (type(selectors) is not tuple or len(selectors)>8 or any(type(s) is not PaEvaluationSelector for s in selectors)
    or len({(s.source_name,s.source_row) for s in selectors})!=len(selectors) or len({s.source_name for s in selectors})>2
    or type(reference_names) is not tuple or any(type(n) is not str or n not in SPECS for n in reference_names)
    or len(set(reference_names))!=len(reference_names) or not selectors and not reference_names):
   raise ValueError("atmost8unique occurrences/2typedshards; explicit unique reference names required")
  entries,refs=self._snapshot()
  names=tuple(dict.fromkeys(reference_names+tuple(s.source_name for s in selectors)))
  if sum(entries[n]["original"]["bytes"] for n in names)>MAX_TOTAL_RAW_BYTES:
   raise ValueError("total original text budget exceeded before source allocation")
  for s in selectors:
   a=entries[s.source_name]["asset"];o=entries[s.source_name]["original"]
   if s.source_file_sha256!=o["sha256"] or s.typed_file_sha256!=a["typed_sha256"] or s.source_row>=a["source_rows"]:
    raise ValueError("wrong exact source/typed/occurrence binding")
  sets={};raw_states={};record_refs={}
  for name in names:
   entry=entries[name];a=entry["asset"];o=entry["original"];receipt,rref=self._receipt(name,entry)
   raw,lines,state=self._read_original(o)
   if len(lines)!=a["source_rows"]:raise ValueError("full original native row conservation required")
   binding=dict(schema="mora3.PaRoutesCompleteEvaluationSet/v1",source_name=name,source_sha256=o["sha256"],
    handoff_sha256=HANDOFF_SHA256,manifest_sha256=MANIFEST_SHA256,coverage="COMPLETE_PINNED_LITERAL_REFERENCE_SET",
    references={**refs,"receipt":rref,"original":dict(path=o["path"],bytes=o["bytes"],sha256=o["sha256"])},
    source_QC_digest=receipt["semantic_rows_digest"],fresh_original_identity=state,full_original_sha_fresh=True,
    model_use_allowed=False,task_admitted=False,
    membership_scope="exact native literal only; not hash algorithm/chemical equivalence/global novelty/heldout safety")
   sets[name]=CompleteEvaluationSet(name,raw,_canonical(binding));raw_states[name]=state;record_refs[name]=binding
  candidates=[];typed_states={}
  for name in dict.fromkeys(s.source_name for s in selectors):
   subset=[s for s in selectors if s.source_name==name];entry=entries[name];a=entry["asset"]
   rows,state=self._rows(a,tuple(s.source_row for s in subset));typed_states[name]=state
   for s in subset:
    line=sets[name].raw_lines[s.source_row]
    binding=dict(schema="mora3.PaRoutesEvaluationOccurrence/v1",handoff_sha256=HANDOFF_SHA256,
     manifest_sha256=MANIFEST_SHA256,source_name=name,complete_reference_binding_sha256=sets[name].binding_sha256,
     references=record_refs[name]["references"],typed_reference=dict(path=a["typed_path"],sha256=a["typed_sha256"],bytes=a["typed_bytes"]),
     fresh_typed_identity=state,full_typed_sha_fresh=True,model_use_allowed=False,task_admitted=False,
     representation="exact original raw line distinct from canonical parsed UTF8 payload; no chemical/hash normalization")
    candidates.append(PaEvaluationCandidate(s,line.byte_start,line.native_bytes,_canonical(rows[s.source_row]),_canonical(binding)))
  for name in names:
   if _canonical(_identity(self._path(entries[name]["original"]["path"],{".txt"}).stat()))!=_canonical(raw_states[name]):
    raise ValueError("original changed during multi-reference call")
  for name,state in typed_states.items():
   if _canonical(_identity(self._path(entries[name]["asset"]["typed_path"],{".parquet"}).stat()))!=_canonical(state):
    raise ValueError("typed source replaced/changed during multi-reference call")
  by_selector={c.selector:c for c in candidates}
  return PaEvaluationRead(tuple(sets[n] for n in names),tuple(by_selector[s] for s in selectors))
 def read_rows(self,selectors):return self.read(selectors).candidates
 def read_reference_sets(self,names):return self.read((),reference_names=names).reference_sets
 def membership(self,source_name,value):
  _literal(value)
  refs=self.read_reference_sets((source_name,));return refs[0].membership(value)
 def verify(self,candidate):
  if type(candidate) is not PaEvaluationCandidate:raise ValueError("exact official native reference occurrence required")
  checked,=self.read_rows((candidate.selector,))
  if checked.binding_sha256!=candidate.binding_sha256:raise ValueError("exact native source/binding replay mismatch")
  return checked
 def project_model_inputs(self,candidate):
  if type(candidate) is not PaEvaluationCandidate:raise ValueError("official native reference occurrence required")
  _refuse("model_input")
