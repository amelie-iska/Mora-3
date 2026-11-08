"""Source-bound PaRoutes native reaction/template tables; no model admission."""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass,field
from datetime import datetime,timezone
import csv,gzip,hashlib,io,json,os,re
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
IDENTITY_PATH=BATCH+"/curated/bounded-PaRoutes-table-native-reader-fixture-v1/six-table-original-and-typed-current-identity-supplement-v1.json"
IDENTITY_SHA256="7fbde520cc3058b34777e48ff219e45127a0542ac2f784d752d9afe353d48a13"
FIXTURE_READER_SHA256="2e33707c13cfccbbfa06fcfbe324dd61e3bd389eb1bf3d9fee90249b5714c7eb"
CANDIDATE="CANDIDATE_REQUIRES_HELDOUT_AND_CHEMICAL_GROUP_EXCLUSION"
COMPUTED="COMPUTED_RULE_CONTEXT_NOT_INDEPENDENT_EXPERIMENT"
TABLE_HEADERS={
 "selected_reactions.csv":("RxnSmilesClean","PseudoHash","classification","RingBreaker"),
 "selected_reactions_all.csv":("id","source","date","rsmi","classification","rsmi_processed","BadMolecules","BadMolecules2",
  "NReactants","NMappedReactants","NReagents","NMappedReagents","NProducts","NMappedProducts","PseudoHash","ElementCount",
  "ProductSize","UnmappedProdAtoms","WidowAtoms","HasUnmappedRadicalAtom","HasUnsanitizableReactants","NRingChange","RingBondMade","RingMadeSize","RingBreaker"),
 "uspto_template_library.csv":("reaction_smiles","reaction_hash","classification","ring_breaker","retro_template","template_hash"),
 "uspto_ringbreaker_template_library.csv":("reaction_smiles","reaction_hash","classification","ring_breaker","retro_template","template_hash"),
 "uspto_unique_templates.csv.gz":("template_code","retro_template","template_hash","classification","library_occurence"),
 "uspto_ringbreaker_unique_templates.csv.gz":("template_code","retro_template","template_hash","classification","library_occurence")}
TABLE_ROLES={name:CANDIDATE if name.startswith("selected_reactions") else COMPUTED for name in TABLE_HEADERS}
REACTION_FIELDS={
 "selected_reactions.csv":(("RxnSmilesClean","source_cleaned_reaction_text"),),
 "selected_reactions_all.csv":(("rsmi","source_raw_reaction_text"),("rsmi_processed","source_processed_reaction_text")),
 "uspto_template_library.csv":(("reaction_smiles","source_reaction_reference_in_computed_rule"),),
 "uspto_ringbreaker_template_library.csv":(("reaction_smiles","source_reaction_reference_in_computed_rule"),),
 "uspto_unique_templates.csv.gz":(),"uspto_ringbreaker_unique_templates.csv.gz":()}
FIELDS=("source_file_sha256","source_row","source_role","payload_json","quality_reasons_json",
 "source_reference_hash_exposure","task_admitted","adapter_supported")
MAX_HEADER_BYTES=65536
MAX_NATIVE_ROW_JSON_BYTES=32*1024**2
MAX_ROW_GROUP_BYTES=256*1024**2
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
 if type(raw) is not bytes:raise ValueError("immutable canonical native JSON required")
 value=_strict_json(raw)
 if _canonical(value)!=raw:raise ValueError("exact canonical native JSON bytes required")
 return value
def _hex(value):return type(value) is str and re.fullmatch("[0-9a-f]{64}",value) is not None
def _int(value):return type(value) is int and value>=0
def _identity(stat):return dict(dev=stat.st_dev,ino=stat.st_ino,size=stat.st_size,mtime_ns=stat.st_mtime_ns,ctime_ns=stat.st_ctime_ns)
def _valid_identity(value):return type(value) is dict and set(value)=={"dev","ino","size","mtime_ns","ctime_ns"} and all(_int(v) for v in value.values())
def _utc_ns(value):
 d=datetime.fromisoformat(value)-datetime(1970,1,1,tzinfo=timezone.utc)
 return ((d.days*86400+d.seconds)*10**6+d.microseconds)*1000
def _stream_sha(handle):
 h=hashlib.sha256()
 for block in iter(lambda:handle.read(8*1024**2),b""):h.update(block)
 return h.hexdigest()

@dataclass(frozen=True)
class NativeTSVHeader:
 columns:tuple
 delimiter:str="\t"
 native_bytes:bytes|None=field(default=None,repr=False)
 def __post_init__(self):
  if (type(self.columns) is not tuple or not self.columns or any(type(c) is not str or not c for c in self.columns)
      or len(set(self.columns))!=len(self.columns) or type(self.delimiter) is not str or self.delimiter!="\t"
      or self.native_bytes is not None and type(self.native_bytes) is not bytes):
   raise ValueError("exact ordered unique native TSV header and literal tab delimiter required")
  if self.native_bytes is not None:
   if not self.native_bytes or len(self.native_bytes)>MAX_HEADER_BYTES:raise ValueError("bounded complete native header required")
   try:rows=list(csv.reader(io.StringIO(self.native_bytes.decode("utf8"),newline=""),delimiter="\t",strict=True))
   except (csv.Error,UnicodeError) as error:raise ValueError("unsupported native header bytes") from error
   if len(rows)!=1 or tuple(rows[0])!=self.columns:raise ValueError("native header bytes/declared column binding mismatch")
def parse_tsv_header(raw,*,delimiter="\t"):
 if type(raw) is not bytes or not raw or len(raw)>MAX_HEADER_BYTES or type(delimiter) is not str or delimiter!="\t":
  raise ValueError("bounded exact native TSV header bytes/tab delimiter required")
 try:rows=list(csv.reader(io.StringIO(raw.decode("utf8"),newline=""),delimiter="\t",strict=True))
 except (csv.Error,UnicodeError) as error:raise ValueError("unsupported native TSV header") from error
 if len(rows)!=1:raise ValueError("one complete native TSV header required")
 return NativeTSVHeader(tuple(rows[0]),delimiter,raw)

@dataclass(frozen=True)
class PaTableSelector:
 source_name:str
 source_row:int
 source_file_sha256:str
 typed_file_sha256:str
 def __post_init__(self):
  if (type(self.source_name) is not str or self.source_name not in TABLE_HEADERS or not _int(self.source_row)
      or not _hex(self.source_file_sha256) or not _hex(self.typed_file_sha256)):
   raise ValueError("exact native table source/row and original/typed hash bindings required")
@dataclass(frozen=True)
class NativeStringEvidence:
 column:str
 syntax:str
 native_text:str
 source_role:str

@dataclass(frozen=True)
class PaTableCandidate:
 selector:PaTableSelector
 header:NativeTSVHeader
 typed_record_bytes:bytes=field(repr=False)
 binding_bytes:bytes=field(repr=False)
 def __post_init__(self):
  if type(self.selector) is not PaTableSelector or type(self.header) is not NativeTSVHeader or self.header.columns!=TABLE_HEADERS[self.selector.source_name]:
   raise ValueError("exact versioned native header ordering/source binding required")
  row=_json_bytes(self.typed_record_bytes);binding=_json_bytes(self.binding_bytes)
  if (type(row) is not dict or set(row)!=set(FIELDS) or type(binding) is not dict
      or binding.get("schema")!="mora3.PaRoutesNativeTableBinding/v1" or binding.get("handoff_sha256")!=HANDOFF_SHA256
      or binding.get("manifest_sha256")!=MANIFEST_SHA256 or row["source_file_sha256"]!=self.selector.source_file_sha256
      or not _int(row["source_row"]) or row["source_row"]!=self.selector.source_row or row["source_role"]!=TABLE_ROLES[self.selector.source_name]
      or type(row["payload_json"]) is not str or len(row["payload_json"].encode())>MAX_NATIVE_ROW_JSON_BYTES
      or type(row["quality_reasons_json"]) is not str
      or row["source_reference_hash_exposure"] is not None and type(row["source_reference_hash_exposure"]) is not bool
      or row["task_admitted"] is not False or row["adapter_supported"] is not False):
   raise ValueError("native table/source/value/type/mask binding mismatch")
  cells=_json_bytes(row["payload_json"].encode());reasons=_json_bytes(row["quality_reasons_json"].encode())
  if type(cells) is not list or any(type(c) is not str for c in cells) or type(reasons) is not list or any(type(r) is not str or not r for r in reasons):
   raise ValueError("complete literal native string cell array and quality reasons required")
  if len(cells)!=len(self.header.columns) and "native_TSV_width_mismatch" not in reasons:
   raise ValueError("native width discrepancy must retain source quarantine")
 def native_record(self):return _json_bytes(self.typed_record_bytes)
 def native_cells(self):return tuple(_strict_json(self.native_record()["payload_json"]))
 def native_fields(self):
  values=self.native_cells()
  if len(values)!=len(self.header.columns):raise ValueError("native width mismatch; no guessed padding/dropping")
  return dict(zip(self.header.columns,values))
 def binding(self):return _json_bytes(self.binding_bytes)
 def field_status(self,column):
  if type(column) is not str:raise ValueError("native column name required")
  if column not in self.header.columns:return "not_encoded"
  if len(self.native_cells())!=len(self.header.columns):return "unresolved_native_width_mismatch"
  return "native_empty_literal" if self.native_fields()[column]=="" else "native_present_literal"
 def reaction_evidence(self):
  fields=self.native_fields()
  return tuple(NativeStringEvidence(c,"REACTION_SMILES",fields[c],role) for c,role in REACTION_FIELDS[self.selector.source_name])
 def template_evidence(self):
  fields=self.native_fields()
  return (NativeStringEvidence("retro_template","SMARTS",fields["retro_template"],"source_computed_template_rule"),) if "retro_template" in fields else ()
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
 def binding_sha256(self):
  return _sha(_canonical(dict(selector=self.selector.__dict__,header_columns=self.header.columns,delimiter=self.header.delimiter,
   header_native_hex=None if self.header.native_bytes is None else self.header.native_bytes.hex(),
   typed_hex=self.typed_record_bytes.hex(),binding_hex=self.binding_bytes.hex())))
 def require_model_use(self,purpose,*,admission=None):
  if type(purpose) is not str or not purpose:raise ValueError("explicit model-use purpose required")
  if self.quality_reasons:raise ValueError("source parsing/query-chemistry quarantine cannot grant supervision")
  raise ValueError("native reaction/template candidate has no constructor/master/model or cross-source heldout admission; computed templates and reference nonmembership are not measured labels or safety proof")
def _product_query(candidate,*,reaction_field=None):
 from mora3.data.retrosynthesis_native import ChemicalIdentity
 if type(candidate) is not PaTableCandidate or candidate.quality_reasons:raise ValueError("nonquarantined native reaction cell required")
 evidence=candidate.reaction_evidence()
 if reaction_field is None:
  if len(evidence)!=1:raise ValueError("explicit native raw/processed reaction field required; template SMARTS is not a molecule")
  selected=evidence[0]
 else:
  if type(reaction_field) is not str:raise ValueError("exact native reaction field required")
  selected=next((e for e in evidence if e.column==reaction_field),None)
  if selected is None:raise ValueError("native reaction-SMILES role required; never infer molecule from SMARTS/template/class")
 pieces=selected.native_text.split(">")
 if len(pieces)!=3 or not pieces[0] or not pieces[2]:raise ValueError("complete native reactant/agent/product sections required; no repair")
 product=ChemicalIdentity(pieces[2]).product_input_text()
 return dict(query_role="product_conditioned_chemical_retrosynthesis",products=[dict(syntax="SMILES",value=product)])

class PaRoutesReactionTablesSource:
 """Six native TSV tables, bounded cell candidates; all model/task use closed."""
 def __init__(self,*,root,handoff=HANDOFF_PATH,expected_manifest_sha256=MANIFEST_SHA256):
  self.root=Path(root).resolve()
  if handoff!=HANDOFF_PATH or expected_manifest_sha256!=MANIFEST_SHA256:raise ValueError("exact versioned full PaRoutes source required")
  self._snapshot()
 def _path(self,path,suffixes):
  if type(path) is not str or not path:raise ValueError("exact source path required")
  p=(self.root/path).resolve()
  if not p.is_relative_to(self.root) or p.suffix.lower() not in suffixes:raise ValueError("unsafe source path/format; no pickle/generated execution")
  return p
 def _json(self,path,sha,size=None):
  p=self._path(path,{".json"});raw=p.read_bytes()
  if len(raw)>8*1024**2 or not _hex(sha) or _sha(raw)!=sha or size is not None and len(raw)!=size:raise ValueError("pinned source proof integrity mismatch")
  return _strict_json(raw),dict(path=str(p),sha256=sha,bytes=len(raw))
 def _snapshot(self):
  h,hr=self._json(HANDOFF_PATH,HANDOFF_SHA256,13107);m,mr=self._json(MANIFEST_PATH,MANIFEST_SHA256,12293)
  plan,pr=self._json(m["plan_path"],PLAN_SHA256);available,ar=self._json(AVAILABLE_PATH,AVAILABLE_SHA256)
  proof,fr=self._json(h["handoff_proof"]["path"],PROOF_SHA256,97338);identity,ir=self._json(IDENTITY_PATH,IDENTITY_SHA256,16926)
  if (h["complete_native_manifests"]["PaRoutes"]["sha256"]!=MANIFEST_SHA256 or m["plan_sha256"]!=PLAN_SHA256
      or plan["code_and_input_bindings"].get(str((self.root/AVAILABLE_PATH).resolve()))!=AVAILABLE_SHA256
      or _canonical(h["source_worker_task_admitted"])!=b"0" or h["model_use_allowed"] is not False or h["adapter_supported"] is not False
      or _canonical(m["task_admitted_rows"])!=b"0" or m["adapter_supported"] is not False
      or proof["source_manifests"]["PaRoutes"]["sha256"]!=MANIFEST_SHA256 or proof["no_new_task_admission"] is not True
      or proof["no_missing_or_inprogress_payloads"] is not True
      or _canonical(m["source_files"])!=b"15" or len(m["assets"])!=15 or len(plan["assets"])!=15
      or _canonical(m["source_rows_nonadditive"])!=b"6032740"
      or identity["whole_QC_manifest_sha256"]!=MANIFEST_SHA256 or identity["whole_QC_completed_utc"]!=m["completed_utc"]
      or identity["no_fresh_raw_table_hashes"] is not True):
   raise ValueError("exact full-release/source-QC/proof/closed-admission chain required")
  originals={Path(a["path"]).name:a for a in plan["assets"]};catalog={}
  for a in m["assets"]:
   name=Path(a["typed_path"]).name.removesuffix(".typed.parquet")
   if name in catalog or name not in originals or not _int(a["source_rows"]) or a["quality_pass_rows"]+a["quarantined_rows"]!=a["source_rows"]:
    raise ValueError("inconsistent native source allocation")
   catalog[name]=dict(asset=a,original=originals[name])
  if len(catalog)!=15:raise ValueError("complete15source catalog required")
  guards={}
  for g in identity["guards"]:
   key=(g["source_name"],g["kind"])
   if key in guards or g["source_name"] not in TABLE_HEADERS or g["kind"] not in ("original","typed"):raise ValueError("exact table identity guard allocation required")
   a=catalog[g["source_name"]]["asset"];o=catalog[g["source_name"]]["original"];typed=g["kind"]=="typed"
   expected=(a["typed_path"],a["typed_sha256"],a["typed_bytes"]) if typed else (o["path"],o["sha256"],o["bytes"])
   info=g["file_identity"];receipt_info=g["QC_receipt_file_identity"]
   if (not _valid_identity(info) or not _valid_identity(receipt_info) or (g["path"],g["expected_whole_sha256"],info["size"])!=expected
       or g["QC_receipt_sha256"]!=a["receipt_sha256"] or g["QC_receipt_path"]!=a["receipt_path"]
       or g["fresh_whole_sha256"] is not None and g["fresh_whole_sha256"]!=expected[1]
       or g["current_mtime_ctime_predate_individual_QC_receipt"] is not True or g["current_mtime_ctime_predate_final_QC_completion"] is not True
       or max(info["mtime_ns"],info["ctime_ns"])>min(receipt_info["mtime_ns"],receipt_info["ctime_ns"],_utc_ns(m["completed_utc"]))):
    raise ValueError("unsupported inherited identity/timestamp/QC lineage")
   guards[key]=g
  if set(guards)!={(n,k) for n in TABLE_HEADERS for k in ("original","typed")}:raise ValueError("complete six-table identity guard required")
  return catalog,guards,dict(handoff=hr,manifest=mr,plan=pr,available=ar,proof=fr,identity_supplement=ir)
 def catalog(self):
  c,_,_=self._snapshot()
  return tuple(dict(source_name=n,source_rows=v["asset"]["source_rows"],source_role=v["asset"]["source_role"],
   source_file_sha256=v["original"]["sha256"],typed_file_sha256=v["asset"]["typed_sha256"],
   supported_view="native_TSV_table" if n in TABLE_HEADERS else "unimplemented_closed") for n,v in c.items())
 def _guard(self,g):
  p=self._path(g["path"],{".csv",".gz",".parquet"})
  if _canonical(_identity(p.stat()))!=_canonical(g["file_identity"]):raise ValueError("source identity changed; inherited proof cannot be reused")
  return p
 def _receipt(self,name,entry):
  a=entry["asset"];o=entry["original"];receipt,ref=self._json(a["receipt_path"],a["receipt_sha256"])
  expected=dict(source_path=o["path"],source_sha256=o["sha256"],source_bytes=o["bytes"],source_rows=a["source_rows"],
   typed_path=a["typed_path"],typed_sha256=a["typed_sha256"],typed_bytes=a["typed_bytes"],source_role=TABLE_ROLES[name],
   quality_pass_rows=a["quality_pass_rows"],quarantined_rows=a["quarantined_rows"],full_typed_readback_rows=a["source_rows"],
   native_TSV_ordered_columns=list(TABLE_HEADERS[name]),source_native_format="TSV",task_admitted_rows=0,adapter_supported=False)
  if any(_canonical(receipt.get(k))!=_canonical(v) for k,v in expected.items()):raise ValueError("exact table/header/source-QC receipt binding mismatch")
  return receipt,ref
 def _header(self,g,columns):
  p=self._guard(g)
  with p.open("rb") as raw:
   before=_identity(os.fstat(raw.fileno()))
   if _canonical(before)!=_canonical(g["file_identity"]):raise ValueError("source header handle identity mismatch")
   if p.suffix==".gz":
    with gzip.GzipFile(fileobj=raw,mode="rb") as stream:line=stream.readline(MAX_HEADER_BYTES+1)
   else:line=raw.readline(MAX_HEADER_BYTES+1)
   physical=raw.tell()
   if _canonical(_identity(os.fstat(raw.fileno())))!=_canonical(before):raise ValueError("source changed during bounded header read")
  self._guard(g);header=parse_tsv_header(line)
  if header.columns!=tuple(columns):raise ValueError("original native TSV header/order/delimiter mismatch")
  return header,dict(native_header_bytes=len(line),logical_source_prefix_bytes_consumed=physical,header_sha256=_sha(line),
   serialization_role="decompressed native header prefix" if p.suffix==".gz" else "literal original TSV header prefix",
   full_raw_hash_performed=False,full_gzip_CRC_recheck=False)
 @contextmanager
 def _typed_file(self,g):
  p=self._guard(g)
  with p.open("rb") as handle:
   before=_identity(os.fstat(handle.fileno()))
   if _canonical(before)!=_canonical(g["file_identity"]) or _stream_sha(handle)!=g["expected_whole_sha256"]:raise ValueError("fresh selected typed integrity mismatch")
   handle.seek(0);yield handle
   if _canonical(_identity(os.fstat(handle.fileno())))!=_canonical(before):raise ValueError("typed file changed during read")
   self._guard(g)
 def _rows(self,g,indices,total):
  import pyarrow as pa,pyarrow.parquet as pq
  with self._typed_file(g) as handle:
   p=pq.ParquetFile(handle)
   expected=pa.schema(list(zip(FIELDS,[pa.string(),pa.int64(),pa.string(),pa.string(),pa.string(),pa.bool_(),pa.bool_(),pa.bool_()])))
   if not p.schema_arrow.equals(expected,check_metadata=True) or p.metadata.num_rows!=total:raise ValueError("exact native typed table schema/axis required")
   wanted=set(indices);offset=0;out={}
   for group in range(p.num_row_groups):
    info=p.metadata.row_group(group);end=offset+info.num_rows;local=[i for i in wanted if offset<=i<end]
    if local:
     if info.total_byte_size>MAX_ROW_GROUP_BYTES:raise ValueError("native rowgroup exceeds bounded runtime; no crop")
     table=p.read_row_group(group,use_threads=False)
     for i in local:out[i]=table.slice(i-offset,1).to_pylist()[0]
    offset=end
   if set(out)!=wanted:raise ValueError("selected table occurrence missing")
   return out
 def read_rows(self,selectors):
  if (type(selectors) is not tuple or not 1<=len(selectors)<=8 or any(type(s) is not PaTableSelector for s in selectors)
      or len({(s.source_name,s.source_row) for s in selectors})!=len(selectors) or len({s.source_name for s in selectors})>2):
   raise ValueError("1..8 unique table occurrences across atmost2typed table shards required")
  catalog,guards,refs=self._snapshot();out={}
  for name in dict.fromkeys(s.source_name for s in selectors):
   entry=catalog[name];a=entry["asset"];o=entry["original"];subset=[s for s in selectors if s.source_name==name]
   if any(s.source_file_sha256!=o["sha256"] or s.typed_file_sha256!=a["typed_sha256"] or s.source_row>=a["source_rows"] for s in subset):
    raise ValueError("wrong exact original/typed/source-row binding")
   receipt,rref=self._receipt(name,entry);header,header_read=self._header(guards[(name,"original")],receipt["native_TSV_ordered_columns"])
   rows=self._rows(guards[(name,"typed")],tuple(s.source_row for s in subset),a["source_rows"]);self._guard(guards[(name,"original")])
   for selector in subset:
    binding=dict(schema="mora3.PaRoutesNativeTableBinding/v1",handoff_sha256=HANDOFF_SHA256,manifest_sha256=MANIFEST_SHA256,
     serialization_role="canonical parsed native TSV cell array; not original source row serialization",source_QC_digest=receipt["semantic_rows_digest"],
     references={**refs,"receipt":rref,"original":dict(path=o["path"],bytes=o["bytes"],sha256=o["sha256"]),
      "typed":dict(path=a["typed_path"],bytes=a["typed_bytes"],sha256=a["typed_sha256"])},
     raw_identity_guard=guards[(name,"original")],typed_identity_guard=guards[(name,"typed")],header_read=header_read,
     raw_authority="prior completeQC wholeSHA under exact current identity/OS timestamp/protected history guard; only header prefix read",
     native_reaction_field_roles=list(REACTION_FIELDS[name]),template_role="source_computed_SMARTS_rule_not_molecular_label",
     model_use_allowed=False,task_admitted=False,chemistry_scope="source parsing/query representation flags, not chemical validity or experimental truth")
    out[(name,selector.source_row)]=PaTableCandidate(selector,header,_canonical(rows[selector.source_row]),_canonical(binding))
  return tuple(out[(s.source_name,s.source_row)] for s in selectors)
 def verify(self,candidate):
  if type(candidate) is not PaTableCandidate:raise ValueError("typed native table candidate required")
  checked,=self.read_rows((candidate.selector,))
  if checked.binding_sha256!=candidate.binding_sha256:raise ValueError("exact native table source replay mismatch")
  return checked
 def project_product_inputs(self,candidate,*,reaction_field=None):
  checked=self.verify(candidate)
  from mora3.data import retrosynthesis_native as dependency
  if _sha(Path(dependency.__file__).read_bytes())!=FIXTURE_READER_SHA256:raise ValueError("reviewed literal-product dependency changed; compatibility review required")
  return _product_query(checked,reaction_field=reaction_field)
