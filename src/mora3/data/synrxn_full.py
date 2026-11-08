"""Pinned native SynRXN synthesis protocol candidates; no task/model admission."""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass,field
from datetime import datetime,timezone
from pathlib import Path
import csv,gzip,hashlib,io,json,os,re
BATCH="outputs/data-sources/20261101-retrosynthesis-source-v1"
HANDOFF_PATH=BATCH+"/handoff-SynRXN-three-complete-native-USPTO-protocol-tables-v1.json"
HANDOFF_SHA256="1fe4681fab6e67ce832093898fa687f0c12f72fe0b60e533879a3bb406c8ac63"
MANIFEST_PATH=BATCH+"/curated/SynRXN-three-whole-USPTO-native-QC-v1/manifest.json"
MANIFEST_SHA256="4692646a4596d24a2c55143ac955f1cfd50cdbe8c2746097fdf2ce4d367b92e2"
PLAN_SHA256="eecee6fac2b6a108f1a59d8a8e98f5274cecaf2778e048331689be487a66e9c8"
SOURCE_COMMIT="e7c03766bd6a405da468e281d922cf1beccdd56a"
RIGHTS_SHA256="5f96c10265969a0db847d99b13d055af517d6c3722352eea1bf042b907a25018"
LOWE_RECEIPT_SHA256="c4d38bba05a1e973330d6c690d3dc8576ba55e08c47d73bbbc7b6fcbb6bfb0fb"
FIXTURE_READER_SHA256="2e33707c13cfccbbfa06fcfbe324dd61e3bd389eb1bf3d9fee90249b5714c7eb"
TRAIN="SOURCE_TRAIN_CANDIDATE_PENDING_EXPOSURE"
EVAL="EVALUATION_EXPOSURE_CONTROL_ONLY"
HEADERS={"uspto_500.csv.gz":("r_id","rxn","reagent","split"),
 "uspto_50k.csv.gz":("r_id","aam","split","source"),"uspto_mit.csv.gz":("r_id","aam","split","rc")}
SPLITS={"uspto_500.csv.gz":{"train":116360,"val":12937,"test":14238},
 "uspto_50k.csv.gz":{"train":40008,"valid":5001,"test":5007},
 "uspto_mit.csv.gz":{"train":409035,"valid":30000,"test":40000}}
SOURCE_SHAS={"uspto_500.csv.gz":"fad362c125f8fc6195d49748bbd9f381ca4c9d3f03935d82f457b350ab46bef1",
 "uspto_50k.csv.gz":"a23d355405f037cc4ecc751cf094003f3b64c115dd212f20d99015bc9ba05ac4",
 "uspto_mit.csv.gz":"74dffc5bb8e5fdfb41c11676deb5fedb9bb63d28325bdfc719ea68d8b8d9f33b"}
STRING_FIELDS=("source_file_sha256","reaction_id","native_columns_json","native_cells_json","native_split","source_protocol_role",
 "gold_target_columns_json","default_predictor_columns_json","quality_reasons_json")
BOOL_FIELDS=("task_admitted","adapter_supported","model_use_allowed","strict_molecule_reaction_parse_pass","query_reaction_parse_pass")
FIELDS=STRING_FIELDS+("source_row",)+BOOL_FIELDS
MAX_HEADER_BYTES=65536
MAX_NATIVE_JSON_BYTES=32*1024**2
MAX_ROW_GROUP_BYTES=64*1024**2
FIXTURE_BASE=BATCH+"/curated/bounded-SynRXN-full-native-reader-fixture-v1"
IDENTITY_PATH=FIXTURE_BASE+"/three-source-original-typed-current-identity-and-schema-supplement-v1.json"
IDENTITY_SHA256="efc6971da61a08ec7180a6eeb24c7419bc20e5f767f1cd672f470649870e26a7"
FIXTURE_HANDOFF_PATH=FIXTURE_BASE+"/handoff-eight-native-SynRXN-USPTO-protocol-occurrences-v1.json"
FIXTURE_HANDOFF_SHA256="0b6101b85cb50136bcc29719b2ddd83ca82eca397f1ecbd75a0d2f72f6c383a4"
FIXTURE_PATH=FIXTURE_BASE+"/eight-native-SynRXN-USPTO-occurrences-two-typed-shards-v1.json"
FIXTURE_SHA256="a572d2b0d56816e7164b5211de015e7326d0a900ea010e27096c0845e8a5d73f"
def _sha(raw):return hashlib.sha256(raw).hexdigest()
def _canonical(value):return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def _int(value):return type(value) is int and value>=0
def _hex(value):return type(value) is str and re.fullmatch("[0-9a-f]{64}",value) is not None
def _strict_json(raw):
 def pairs(items):
  result={}
  for k,v in items:
   if k in result:raise ValueError("duplicate native/proof JSON key")
   result[k]=v
  return result
 def invalid(value):raise ValueError("nonfinite JSON literal")
 try:return json.loads(raw,object_pairs_hook=pairs,parse_constant=invalid)
 except (UnicodeError,json.JSONDecodeError,RecursionError) as e:raise ValueError("invalid native/proof JSON") from e
def _json_bytes(raw):
 if type(raw) is not bytes or len(raw)>MAX_NATIVE_JSON_BYTES:raise ValueError("bounded immutable canonical bytes required")
 value=_strict_json(raw)
 if _canonical(value)!=raw:raise ValueError("exact canonical native/proof bytes required")
 return value
def _identity(s):return dict(dev=s.st_dev,ino=s.st_ino,size=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
def _valid_identity(v):return type(v) is dict and set(v)=={"dev","ino","size","mtime_ns","ctime_ns"} and all(_int(x) for x in v.values())
def _utc_ns(value):
 d=datetime.fromisoformat(value)-datetime(1970,1,1,tzinfo=timezone.utc)
 return ((d.days*86400+d.seconds)*10**6+d.microseconds)*1000
def arrow_schema():
 import pyarrow as pa
 return pa.schema([(x,pa.string()) for x in STRING_FIELDS]+[("source_row",pa.int64())]+[(x,pa.bool_()) for x in BOOL_FIELDS])

@dataclass(frozen=True)
class NativeCSVHeader:
 columns:tuple
 native_bytes:bytes=field(repr=False)
 def __post_init__(self):
  if (type(self.columns) is not tuple or not self.columns or any(type(c) is not str or not c for c in self.columns)
   or len(set(self.columns))!=len(self.columns) or type(self.native_bytes) is not bytes
   or not self.native_bytes or len(self.native_bytes)>MAX_HEADER_BYTES):raise ValueError("bounded ordered unique native CSV header required")
  try:rows=list(csv.reader(io.StringIO(self.native_bytes.decode("utf8"),newline=""),delimiter=",",strict=True))
  except (csv.Error,UnicodeError) as e:raise ValueError("unsupported native CSV header") from e
  if len(rows)!=1 or tuple(rows[0])!=self.columns:raise ValueError("native header bytes/order binding mismatch")
def parse_csv_header(raw,*,expected_columns=None):
 if type(raw) is not bytes or not raw or len(raw)>MAX_HEADER_BYTES:raise ValueError("bounded complete native CSV header bytes required")
 try:rows=list(csv.reader(io.StringIO(raw.decode("utf8"),newline=""),delimiter=",",strict=True))
 except (csv.Error,UnicodeError) as e:raise ValueError("unsupported native CSV header") from e
 if len(rows)!=1:raise ValueError("exact one native CSV header record required")
 header=NativeCSVHeader(tuple(rows[0]),raw)
 if expected_columns is not None and header.columns!=expected_columns:raise ValueError("exact native table header/schema required")
 return header

@dataclass(frozen=True)
class SynRXNSelector:
 source_name:str
 source_row:int
 source_file_sha256:str
 typed_file_sha256:str
 def __post_init__(self):
  if (type(self.source_name) is not str or self.source_name not in HEADERS or not _int(self.source_row)
   or not _hex(self.source_file_sha256) or not _hex(self.typed_file_sha256)):raise ValueError("exact synthesis-table source/occurrence and original/typed SHA required")

@dataclass(frozen=True)
class SynRXNCandidate:
 selector:SynRXNSelector
 header:NativeCSVHeader
 typed_record_bytes:bytes=field(repr=False)
 binding_bytes:bytes=field(repr=False)
 def __post_init__(self):
  if (type(self.selector) is not SynRXNSelector or type(self.header) is not NativeCSVHeader
   or self.header.columns!=HEADERS[self.selector.source_name]):raise ValueError("exact native table/header selector required")
  row=_json_bytes(self.typed_record_bytes);binding=_json_bytes(self.binding_bytes)
  if (type(row) is not dict or set(row)!=set(FIELDS) or any(type(row[x]) is not str for x in STRING_FIELDS)
   or any(type(row[x]) is not bool for x in BOOL_FIELDS) or not _int(row["source_row"])
   or row["source_row"]!=self.selector.source_row or row["source_file_sha256"]!=self.selector.source_file_sha256
   or row["task_admitted"] is not False or row["adapter_supported"] is not False or row["model_use_allowed"] is not False
   or type(binding) is not dict or binding.get("schema")!="mora3.SynRXNNativeCandidateBinding/v1"
   or binding.get("handoff_sha256")!=HANDOFF_SHA256 or binding.get("manifest_sha256")!=MANIFEST_SHA256
   or binding.get("source_commit")!=SOURCE_COMMIT or binding.get("source_family")!="synthesis"
   or binding.get("source_name")!=self.selector.source_name
   or binding.get("model_use_allowed") is not False or binding.get("task_admitted") is not False):
   raise ValueError("exact native/source/version/type/closed-mask binding mismatch")
  columns=_json_bytes(row["native_columns_json"].encode());cells=_json_bytes(row["native_cells_json"].encode())
  if (type(columns) is not list or tuple(columns)!=self.header.columns or type(cells) is not list
   or len(cells)!=len(columns) or any(type(v) is not str for v in cells)):raise ValueError("complete native ordered string axes required; no null/float coercion")
  fields=dict(zip(columns,cells));split=fields["split"]
  if split not in SPLITS[self.selector.source_name] or row["native_split"]!=split:
   raise ValueError("exact source-table split label required; no val/valid aliases or invented protocol")
  role=TRAIN if split=="train" else EVAL
  gold=[c for c in columns if c in ("rc","reagent","ground_truth","label","class","reaction_center")]
  reasons=["missing_native_row_identifier"] if fields["r_id"]=="" else []
  if (row["reaction_id"]!=fields["r_id"] or row["source_protocol_role"]!=role
   or row["gold_target_columns_json"].encode()!=_canonical(gold) or row["default_predictor_columns_json"]!="[]"
   or row["quality_reasons_json"].encode()!=_canonical(reasons)):
   raise ValueError("native ID/split/role/gold-target/default-predictor/quality mask binding mismatch")
 def native_record(self):return _json_bytes(self.typed_record_bytes)
 def native_cells(self):return tuple(_strict_json(self.native_record()["native_cells_json"]))
 def native_fields(self):return dict(zip(self.header.columns,self.native_cells()))
 def binding(self):return _json_bytes(self.binding_bytes)
 def field_status(self,column):
  if type(column) is not str:raise ValueError("literal native column required")
  if column not in self.header.columns:return "not_encoded"
  return "native_empty_literal" if self.native_fields()[column]=="" else "native_present_literal"
 @property
 def native_split(self):return self.native_record()["native_split"]
 @property
 def source_protocol_role(self):return self.native_record()["source_protocol_role"]
 @property
 def default_predictor_columns(self):return tuple(_strict_json(self.native_record()["default_predictor_columns_json"]))
 @property
 def gold_target_columns(self):return tuple(_strict_json(self.native_record()["gold_target_columns_json"]))
 @property
 def quality_reasons(self):return tuple(_strict_json(self.native_record()["quality_reasons_json"]))
 @property
 def status(self):
  row=self.native_record()
  if self.quality_reasons:return "native_quality_quarantined_unadmitted"
  if not row["strict_molecule_reaction_parse_pass"] or not row["query_reaction_parse_pass"]:return "native_parser_rejected_unadmitted"
  return "native_train_candidate_pending_exposure" if self.native_split=="train" else "native_heldout_evaluation_only"
 @property
 def neural_token_count(self):return None
 @property
 def binding_sha256(self):
  return _sha(_canonical(dict(selector=self.selector.__dict__,columns=self.header.columns,header_hex=self.header.native_bytes.hex(),
   typed_hex=self.typed_record_bytes.hex(),binding_hex=self.binding_bytes.hex())))
 def require_model_use(self,purpose,*,admission=None):
  if type(purpose) is not str or not purpose:raise ValueError("explicit use purpose required")
  raise ValueError("native source candidate/official heldout has no constructor/master/model or cross-source exposure admission")

def _product_preview(candidate):
 if type(candidate) is not SynRXNCandidate or candidate.quality_reasons:raise ValueError("exact nonquarantined native source candidate required")
 row=candidate.native_record()
 if not row["strict_molecule_reaction_parse_pass"] or not row["query_reaction_parse_pass"]:
  raise ValueError("source parser-rejected representation has no product-molecule preview; no repair")
 from mora3.data import retrosynthesis_native as dependency
 if _sha(Path(dependency.__file__).read_bytes())!=FIXTURE_READER_SHA256:raise ValueError("reviewed literal-product dependency changed")
 fields=candidate.native_fields();text=fields["rxn" if "rxn" in fields else "aam"];pieces=text.split(">")
 if len(pieces)!=3 or not pieces[0] or not pieces[2]:raise ValueError("exact native three-section source reaction required")
 product=dependency.ChemicalIdentity(pieces[2]).product_input_text()
 return dict(query_role="source_derived_product_preview_not_model_admission",products=[dict(syntax="SMILES",value=product)])

class SynRXNSource:
 """Three exact synthesis/protocol tables; bounded native candidates only."""
 def __init__(self,*,root,handoff=HANDOFF_PATH,expected_manifest_sha256=MANIFEST_SHA256):
  self.root=Path(root).resolve()
  if handoff!=HANDOFF_PATH or expected_manifest_sha256!=MANIFEST_SHA256:raise ValueError("exact pinned three-table SynRXN release required")
  self._snapshot()
 def _path(self,path,suffixes):
  if type(path) is not str or not path:raise ValueError("exact pinned path required")
  p=(self.root/path).resolve()
  if not p.is_relative_to(self.root) or p.suffix.lower() not in suffixes:raise ValueError("unsafe source path/format")
  return p
 def _blob(self,path,sha,size=None):
  p=self._path(path,{".json",".payload"})
  with p.open("rb") as h:raw=h.read(8*1024**2+1)
  if not _hex(sha) or _sha(raw)!=sha or len(raw)>8*1024**2 or size is not None and len(raw)!=size:
   raise ValueError("pinned proof/rights integrity mismatch")
  return raw,dict(path=str(p),sha256=sha,bytes=len(raw))
 def _json(self,path,sha,size=None):
  raw,ref=self._blob(path,sha,size);return _strict_json(raw),ref
 def _base_snapshot(self):
  h,hr=self._json(HANDOFF_PATH,HANDOFF_SHA256,8801);m,mr=self._json(MANIFEST_PATH,MANIFEST_SHA256,3365)
  plan,pr=self._json(m["plan_path"],PLAN_SHA256)
  grant=h["source_data_rights"]["grant_primary"];_,rr=self._blob(grant["path"],RIGHTS_SHA256,8007)
  lowe=h["source_data_rights"]["source_Lowe"];lr,lrref=self._json(lowe["path"],LOWE_RECEIPT_SHA256,3261)
  _,original_rights=self._blob(lr["path"],"a3eecb1ea24628600c2cdc93e499d965e170d87d5797d33738fe7b6106e623e4",5925)
  if (h["source_commit"]!=SOURCE_COMMIT or h["source_manifest_version"]!="1.2.1 exact pinned commit"
   or plan["source_commit"]!=SOURCE_COMMIT or h["source_and_native_QC_manifest"]["sha256"]!=MANIFEST_SHA256
   or m["plan_sha256"]!=PLAN_SHA256 or grant["sha256"]!=RIGHTS_SHA256 or lowe["sha256"]!=LOWE_RECEIPT_SHA256
   or _canonical(lr["original_metadata"]["version"])!=b"1" or lr["original_metadata"]["license"]["name"]!="CC0"
   or _canonical(h["task_admitted"])!=b"0" or h["adapter_supported"] is not False or h["model_use_allowed"] is not False
   or h["no_substantive_training"] is not True or _canonical(m["task_admitted"])!=b"0"
   or m["adapter_supported"] is not False or m["model_use_allowed"] is not False
   or m["all_native_splits_preserved"] is not True or m["all_validation_test_EVAL_exposure_only"] is not True
   or _canonical(m["source_files"])!=b"3" or _canonical(m["native_record_views_nonadditive"])!=b"672586"
   or _canonical(h["native_record_views_nonadditive"])!=b"672586" or _canonical(plan["expected_records_nonadditive"])!=b"672586"
   or len(m["files"])!=3 or len(plan["assets"])!=3 or len(h["whole_native_files"])!=3):
   raise ValueError("exact source-version/rights/three-table/protocol/closed-admission proof required")
  originals={Path(a["path"]).name:a for a in plan["assets"]}
  assets={Path(a["typed_path"]).name.removesuffix(".typed.parquet"):a for a in m["files"]}
  native={a["native_file"]:a for a in h["whole_native_files"]}
  if set(originals)!=set(HEADERS) or set(assets)!=set(HEADERS) or set(native)!=set(HEADERS):raise ValueError("exact unique three synthesis-table allocation required")
  result={}
  for name in HEADERS:
   o=originals[name];a=assets[name];n=native[name];rows=sum(SPLITS[name].values())
   expected_roles={TRAIN:SPLITS[name]["train"],EVAL:rows-SPLITS[name]["train"]}
   if (o["source_key"]!="synthesis/"+name or o["license"]!="CC BY 4.0" or tuple(o["columns"])!=HEADERS[name]
    or o["sha256"]!=SOURCE_SHAS[name] or _canonical(o["expected_rows"])!=_canonical(rows)
    or _canonical(a["source_rows"])!=_canonical(rows) or _canonical(a["native_split_counts"])!=_canonical(SPLITS[name])
    or _canonical(a["source_protocol_role_counts"])!=_canonical(expected_roles)
    or n["source_raw_SHA256"]!=o["sha256"] or _canonical(n["native_columns"])!=_canonical(list(HEADERS[name]))
    or n["typed_file"]["sha256"]!=a["typed_sha256"] or _canonical(n["typed_file"]["bytes"])!=_canonical(a["typed_bytes"])
    or n["QC_receipt"]["sha256"]!=a["receipt_sha256"]):
    raise ValueError("exact native columns/split/count/typed/QC source allocation required")
   self._path(o["path"],{".gz"});self._path(a["typed_path"],{".parquet"});result[name]=dict(original=o,asset=a)
  return result,dict(handoff=hr,manifest=mr,plan=pr,rights=rr,Lowe_receipt=lrref,Lowe_original_metadata=original_rights)
 def _snapshot(self):
  entries,refs=self._base_snapshot()
  identity,ir=self._json(IDENTITY_PATH,IDENTITY_SHA256,10021)
  handoff,hr=self._json(FIXTURE_HANDOFF_PATH,FIXTURE_HANDOFF_SHA256,1174)
  manifest,_=self._json(MANIFEST_PATH,MANIFEST_SHA256,3365)
  if (identity["source_commit"]!=SOURCE_COMMIT or identity["wholeQC_manifest_sha256"]!=MANIFEST_SHA256
   or identity["wholeQC_manifest_path"]!=str((self.root/MANIFEST_PATH).resolve())
   or identity["wholeQC_completed_utc"]!=manifest["completed_utc"]
   or _canonical(handoff["task_admitted"])!=b"0" or _canonical(handoff["model_used"])!=b"0"
   or handoff["native_cell_and_typed_fingerprint_roundtrip"] is not True
   or handoff["original_late_row_serialization_reconstruction"] is not False
   or _canonical(handoff["source_rawwhole_scans"])!=b"0"
   or handoff["identity_supplement"]!={"path":str((self.root/IDENTITY_PATH).resolve()),"sha256":IDENTITY_SHA256,"bytes":10021}):
   raise ValueError("exact source-owned native/identity/closed-scope fixture proof required")
  headers={}
  for h in identity["bounded_headers"]:
   name=h["source_name"]
   if (name in headers or name not in HEADERS or _canonical(h["native_ordered_columns"])!=_canonical(list(HEADERS[name]))
    or type(h["header_decompressed_bytes_hex"]) is not str):
    raise ValueError("unique source-native header allocation required")
   raw=bytes.fromhex(h["header_decompressed_bytes_hex"])
   parsed=parse_csv_header(raw,expected_columns=HEADERS[name])
   if _sha(parsed.native_bytes)!=h["header_sha256"]:raise ValueError("exact bounded native header fingerprint required")
   headers[name]=h
  if set(headers)!=set(HEADERS):raise ValueError("all three native header/schema witnesses required")
  guards={}
  for g in identity["guards"]:
   name=g["source_name"]
   if name not in entries or (name,"original") in guards:raise ValueError("unique source identity allocation required")
   a=entries[name]["asset"];o=entries[name]["original"];receipt_info=g["QC_receipt_current_identity"]
   if (not _valid_identity(receipt_info) or g["QC_receipt_path"]!=a["receipt"] or g["QC_receipt_sha256"]!=a["receipt_sha256"]
    or _canonical(_identity(self._path(a["receipt"],{".json"}).stat()))!=_canonical(receipt_info)
    or g["current_raw_and_typed_mtime_ctime_predate_QC"] is not True
    or _canonical(g["full_source_rows_inherited"])!=_canonical(a["source_rows"])
    or _canonical(g["native_ordered_columns"])!=_canonical(list(HEADERS[name]))
    or type(g["typed_whole_hash_fresh"]) is not bool):
    raise ValueError("exact preQC/native-count/receipt identity guard required")
   for kind in ("original","typed"):
    info=g[kind+"_current_identity"]
    path=g["original_path"] if kind=="original" else g["typed_path"]
    expected_sha=g["original_expected_whole_sha256"] if kind=="original" else g["typed_whole_sha256"]
    expected=(o["path"],o["sha256"],o["bytes"]) if kind=="original" else (a["typed_path"],a["typed_sha256"],a["typed_bytes"])
    if (not _valid_identity(info) or (path,expected_sha,info["size"])!=expected
     or max(info["mtime_ns"],info["ctime_ns"])>min(receipt_info["mtime_ns"],receipt_info["ctime_ns"],_utc_ns(manifest["completed_utc"]))):
     raise ValueError("exact original/typed current identity and preQC timestamp required")
    guard=dict(path=path,file_identity=info,expected_whole_sha256=expected_sha,
     source_identity_proof=ir,QC_receipt_path=a["receipt"],QC_receipt_sha256=a["receipt_sha256"],
     native_header_sha256=headers[name]["header_sha256"],
     original_authority="inherited complete raw SHA/gzipCRC/semanticQC + current identity/preQCtimestamps/protected history")
    guards[(name,kind)]=guard
  if set(guards)!={(n,k) for n in HEADERS for k in ("original","typed")}:raise ValueError("complete three-source guard set required")
  return entries,guards,{**refs,"identity_supplement":ir,"bounded_fixture_handoff":hr}
 def _guard(self,g):
  p=self._path(g["path"],{".gz",".parquet"})
  if _canonical(_identity(p.stat()))!=_canonical(g["file_identity"]):raise ValueError("source identity changed; inherited proof cannot be reused")
  return p
 def _receipt(self,name,entry):
  a=entry["asset"];o=entry["original"];receipt,ref=self._json(a["receipt"],a["receipt_sha256"])
  rows=a["source_rows"];strict={"(True, True)":rows} if name!="uspto_mit.csv.gz" else {"(False, False)":11,"(True, True)":479024}
  expected=dict(source_path=o["path"],source_sha256=o["sha256"],source_bytes=o["bytes"],source_rows=rows,
   native_ordered_columns=list(HEADERS[name]),native_split_counts=SPLITS[name],
   typed_path=a["typed_path"],typed_sha256=a["typed_sha256"],typed_bytes=a["typed_bytes"],
   full_typed_readback_rows=rows,native_ID_unique=rows,duplicate_native_ID_occurrences=0,
   parser_acceptance_counts_strict_molecule_vs_query=strict,gzip_full_read_CRC_passed=True,
   task_admitted_rows=0,adapter_supported=False,model_use_allowed=False)
  if any(_canonical(receipt.get(k))!=_canonical(v) for k,v in expected.items()):raise ValueError("exact native/source/protocol/parser QC receipt binding required")
  return receipt,ref
 def _header(self,g,columns):
  p=self._guard(g)
  with p.open("rb") as raw:
   before=_identity(os.fstat(raw.fileno()))
   if _canonical(before)!=_canonical(g["file_identity"]):raise ValueError("raw header handle identity mismatch")
   with gzip.GzipFile(fileobj=raw,mode="rb") as stream:line=stream.readline(MAX_HEADER_BYTES+1)
   consumed=raw.tell()
   header=parse_csv_header(line,expected_columns=columns)
   if g.get("native_header_sha256") is not None and _sha(line)!=g["native_header_sha256"]:
    raise ValueError("fresh native header fingerprint mismatch")
   if _canonical(_identity(os.fstat(raw.fileno())))!=_canonical(before):raise ValueError("raw source changed during bounded header read")
  self._guard(g)
  return header,dict(header_bytes=len(line),header_sha256=_sha(line),logical_compressed_prefix_bytes_consumed=consumed,
   full_raw_SHA_fresh=False,full_gzip_CRC_rechecked=False)
 @contextmanager
 def _typed_file(self,g):
  p=self._guard(g)
  with p.open("rb") as h:
   before=_identity(os.fstat(h.fileno()));digest=hashlib.sha256()
   if _canonical(before)!=_canonical(g["file_identity"]):raise ValueError("typed handle identity mismatch")
   for block in iter(lambda:h.read(8*1024**2),b""):digest.update(block)
   if digest.hexdigest()!=g["expected_whole_sha256"]:raise ValueError("fresh selected typed integrity mismatch")
   h.seek(0);yield h
   if _canonical(_identity(os.fstat(h.fileno())))!=_canonical(before):raise ValueError("typed source changed during read")
  self._guard(g)
 def _rows(self,g,indices,total):
  import pyarrow.parquet as pq
  with self._typed_file(g) as handle:
   p=pq.ParquetFile(handle)
   if not p.schema_arrow.equals(arrow_schema(),check_metadata=True) or p.metadata.num_rows!=total:raise ValueError("exact typed schema/complete row axis required")
   wanted=set(indices);offset=0;out={}
   for group in range(p.num_row_groups):
    info=p.metadata.row_group(group);end=offset+info.num_rows;selected=[i for i in wanted if offset<=i<end]
    if selected:
     if info.total_byte_size>MAX_ROW_GROUP_BYTES:raise ValueError("native rowgroup exceeds bounded runtime; no crop")
     table=p.read_row_group(group,use_threads=False)
     for i in selected:out[i]=table.slice(i-offset,1).to_pylist()[0]
    offset=end
   if set(out)!=wanted:raise ValueError("selected native occurrence missing")
   return out
 def read_rows(self,selectors):
  if (type(selectors) is not tuple or not 1<=len(selectors)<=8 or any(type(s) is not SynRXNSelector for s in selectors)
   or len({(s.source_name,s.source_row) for s in selectors})!=len(selectors) or len({s.source_name for s in selectors})>2):
   raise ValueError("1..8unique native occurrences/atmost2typedshards required")
  entries,guards,refs=self._snapshot();out={}
  for name in dict.fromkeys(s.source_name for s in selectors):
   entry=entries[name];a=entry["asset"];o=entry["original"];subset=[s for s in selectors if s.source_name==name]
   if any(s.source_row>=a["source_rows"] or s.source_file_sha256!=o["sha256"] or s.typed_file_sha256!=a["typed_sha256"] for s in subset):
    raise ValueError("wrong exact original/typed/source-row binding")
   receipt,rref=self._receipt(name,entry);header,header_read=self._header(guards[(name,"original")],HEADERS[name])
   rows=self._rows(guards[(name,"typed")],tuple(s.source_row for s in subset),a["source_rows"])
   for s in subset:
    binding=dict(schema="mora3.SynRXNNativeCandidateBinding/v1",source_name=name,source_family="synthesis",source_commit=SOURCE_COMMIT,
     handoff_sha256=HANDOFF_SHA256,manifest_sha256=MANIFEST_SHA256,source_QC_digest=receipt["semantic_rows_digest"],
     references={**refs,"receipt":rref,"original":dict(path=o["path"],bytes=o["bytes"],sha256=o["sha256"]),
      "typed":dict(path=a["typed_path"],bytes=a["typed_bytes"],sha256=a["typed_sha256"])},
     raw_identity_guard=guards[(name,"original")],typed_identity_guard=guards[(name,"typed")],header_read=header_read,
     model_use_allowed=False,task_admitted=False,
     original_authority="inherited complete raw SHA+gzipCRC/nativeQC under exact current identity/preQCtimestamps/protected history; bounded header only",
     serialization_role="source parsed ordered CSV string cells; not original CSV quoting/row bytes",
     parser_role="native source strict-molecule/query flags, not chemical validity or feasibility",
     protocol_role="exact source labels; train candidate still requires cross-source exposure/admission; official heldouts never train")
    out[(name,s.source_row)]=SynRXNCandidate(s,header,_canonical(rows[s.source_row]),_canonical(binding))
  for name in dict.fromkeys(s.source_name for s in selectors):
   self._guard(guards[(name,"original")]);self._guard(guards[(name,"typed")])
  return tuple(out[(s.source_name,s.source_row)] for s in selectors)
 def verify(self,candidate):
  if type(candidate) is not SynRXNCandidate:raise ValueError("exact native SynRXN candidate required")
  checked,=self.read_rows((candidate.selector,))
  if checked.binding_sha256!=candidate.binding_sha256:raise ValueError("exact native/source/header/binding replay mismatch")
  return checked
 def project_product_query(self,candidate):return _product_preview(self.verify(candidate))
 def project_model_inputs(self,candidate):
  if type(candidate) is not SynRXNCandidate:raise ValueError("native candidate required")
  candidate.require_model_use("model_input")
