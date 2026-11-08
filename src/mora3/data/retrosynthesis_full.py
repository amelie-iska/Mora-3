"""Pinned whole-release ORD native candidates; no task/model admission.

Safe JSON/Parquet only. Full-source semantic QC is inherited through exact pinned
evidence; selected original protobuf bytes remain authoritative and are replayed.
"""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass,field
import hashlib,io,json,os,re
from pathlib import Path

BATCH="outputs/data-sources/20261101-retrosynthesis-source-v1"
HANDOFF_PATH=BATCH+"/handoff-complete-ORD-PaRoutes-native-reaction-sources-v1.json"
HANDOFF_SHA256="62ea114eeab722e8363884b4f3704143b7f0d38928791b01d2fa39694dad72d7"
MANIFEST_PATH=BATCH+"/curated/ORD-whole-93-native-semantic-QC-v1/manifest.json"
MANIFEST_SHA256="7aadf1919e7e2650cf58c3a9023bdead0fc89d035d5d72197c03aa47341324d1"
PROOF_SHA256="c4f0bfe52bab987929db580a5b8b57e33e92d2c7edd21abe08a32c456edfb119"
PROTOCOL_SHA256="ee445b4c2020b63be62cdcaaedb0dfb3c6f5c0ae6fed75c8a7f02a6d57762be6"
AVAILABLE_SHA256="f934d654f7439b0e991381c7c196cc7e2dac6d355cdd1430b5baa3646dfba542"
PLAN_SHA256="208fddbf1508c4fd693a2f32cd0febbd688087d74a35776df3a33f0a2cf7dc18"
DESCRIPTOR_SHA256="f33ac383b4c0517355188a6e2b4d77bc679dc9ea553b16d467043e478f4f16dd"
SOURCE_REVISION="b6c89761dacf809d4082acbdbebe72042a08db91"
SCHEMA_REVISION="a9c1603a4fb040f251c1f174548d1747838fb247"
FIXTURE_READER_SHA256="2e33707c13cfccbbfa06fcfbe324dd61e3bd389eb1bf3d9fee90249b5714c7eb"
MAX_ROW_GROUP_BYTES=256*1024**2
TYPED_FIELDS=("dataset_id","source_file_sha256","source_row","reaction_id","source_proto_sha256",
 "source_proto_bytes","semantic_json","quality_reasons_json","unknown_wire_fields",
 "semantic_json_roundtrip_equal","task_admitted","adapter_supported")
def _sha(raw):return hashlib.sha256(raw).hexdigest()
def _canonical(value):return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def _strict_json(raw):
 def pairs(items):
  result={}
  for k,v in items:
   if k in result:raise ValueError("duplicate native JSON key")
   result[k]=v
  return result
 def invalid(value):raise ValueError("nonfinite native JSON literal")
 return json.loads(raw,object_pairs_hook=pairs,parse_constant=invalid)
def _canonical_json(raw):
 if type(raw) is not bytes:raise ValueError("immutable canonical native JSON bytes required")
 value=_strict_json(raw)
 if _canonical(value)!=raw:raise ValueError("exact source canonical JSON bytes required")
 return value
def _hex(value):return type(value) is str and re.fullmatch("[0-9a-f]{64}",value) is not None
def _integer(value):return type(value) is int and value>=0
def _stream_sha(handle):
 h=hashlib.sha256()
 for block in iter(lambda:handle.read(8*1024**2),b""):h.update(block)
 return h.hexdigest()
def _identity(stat):return (stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)

@dataclass(frozen=True)
class ORDFileRef:
 path:str
 sha256:str
 bytes:int|None=None
 def __post_init__(self):
  if (type(self.path) is not str or not self.path or not _hex(self.sha256)
      or self.bytes is not None and (type(self.bytes) is not int or self.bytes<=0)):
   raise ValueError("exact typed source file path/SHA/size required")
 def mapping(self):return dict(path=self.path,sha256=self.sha256,bytes=self.bytes)

@dataclass(frozen=True)
class ORDOccurrenceSelector:
 dataset_id:str
 source_row:int
 source_file_sha256:str
 typed_file_sha256:str
 def __post_init__(self):
  if (type(self.dataset_id) is not str or re.fullmatch("ord_dataset-[0-9a-f]{32}",self.dataset_id) is None
      or not _integer(self.source_row) or not _hex(self.source_file_sha256) or not _hex(self.typed_file_sha256)):
   raise ValueError("exact typed ORD dataset/source occurrence and original/typed file hashes required")

@dataclass(frozen=True)
class ORDNativeCandidate:
 selector:ORDOccurrenceSelector
 source_proto:bytes=field(repr=False)
 typed_record_bytes:bytes=field(repr=False)
 binding_bytes:bytes=field(repr=False)
 def __post_init__(self):
  if type(self.selector) is not ORDOccurrenceSelector or type(self.source_proto) is not bytes or len(self.source_proto)>64*1024**2:
   raise ValueError("typed bounded source-native protobuf bytes required")
  row=_canonical_json(self.typed_record_bytes);binding=_canonical_json(self.binding_bytes)
  if type(row) is not dict or set(row)!=set(TYPED_FIELDS) or type(binding) is not dict:
   raise ValueError("complete exact native typed record/binding required")
  if (binding.get("schema")!="mora3.FullORDNativeCandidateBinding/v1"
      or binding.get("handoff_sha256")!=HANDOFF_SHA256 or binding.get("manifest_sha256")!=MANIFEST_SHA256
      or type(binding.get("source_protocol_role")) is not str):
   raise ValueError("exact full-release candidate chain binding required; no admission")
  if (row["dataset_id"]!=self.selector.dataset_id or row["source_file_sha256"]!=self.selector.source_file_sha256
      or not _integer(row["source_row"]) or row["source_row"]!=self.selector.source_row
      or type(row["reaction_id"]) is not str or not _integer(row["source_proto_bytes"])
      or row["source_proto_bytes"]!=len(self.source_proto) or row["source_proto_sha256"]!=_sha(self.source_proto)
      or row["task_admitted"] is not False or row["adapter_supported"] is not False
      or type(row["semantic_json_roundtrip_equal"]) is not bool
      or row["unknown_wire_fields"] is not None and type(row["unknown_wire_fields"]) is not bool
      or row["semantic_json"] is not None and type(row["semantic_json"]) is not str
      or type(row["quality_reasons_json"]) is not str):
   raise ValueError("native source/row/protobuf/value/type/mask binding mismatch")
  reasons=_canonical_json(row["quality_reasons_json"].encode())
  if type(reasons) is not list or any(type(x) is not str or not x for x in reasons):
   raise ValueError("exact native quality-reason list required")
  semantic=None if row["semantic_json"] is None else _canonical_json(row["semantic_json"].encode())
  if semantic is not None and type(semantic) is not dict:raise ValueError("native semantic object or absent required")
  if not reasons and (row["unknown_wire_fields"] is not False or row["semantic_json_roundtrip_equal"] is not True
      or semantic is None or not row["reaction_id"] or semantic.get("reaction_id")!=row["reaction_id"]):
   raise ValueError("passing semantic/source reaction identity requires exact full QC flags")
  if semantic is not None and semantic.get("reaction_id")!=row["reaction_id"] and "column_message_reaction_id_mismatch" not in reasons:
   raise ValueError("native semantic/source ID discrepancy must retain source quarantine reason")
 def native_record(self):return _canonical_json(self.typed_record_bytes)
 def semantic(self):
  value=self.native_record()["semantic_json"]
  return None if value is None else _canonical_json(value.encode())
 def binding(self):return _canonical_json(self.binding_bytes)
 @property
 def quality_reasons(self):return tuple(_strict_json(self.native_record()["quality_reasons_json"]))
 @property
 def status(self):return "native_quarantined_unadmitted" if self.quality_reasons else "native_candidate_unadmitted"
 @property
 def source_protocol_role(self):return self.binding()["source_protocol_role"]
 @property
 def neural_token_count(self):return None
 def field_present(self,path):
  if type(path) is not tuple or any(type(k) is not str for k in path):raise ValueError("exact native field-key path required")
  current=self.semantic()
  for key in path:
   if type(current) is not dict or key not in current:return False
   current=current[key]
  return True
 @property
 def binding_sha256(self):
  return _sha(_canonical(dict(selector=self.selector.__dict__,source_proto_hex=self.source_proto.hex(),
   typed_record_hex=self.typed_record_bytes.hex(),binding_hex=self.binding_bytes.hex())))
 def require_model_use(self,purpose,*,admission=None):
  if type(purpose) is not str or not purpose:raise ValueError("explicit model-use purpose required")
  if self.quality_reasons:raise ValueError("native ORD quality quarantine cannot construct model supervision")
  if purpose=="training" and ("OFFICIAL_TEST_" in self.source_protocol_role or "OFFICIAL_VALIDATION_" in self.source_protocol_role):
   raise ValueError("official source heldout/evaluation exposure cannot enter training")
  raise ValueError("ORD native candidate has no constructor/master/model task admission or cross-corpus heldout proof")

def _product_query(candidate,*,product_pointers=None,known_conditions=()):
 """Literal query projection only. Public reader projection first replays source."""
 from mora3.data.retrosynthesis_native import ChemicalIdentity,ReactionCondition
 if type(candidate) is not ORDNativeCandidate or candidate.quality_reasons:raise ValueError("nonquarantined native candidate query required")
 semantic=candidate.semantic();available={}
 for oi,outcome in enumerate(semantic.get("outcomes",[])):
  for pi,product in enumerate(outcome.get("products",[])):
   if product.get("reaction_role")!="PRODUCT":continue
   for ii,identifier in enumerate(product.get("identifiers",[])):
    if identifier.get("type")=="SMILES" and type(identifier.get("value")) is str:
     available[f"/outcomes/{oi}/products/{pi}/identifiers/{ii}"]=identifier["value"]
 if product_pointers is None:
  if len(available)!=1:raise ValueError("one unambiguous native SMILES product or explicit product selector required")
  product_pointers=tuple(available)
 if (type(product_pointers) is not tuple or not product_pointers or len(product_pointers)>8
     or any(type(p) is not str or p not in available for p in product_pointers)
     or len(set(product_pointers))!=len(product_pointers) or type(known_conditions) is not tuple
     or any(type(c) is not ReactionCondition or c.role!="known_input" or c.availability!="explicit_caller_known" for c in known_conditions)):
  raise ValueError("source product-only/caller-known context allowlist required")
 out=dict(query_role="product_conditioned_chemical_retrosynthesis",products=[
  dict(syntax="SMILES",value=ChemicalIdentity(available[p]).product_input_text()) for p in product_pointers])
 if known_conditions:out["known_conditions"]=[c.as_mapping() for c in known_conditions]
 return out

class FullORDNativeSource:
 """Explicit whole93-release catalog, bounded replay; never task admission."""
 def __init__(self,*,root,handoff=HANDOFF_PATH,expected_manifest_sha256=MANIFEST_SHA256):
  self.root=Path(root).resolve()
  if handoff!=HANDOFF_PATH or expected_manifest_sha256!=MANIFEST_SHA256:raise ValueError("exact pinned complete ORD release required")
  self._snapshot()
 def _path(self,ref,suffixes):
  if type(ref) is not ORDFileRef:raise ValueError("typed exact native file reference required")
  path=(self.root/ref.path).resolve()
  if not path.is_relative_to(self.root) or path.suffix.lower() not in suffixes:
   raise ValueError("unsafe/unsupported native source path/format; no pickle/generated-code execution")
  return path
 def _json(self,path,sha256,size=None):
  ref=ORDFileRef(str(path),sha256,size);p=self._path(ref,{".json"})
  raw=p.read_bytes()
  if len(raw)>8*1024**2 or _sha(raw)!=ref.sha256 or ref.bytes is not None and len(raw)!=ref.bytes:
   raise ValueError("pinned source JSON integrity mismatch")
  return _strict_json(raw),ORDFileRef(str(p),sha256,len(raw))
 def _snapshot(self):
  hand,href=self._json(HANDOFF_PATH,HANDOFF_SHA256,13107)
  manifest,mref=self._json(MANIFEST_PATH,MANIFEST_SHA256,71998)
  proof,pref=self._json(hand["handoff_proof"]["path"],PROOF_SHA256,97338)
  protocol,rref=self._json(hand["ORD"]["protocol_map"]["path"],PROTOCOL_SHA256,86851)
  available,aref=self._json(manifest["original_available_manifest"],AVAILABLE_SHA256)
  plan,nref=self._json(manifest["plan_path"],PLAN_SHA256)
  if (hand["complete_native_manifests"]["ORD"]["sha256"]!=MANIFEST_SHA256
      or hand["handoff_proof"]["sha256"]!=PROOF_SHA256 or hand["ORD"]["protocol_map"]["sha256"]!=PROTOCOL_SHA256
      or manifest["original_available_manifest_sha256"]!=AVAILABLE_SHA256 or manifest["plan_sha256"]!=PLAN_SHA256
      or proof["source_manifests"]["ORD"]["sha256"]!=MANIFEST_SHA256
      or hand["source_worker_task_admitted"]!=0 or type(hand["source_worker_task_admitted"]) is not int
      or hand["model_use_allowed"] is not False or hand["adapter_supported"] is not False
      or manifest["task_admitted_rows"]!=0 or type(manifest["task_admitted_rows"]) is not int or manifest["adapter_supported"] is not False
      or proof["no_missing_or_inprogress_payloads"] is not True or proof["no_new_task_admission"] is not True
      or available["whole_original_LFS_SHA_size_and_GitHub_pointer_matches"] is not True
      or available["revision"]!=SOURCE_REVISION or hand["ORD"]["primary_schema_revision"]!=SCHEMA_REVISION
      or plan["reaction_pb2_sha256"]!=DESCRIPTOR_SHA256
      or len(manifest["assets"])!=93 or manifest["source_files"]!=93 or type(manifest["source_files"]) is not int
      or manifest["source_rows"]!=2428291 or type(manifest["source_rows"]) is not int
      or manifest["native_semantic_quality_pass"]!=2428287 or manifest["quarantined_rows"]!=4
      or len(available["assets"])!=93 or len(protocol["files"])!=93):
   raise ValueError("exact source/schema/QC/protocol/full-release closed-gate binding required")
  descriptor=ORDFileRef(plan["reaction_pb2_source"],DESCRIPTOR_SHA256)
  dp=self._path(descriptor,{".payload"})
  if dp.stat().st_size>1024**2 or _sha(dp.read_bytes())!=DESCRIPTOR_SHA256:raise ValueError("pinned descriptor bytes mismatch; never executed")
  originals={a["id"]:a for a in available["assets"]};roles={p["id"]:p for p in protocol["files"]};catalog={}
  if len(originals)!=93 or len(roles)!=93:raise ValueError("duplicate native source/protocol allocation")
  for a in manifest["assets"]:
   dataset=Path(a["typed_path"]).name.removesuffix(".typed.parquet")
   if (dataset in catalog or dataset not in originals or dataset not in roles
       or not _integer(a["source_rows"]) or a["source_rows"]!=roles[dataset]["footer_rows"]
       or a["native_semantic_quality_pass"]+a["quarantined_rows"]!=a["source_rows"]
       or roles[dataset]["path"]!=originals[dataset]["path"] or roles[dataset]["sha256"]!=originals[dataset]["sha256"]):
    raise ValueError("native source/typed/protocol occurrence allocation mismatch")
   catalog[dataset]=dict(asset=a,original=originals[dataset],protocol=roles[dataset])
  refs=dict(handoff=href.mapping(),manifest=mref.mapping(),proof=pref.mapping(),protocol=rref.mapping(),available=aref.mapping(),
   plan=nref.mapping(),descriptor=dict(path=str(dp),sha256=DESCRIPTOR_SHA256,bytes=dp.stat().st_size))
  return catalog,refs
 def catalog(self):
  catalog,_=self._snapshot()
  return tuple(dict(dataset_id=k,source_rows=v["asset"]["source_rows"],source_file_sha256=v["original"]["sha256"],
   typed_file_sha256=v["asset"]["typed_sha256"],source_protocol_role=v["protocol"]["source_protocol_role"]) for k,v in catalog.items())
 @contextmanager
 def _verified_parquet(self,ref):
  path=self._path(ref,{".parquet"})
  with path.open("rb") as handle:
   before=_identity(os.fstat(handle.fileno()))
   if ref.bytes is None or before[2]!=ref.bytes or _stream_sha(handle)!=ref.sha256:
    raise ValueError("native selected Parquet source/typed byte integrity mismatch")
   handle.seek(0)
   yield handle
   if _identity(os.fstat(handle.fileno()))!=before or _identity(path.stat())!=before:
    raise ValueError("native source changed/replaced during bounded read")
 def _rows(self,ref,indices,expected_rows,*,typed,metadata=None,source_schema=None):
  import pyarrow as pa,pyarrow.parquet as pq
  with self._verified_parquet(ref) as handle:
   parquet=pq.ParquetFile(handle);schema=parquet.schema_arrow
   if typed:
    expected=pa.schema([(n,t) for n,t in zip(TYPED_FIELDS,[pa.string(),pa.string(),pa.int64(),pa.string(),pa.string(),
     pa.int64(),pa.string(),pa.string(),pa.bool_(),pa.bool_(),pa.bool_(),pa.bool_()])])
    if not schema.equals(expected,check_metadata=True):raise ValueError("exact native typed ORD schema required")
   else:
    if schema.names!=["reaction_id","reaction"] or not pa.types.is_string(schema.field(0).type) or not pa.types.is_binary(schema.field(1).type):
     raise ValueError("exact original ORD column schema required")
    observed={k.decode():v.decode() for k,v in (schema.metadata or {}).items()}
    if type(source_schema) is not str or str(schema)!=source_schema:
     raise ValueError("exact original source schema/nullability/metadata binding required")
    if _canonical(observed)!=_canonical(metadata):raise ValueError("native original source metadata binding mismatch")
   if parquet.metadata.num_rows!=expected_rows or any(i>=expected_rows for i in indices):raise ValueError("native occurrence outside source axis")
   wanted=set(indices);offset=0;out={}
   for group in range(parquet.num_row_groups):
    info=parquet.metadata.row_group(group);end=offset+info.num_rows;local=[i for i in wanted if offset<=i<end]
    if local:
     if info.total_byte_size>MAX_ROW_GROUP_BYTES:raise ValueError("native row group exceeds bounded runtime; no crop")
     table=parquet.read_row_group(group,use_threads=False)
     for i in local:out[i]=table.slice(i-offset,1).to_pylist()[0]
    offset=end
   if set(out)!=wanted:raise ValueError("missing original/typed selected occurrence")
   return out
 def read_occurrences(self,selectors):
  if (type(selectors) is not tuple or not 1<=len(selectors)<=8 or any(type(s) is not ORDOccurrenceSelector for s in selectors)
      or len({(s.dataset_id,s.source_row) for s in selectors})!=len(selectors) or len({s.dataset_id for s in selectors})>2):
   raise ValueError("1..8 unique native occurrences across at most2 typed shards required")
  catalog,refs=self._snapshot();out={}
  for dataset in dict.fromkeys(s.dataset_id for s in selectors):
   if dataset not in catalog:raise ValueError("unrelated native dataset outside pinned full ORD catalog")
   entry=catalog[dataset];a=entry["asset"];original=entry["original"];protocol=entry["protocol"]
   subset=[s for s in selectors if s.dataset_id==dataset]
   for s in subset:
    if s.source_file_sha256!=original["sha256"] or s.typed_file_sha256!=a["typed_sha256"] or s.source_row>=a["source_rows"]:
     raise ValueError("wrong original/typed native source/row binding")
   receipt,receiptref=self._json(a["receipt_path"],a["receipt_sha256"])
   comparisons=dict(source_id=dataset,source_path=original["path"],source_sha256=original["sha256"],source_bytes=original["bytes"],
    source_rows=a["source_rows"],typed_path=a["typed_path"],typed_sha256=a["typed_sha256"],typed_bytes=a["typed_bytes"],
    full_typed_readback_rows=a["source_rows"],native_semantic_quality_pass=a["native_semantic_quality_pass"],quarantined_rows=a["quarantined_rows"],
    task_admitted_rows=0,adapter_supported=False,source_metadata=protocol["native_metadata"])
   if any(_canonical(receipt.get(k))!=_canonical(v) for k,v in comparisons.items()):raise ValueError("exact selected whole-QC receipt/source/typed/protocol binding mismatch")
   original_ref=ORDFileRef(original["path"],original["sha256"],original["bytes"])
   typed_ref=ORDFileRef(a["typed_path"],a["typed_sha256"],a["typed_bytes"]);indices=tuple(s.source_row for s in subset)
   typed_rows=self._rows(typed_ref,indices,a["source_rows"],typed=True)
   originals=self._rows(original_ref,indices,a["source_rows"],typed=False,metadata=receipt["source_metadata"],source_schema=receipt["source_schema"])
   for selector in subset:
    row=typed_rows[selector.source_row];native=originals[selector.source_row]
    if (type(native["reaction_id"]) is not str or type(native["reaction"]) is not bytes or row.get("reaction_id")!=native["reaction_id"]):
     raise ValueError("original/typed native row ID/protobuf binding mismatch")
    binding=dict(schema="mora3.FullORDNativeCandidateBinding/v1",handoff_sha256=HANDOFF_SHA256,manifest_sha256=MANIFEST_SHA256,
     source_revision=SOURCE_REVISION,schema_revision=SCHEMA_REVISION,source_protocol_role=protocol["source_protocol_role"],
     original_native_metadata=receipt["source_metadata"],semantic_full_QC_digest=receipt["semantic_rows_digest"],
     references={**refs,**dict(original=original_ref.mapping(),typed=typed_ref.mapping(),receipt=receiptref.mapping())},
     source_roles="native protobuf authority; semantic JSON source-QC projection; no chemistry/experiment/independence assertion",
     task_admitted=False,model_use_allowed=False)
    out[(dataset,selector.source_row)]=ORDNativeCandidate(selector,native["reaction"],_canonical(row),_canonical(binding))
  return tuple(out[(s.dataset_id,s.source_row)] for s in selectors)
 def verify(self,candidate):
  if type(candidate) is not ORDNativeCandidate:raise ValueError("typed full ORD native candidate required")
  checked,=self.read_occurrences((candidate.selector,))
  if checked.binding_sha256!=candidate.binding_sha256:raise ValueError("exact native source replay/binding mismatch")
  return checked
 def project_product_inputs(self,candidate,*,product_pointers=None,known_conditions=()):
  checked=self.verify(candidate)
  from mora3.data import retrosynthesis_native as dependency
  if _sha(Path(dependency.__file__).read_bytes())!=FIXTURE_READER_SHA256:
   raise ValueError("reviewed product/context projection dependency changed; explicit compatibility review required")
  return _product_query(checked,product_pointers=product_pointers,known_conditions=known_conditions)
