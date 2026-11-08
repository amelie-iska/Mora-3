"""Whole-native RNAcentral task candidates; independent proof/admission pending.

Literal source symbols are targets. Native case/T/U and ambiguity are never
normalized. This module does not return an admitted NucleotidePriorView.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

SOURCE="RNAcentral-reference-priors"
TASK="source_native_nucleotide_sequence_prior"
DOMAIN="RNA_reference_native_symbols"
REVISION="officialRNAcentral28.0active+inactive"
POLICY="referenceprior_trainonly;knownheldoutguards;homology_and_genuine_contextheldout_unresolved"
EVIDENCE="releasedRNAreference_sequenceprior;notnewindependentexperiment"
ALPHABET=frozenset("ACGTURYSWKMBDHVNacgturyswkmbdhvn")
CONDITIONS=dict(prior_domain=DOMAIN,source_organism=None)
MISSING=["native_neural_RNAprior_reader","nativeheader_taxonomy_function_scope_verification",
 "homology_and_genuine_heldout_protocol","cellular_response_or_imagepair"]
CONTEXT=dict(source_status_notactivitytruth=True,sourceIDsheadersauditonly=True,
 all_unlisted_annotations_unknown=True,physical_material_modification_context_known=False,
 source_symbol_knownmask_rule="biological exact-base objective only where native.upper in ACGTU; source ambiguity and case remain observed native symbols; no imputation")
CENSUS_SHA256="d6240751e8ba21744bd9869ccac0c4ea74809bbaa3dca3c456144b176e1020bd"
BASE="outputs/data-intake/20261101-RNAcentral-full-native-corpus-v1"
FIXTURE=dict(path=BASE+"/closed-native-reader-fixture-v1/manifest.json",bytes=18802,
 sha256="7af114e8897a07d712146e3a35d70ea3a2eb5c890fb980edc525916c41c2d029")
_COMPLEMENT=str.maketrans("ACGTRYSWKMBDHVNX","TGCAYRSWMKVHDBNX")

def _canonical(value):
 return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
def _sha(raw):return hashlib.sha256(raw).hexdigest()
def _is_sha(value):return type(value) is str and re.fullmatch("[0-9a-f]{64}",value) is not None
def _integer(value):return type(value) is int and value>=0
def _text(value):return type(value) is str and bool(value)
def _json(raw):
 if type(raw) not in (str,bytes):raise ValueError("serialized native JSON required")
 def pairs(items):
  out={}
  for k,v in items:
   if k in out:raise ValueError("duplicate native JSON key")
   out[k]=v
  return out
 def invalid(value):raise ValueError("nonfinite native JSON")
 return json.loads(raw,object_pairs_hook=pairs,parse_constant=invalid)
def _same(actual,expected,message):
 if _canonical(actual)!=_canonical(expected):raise ValueError(message)
def _groups(sequence):
 if type(sequence) is not str or not sequence or set(sequence)-ALPHABET:
  raise ValueError("complete source-native IUPAC sequence required; no crop/repair/X replacement")
 upper=sequence.upper();base=upper.replace("U","T");rc=base.translate(_COMPLEMENT)[::-1]
 return dict(native_sha=_sha(sequence.encode()),upper_sha=_sha(upper.encode()),rc_sha=_sha(min(base,rc).encode()))
def _task_id(native,reuse_kind="new"):
 domain="RNA_family_reference_native_symbols" if reuse_kind=="Rfam_existing" else DOMAIN
 return _sha(_canonical([domain,None,native]))
def _model_gate():
 raise ValueError("whole RNAcentral independent source/task proof and authoritative master admission pending")

@dataclass(frozen=True)
class WholeRNAcentralCandidate:
 sequence: str
 condition_bytes: bytes
 provenance_bytes: bytes
 def __post_init__(self):
  g=_groups(self.sequence)
  if type(self.condition_bytes) is not bytes or type(self.provenance_bytes) is not bytes:
   raise ValueError("immutable source/condition provenance bytes required")
  if self.condition_bytes!=_canonical(CONDITIONS):raise ValueError("target-free reference-prior conditions required")
  p=_json(self.provenance_bytes)
  if (p.get("sequence_sha256")!=g["native_sha"] or p.get("condition_sha256")!=_sha(self.condition_bytes)
      or p.get("whole_master_admitted") is not False):
   raise ValueError("native target/condition pending-admission binding mismatch")
 def inputs(self):return _json(self.condition_bytes)
 def input_bytes(self):return self.condition_bytes
 def native_bytes(self):return self.sequence.encode()
 def provenance(self):return _json(self.provenance_bytes)
 def target_masks(self):
  return (True,)*len(self.sequence),tuple(c.upper() in "ACGTU" for c in self.sequence)
 def check_context_capacity(self,*,max_symbols):
  if type(max_symbols) is not int or max_symbols<1:raise ValueError("explicit positive integer context capacity required")
  if len(self.sequence)>max_symbols:raise ValueError("complete native axis exceeds caller context capacity; no crop")
 def require_model_use(self,purpose,*,admission=None):_model_gate()
 def as_neural_prior(self):_model_gate()

@dataclass(frozen=True)
class WholeRNAcentralSelector:
 kind: str
 indices: tuple
 def __post_init__(self):
  if (self.kind not in ("new","reuse") or type(self.indices) is not tuple or not 1<=len(self.indices)<=16
      or any(not _integer(i) for i in self.indices) or len(set(self.indices))!=len(self.indices)):
   raise ValueError("explicit unique bounded native candidate selectors required")

def _source_record(source,source_file,offset,native_row):
 if (type(source) is not dict or type(source_file) is not dict
     or not _integer(offset) or not _integer(native_row)
     or not _integer(source_file.get("rows")) or native_row>=source_file["rows"]
     or source_file.get("component") not in ("active","inactive")
     or not _text(source_file.get("path")) or not _is_sha(source_file.get("sha256"))
     or not _integer(source_file.get("bytes")) or source_file["bytes"]<1
     or not _text(source_file.get("original_FASTA"))):
  raise ValueError("exact native source-file allocation/offset required")
 seq=source["native_sequence"];g=_groups(seq);header=source["native_header"]
 if not _text(header) or not header.split():raise ValueError("complete original native header required")
 identifier=header.split()[0];index=offset+native_row;status=source_file["component"]
 rfam=re.search(r"(?<![A-Za-z0-9])RF[0-9]{5}(?![0-9])",header)
 expected=dict(source_record_id=f"official_RNAcentral28_{status}:DSE3-RNACENTRAL28-{status}:{index}",
  source_record_index=index,source_payload=source_file["original_FASTA"],source_identifier=identifier,native_header=header,
  native_sequence=seq,native_length=len(seq),native_sequence_sha256=g["native_sha"],uppercase_sequence_sha256=g["upper_sha"],
  sequence_alphabet_pass=True,native_T_count=seq.upper().count("T"),native_U_count=seq.upper().count("U"),
  ambiguity_symbol_count=sum(c.upper() not in "ACGTU" for c in seq),explicit_Rfam_ID_in_header=rfam[0] if rfam else None,
  source_role="released RNAfamily/reference-sequence evidence; exact family/region join requires source metadata",task_admitted=False)
 _same(source,expected,"literal native row/header/source ID/axis/mask/types mismatch")
 ref=dict(native_payload=source_file["path"],native_payload_sha256=source_file["sha256"],native_row_index=native_row,
  original_record_index=index,original_FASTA=source_file["original_FASTA"],source_identifier=identifier,native_header=header,
  source_status=status,source_release="RNAcentral28.0",source_rights="CC0-1.0",original_source_record_id=expected["source_record_id"])
 return seq,g,ref

def rnacentral_full_candidate(row,native_record,occurrence,state,index_row,identifiers,group_row,*,
 source_file,source_file_ordinal,source_record_offset,binding):
 """Strict in-memory native recipe validation; file hashes are reader responsibility."""
 try:
  return _project(row,native_record,occurrence,state,index_row,identifiers,group_row,source_file,
   source_file_ordinal,source_record_offset,binding)
 except (KeyError,TypeError,UnicodeError,OverflowError,AttributeError) as exc:
  raise ValueError("malformed whole RNAcentral native candidate") from exc

def _project(row,source,occ,state,index,identifiers,group,source_file,file_ordinal,offset,binding):
 if not _integer(file_ordinal) or type(occ) is not dict or not _integer(occ.get("native_row_index")):
  raise ValueError("exact source ordinal/occurrence row required")
 local=occ["native_row_index"];seq,g,ref=_source_record(source,source_file,offset,local)
 tid=_task_id(g["native_sha"])
 if (type(identifiers) is not list or not identifiers or any(not _text(x) or any(c.isspace() for c in x) for x in identifiers)
     or identifiers!=sorted(set(identifiers)) or ref["source_identifier"] not in identifiers):
  raise ValueError("literal unique ordered native identifier memberships required")
 expected_state=dict(native_sha=bytes.fromhex(g["native_sha"]),uppercase_sha=bytes.fromhex(g["upper_sha"]),
  rc_sha=bytes.fromhex(g["rc_sha"]),task_id=tid,reuse_kind="new",first_file=file_ordinal,first_row=local,
  occurrences=state["occurrences"])
 if not _integer(state["occurrences"]) or state["occurrences"]<1:raise ValueError("positive exact source multiplicity required")
 for key in ("native_sha","uppercase_sha","rc_sha"):
  if type(state.get(key)) is not bytes or state[key]!=expected_state[key]:raise ValueError("typed literal native/guard digest required")
 _same({k:v for k,v in state.items() if k not in ("native_sha","uppercase_sha","rc_sha")},
  {k:v for k,v in expected_state.items() if k not in ("native_sha","uppercase_sha","rc_sha")},"new identity/first-occurrence/reuse state mismatch")
 expected_occ=dict(source_file_ordinal=file_ordinal,native_row_index=local,original_record_index=source["source_record_index"],
  source_identifier=source["source_identifier"],native_sha=bytes.fromhex(g["native_sha"]),uppercase_sha=bytes.fromhex(g["upper_sha"]),
  rc_guard_sha=bytes.fromhex(g["rc_sha"]),native_length=len(seq),source_alphabet_pass=True,first_native_occurrence=True,
  reuse_kind="new",task_identity_digest=bytes.fromhex(tid),local_existing_heldout_group_match=False,nonstandard_symbols=None)
 for key in ("native_sha","uppercase_sha","rc_guard_sha","task_identity_digest"):
  if type(occ.get(key)) is not bytes or occ[key]!=expected_occ[key]:raise ValueError("exact typed occurrence/native task digest required")
 _same({k:v for k,v in occ.items() if k not in ("native_sha","uppercase_sha","rc_guard_sha","task_identity_digest")},
  {k:v for k,v in expected_occ.items() if k not in ("native_sha","uppercase_sha","rc_guard_sha","task_identity_digest")},
  "source occurrence/type/reuse/heldout allocation mismatch")
 if (type(index.get("task_id")) is not bytes or index["task_id"]!=bytes.fromhex(tid)
     or type(index.get("native_sha")) is not bytes or index["native_sha"]!=bytes.fromhex(g["native_sha"])
     or not _integer(index.get("shard")) or not _integer(index.get("local_row"))):
  raise ValueError("exact closed-shard serialization index binding required")
 expected_index=dict(task_id=index["task_id"],native_sha=index["native_sha"],source_file=file_ordinal,source_row=local,
  reuse_kind="new",shard=index["shard"],local_row=index["local_row"])
 _same({k:v for k,v in index.items() if k not in ("task_id","native_sha")},
  {k:v for k,v in expected_index.items() if k not in ("task_id","native_sha")},"source/task index location mismatch")
 groups=sorted({"nucleotide_native:"+g["native_sha"],"nucleotide_upper:"+g["upper_sha"],
  "nucleotide_derived_RC_baseorder:"+g["rc_sha"]}|{"rnacentral_native_identifier:"+x for x in identifiers})
 _same(group,dict(task_id=tid,effective_groups_json=group["effective_groups_json"]),"sidecar source allocation mismatch")
 _same(_json(group["effective_groups_json"]),groups,"closed group/URS membership mismatch")
 audit=dict(source_ref=ref,source_role="releasedRNAreference_sequence_prior;notindependentexperimentalmeasurement",
  source_record_status_is_not_activity_label=True,source_symbols="nativecase_T_U_IUPACretainedwithoutcroporrepair",
  source_symbol_loss_mask="allnativevalidcharactersobserved;biologicalbase-resolutionlossonlyupperACGTU;ambiguitynotimputed",
  excluded_predictors=["sourceidentifiers_headers_status_classifications","targetsequence_length_hash","sourceauditmetadata"],
  native_reader_supported=False)
 payload=_canonical(dict(schema="mora3.CorpusTask/v1",task=TASK,conditions=CONDITIONS,
  targets=dict(source_native_sequence=seq),audit=audit,missing=MISSING))
 if type(row.get("native_payload")) is not bytes or row["native_payload"]!=payload:
  raise ValueError("exact native task/audit/condition/target/missingness bytes mismatch")
 expected_envelope=dict(task_id=tid,record_id="RNAcentral28-native-sequence:"+g["native_sha"],source_family=SOURCE,
  source_revision=REVISION,source_sha256=source_file["sha256"],task_name=TASK,native_schema="mora3.CorpusTask/v1",
  native_payload_sha256=_sha(payload),split="train",split_policy=POLICY,evidence_class=EVIDENCE,
  admitted_for_task=True,model_adapter_supported=False,full_axis_values=len(seq))
 excluded={"native_payload","admission_receipt_json","missing_context_json","condition_context_json","group_ids_json"}
 _same({k:v for k,v in row.items() if k not in excluded},expected_envelope,"exact envelope source/train/types/axis mismatch")
 receipt=dict(source_ref=ref,native_sequence_sha256=g["native_sha"],exact_full_axis=True,
  source_case_T_U_and_ambiguity_preserved=True,source_license="CC0-1.0",source_status_is_not_activity_label=True,
  new_independent_experiments=0,task_admission="eligible native reference prior; full independent master proof pending",
  record_ID_scope="literal native sequence identity; original source record ID and all occurrences retained separately")
 _same(_json(row["admission_receipt_json"]),receipt,"exact receipt/source/types/admission scope mismatch")
 _same(_json(row["condition_context_json"]),CONTEXT,"target-free context/mask role mismatch")
 _same(_json(row["missing_context_json"]),MISSING,"native missingness scope mismatch")
 _same(_json(row["group_ids_json"]),groups,"complete literal identity/derived guard/URS groups mismatch")
 if (type(binding) is not dict or set(binding)!={"census_sha256","task_shard_sha256","groups_sha256","source_shard_sha256","occurrence_sha256"}
     or any(not _is_sha(v) for v in binding.values()) or binding["census_sha256"]!=CENSUS_SHA256
     or binding["source_shard_sha256"]!=source_file["sha256"]):
  raise ValueError("exact frozen census/closed task/group/source/occurrence references required")
 p=dict(schema="mora3.WholeRNAcentralNativeCandidate/v1",sequence_sha256=g["native_sha"],condition_sha256=_sha(_canonical(CONDITIONS)),
  task_id=tid,source_ref=ref,envelope={k:v for k,v in row.items() if k!="native_payload"},source_native_record=source,
  native_payload=_json(payload),effective_groups=groups,source_identity_occurrences=state["occurrences"],binding=binding,
  closed_shard_index=index["shard"],closed_shard_local_row=index["local_row"],source_file_ordinal=file_ordinal,
  whole_master_admitted=False,independent_whole_source_task_proof=False,neural_model_tokens=None,actual_model_use_allowed=False,
  source_record_status_is_not_activity_label=True,derived_T_U_RC_guard_is_material_identity=False,
  homology_independence_established=False,source_file_bytes_verified=False)
 return WholeRNAcentralCandidate(seq,_canonical(CONDITIONS),_canonical(p))


@dataclass(frozen=True)
class WholeRNAcentralReuseWitness:
 sequence: str
 reuse_kind: str
 existing_task_id: str
 provenance_bytes: bytes
 def __post_init__(self):
  g=_groups(self.sequence)
  if (self.reuse_kind not in ("RNAcentral_pilot_existing","Rfam_existing")
      or self.existing_task_id!=_task_id(g["native_sha"],self.reuse_kind)
      or type(self.provenance_bytes) is not bytes):
   raise ValueError("literal native identity/existing prior reuse required")
  p=_json(self.provenance_bytes)
  if p.get("new_task_serialized") is not False or p.get("whole_master_admitted") is not False:
   raise ValueError("reuse is not duplicate allocation or whole-source admission")
 def native_bytes(self):return self.sequence.encode()
 def provenance(self):return _json(self.provenance_bytes)
 def as_new_candidate(self):raise ValueError("existing prior reuse cannot serialize a duplicate new task")
 def require_model_use(self,purpose,*,admission=None):_model_gate()

def _typed_native(value):
 if type(value) is bytes:return {"native_bytes_hex":value.hex()}
 if type(value) is dict:return {k:_typed_native(v) for k,v in value.items()}
 if type(value) in (tuple,list):return [_typed_native(v) for v in value]
 return value
def _same_native(actual,expected,message):_same(_typed_native(actual),_typed_native(expected),message)
def _occurrence_bytes(occ):
 out=dict(occ)
 for key in ("native_sha","uppercase_sha","rc_guard_sha","task_identity_digest"):
  if not _is_sha(out.get(key)):raise ValueError("typed hexadecimal fixture occurrence digest required")
  out[key]=bytes.fromhex(out[key])
 return out

class WholeRNAcentralNativeSource:
 """Pinned closed-prefix candidate reader; no future/master admission guessing."""
 def __init__(self,*,root,manifest=FIXTURE["path"],expected_manifest_sha256=FIXTURE["sha256"]):
  self.root=Path(root).resolve();self.manifest=Path(manifest)
  if expected_manifest_sha256!=FIXTURE["sha256"]:raise ValueError("exact immutable closed native fixture required")
  self._snapshot()
 def _verified(self,ref,*,suffix):
  if (type(ref) is not dict or not _text(ref.get("path")) or not _is_sha(ref.get("sha256"))
      or not _integer(ref.get("bytes")) or ref["bytes"]<1):
   raise ValueError("typed exact source file/hash/size reference required")
  p=(self.root/ref["path"]).resolve()
  if not p.is_relative_to(self.root) or p.suffix.lower()!=suffix or suffix not in (".json",".parquet"):
   raise ValueError("safe native JSON/Parquet only; no FASTA/pickle/SQLite/opaque-format fallback")
  raw=p.read_bytes()
  if len(raw)!=ref["bytes"] or _sha(raw)!=ref["sha256"]:raise ValueError("closed source/census/shard integrity mismatch")
  return raw
 def _rows(self,ref):
  import pyarrow as pa
  import pyarrow.parquet as pq
  raw=self._verified(ref,suffix=".parquet")
  table=pq.read_table(pa.BufferReader(raw),use_threads=False)
  if "rows" in ref and (not _integer(ref["rows"]) or table.num_rows!=ref["rows"]):
   raise ValueError("native fixture row allocation mismatch")
  return table.to_pylist()
 def _row(self,ref,index):
  import pyarrow as pa
  import pyarrow.parquet as pq
  if not _integer(index):raise ValueError("exact native shard ordinal required")
  pf=pq.ParquetFile(pa.BufferReader(self._verified(ref,suffix=".parquet")))
  if index>=pf.metadata.num_rows:raise ValueError("closed shard row outside native allocation")
  for group in range(pf.num_row_groups):
   n=pf.metadata.row_group(group).num_rows
   if index<n:return pf.read_row_group(group,use_threads=False).slice(index,1).to_pylist()[0]
   index-=n
  raise ValueError("closed native row missing")
 def _snapshot(self):
  ref=dict(FIXTURE,path=str(self.manifest))
  m=_json(self._verified(ref,suffix=".json"))
  if (m.get("schema")!="mora3.ClosedRNAcentralNativeReaderFixture/v1"
      or any(m.get(k) is not False for k in ("global_source_task_proof_complete","master_union_admission","actual_model_use_allowed","model_training"))
      or m.get("current_authoritative_master")!="v22" or m["source_census"]["sha256"]!=CENSUS_SHA256):
   raise ValueError("candidate fixture must preserve pending global proof/master/model gates")
  census=_json(self._verified(m["source_census"],suffix=".json"))
  if (census.get("schema")!="mora3.FullRNAcentralNativeCensus/v1"
      or census.get("status")!="complete_full_source_census_pending_materialization_and_independent_proof"
      or census["source_counts"]["source_occurrences"]!=61026468
      or census["new_native_sequences_eligible_after_known_heldout_guards"]!=54732469):
   raise ValueError("exact frozen whole native census required")
  _same(m["frozen_census_index"],census["native_identity_URS_index"],"frozen identity/URS index receipt mismatch")
  plan=_json(self._verified(census["build_plan"],suffix=".json"))
  natives=self._rows(m["native_source_records"])
  if len(m["new_task_cases"])!=3 or len(m["old_reuse_witnesses"])!=2 or len(natives)!=5:
   raise ValueError("exact bounded task/reuse/source allocation required")
  task_ids=[x["selected_group_row"]["task_id"] for x in m["new_task_cases"]]+[x["existing_task_id"] for x in m["old_reuse_witnesses"]]
  if len(set(task_ids))!=5:raise ValueError("duplicate new/reused task allocation")
  return m,census,plan,natives
 @staticmethod
 def _source_binding(case,census):
  f=case["native_source_file"];o=case["census_occurrence_file"];ordinal=o["source_file_ordinal"]
  if not _integer(ordinal):raise ValueError("literal native source file ordinal required")
  entries=census["source_file_dictionary"]
  if ordinal>=len(entries):raise ValueError("source outside whole census")
  entry=entries[ordinal]
  _same(entry,dict(source_file_ordinal=ordinal,source=f,occurrences=o),"source/occurrence exact census membership mismatch")
  offset=sum(x["source"]["rows"] for x in entries[:ordinal] if x["source"]["component"]==f["component"])
  return ordinal,offset
 @staticmethod
 def _native(case,natives):
  occ=case["selected_occurrence"];sha=occ["native_sha"]
  found=[r for r in natives if r["native_sequence_sha256"]==sha and r["source_record_index"]==occ["original_record_index"]
   and r["source_identifier"]==occ["source_identifier"]]
  if len(found)!=1:raise ValueError("exact copied source occurrence required")
  return found[0]
 def _document_ref(self,path,digest):
  p=(self.root/path).resolve()
  if not p.is_relative_to(self.root) or p.suffix.lower()!=".json":raise ValueError("retained source QC JSON required")
  return dict(path=str(p),bytes=p.stat().st_size,sha256=digest)
 def _lineage(self,case,census,plan):
  handoff=_json(self._verified(census["source_handoff"],suffix=".json"))
  f=case["native_source_file"];component=handoff["components"][f["component"]]
  native_ref=self._document_ref(component["native_manifest"],component["native_manifest_sha256"])
  proof_ref=self._document_ref(component["independent_full_readback"],component["independent_full_readback_sha256"])
  native=_json(self._verified(native_ref,suffix=".json"));proof=_json(self._verified(proof_ref,suffix=".json"))
  if (handoff["available_source_occurrences"]!=61026468 or handoff["task_admitted_records"]!=0
      or any(handoff["full_independent_replay"].get(k) is not True for k in
       ("all_compressed_source_SHA_checked","whole_source_gzip_CRC_read_to_EOF","all_typed_file_SHA_and_all_columns_read",
        "all_native_headers_full_sequences_source_order_exact","all_native_and_normalized_sequence_SHA_length_sourceIDs_recomputed"))):
   raise ValueError("full-native source replay receipt scope mismatch")
  if (proof["source_native_manifest_sha256"]!=component["native_manifest_sha256"]
      or native["source_asset"]["path"]!=f["original_FASTA"] or proof["source_path"]!=f["original_FASTA"]
      or native["source_asset"]["sha256"]!=component["source_sha256"] or proof["source_sha256"]!=component["source_sha256"]
      or native["source_asset"]["license"]!="CC0-1.0"
      or any(native.get(k) is not True for k in ("whole_gzip_stream_scanned","full_native_sequence_axis_preserved","no_T_U_conversion","source_sequence_case_preserved","all_original_headers_preserved"))
      or any(proof.get(k) is not True for k in ("all_compressed_source_bytes_SHA_checked","whole_source_gzip_CRC_read_to_EOF",
       "all_native_rows_headers_sequences_SHA_axes_source_order_compared","all_typed_payload_bytes_SHA_checked","all_typed_columns_decoded"))):
   raise ValueError("source/raw-native QC lineage scope mismatch")
  selected=(self.root/f["path"]).resolve()
  for rows in (native["native_payloads"],proof["source_payloads"]):
   matched=[r for r in rows if Path(r["path"]).resolve()==selected]
   if len(matched)!=1 or any(matched[0][k]!=f[k] for k in ("sha256","bytes","rows")):
    raise ValueError("source shard absent from full-native raw replay lineage")
  return dict(source_handoff=census["source_handoff"],native_manifest=native_ref,raw_QC=proof_ref,
   original_FASTA_sha256=component["source_sha256"],full_native_QC_receipts_checked=True,
   raw_FASTA_reread=False,proof_scope="all raw-native source rows; not independent whole task/group/master proof")
 def _new(self,index,snapshot):
  m,census,plan,natives=snapshot
  if index>=len(m["new_task_cases"]):raise ValueError("new task not available in immutable closed fixture")
  case=m["new_task_cases"][index]
  original=self._row(case["closed_original_shard"],case["source_shard_row_index"])
  copied,=self._rows(case["case"])
  _same_native(copied,original,"copied full native envelope differs from closed original task bytes/types")
  group=self._row(case["closed_original_group"],case["source_shard_row_index"])
  _same(group,case["selected_group_row"],"closed original group/selected fixture allocation mismatch")
  ordinal,offset=self._source_binding(case,census);source=self._native(case,natives)
  seq,g,ref=_source_record(source,case["native_source_file"],offset,case["native_source_row_index"])
  if (case["native_source_row_index"]!=case["selected_occurrence"]["native_row_index"]
      or case["identifier_membership_proved_by_exact_frozen_PK_lookup"] is not True
      or case["complete_identifier_enumeration_not_independently_replayed_for_fixture"] is not True):
   raise ValueError("literal native/URS first-occurrence fixture qualification mismatch")
  state_ref=case["frozen_identity_state"]
  state=dict(native_sha=bytes.fromhex(g["native_sha"]),uppercase_sha=bytes.fromhex(state_ref["uppercase_sha256"]),
   rc_sha=bytes.fromhex(state_ref["rc_guard_sha256"]),task_id=copied["task_id"],reuse_kind=state_ref["reuse_kind"],
   first_file=state_ref["first_file"],first_row=state_ref["first_row"],occurrences=state_ref["occurrences"])
  shard=Path(case["closed_original_shard"]["path"]).stem
  if re.fullmatch("[0-9]{6}",shard) is None:raise ValueError("explicit native materializer shard index required")
  idx=dict(task_id=bytes.fromhex(copied["task_id"]),native_sha=state["native_sha"],source_file=ordinal,
   source_row=case["native_source_row_index"],reuse_kind="new",shard=int(shard),local_row=case["source_shard_row_index"])
  binding=dict(census_sha256=CENSUS_SHA256,task_shard_sha256=case["closed_original_shard"]["sha256"],
   groups_sha256=case["closed_original_group"]["sha256"],source_shard_sha256=case["native_source_file"]["sha256"],
   occurrence_sha256=case["census_occurrence_file"]["sha256"])
  v=rnacentral_full_candidate(copied,source,_occurrence_bytes(case["selected_occurrence"]),state,idx,
   case["literal_native_identifiers_from_closed_group"],group,source_file=case["native_source_file"],
   source_file_ordinal=ordinal,source_record_offset=offset,binding=binding)
  p=v.provenance()
  p.update(raw_source_lineage=self._lineage(case,census,plan),reader_binding=dict(fixture_sha256=FIXTURE["sha256"],kind="new",case_index=index),
   fixture_case=case["case"],source_fixture_ref=m["native_source_records"],source_fixture_bytes_verified=True,
   source_file_bytes_verified=False,census_manifest_bytes_verified=True,closed_original_task_and_groups_bytes_verified=True,
   original_source_occurrence_bytes_reread=False,raw_FASTA_reread=False,
   identifier_PK_membership_inherited=True,identifier_enumeration_independently_replayed=False,
   source_identity_scope="literal reference symbols; no material/modification/taxon/function identity")
  return WholeRNAcentralCandidate(v.sequence,v.condition_bytes,_canonical(p))
 def _reuse(self,index,snapshot):
  m,census,plan,natives=snapshot
  if index>=len(m["old_reuse_witnesses"]):raise ValueError("reuse witness outside immutable fixture")
  case=m["old_reuse_witnesses"][index];kind=case["reuse_kind"]
  if kind not in ("RNAcentral_pilot_existing","Rfam_existing") or case["new_task_serialized"] is not False:
   raise ValueError("old exact prior reuse cannot allocate a new task")
  ordinal,offset=self._source_binding(case,census);source=self._native(case,natives)
  seq,g,ref=_source_record(source,case["native_source_file"],offset,case["native_source_row_index"])
  tid=_task_id(g["native_sha"],kind);occ=_occurrence_bytes(case["selected_occurrence"])
  expected=dict(source_file_ordinal=ordinal,native_row_index=case["native_source_row_index"],
   original_record_index=source["source_record_index"],source_identifier=source["source_identifier"],
   native_sha=bytes.fromhex(g["native_sha"]),uppercase_sha=bytes.fromhex(g["upper_sha"]),rc_guard_sha=bytes.fromhex(g["rc_sha"]),
   native_length=len(seq),source_alphabet_pass=True,first_native_occurrence=True,reuse_kind=kind,
   task_identity_digest=bytes.fromhex(tid),local_existing_heldout_group_match=False,nonstandard_symbols=None)
  _same_native(occ,expected,"literal reused source occurrence/type/group/task identity mismatch")
  if tid!=case["existing_task_id"] or g["native_sha"]!=case["source_native_sequence_sha256"]:
   raise ValueError("existing prior native identity mismatch")
  old=plan["old_Rfam_index" if kind=="Rfam_existing" else "old_pilot_index"]
  if case["existing_readonly_index"]!=old["path"]:raise ValueError("existing source index reference mismatch")
  result=["Rfam15.1-native-reference-prior"] if kind=="Rfam_existing" else [tid,0]
  _same(case["existing_index_key_result"],result,"existing readonly-index native reuse witness mismatch")
  p=dict(schema="mora3.WholeRNAcentralReuseWitness/v1",new_task_serialized=False,whole_master_admitted=False,
   existing_task_id=tid,reuse_kind=kind,source_ref=ref,native_sha256=g["native_sha"],
   occurrence_fixture=case,existing_index_ref=old,old_index_reread=False,
   raw_source_lineage=self._lineage(case,census,plan),
   sequence_or_annotation_labels_transferred=False,source_status_is_activity_label=False,
   actual_model_use_allowed=False,neural_model_tokens=None,
   reader_binding=dict(fixture_sha256=FIXTURE["sha256"],kind="reuse",case_index=index))
  return WholeRNAcentralReuseWitness(seq,kind,tid,_canonical(p))
 def read(self,selector):
  if type(selector) is not WholeRNAcentralSelector:raise ValueError("typed bounded selector required")
  snapshot=self._snapshot();method=self._new if selector.kind=="new" else self._reuse
  try:return tuple(method(i,snapshot) for i in selector.indices)
  except (KeyError,TypeError,UnicodeError,OverflowError,IndexError) as exc:
   raise ValueError("malformed closed source candidate binding") from exc
 def verify(self,value):
  if type(value) not in (WholeRNAcentralCandidate,WholeRNAcentralReuseWitness):
   raise ValueError("typed whole-native task/reuse candidate required")
  binding=value.provenance().get("reader_binding",{})
  if (binding.get("fixture_sha256")!=FIXTURE["sha256"] or binding.get("kind") not in ("new","reuse")
      or not _integer(binding.get("case_index"))):
   raise ValueError("exact closed native reader selector binding required")
  checked,=self.read(WholeRNAcentralSelector(binding["kind"],(binding["case_index"],)))
  if value!=checked:raise ValueError("fresh source/census/task/group/receipt/type replay mismatch")
  return checked
 def source_references(self):
  m,census,plan,_=self._snapshot();cases=m["new_task_cases"]
  lineage=self._lineage(cases[0],census,plan)
  return dict(source_handoff=lineage["source_handoff"],native_manifest=lineage["native_manifest"],raw_QC=lineage["raw_QC"],
   fixture=dict(FIXTURE,path=str(self.manifest)),census=m["source_census"],
   source_records=m["native_source_records"],case0=cases[0]["case"],case2=cases[2]["case"],
   shard0=cases[0]["closed_original_shard"],shard2=cases[2]["closed_original_shard"],
   group0=cases[0]["closed_original_group"],group2=cases[2]["closed_original_group"],build_plan=census["build_plan"])
 def gates(self):
  m,*_=self._snapshot()
  return dict(global_source_task_proof_complete=False,master_union_admission=False,actual_model_use_allowed=False,
   current_authoritative_master="v22",whole_neural_token_totals=None,whole_optimizer_exposure=None,
   native_U_case_available=m["native_U_case_available"],homology_independence_established=False,
   identifier_enumeration_independently_replayed=False,model_ready=False,full_dataset_ready=False,
   scope="three closed new task candidates plus two exact old-prior reuse witnesses only")
