"""Synthetic upstream-wire tests; real-source coverage is tested separately."""
from dataclasses import replace
import copy,hashlib,importlib.util,json,os,sys
from pathlib import Path
import pytest
ROOT=Path(os.environ.get("MORA3_REPO_ROOT",Path.cwd()))
def canonical(x):return json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
def sha(x):return hashlib.sha256(x).hexdigest()
@pytest.fixture(scope="module")
def api():
 p=Path(os.environ.get("MORA3_RNACENTRAL_FULL_MODULE","src/mora3/data/rnacentral_full.py"))
 if not p.is_file():pytest.fail("whole RNAcentral native reader is absent")
 spec=importlib.util.spec_from_file_location("mora3.data.rnacentral_full",p);m=importlib.util.module_from_spec(spec)
 sys.modules[spec.name]=m;spec.loader.exec_module(m);return m
def wire(sequence="aCTuRNy"):
 # Frozen constructor recipe is an independent native-wire producer, not model admission.
 sys.path.insert(0,str(ROOT/"scripts"))
 from data_intake_full_rnacentral_tasks import full_native_prior_envelope
 from data_intake_nucleotide_prior import sequence_groups
 g=sequence_groups(sequence);native=bytes.fromhex(g["native_sha256"])
 tid=sha(canonical(["RNA_reference_native_symbols",None,native.hex()]))
 file=dict(path="synthetic/native.parquet",bytes=10,sha256="1"*64,rows=10,component="inactive",original_FASTA="/synthetic/inactive.fasta.gz")
 source=dict(source_record_id="official_RNAcentral28_inactive:DSE3-RNACENTRAL28-inactive:4",source_record_index=4,
  source_payload=file["original_FASTA"],source_identifier="URS_SYNTHETIC",native_header="URS_SYNTHETIC exact header",
  native_sequence=sequence,native_length=len(sequence),native_sequence_sha256=g["native_sha256"],
  uppercase_sequence_sha256=g["uppercase_sha256"],sequence_alphabet_pass=True,native_T_count=sequence.upper().count("T"),
  native_U_count=sequence.upper().count("U"),ambiguity_symbol_count=sum(x.upper() not in "ACGTU" for x in sequence),
  explicit_Rfam_ID_in_header=None,source_role="released RNAfamily/reference-sequence evidence; exact family/region join requires source metadata",
  task_admitted=False)
 state=dict(native_sha=native,uppercase_sha=bytes.fromhex(g["uppercase_sha256"]),rc_sha=bytes.fromhex(g["reverse_complement_base_order_group"]),
  task_id=tid,reuse_kind="new",first_file=0,first_row=4,occurrences=2)
 ids=["URS_SYNTHETIC","URS_SYNTHETIC_ALIAS"]
 ref=dict(native_payload=file["path"],native_payload_sha256=file["sha256"],native_row_index=4,original_record_index=4,
  original_FASTA=file["original_FASTA"],source_identifier=source["source_identifier"],native_header=source["native_header"],
  source_status="inactive",source_release="RNAcentral28.0",source_rights="CC0-1.0",original_source_record_id=source["source_record_id"])
 row=full_native_prior_envelope(sequence,state,ref,ids)
 occurrence=dict(source_file_ordinal=0,native_row_index=4,original_record_index=4,source_identifier=source["source_identifier"],
  native_sha=native,uppercase_sha=state["uppercase_sha"],rc_guard_sha=state["rc_sha"],native_length=len(sequence),
  source_alphabet_pass=True,first_native_occurrence=True,reuse_kind="new",task_identity_digest=bytes.fromhex(tid),
  local_existing_heldout_group_match=False,nonstandard_symbols=None)
 index=dict(task_id=bytes.fromhex(tid),native_sha=native,source_file=0,source_row=4,reuse_kind="new",shard=0,local_row=2)
 group=dict(task_id=tid,effective_groups_json=row["group_ids_json"])
 binding=dict(census_sha256="d6240751e8ba21744bd9869ccac0c4ea74809bbaa3dca3c456144b176e1020bd",
  task_shard_sha256="2"*64,groups_sha256="3"*64,source_shard_sha256=file["sha256"],occurrence_sha256="4"*64)
 return dict(row=row,native_record=source,occurrence=occurrence,state=state,index_row=index,identifiers=ids,
  group_row=group,source_file=file,source_file_ordinal=0,source_record_offset=0,binding=binding)
def project(api,w=None):return api.rnacentral_full_candidate(**(wire() if w is None else w))

def test_upstream_wire_roundtrip_preserves_literal_case_T_U_and_ambiguity_without_admission(api):
 w=wire();v=project(api,w)
 assert v.sequence=="aCTuRNy"
 assert v.native_bytes()==b"aCTuRNy"
 assert v.inputs()=={"prior_domain":"RNA_reference_native_symbols","source_organism":None}
 observed,known=v.target_masks()
 assert observed==(True,)*7 and known==(True,True,True,True,False,False,False)
 assert v.provenance()["whole_master_admitted"] is False
 assert v.provenance()["neural_model_tokens"] is None
 assert v.provenance()["source_record_status_is_not_activity_label"] is True

@pytest.mark.parametrize("sequence",("ACT","ACU","act","ACN"))
def test_synthetic_literal_identity_never_aliases_T_U_case_or_ambiguity(api,sequence):
 v=project(api,wire(sequence))
 assert v.sequence==sequence
 assert v.provenance()["sequence_sha256"]==sha(sequence.encode())

@pytest.mark.parametrize("field",("split","full_axis_values","admitted_for_task","model_adapter_supported","source_family","source_revision","task_id","record_id","source_sha256","evidence_class","split_policy"))
def test_synthetic_envelope_type_scope_identity_and_split_mutations_refuse(api,field):
 w=wire();r=w["row"]
 r[field]={"split":"test","full_axis_values":True,"admitted_for_task":1,"model_adapter_supported":0,
  "source_family":"Rfam15.1-native-reference-prior","source_revision":"fake","task_id":"0"*64,"record_id":"fake",
  "source_sha256":"0"*64,"evidence_class":"measured_function","split_policy":"heldout"}[field]
 with pytest.raises(ValueError):project(api,w)

@pytest.mark.parametrize("field",("source_record_index","source_identifier","native_header","native_sequence","native_length","sequence_alphabet_pass","native_T_count","source_record_id","task_admitted"))
def test_synthetic_native_row_types_full_axis_and_original_identity_mutations_refuse(api,field):
 w=wire();source=w["native_record"]
 source[field]={"source_record_index":True,"source_identifier":"wrong","native_header":"wrong header","native_sequence":"ACTU",
  "native_length":True,"sequence_alphabet_pass":1,"native_T_count":True,"source_record_id":"wrong","task_admitted":0}[field]
 with pytest.raises(ValueError):project(api,w)

@pytest.mark.parametrize("mutation",("hidden_target","source_status","native_reader","condition_target","source_ID_condition","unknown_target","missing"))
def test_synthetic_rehashed_payload_audit_condition_target_and_missingness_mutations_refuse(api,mutation):
 w=wire();r=w["row"];d=json.loads(r["native_payload"])
 if mutation=="hidden_target":d["targets"]["source_native_sequence"]="AAAAAAA"
 elif mutation=="source_status":d["audit"]["source_record_status_is_not_activity_label"]=1
 elif mutation=="native_reader":d["audit"]["native_reader_supported"]=0
 elif mutation=="condition_target":d["conditions"]["target_length"]=7
 elif mutation=="source_ID_condition":d["conditions"]["source_identifier"]="URS_SYNTHETIC"
 elif mutation=="unknown_target":d["targets"]["function"]="inferred"
 else:d["missing"]=[]
 r["native_payload"]=canonical(d);r["native_payload_sha256"]=sha(r["native_payload"])
 with pytest.raises(ValueError):project(api,w)

@pytest.mark.parametrize("field",("exact_full_axis","new_independent_experiments","source_status_is_not_activity_label","source_ref","record_ID_scope"))
def test_synthetic_rehashed_receipt_numeric_bool_and_source_role_aliases_refuse(api,field):
 w=wire();d=json.loads(w["row"]["admission_receipt_json"])
 if field=="source_ref":d[field]["native_row_index"]=True
 elif field=="new_independent_experiments":d[field]=False
 elif field=="record_ID_scope":d[field]="experimental assay"
 else:d[field]=1
 w["row"]["admission_receipt_json"]=json.dumps(d)
 with pytest.raises(ValueError):project(api,w)

@pytest.mark.parametrize("mutation",("phantom_identifier","missing_identifier","group_order","duplicate_group","first_row","reuse_kind","index_location","occurrence_type","heldout","source_file_ordinal","source_offset"))
def test_synthetic_group_first_occurrence_reuse_serialization_and_source_location_refuse(api,mutation):
 w=wire()
 if mutation=="phantom_identifier":w["identifiers"].append("phantom")
 elif mutation=="missing_identifier":w["identifiers"].pop()
 elif mutation=="group_order":
  w["group_row"]["effective_groups_json"]=json.dumps(list(reversed(json.loads(w["row"]["group_ids_json"]))))
 elif mutation=="duplicate_group":
  g=json.loads(w["row"]["group_ids_json"]);w["row"]["group_ids_json"]=json.dumps(g+[g[0]])
 elif mutation=="first_row":w["state"]["first_row"]=3
 elif mutation=="reuse_kind":w["state"]["reuse_kind"]="RNAcentral_pilot_existing"
 elif mutation=="index_location":w["index_row"]["source_row"]=3
 elif mutation=="occurrence_type":w["occurrence"]["first_native_occurrence"]=1
 elif mutation=="heldout":w["occurrence"]["local_existing_heldout_group_match"]=True
 elif mutation=="source_file_ordinal":w["source_file_ordinal"]=False
 else:w["source_record_offset"]=1
 with pytest.raises(ValueError):project(api,w)

def test_synthetic_context_projection_is_target_audit_length_status_group_and_source_ID_free(api):
 a=project(api);b=project(api,wire("UGAAC"*2))
 assert a.input_bytes()==b.input_bytes()
 hidden=replace(a,provenance_bytes=canonical(dict(sequence_sha256=sha(a.sequence.encode()),
  condition_sha256=sha(a.condition_bytes),hidden_target_hash="secret",source_status="active",whole_master_admitted=False)))
 assert hidden.input_bytes()==a.input_bytes()

@pytest.mark.parametrize("purpose",("training","evaluation","teacher_forcing","generation","loss","split_use"))
def test_unadmitted_whole_source_model_use_never_granted_by_flag_digest_or_pilot_domain(api,purpose):
 v=project(api)
 for fake in (None,True,{"verified":True,"manifest_sha256":"0"*64}):
  with pytest.raises(ValueError,match="admission|proof|master"):v.require_model_use(purpose,admission=fake)
 with pytest.raises(ValueError):v.as_neural_prior()

def test_synthetic_long_native_axis_masks_retained_context_overflow_refuses_without_crop(api):
 v=project(api,wire("aCTuN"*2000))
 assert len(v.sequence)==10000 and len(v.target_masks()[0])==10000
 with pytest.raises(ValueError,match="context|crop|capacity"):v.check_context_capacity(max_symbols=8192)
 assert v.sequence=="aCTuN"*2000
 v.check_context_capacity(max_symbols=10000)
 for cap in (True,0,1.0):
  with pytest.raises(ValueError):v.check_context_capacity(max_symbols=cap)

@pytest.mark.parametrize("ids",((True,),(),(0,0),(-1,),[0],tuple(range(17))))
def test_selector_exact_unique_bounded_types(api,ids):
 with pytest.raises(ValueError):api.WholeRNAcentralSelector("new",ids)


FIXTURE=Path("outputs/data-intake/20261101-RNAcentral-full-native-corpus-v1/closed-native-reader-fixture-v1/manifest.json")
FIXTURE_SHA="7af114e8897a07d712146e3a35d70ea3a2eb5c890fb980edc525916c41c2d029"
@pytest.fixture(scope="module")
def real_source(api):
 if not (ROOT/FIXTURE).is_file():pytest.skip("constructor closed native fixture unavailable")
 return api.WholeRNAcentralNativeSource(root=ROOT,manifest=FIXTURE,expected_manifest_sha256=FIXTURE_SHA)
@pytest.fixture(scope="module")
def real(api,real_source):
 return real_source.read(api.WholeRNAcentralSelector("new",(0,1,2)))

def test_actual_closed_task_source_roundtrip_and_complete_long_axis(api,real_source,real):
 assert [len(v.sequence) for v in real]==[501,1477,15466]
 for i,v in enumerate(real):
  assert real_source.verify(v)==v
  p=v.provenance()
  assert p["whole_master_admitted"] is False and p["actual_model_use_allowed"] is False
  assert p["identifier_enumeration_independently_replayed"] is False
  assert p["source_fixture_bytes_verified"] is True
  assert p["source_file_bytes_verified"] is False
  assert len(v.target_masks()[0])==len(v.sequence)
  assert v.inputs()=={"prior_domain":"RNA_reference_native_symbols","source_organism":None}
  assert set(v.inputs())=={"prior_domain","source_organism"}
 with pytest.raises(ValueError,match="context|capacity|crop"):real[2].check_context_capacity(max_symbols=8192)

def test_actual_native_ambiguity_T_and_absent_U_scope_are_honest(real_source,real):
 assert real[1].target_masks()[1].count(False)==1
 assert all(v.sequence.count("T")>0 for v in real)
 assert real_source.gates()["native_U_case_available"] is False
 assert all("U" not in v.sequence for v in real)

def test_actual_byte_tokenizer_roundtrip_is_data_codec_only_not_neural_counts(real):
 from mora3.tokenization.shortest_path import ByteTokenizer
 t=ByteTokenizer()
 for v in real:
  assert t.decode(t.encode(v.native_bytes()))==v.native_bytes()
  assert v.provenance()["neural_model_tokens"] is None

def test_actual_old_pilot_and_Rfam_are_reuse_witnesses_not_new_tasks(api,real_source):
 reuse=real_source.read(api.WholeRNAcentralSelector("reuse",(0,1)))
 assert [r.reuse_kind for r in reuse]==["RNAcentral_pilot_existing","Rfam_existing"]
 assert [len(r.sequence) for r in reuse]==[772,337]
 for r in reuse:
  assert r.provenance()["new_task_serialized"] is False
  assert r.existing_task_id
  assert real_source.verify(r)==r
  with pytest.raises(ValueError,match="reuse|existing|duplicate"):r.as_new_candidate()
  with pytest.raises(ValueError):r.require_model_use("training",admission=True)

@pytest.mark.parametrize("purpose",("training","evaluation","teacher_forcing","loss","generation","split_use"))
def test_actual_eligibility_flag_does_not_allow_model_entry(real,purpose):
 for v in real:
  assert v.provenance()["envelope"]["admitted_for_task"] is True
  for fake in (True,{"verified":True,"master_version":23,"sha256":FIXTURE_SHA}):
   with pytest.raises(ValueError):v.require_model_use(purpose,admission=fake)

@pytest.mark.parametrize("field",("sequence","source_audit","receipt_type","binding","group","reader_selector"))
def test_actual_tamper_and_rehashed_hidden_metadata_keep_inputs_but_source_replay_refuses(real_source,real,field):
 v=real[0];p=v.provenance()
 if field=="sequence":
  p["sequence_sha256"]=sha((v.sequence+"N").encode())
  changed=replace(v,sequence=v.sequence+"N",provenance_bytes=canonical(p))
 else:
  if field=="source_audit":p["source_native_record"]["native_header"]="hidden changed audit"
  elif field=="receipt_type":
   d=json.loads(p["envelope"]["admission_receipt_json"]);d["exact_full_axis"]=1
   p["envelope"]["admission_receipt_json"]=json.dumps(d)
  elif field=="binding":p["binding"]["task_shard_sha256"]="0"*64
  elif field=="group":p["effective_groups"].append("phantom")
  else:p["reader_binding"]["case_index"]=True
  changed=replace(v,provenance_bytes=canonical(p))
 assert changed.input_bytes()==v.input_bytes()
 with pytest.raises(ValueError):real_source.verify(changed)

@pytest.mark.parametrize("artifact",("fixture","census","source_records","case0","case2","shard0","shard2","group0","group2","source_handoff","native_manifest","raw_QC"))
def test_actual_consumed_closed_source_and_task_artifacts_are_hash_bound(api,real_source,artifact,monkeypatch):
 target=(ROOT/real_source.source_references()[artifact]["path"]).resolve()
 old=Path.read_bytes;touched=[]
 def changed(path):
  raw=old(path)
  if path.resolve()==target:touched.append(True);return raw+b" "
  return raw
 monkeypatch.setattr(Path,"read_bytes",changed)
 with pytest.raises(ValueError):
  s=api.WholeRNAcentralNativeSource(root=ROOT)
  s.read(api.WholeRNAcentralSelector("new",(0,1,2)))
 assert touched

def test_runtime_never_opens_writing_index_raw_FASTA_pickle_or_opaque_source(api,real_source,monkeypatch):
 old=Path.read_bytes;paths=[]
 def guard(p):
  assert p.suffix.lower() in (".json",".parquet")
  assert "task-identity-serialization-index.sqlite" not in str(p)
  assert "native-identity-URS.sqlite" not in str(p)
  paths.append(str(p));return old(p)
 monkeypatch.setattr(Path,"read_bytes",guard)
 real_source.read(api.WholeRNAcentralSelector("new",(0,1,2)))
 real_source.read(api.WholeRNAcentralSelector("reuse",(0,1)))
 assert paths

def test_unknown_closed_candidate_and_unrelated_or_rehashed_manifest_refuse(api,real_source,tmp_path):
 with pytest.raises(ValueError):real_source.read(api.WholeRNAcentralSelector("new",(3,)))
 with pytest.raises(ValueError):real_source.read(api.WholeRNAcentralSelector("reuse",(2,)))
 fake=tmp_path/"fake.json";fake.write_text('{"master_union_admission":true}')
 with pytest.raises(ValueError):api.WholeRNAcentralNativeSource(root=tmp_path,manifest="fake.json")
 with pytest.raises(ValueError):api.WholeRNAcentralNativeSource(root=ROOT,expected_manifest_sha256=sha(fake.read_bytes()))
 gates=real_source.gates()
 assert gates["global_source_task_proof_complete"] is False
 assert gates["master_union_admission"] is False and gates["actual_model_use_allowed"] is False
 assert gates["whole_neural_token_totals"] is None


def test_actual_raw_source_lineage_inherits_pinned_full_native_QC_without_rereading_FASTA(real_source,real):
 for v in real:
  p=v.provenance()
  assert p["raw_source_lineage"]["full_native_QC_receipts_checked"] is True
  assert p["raw_source_lineage"]["original_FASTA_sha256"]=="f6579aa6d95623675d6f7233fe9ce450bccffa9d5d1c7d65145d13aaf09ef725"
  assert p["raw_FASTA_reread"] is False
  assert p["independent_whole_source_task_proof"] is False
  assert p["homology_independence_established"] is False

@pytest.mark.parametrize("symbol",("X","?","-","1"," "))
def test_synthetic_nonstandard_native_symbols_are_quarantined_not_repaired(api,symbol):
 w=wire()
 s=w["native_record"];s["native_sequence"]="AC"+symbol
 s["native_sequence_sha256"]=sha(s["native_sequence"].encode())
 with pytest.raises(ValueError):project(api,w)
