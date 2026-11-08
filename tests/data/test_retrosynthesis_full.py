"""Full ORD candidate tests. Synthetic edge cases do not establish source coverage."""
import dataclasses,hashlib,importlib.util,json,os,sys
from pathlib import Path
import pytest
ROOT=Path(os.environ.get("MORA3_REPO_ROOT",Path.cwd()))
PACKAGE=ROOT/"outputs/implementation/aux-native-readers-20261101/Retrosynthesis-full-native-v1"
def load():
 path=Path(os.environ.get("MORA3_RETRO_FULL_MODULE",ROOT/"src/mora3/data/retrosynthesis_full.py"))
 try:
  spec=importlib.util.spec_from_file_location("mora3.data.retro_full_test",path);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
 except FileNotFoundError:pytest.fail("source-bound whole ORD candidate reader is missing")
 return m
def test_whole_ORD_entrypoint_rejects_boolean_source_row_coordinate():
 api=load()
 with pytest.raises(ValueError):api.ORDOccurrenceSelector("ord_dataset-"+"a"*32,True,"b"*64,"c"*64)
@pytest.fixture(scope="module")
def api():return load()
def canonical(value):return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def synthetic(api,semantic=None,**mutations):
 semantic=semantic if semantic is not None else {"reaction_id":"synthetic","inputs":{},"outcomes":[{"products":[{"reaction_role":"PRODUCT","identifiers":[{"type":"SMILES","value":"[13CH3:7][C@@H:8](F)Cl"}]}]}]}
 proto=b"\x00synthetic-native-proto\x80";selector=api.ORDOccurrenceSelector("ord_dataset-"+"a"*32,0,"b"*64,"c"*64)
 row=dict(dataset_id=selector.dataset_id,source_file_sha256=selector.source_file_sha256,source_row=0,reaction_id="synthetic",
  source_proto_sha256=hashlib.sha256(proto).hexdigest(),source_proto_bytes=len(proto),semantic_json=canonical(semantic).decode(),
  quality_reasons_json="[]",unknown_wire_fields=False,semantic_json_roundtrip_equal=True,task_admitted=False,adapter_supported=False)
 row.update(mutations)
 binding=dict(schema="mora3.FullORDNativeCandidateBinding/v1",handoff_sha256=api.HANDOFF_SHA256,manifest_sha256=api.MANIFEST_SHA256,
  source_protocol_role="unassigned original dataset/study; requires corpus rights/provenance/group review")
 return api.ORDNativeCandidate(selector,proto,canonical(row),canonical(binding))
@pytest.mark.parametrize("field,value",[
 ("dataset_id","../../fake"),("source_row",True),("source_row",1.0),("source_row",-1),
 ("source_file_sha256",False),("source_file_sha256","0"*63),("typed_file_sha256","BAD")
])
def test_typed_occurrence_rejects_wrong_source_coordinate_or_numeric_alias(api,field,value):
 values=dict(dataset_id="ord_dataset-"+"a"*32,source_row=0,source_file_sha256="b"*64,typed_file_sha256="c"*64);values[field]=value
 with pytest.raises(ValueError):api.ORDOccurrenceSelector(**values)

def test_synthetic_native_bytes_semantic_order_missingness_and_literal_product_maps_are_separate(api):
 native=synthetic(api);semantic=native.semantic()
 assert native.source_proto==b"\x00synthetic-native-proto\x80"
 assert semantic["outcomes"][0]["products"][0]["identifiers"][0]["value"]=="[13CH3:7][C@@H:8](F)Cl"
 assert native.field_present(("conditions",)) is False
 assert native.field_present(("inputs",)) is True
 assert native.quality_reasons==() and native.status=="native_candidate_unadmitted"
 assert api._product_query(native)=={"query_role":"product_conditioned_chemical_retrosynthesis","products":[{"syntax":"SMILES","value":"[13CH3][C@@H](F)Cl"}]}
 assert native.neural_token_count is None
 for purpose in ("training","evaluation","positive_loss","generation"):
  with pytest.raises(ValueError):native.require_model_use(purpose,admission={"verified":True,"sha256":"a"*64})

@pytest.mark.parametrize("field,value",[
 ("source_row",False),("source_proto_bytes",True),("task_admitted",0),("adapter_supported",0),
 ("unknown_wire_fields",0),("semantic_json_roundtrip_equal",1),("source_file_sha256","0"*64),
 ("source_proto_sha256","0"*64),("reaction_id","wrong"),("quality_reasons_json","{}")
])
def test_synthetic_rehashed_native_record_type_value_and_binding_aliases_refuse(api,field,value):
 with pytest.raises(ValueError):synthetic(api,**{field:value})

def test_synthetic_quarantine_keeps_original_bytes_but_refuses_product_projection(api):
 candidate=synthetic(api,quality_reasons_json='["semantic_json_roundtrip_mismatch"]',semantic_json_roundtrip_equal=False)
 assert candidate.status=="native_quarantined_unadmitted"
 assert candidate.source_proto==b"\x00synthetic-native-proto\x80"
 with pytest.raises(ValueError):api._product_query(candidate)
 with pytest.raises(ValueError):candidate.require_model_use("training",admission=True)

@pytest.mark.parametrize("raw",(b'{"a":1,"a":2}',b'{"a":NaN}',b'{"a":Infinity}'))
def test_synthetic_duplicate_or_nonfinite_json_cannot_relabel_a_native_record(api,raw):
 with pytest.raises(ValueError):api._strict_json(raw)

def test_synthetic_hidden_precursors_conditions_outcomes_goldclass_and_audit_do_not_enter_query(api):
 base=synthetic(api);semantic=base.semantic()
 semantic["inputs"]={"unknown":{"components":[{"identifiers":[{"type":"SMILES","value":"TARGET_SECRET"}]}]}}
 semantic["identifiers"]=[{"type":"REACTION_TYPE","value":"GOLD_SECRET"}]
 semantic["conditions"]={"temperature":{"setpoint":{"value":999}}}
 semantic["provenance"]={"patent":"SOURCE_SECRET"}
 semantic["outcomes"][0]["conversion"]={"value":12345}
 other=synthetic(api,semantic=semantic)
 assert api._product_query(base)==api._product_query(other)
 assert "SECRET" not in json.dumps(api._product_query(other))
 with pytest.raises(ValueError):api._product_query(base,product_pointers=("/inputs/unknown/components/0/identifiers/0",))

@pytest.mark.parametrize("semantic",[
 {"reaction_id":"synthetic","outcomes":[]},
 {"reaction_id":"synthetic","outcomes":[{"products":[{"identifiers":[{"type":"SMILES","value":"CC"}]}]}]},
 {"reaction_id":"synthetic","outcomes":[{"products":[{"reaction_role":"PRODUCT","identifiers":[{"type":"NAME","value":"unresolved"}]}]}]},
])
def test_synthetic_missing_or_unknown_native_product_roles_are_not_guessed(api,semantic):
 with pytest.raises(ValueError):api._product_query(synthetic(api,semantic=semantic))

def test_synthetic_product_alternatives_require_explicit_selection_and_no_CXSMILES_flattening(api):
 semantic=synthetic(api).semantic()
 semantic["outcomes"][0]["products"].append({"reaction_role":"PRODUCT","identifiers":[{"type":"SMILES","value":"C[NH3+:9]"}]})
 candidate=synthetic(api,semantic=semantic)
 with pytest.raises(ValueError):api._product_query(candidate)
 result=api._product_query(candidate,product_pointers=("/outcomes/0/products/1/identifiers/0",))
 assert result["products"]==[{"syntax":"SMILES","value":"C[NH3+]"}]
 semantic["outcomes"][0]["products"][1]["identifiers"][0]["value"]="CC |enhanced_stereo|"
 with pytest.raises(ValueError):api._product_query(synthetic(api,semantic=semantic),product_pointers=("/outcomes/0/products/1/identifiers/0",))

def test_full_reader_pins_entire_release_and_refuses_foreign_manifest_or_fake_admission(api):
 with pytest.raises(ValueError):api.FullORDNativeSource(root=ROOT,expected_manifest_sha256="0"*64)
 with pytest.raises(ValueError):api.FullORDNativeSource(root=ROOT,handoff="fake.pkl")

@pytest.fixture(scope="module")
def actual(api):
 pointer=PACKAGE/"source-fixture-reference.json"
 if not pointer.exists():pytest.skip("source-owner bounded full ORD fixture not yet delivered")
 reference=json.loads(pointer.read_text());raw=(ROOT/reference["path"]).read_bytes()
 assert hashlib.sha256(raw).hexdigest()==reference["sha256"]
 fixture=json.loads(raw)
 files={f["source_id"]:f for f in fixture["files"]};rows=[]
 for record in fixture["records"]:
  typed=record["typed_row_exact_fields"];source=files[record["source_id"]]
  rows.append(dict(selector=dict(dataset_id=record["source_id"],source_row=record["original_source_row"],
   source_file_sha256=source["original"]["sha256"],typed_file_sha256=source["typed"]["sha256"]),
   source_proto_sha256=record["native_proto"]["sha256"],source_proto_bytes=record["native_proto"]["bytes"],
   semantic_json=typed["semantic_json"],quality_reasons_json=typed["quality_reasons_json"],
   source_protocol_role=record["source_protocol_role"]))
 return dict(rows=rows,source_fixture=fixture)

def actual_selectors(api,fixture):
 return tuple(api.ORDOccurrenceSelector(**r["selector"]) for r in fixture["rows"])
def test_actual_at_most_eight_two_shard_occurrences_preserve_original_and_typed_fingerprints(api,actual):
 selectors=actual_selectors(api,actual);assert len(selectors)<=8 and len({s.dataset_id for s in selectors})<=2
 source=api.FullORDNativeSource(root=ROOT);values=source.read_occurrences(selectors)
 for expected,value in zip(actual["rows"],values):
  assert hashlib.sha256(value.source_proto).hexdigest()==expected["source_proto_sha256"]
  assert len(value.source_proto)==expected["source_proto_bytes"]
  assert value.native_record()["semantic_json"]==expected["semantic_json"]
  assert value.native_record()["quality_reasons_json"]==expected["quality_reasons_json"]
  assert value.source_protocol_role==expected["source_protocol_role"]
  assert source.verify(value).binding_sha256==value.binding_sha256
  for purpose in ("training","evaluation","positive_loss"):
   with pytest.raises(ValueError):value.require_model_use(purpose,admission={"approved":True})

def test_actual_source_replay_rejects_rehashed_hidden_target_or_audit_mutation(api,actual):
 source=api.FullORDNativeSource(root=ROOT);candidate=source.read_occurrences(actual_selectors(api,actual)[:1])[0]
 row=candidate.native_record();semantic=candidate.semantic()
 semantic["provenance"]={"synthetic_attack":"changed"};row["semantic_json"]=canonical(semantic).decode()
 mutated=dataclasses.replace(candidate,typed_record_bytes=canonical(row))
 with pytest.raises(ValueError):source.verify(mutated)
 with pytest.raises(ValueError):source.project_product_inputs(mutated)

def test_actual_selector_allocation_order_limits_and_wrong_native_bindings(api,actual):
 source=api.FullORDNativeSource(root=ROOT);selectors=actual_selectors(api,actual)
 with pytest.raises(ValueError):source.read_occurrences((selectors[0],selectors[0]))
 with pytest.raises(ValueError):source.read_occurrences(tuple([selectors[0]]*9))
 with pytest.raises(ValueError):source.read_occurrences((dataclasses.replace(selectors[0],source_file_sha256="0"*64),))
 with pytest.raises(ValueError):source.read_occurrences((dataclasses.replace(selectors[0],typed_file_sha256="0"*64),))
 with pytest.raises(ValueError):source.read_occurrences((dataclasses.replace(selectors[0],source_row=10**12),))
 before=source.read_occurrences(selectors);after=source.read_occurrences(tuple(reversed(selectors)))
 assert tuple(x.binding_sha256 for x in after)==tuple(reversed([x.binding_sha256 for x in before]))

def test_actual_NaN_quarantine_official_heldout_and_product_projection_gates(api,actual):
 source=api.FullORDNativeSource(root=ROOT);values=source.read_occurrences(actual_selectors(api,actual))
 assert any(v.status=="native_quarantined_unadmitted" for v in values)
 assert any("OFFICIAL_" in v.source_protocol_role and "TRAIN_SOURCE" not in v.source_protocol_role for v in values)
 successes=0
 for v in values:
  if v.status=="native_quarantined_unadmitted":
   with pytest.raises(ValueError):source.project_product_inputs(v)
  else:
   try:inputs=source.project_product_inputs(v)
   except ValueError:continue
   successes+=1;assert set(inputs)=={"query_role","products"}
 assert successes>0

@pytest.mark.parametrize("file_kind",("original","typed","receipt","protocol","handoff"))
def test_actual_source_file_or_proof_freshness_tamper_rejects_without_source_writes(api,actual,monkeypatch,file_kind):
 source=api.FullORDNativeSource(root=ROOT);selector=actual_selectors(api,actual)[0]
 candidate=source.read_occurrences((selector,))[0];refs=candidate.binding()["references"];target=Path(refs[file_kind]["path"]).resolve()
 if file_kind in ("original","typed"):
  original=api._stream_sha
  def changed(handle):return "0"*64 if Path(handle.name).resolve()==target else original(handle)
  monkeypatch.setattr(api,"_stream_sha",changed)
 else:
  original=Path.read_bytes
  def changed(path):
   raw=original(path);return raw+b" " if path.resolve()==target else raw
  monkeypatch.setattr(Path,"read_bytes",changed)
 with pytest.raises(ValueError):source.read_occurrences((selector,))

def test_actual_original_schema_nullable_or_metadata_substitution_refuses(api,actual,monkeypatch):
 import pyarrow as pa,pyarrow.parquet as pq
 source=api.FullORDNativeSource(root=ROOT);selector=actual_selectors(api,actual)[0];factory=pq.ParquetFile
 def changed(*args,**kwargs):
  original=factory(*args,**kwargs)
  if original.schema_arrow.names!=["reaction_id","reaction"]:return original
  class File:
   def __getattr__(self,key):return getattr(original,key)
   @property
   def schema_arrow(self):
    return pa.schema([pa.field(f.name,f.type,nullable=True) for f in original.schema_arrow],metadata=original.schema_arrow.metadata)
  return File()
 monkeypatch.setattr(pq,"ParquetFile",changed)
 with pytest.raises(ValueError):source.read_occurrences((selector,))

@pytest.mark.parametrize("field,value",(("source_row",False),("task_admitted",0),("adapter_supported",0),("unknown_wire_fields",0),("source_proto_bytes",True)))
def test_actual_postdecode_numeric_alias_and_mask_corruption_rejects(api,actual,monkeypatch,field,value):
 import pyarrow.parquet as pq
 source=api.FullORDNativeSource(root=ROOT);selector=actual_selectors(api,actual)[0];factory=pq.ParquetFile
 def changed(*args,**kwargs):
  original=factory(*args,**kwargs)
  if original.schema_arrow.names!=list(api.TYPED_FIELDS):return original
  class File:
   def __getattr__(self,key):return getattr(original,key)
   def read_row_group(self,*a,**kw):
    table=original.read_row_group(*a,**kw)
    class Table:
     def slice(self,*a,**kw):
      piece=table.slice(*a,**kw)
      class Rows:
       def to_pylist(self):
        rows=piece.to_pylist();rows[0][field]=value;return rows
      return Rows()
    return Table()
  return File()
 monkeypatch.setattr(pq,"ParquetFile",changed)
 with pytest.raises(ValueError):source.read_occurrences((selector,))

def test_actual_rehashed_audit_receipt_type_alias_is_rejected_on_source_replay(api,actual):
 source=api.FullORDNativeSource(root=ROOT);candidate=source.read_occurrences(actual_selectors(api,actual)[:1])[0]
 audit=candidate.binding();audit["model_use_allowed"]=0
 mutated=dataclasses.replace(candidate,binding_bytes=canonical(audit))
 assert mutated.binding_sha256!=candidate.binding_sha256
 with pytest.raises(ValueError):source.verify(mutated)
 with pytest.raises(ValueError):source.project_product_inputs(mutated)

def test_actual_file_replacement_during_read_is_caught_without_changing_source(api,actual,monkeypatch):
 source=api.FullORDNativeSource(root=ROOT);selector=actual_selectors(api,actual)[0]
 candidate=source.read_occurrences((selector,))[0];refs=candidate.binding()["references"]
 target=Path(refs["typed"]["path"]).resolve();other=Path(refs["original"]["path"]).resolve()
 stat=Path.stat
 def changed(path,*a,**kw):return stat(other,*a,**kw) if path==target else stat(path,*a,**kw)
 monkeypatch.setattr(Path,"stat",changed)
 with pytest.raises(ValueError):source.read_occurrences((selector,))

@pytest.mark.parametrize("path",("../unsafe.json","fake.pkl","source.py","input.pb"))
def test_unsafe_or_generated_source_formats_never_read(api,path):
 source=api.FullORDNativeSource(root=ROOT)
 with pytest.raises(ValueError):source._path(api.ORDFileRef(path,"0"*64),{".parquet"})

def test_actual_original_rowgroup_bound_refuses_without_cropping_or_large_read(api,actual,monkeypatch):
 source=api.FullORDNativeSource(root=ROOT);selector=actual_selectors(api,actual)[0]
 monkeypatch.setattr(api,"MAX_ROW_GROUP_BYTES",1)
 with pytest.raises(ValueError,match="no crop"):source.read_occurrences((selector,))

def test_actual_public_known_context_is_explicit_and_desired_or_gold_context_is_not_observed(api,actual):
 from mora3.data.retrosynthesis_native import ReactionCondition
 source=api.FullORDNativeSource(root=ROOT);candidate=source.read_occurrences(actual_selectors(api,actual)[:1])[0]
 before=source.project_product_inputs(candidate)
 known=ReactionCondition("temperature","42","CELSIUS","known_input","explicit_caller_known")
 after=source.project_product_inputs(candidate,known_conditions=(known,))
 assert after["products"]==before["products"] and after["known_conditions"]==[known.as_mapping()]
 desired=ReactionCondition("temperature","42","CELSIUS","desired","caller_desired")
 with pytest.raises(ValueError):source.project_product_inputs(candidate,known_conditions=(desired,))
 with pytest.raises(ValueError):ReactionCondition("reaction_class","GOLD",None,"known_input","explicit_caller_known")
