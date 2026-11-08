"""Isolated synthetic native-text/IO tests; real readback is separate evidence."""
from dataclasses import replace
from pathlib import Path
import hashlib,importlib.util,json,sys
import pytest
def canonical(value):return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def sha(raw):return hashlib.sha256(raw).hexdigest()
@pytest.fixture
def api():
 path=Path(__file__).resolve().parents[2]/"src/mora3/data/paroutes_evaluation.py"
 name="mora3.data._staged_paroutes_evaluation_test"
 previous=sys.modules.get(name);spec=importlib.util.spec_from_file_location(name,path)
 module=importlib.util.module_from_spec(spec);sys.modules[name]=module
 try:spec.loader.exec_module(module);yield module
 finally:
  if previous is None:sys.modules.pop(name,None)
  else:sys.modules[name]=previous

def test_native_line_terminators_order_duplicates_and_empty_missingness(api):
 raw=b"A\r\nB\rA\n\nZ"
 lines=api.parse_native_lines(raw)
 assert [l.source_row for l in lines]==list(range(5))
 assert [l.native_bytes for l in lines]==[b"A\r\n",b"B\r",b"A\n",b"\n",b"Z"]
 assert [l.literal for l in lines]==["A","B","A","","Z"]
 assert [l.terminator for l in lines]==[b"\r\n",b"\r",b"\n",b"\n",b""]
 assert lines[3].status=="native_empty_literal"
 assert b"".join(l.native_bytes for l in lines)==raw
 assert lines[-1].byte_stop==len(raw)
 assert api.parse_native_lines(b"")==()
 assert len(api.parse_native_lines(b"A\n"))==1

def test_text_preserves_unicode_spaces_bom_hash_case_and_no_chemical_normalization(api):
 raw=" \ufeff[C@@H:7](F)Cl \nABCdef\r\nabcDEF\n".encode()
 lines=api.parse_native_lines(raw)
 assert lines[0].literal==" \ufeff[C@@H:7](F)Cl "
 assert lines[1].literal!="abcdef"
 assert lines[1].literal!=lines[2].literal
 for bad in [None,"x",bytearray(b"A"),b"\xff"]:
  with pytest.raises(ValueError):api.parse_native_lines(bad)

@pytest.mark.parametrize("bad",[True,False,-1,1.0,"1"])
def test_occurrence_and_version_types_are_exact(api,bad):
 with pytest.raises(ValueError):api.PaEvaluationSelector("targets_n1.txt",bad,"a"*64,"b"*64)
def test_unsupported_names_hashes_and_unsafe_formats(api):
 for name in ["targets_n2.txt","fake.pkl",False]:
  with pytest.raises(ValueError):api.PaEvaluationSelector(name,0,"a"*64,"b"*64)
 for bad in [False,"g"*64,"0"*63]:
  with pytest.raises(ValueError):api.PaEvaluationSelector("targets_n1.txt",0,bad,"b"*64)

@pytest.fixture
def store(api,tmp_path,monkeypatch):
 import pyarrow as pa,pyarrow.parquet as pq
 pa.set_cpu_count(1);pa.set_io_thread_count(1)
 def make(raws=None,overrides=None):
  raws=raws or {"targets_n1.txt":b"[C@@H:9](F)Cl\r\nC\nC\rZ",
                "targets_n5.txt":b"N\nO\n","ref_routes_reaction_hashes.txt":b"OPAQUE-HASH\nMixedCase\n"}
  entries={};specs={};selectors={}
  for name,raw in raws.items():
   p=tmp_path/name;p.write_bytes(raw)
   lines=api.parse_native_lines(raw)
   records=[dict(source_file_sha256=sha(raw),source_row=l.source_row,source_role=api.EVALUATION,
      payload_json=canonical(l.literal).decode(),quality_reasons_json="[]",
      source_reference_hash_exposure=None,task_admitted=False,adapter_supported=False) for l in lines]
   if overrides and name in overrides:
    for i,changes in overrides[name].items():records[i].update(changes)
   typed=p.with_suffix(".parquet")
   schema=pa.schema(list(zip(api.FIELDS,[pa.string(),pa.int64(),pa.string(),pa.string(),pa.string(),pa.bool_(),pa.bool_(),pa.bool_()])))
   pq.write_table(pa.Table.from_pylist(records,schema=schema),typed,row_group_size=2)
   specs[name]=dict(source_rows=len(lines),source_sha256=sha(raw),source_bytes=len(raw),
     typed_sha256=sha(typed.read_bytes()),typed_bytes=typed.stat().st_size)
   entries[name]=dict(original=dict(path=str(p),bytes=len(raw),sha256=sha(raw)),
     asset=dict(typed_path=str(typed),typed_sha256=sha(typed.read_bytes()),typed_bytes=typed.stat().st_size,source_rows=len(lines)))
   selectors[name]=tuple(api.PaEvaluationSelector(name,i,sha(raw),sha(typed.read_bytes())) for i in range(len(lines)))
  monkeypatch.setattr(api,"SPECS",specs)
  reader=api.PaRoutesEvaluationSource.__new__(api.PaRoutesEvaluationSource);reader.root=tmp_path.resolve()
  reader._snapshot=lambda:(entries,dict(synthetic=True))
  reader._receipt=lambda name,entry:(dict(semantic_rows_digest="d"*64),dict(synthetic=True))
  return reader,selectors,entries
 return make

def test_complete_native_set_and_partial_membership_have_distinct_truth_values(api,store):
 reader,sel,entries=store()
 result=reader.read((sel["targets_n1.txt"][2],sel["targets_n1.txt"][0]),reference_names=("targets_n1.txt",))
 ref=result.reference_sets[0]
 assert ref.occurrences("C")== (1,2)
 assert ref.membership("C") is True and ref.membership("c") is False
 assert ref.membership("NOT-SEEN") is False
 assert ref.row_count==4 and ref.unique_literal_count==3
 assert [c.selector.source_row for c in result.candidates]==[2,0]
 assert api.partial_membership(result.candidates,"C") is True
 assert api.partial_membership(result.candidates,"Z") is None
 assert api.partial_membership(result.candidates,"NOT-SEEN") is None
 assert result.candidates[1].native_line_bytes==b"[C@@H:9](F)Cl\r\n"
 assert result.candidates[1].literal=="[C@@H:9](F)Cl"
 assert ref.neural_token_count is None
 for bad in [None,0,False,b"C"]:
  with pytest.raises(ValueError):ref.membership(bad)

def test_multiple_complete_sources_and_requested_order_with_third_raw_auxiliary(api,store):
 reader,sel,entries=store()
 request=(sel["targets_n5.txt"][1],sel["targets_n1.txt"][0],sel["targets_n5.txt"][0])
 result=reader.read(request,reference_names=("ref_routes_reaction_hashes.txt","targets_n5.txt","targets_n1.txt"))
 assert [r.source_name for r in result.reference_sets]==["ref_routes_reaction_hashes.txt","targets_n5.txt","targets_n1.txt"]
 assert tuple(c.selector for c in result.candidates)==request
 assert result.reference_sets[0].membership("opaque-hash") is False
 assert result.reference_sets[0].membership("OPAQUE-HASH") is True
 assert result.reference_sets[0].raw_lines[0].terminator==b"\n"
 with pytest.raises(ValueError):api.partial_membership(result.candidates,"N")

@pytest.mark.parametrize("purpose",["training","positive_loss","model_input","gold_class","evaluation","heldout_protocol"])
def test_all_uses_close_regardless_fabricated_admission_or_complete_membership(api,store,purpose):
 reader,sel,entries=store();result=reader.read((sel["targets_n1.txt"][0],))
 ref=result.reference_sets[0];candidate=result.candidates[0]
 for item in [ref,candidate]:
  with pytest.raises(ValueError,match="evaluation|admission"):item.require_model_use(purpose,admission={"verified":True,"master":"a"*64})
 with pytest.raises(ValueError):reader.project_model_inputs(candidate)

@pytest.mark.parametrize("change",[
 {"source_file_sha256":"a"*64},{"source_row":9},{"source_role":"TRAIN"},
 {"payload_json":'"CHANGED"'},{"payload_json":"0"},{"source_reference_hash_exposure":False},
 {"task_admitted":True},{"adapter_supported":True},{"quality_reasons_json":"[0]"}])
def test_rehashed_typed_source_target_role_mask_and_type_mutations_reject(api,store,change):
 reader,sel,entries=store(overrides={"targets_n1.txt":{0:change}})
 with pytest.raises(ValueError):reader.read((sel["targets_n1.txt"][0],))

def test_fresh_raw_and_typed_hashes_between_calls(api,store):
 reader,sel,entries=store();c=reader.read((sel["targets_n1.txt"][0],)).candidates[0]
 p=Path(entries["targets_n1.txt"]["original"]["path"])
 p.write_bytes(p.read_bytes().replace(b"F",b"N"))
 with pytest.raises(ValueError,match="integrity"):reader.verify(c)
 reader,sel,entries=store();p=Path(entries["targets_n1.txt"]["asset"]["typed_path"])
 raw=bytearray(p.read_bytes());raw[len(raw)//2]^=1;p.write_bytes(raw)
 with pytest.raises(ValueError,match="integrity"):reader.read((sel["targets_n1.txt"][0],))

def test_full_raw_tamper_cannot_be_hidden_by_rehashed_candidate_or_boolean_complete_flag(api,store):
 reader,sel,entries=store();result=reader.read((sel["targets_n1.txt"][0],))
 ref=result.reference_sets[0];c=result.candidates[0]
 with pytest.raises(ValueError):replace(ref,native_file_bytes=ref.native_file_bytes.replace(b"Z",b"X"))
 binding=json.loads(ref.binding_bytes);binding["model_use_allowed"]=0
 with pytest.raises(ValueError):replace(ref,binding_bytes=canonical(binding))
 row=json.loads(c.typed_record_bytes);row["payload_json"]='"changed"'
 with pytest.raises(ValueError):replace(c,typed_record_bytes=canonical(row))
 with pytest.raises(ValueError):replace(c,native_line_bytes=b"CHANGED\n")

def test_raw_source_changes_during_parse_and_replace_during_read_refuse(api,store,monkeypatch):
 reader,sel,entries=store();p=Path(entries["targets_n1.txt"]["original"]["path"])
 original=api.parse_native_lines
 def changed(raw):
  p.touch();return original(raw)
 monkeypatch.setattr(api,"parse_native_lines",changed)
 with pytest.raises(ValueError,match="identity|changed"):reader.read((sel["targets_n1.txt"][0],))

def test_raw_total_budget_and_typed_rowgroup_caps_fail_before_allocation(api,store,monkeypatch):
 reader,sel,entries=store()
 monkeypatch.setattr(api,"MAX_TOTAL_RAW_BYTES",1)
 with pytest.raises(ValueError,match="budget|bound"):reader.read((),reference_names=tuple(sel))
 monkeypatch.setattr(api,"MAX_TOTAL_RAW_BYTES",16*1024**2)
 monkeypatch.setattr(api,"MAX_ROW_GROUP_BYTES",1)
 with pytest.raises(ValueError,match="rowgroup"):reader.read((sel["targets_n1.txt"][0],))

def test_selector_limits_order_absence_and_third_typed_source_refuse(api,store):
 reader,sel,entries=store()
 for request,names in [((),()),([],()),((sel["targets_n1.txt"][0],)*2,()),
  ((sel["targets_n1.txt"][0],)*9,()),(tuple(s[0] for s in sel.values()),())]:
  with pytest.raises(ValueError):reader.read(request,reference_names=names)
 with pytest.raises(ValueError):reader.read((),reference_names=("targets_n1.txt","targets_n1.txt"))
 with pytest.raises(ValueError):reader.read((replace(sel["targets_n1.txt"][0],source_row=99),))
 with pytest.raises(ValueError):reader.read((replace(sel["targets_n1.txt"][0],source_file_sha256="f"*64),))

def test_native_empty_line_is_not_omitted_or_filled(api,store):
 reader,sel,entries=store({"targets_n1.txt":b"A\n\nB"})
 result=reader.read((sel["targets_n1.txt"][1],))
 assert result.candidates[0].literal=="" and result.candidates[0].status=="native_empty_literal_evaluation_only"
 assert result.reference_sets[0].row_count==3
 assert result.reference_sets[0].membership("") is True
 assert api.partial_membership(result.candidates,"A") is None

@pytest.fixture
def metadata_source(api):
 root=next(p for p in Path(__file__).resolve().parents if (p/"AGENTS.md").is_file())
 source=api.PaRoutesEvaluationSource.__new__(api.PaRoutesEvaluationSource);source.root=root;return source

def test_exact_whole15_and_three_official_reference_metadata_bindings(api,metadata_source):
 entries,refs=metadata_source._snapshot()
 assert set(entries)==set(api.SPECS)
 assert sum(e["original"]["bytes"] for e in entries.values())==5350893
 assert refs["manifest"]["sha256"]==api.MANIFEST_SHA256
 for name,entry in entries.items():
  receipt,_=metadata_source._receipt(name,entry)
  assert receipt["source_role"]==api.EVALUATION and receipt["task_admitted_rows"]==0

@pytest.mark.parametrize("target,key,value",[
 ("handoff","source_worker_task_admitted",False),("handoff","model_use_allowed",0),
 ("manifest","task_admitted_rows",False),("manifest","source_files",True),
 ("proof","no_new_task_admission",1),("proof","no_missing_or_inprogress_payloads",False)])
def test_rehashed_source_proof_type_and_closed_scope_mutations_refuse(api,metadata_source,monkeypatch,target,key,value):
 old=metadata_source._json;paths={"handoff":api.HANDOFF_PATH,"manifest":api.MANIFEST_PATH}
 def changed(path,sha,size=None):
  result,ref=old(path,sha,size)
  if path==paths.get(target) or target=="proof" and sha==api.PROOF_SHA256:result[key]=value
  return result,ref
 monkeypatch.setattr(metadata_source,"_json",changed)
 with pytest.raises(ValueError):metadata_source._snapshot()

def test_unknown_release_unsafe_path_and_admission_arguments_refuse(api,tmp_path):
 with pytest.raises(ValueError):api.PaRoutesEvaluationSource(root=tmp_path,handoff="fake.json")
 with pytest.raises(ValueError):api.PaRoutesEvaluationSource(root=tmp_path,expected_manifest_sha256="a"*64)
 source=api.PaRoutesEvaluationSource.__new__(api.PaRoutesEvaluationSource);source.root=tmp_path
 for p in ["bad.pkl","bad.h5","bad.py","../escape.txt"]:
  with pytest.raises(ValueError):source._path(p,{".json",".txt",".parquet"})

def test_first_typed_source_replacement_during_second_source_read_is_rejected(api,store,monkeypatch):
 reader,sel,entries=store()
 first=Path(entries["targets_n1.txt"]["asset"]["typed_path"])
 original=reader._rows
 def changed(asset,indices):
  result=original(asset,indices)
  if asset["typed_path"]==entries["targets_n5.txt"]["asset"]["typed_path"]:
   first.touch()
  return result
 monkeypatch.setattr(reader,"_rows",changed)
 with pytest.raises(ValueError,match="typed.*changed|typed.*replaced"):
  reader.read((sel["targets_n1.txt"][0],sel["targets_n5.txt"][0]))

def test_complete_snapshot_unchanged_if_future_source_changes_but_fresh_query_refuses(api,store):
 reader,sel,entries=store()
 snapshot=reader.read_reference_sets(("targets_n1.txt",))[0]
 assert snapshot.membership("Z") is True
 p=Path(entries["targets_n1.txt"]["original"]["path"]);p.write_bytes(p.read_bytes().replace(b"Z",b"Y"))
 assert snapshot.membership("Z") is True
 with pytest.raises(ValueError,match="integrity"):reader.membership("targets_n1.txt","Z")

def test_empty_partial_selection_cannot_manufacture_negative_membership(api):
 with pytest.raises(ValueError):api.partial_membership((),"UNKNOWN")
