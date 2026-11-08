"""Synthetic contract/IO tests; actual source coverage is in sealed readback evidence."""
from dataclasses import replace
import hashlib, importlib.util, json, os, sys
from pathlib import Path
import pytest

def canonical(v):
    return json.dumps(v, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()

@pytest.fixture
def api():
    path=Path(__file__).resolve().parents[2]/"src/mora3/data/paroutes_lookup.py"
    name="mora3.data._staged_paroutes_lookup_test"
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec); sys.modules[name]=module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(name,None)

def test_native_member_order_duplicate_keys_and_number_lexemes(api):
    raw=b' { "z":1, "a":null, "z":2, "nest":{"b":false,"a":[-0,1.00,1e2]} } '
    value=api.parse_native_entry(raw)
    assert tuple(k for k,v in value.members)==("z","a","z","nest")
    assert value.values("z")== (api.NativeJSONNumber("1"),api.NativeJSONNumber("2"))
    with pytest.raises(ValueError,match="ambiguous"):value.unique("z")
    assert value.unique("a") is None
    assert value.unique("nest").unique("a")==tuple(api.NativeJSONNumber(x) for x in ("-0","1.00","1e2"))
    assert value.unique("nest").unique("b") is False

@pytest.mark.parametrize("raw",[
 b'[]',b'null',b'{"x":NaN}',b'{"x":Infinity}',b'{"x":-Infinity}',b'{"x":1} trailing',b'{"x":1,}',b'{"x":"\\ud800"}',b'\xff'])
def test_native_entry_rejects_unsupported_or_unsafe_json(api,raw):
    with pytest.raises(ValueError):api.parse_native_entry(raw)

@pytest.mark.parametrize("lexical",["NaN","Infinity","+1","01","1.","True",1,False])
def test_native_number_rejects_type_and_lexical_aliases(api,lexical):
    with pytest.raises(ValueError):api.NativeJSONNumber(lexical)

def test_immutable_native_object_and_missing_vs_null(api):
    obj=api.parse_native_entry(b'{"reaction_smarts":"[C:1]>>[C:1]","missing":null,"empty":"","zero":0,"false":false}')
    assert obj.status("absent")=="not_encoded"
    assert obj.status("missing")=="native_null"
    assert obj.status("empty")=="native_empty_string"
    assert obj.status("zero")=="native_present"
    assert obj.unique("false") is False
    with pytest.raises((AttributeError,TypeError)):obj.members+=(("x",None),)
    with pytest.raises(ValueError):api.NativeJSONObject([("x",1)])
    with pytest.raises(ValueError):api.NativeJSONObject((("x",[]),))

@pytest.mark.parametrize("row",[True,False,-1,1.0,"0",42554])
def test_selector_requires_exact_occurrence(api,row):
    with pytest.raises(ValueError):api.PaLookupSelector(row,"a"*64,"b"*64)

def test_selector_pin_and_format_types(api):
    for sha in ["x"*64,True,"a"*63]:
        with pytest.raises(ValueError):api.PaLookupSelector(0,sha,"b"*64)

@pytest.fixture
def store(api,tmp_path):
    """Independent synthetic JSON_ARRAY/Parquet pair, never an admission fixture."""
    import pyarrow as pa,pyarrow.parquet as pq
    pa.set_cpu_count(1);pa.set_io_thread_count(1)
    def make(entries=None,overrides=None,raw=None):
        entries=entries or [b'{"z":1,"reaction_smarts":"[C:1]>>[C:1]","classification":"gold","optional":null}',
                            b'{"reaction_smarts":"[O:2]>>[O:2]","list":[3,2,3],"context":{}}',
                            b'{"reaction_smarts":"[N:3]>>[N:3]","known":false}']
        source=tmp_path/("source"+str(len(list(tmp_path.glob("source*"))))+".json")
        source.write_bytes(raw if raw is not None else b"[\n"+b",\n".join(entries)+b"\n]")
        sha=hashlib.sha256(source.read_bytes()).hexdigest()
        rows=[]
        for i,e in enumerate(entries):
            row=dict(source_file_sha256=sha,source_row=i,source_role=api.COMPUTED,
                     payload_json=canonical(json.loads(e)).decode(),quality_reasons_json="[]",
                     source_reference_hash_exposure=None,task_admitted=False,adapter_supported=False)
            if overrides and i in overrides:row.update(overrides[i])
            rows.append(row)
        typed=source.with_suffix(".parquet")
        schema=pa.schema(list(zip(api.FIELDS,[pa.string(),pa.int64(),pa.string(),pa.string(),pa.string(),pa.bool_(),pa.bool_(),pa.bool_()])))
        pq.write_table(pa.Table.from_pylist(rows,schema=schema),typed,row_group_size=2)
        typed_sha=hashlib.sha256(typed.read_bytes()).hexdigest()
        identity=lambda p:api._identity(p.stat())
        original=dict(path=str(source),sha256=sha,bytes=source.stat().st_size)
        asset=dict(typed_path=str(typed),typed_sha256=typed_sha,typed_bytes=typed.stat().st_size,source_rows=len(entries))
        guards={k:dict(path=str(p),file_identity=identity(p),expected_whole_sha256=sh)
                for k,p,sh in [("original",source,sha),("typed",typed,typed_sha)]}
        source_reader=api.PaRoutesLookupSource.__new__(api.PaRoutesLookupSource);source_reader.root=tmp_path.resolve()
        source_reader._snapshot=lambda:(dict(asset=asset,original=original),guards,dict(synthetic=True))
        source_reader._receipt=lambda entry:(dict(semantic_rows_digest="d"*64),dict(synthetic=True))
        selectors=tuple(api.PaLookupSelector(i,sha,typed_sha) for i in range(len(entries)))
        return source_reader,selectors,source,typed,guards,rows
    return make

def test_bounded_source_replays_exact_order_raw_types_and_rule_role(api,store):
    reader,selectors,raw,typed,guards,rows=store()
    candidates=reader.read_entries((selectors[2],selectors[0],selectors[1]))
    assert tuple(c.selector.source_row for c in candidates)==(2,0,1)
    first=candidates[1]
    assert tuple(k for k,v in first.native_value().members)==("z","reaction_smarts","classification","optional")
    assert first.field_status("optional")=="native_null"
    assert first.field_status("none")=="not_encoded"
    assert first.rule_evidence().syntax=="SMARTS"
    assert first.rule_evidence().role=="source_computed_rule_not_molecule_or_measured_reaction"
    assert first.neural_token_count is None
    assert first.source_reference_hash_exposure is None
    assert first.status=="native_computed_candidate_unadmitted"
    assert first.native_entry_bytes.startswith(b'{"z":1,')
    assert first.native_record()==rows[0]
    assert reader.verify(first).binding_sha256==first.binding_sha256
    assert first.binding()["native_read"]["entries_parsed"]==1
    assert first.binding()["native_read"]["full_raw_hash_performed"] is False
    assert first.native_value().unique("z")==api.NativeJSONNumber("1")

def test_native_duplicate_and_list_multiplicity_survive_canonical_projection(api,store):
    reader,sel,*_=store([b'{"reaction_smarts":"[C]>>[C]","x":1,"x":2,"a":[1,1,2]}'])
    c,=reader.read_entries(sel)
    assert len(c.native_value().values("x"))==2
    assert c.native_value().unique("a")==tuple(api.NativeJSONNumber(x) for x in ("1","1","2"))
    assert c.native_record()["payload_json"]==canonical(json.loads(c.native_entry_bytes)).decode()
    assert c.duplicate_member_paths==("$:x",)
    with pytest.raises(ValueError):c.rule_evidence()

@pytest.mark.parametrize("purpose",["training","model_input","positive_loss","evaluation","join_template","reaction_label"])
def test_all_use_refuses_even_pretend_admission_and_hidden_targets(api,store,purpose):
    reader,sel,*_=store();c=reader.read_entries(sel)[0]
    with pytest.raises(ValueError,match="admission|computed|unsupported"):c.require_model_use(purpose,admission={"verified":True,"master":"a"*64})
    with pytest.raises(ValueError,match="product|molecule|computed"):reader.project_product_inputs(c)
    with pytest.raises(ValueError):reader.join_context(c,"a"*64)
    assert "classification" not in c.rule_evidence().__dict__

@pytest.mark.parametrize("change",[
 {"source_file_sha256":"e"*64},{"source_row":99},{"source_role":"TRAIN"},
 {"task_admitted":True},{"adapter_supported":True},{"source_reference_hash_exposure":False},
 {"payload_json":'{"reaction_smarts":"DIFFERENT"}'},{"quality_reasons_json":"[0]"}])
def test_rehashed_typed_payload_source_mask_role_or_target_mutation_rejected(api,store,change):
    reader,sel,*_=store(overrides={0:change})
    with pytest.raises(ValueError):reader.read_entries((sel[0],))

def test_source_and_typed_freshness_between_calls(api,store):
    reader,sel,raw,typed,guards,rows=store();c=reader.read_entries((sel[0],))[0]
    raw.write_bytes(raw.read_bytes().replace(b'"gold"',b'"evil"'))
    with pytest.raises(ValueError,match="identity"):reader.verify(c)

def test_selected_typed_fresh_hash_even_if_current_stat_guard_rebound(api,store):
    reader,sel,raw,typed,guards,rows=store()
    payload=bytearray(typed.read_bytes());payload[len(payload)//2]^=1;typed.write_bytes(payload)
    guards["typed"]["file_identity"]=api._identity(typed.stat())
    with pytest.raises(ValueError,match="integrity"):reader.read_entries((sel[0],))

def test_source_entry_mutation_even_if_current_stat_guard_rebound(api,store):
    reader,sel,raw,typed,guards,rows=store()
    raw.write_bytes(raw.read_bytes().replace(b'"gold"',b'"evil"'));guards["original"]["file_identity"]=api._identity(raw.stat())
    with pytest.raises(ValueError,match="canonical|native"):reader.read_entries((sel[0],))

def test_candidate_rehashed_payload_and_nested_bool_int_alias(api,store):
    reader,sel,*_=store();c=reader.read_entries((sel[2],))[0]
    row=c.native_record();row["payload_json"]=row["payload_json"].replace("false","0")
    with pytest.raises(ValueError):replace(c,typed_record_bytes=canonical(row))
    binding=c.binding();binding["model_use_allowed"]=0
    with pytest.raises(ValueError):replace(c,binding_bytes=canonical(binding))

def test_candidate_wrong_original_order_is_not_reconstructed_from_sorted_typed(api,store):
    reader,sel,*_=store();c=reader.read_entries((sel[0],))[0]
    changed=replace(c,native_entry_bytes=canonical(json.loads(c.native_entry_bytes)))
    with pytest.raises(ValueError,match="replay"):reader.verify(changed)

def test_caller_selection_bounds_and_duplicates(api,store):
    reader,sel,*_=store()
    for request in [[],(),(sel[0],sel[0]),tuple(sel[0] for _ in range(9)),(True,)]:
        with pytest.raises(ValueError):reader.read_entries(request)
    with pytest.raises(ValueError):reader.read_entries((replace(sel[0],source_file_sha256="f"*64),))

def test_unsupported_late_access_and_prefix_extent_refuse_without_crop(api,store,monkeypatch):
    reader,sel,*_=store()
    monkeypatch.setattr(api,"MAX_PREFIX_ENTRIES",2)
    with pytest.raises(ValueError,match="prefix"):reader.read_entries((sel[2],))
    monkeypatch.setattr(api,"MAX_PREFIX_BYTES",8)
    with pytest.raises(ValueError,match="bound|prefix"):reader.read_entries((sel[0],))

def test_original_array_shape_trailing_or_truncated_entry_reject(api,store):
    for raw in [b'{}',b'[{"reaction_smarts":"x"}',b'[1]',b'[,{}]',b'[{"reaction_smarts":"x"},]']:
        reader,sel,*_=store([b'{"reaction_smarts":"x"}'],raw=raw)
        with pytest.raises(ValueError):reader.read_entries(sel)

def test_source_mutation_during_bounded_original_replay(api,store,monkeypatch):
    reader,sel,raw,typed,guards,rows=store()
    original=api.parse_native_entry
    def changing(payload):
        raw.touch()
        return original(payload)
    monkeypatch.setattr(api,"parse_native_entry",changing)
    with pytest.raises(ValueError,match="changed|identity"):reader.read_entries((sel[0],))

def test_rowgroup_allocation_cap_and_unsafe_source_format(api,store,monkeypatch):
    reader,sel,*_=store()
    monkeypatch.setattr(api,"MAX_ROW_GROUP_BYTES",1)
    with pytest.raises(ValueError,match="rowgroup"):reader.read_entries((sel[0],))
    for path in ["escape.pkl","unsafe.py","data.h5"]:
        with pytest.raises(ValueError):reader._path(path,{".json",".parquet"})

def test_finalized_release_pins_and_unrelated_handoff_rejected(api,tmp_path):
    with pytest.raises(ValueError):api.PaRoutesLookupSource(root=tmp_path,handoff="fake.json")
    with pytest.raises(ValueError):api.PaRoutesLookupSource(root=tmp_path,expected_manifest_sha256="f"*64)

def test_complete_array_tail_is_checked_within_bound(api,store,monkeypatch):
    entry=b'{"reaction_smarts":"[C]>>[C]"}'
    reader,sel,*_=store([entry],raw=b"["+entry+b"]"+b" "*140000+b"UNEXPECTED")
    with pytest.raises(ValueError,match="trailing|extent"):reader.read_entries(sel)

def test_final_tail_cannot_exceed_prefix_read_budget(api,store,monkeypatch):
    entry=b'{"reaction_smarts":"[C]>>[C]"}'
    reader,sel,*_=store([entry],raw=b"["+entry+b"]"+b" "*100)
    monkeypatch.setattr(api,"MAX_PREFIX_BYTES",len(entry)+5)
    with pytest.raises(ValueError,match="bound|prefix"):reader.read_entries(sel)

def test_optional_rule_missingness_and_native_quarantine_are_not_repaired(api,store):
    reader,sel,*_=store([b'{"empty":"","number":1.00,"unknown":{"a":[null,false]}}'])
    c,=reader.read_entries(sel)
    assert c.field_status("reaction_smarts")=="not_encoded"
    assert c.native_value().unique("number").lexical=="1.00"
    with pytest.raises(ValueError):c.rule_evidence()
    reader,sel,*_=store(overrides={0:{"quality_reasons_json":'["native_computed_template_SMARTS_parse_failed"]'}})
    c,=reader.read_entries((sel[0],))
    assert c.status=="native_computed_quarantined_unadmitted"
    with pytest.raises(ValueError):c.rule_evidence()

@pytest.fixture
def metadata_source(api):
    root=next(p for p in Path(__file__).resolve().parents if (p/"AGENTS.md").is_file())
    source=api.PaRoutesLookupSource.__new__(api.PaRoutesLookupSource);source.root=root
    return source

def test_pinned_full_release_metadata_is_not_random_original_order_access(api,metadata_source):
    entry,refs=metadata_source._base_snapshot()
    assert entry["asset"]["source_rows"]==42554
    assert entry["original"]["sha256"]==api.SOURCE_SHA256
    assert entry["asset"]["typed_sha256"]==api.TYPED_SHA256
    assert refs["manifest"]["sha256"]==api.MANIFEST_SHA256

@pytest.mark.parametrize("target,key,value",[
 ("handoff","source_worker_task_admitted",False),
 ("handoff","model_use_allowed",0),
 ("manifest","task_admitted_rows",False),
 ("manifest","source_files",True),
 ("proof","no_new_task_admission",1),
 ("proof","no_missing_or_inprogress_payloads",False)])
def test_rehashed_proof_numeric_bool_and_scope_aliases_refuse(api,metadata_source,monkeypatch,target,key,value):
    original=metadata_source._json
    paths={"handoff":api.HANDOFF_PATH,"manifest":api.MANIFEST_PATH}
    def changed(path,sha,size=None):
        value_in,ref=original(path,sha,size)
        if path==paths.get(target) or target=="proof" and sha==api.PROOF_SHA256:
            value_in[key]=value
        return value_in,ref
    monkeypatch.setattr(metadata_source,"_json",changed)
    with pytest.raises(ValueError):metadata_source._base_snapshot()

def test_frozen_original_order_identity_handoff_is_bound(api,metadata_source):
    entry,guards,refs=metadata_source._snapshot()
    assert set(guards)=={"original","typed"}
    assert entry["asset"]["source_rows"]==42554
    assert refs["identity_supplement"]["sha256"]==api.IDENTITY_SHA256
    assert guards["original"]["bounded_prefix_anchor"]["bytes"]==65536
    assert guards["original"]["bounded_prefix_anchor"]["sha256"]==api.PREFIX_SHA256

@pytest.mark.parametrize("mutation",["mtime","bool_identity","lineage","raw_fresh","admission_alias"])
def test_source_identity_receipt_lineage_and_fake_admission_aliases_refuse(api,metadata_source,monkeypatch,mutation):
    original=metadata_source._json
    def changed(path,sha,size=None):
        value,ref=original(path,sha,size)
        if path==api.IDENTITY_PATH:
            if mutation=="mtime":value["guards"][0]["file_identity"]["mtime_ns"]=value["guards"][0]["QC_receipt_file_identity"]["mtime_ns"]+1
            if mutation=="bool_identity":value["guards"][0]["file_identity"]["ino"]=True
            if mutation=="lineage":value["guards"][0]["current_mtime_ctime_predate_individual_QC_receipt"]=1
            if mutation=="raw_fresh":value["original_raw_hash_newly_computed"]=True
        if path==api.FIXTURE_HANDOFF_PATH and mutation=="admission_alias":value["task_admitted"]=False
        return value,ref
    monkeypatch.setattr(metadata_source,"_json",changed)
    with pytest.raises(ValueError):metadata_source._snapshot()

def test_fresh_prefix_anchor_and_literal_computed_context_roles(api,store):
    reader,sel,raw,typed,guards,rows=store()
    guards["original"]["bounded_prefix_anchor"]={"bytes":raw.stat().st_size,"sha256":"a"*64}
    with pytest.raises(ValueError,match="prefix.*integrity"):reader.read_entries((sel[0],))
    reader,sel,*_=store([b'{"reaction_smarts":"[C]>>[C]","count":3,"index":0,"dimer_only":false,"intra_only":true,"necessary_reagent":"","references":["r","r"],"unknown":null}'])
    c,=reader.read_entries(sel)
    assert c.field_role("count")=="source_template_count_not_independent_measurements"
    assert c.field_role("dimer_only")=="source_computed_applicability_flag_not_validated_feasibility"
    assert c.field_role("index")=="native_source_index_no_automatic_join"
    assert c.field_role("unknown")=="native_unqualified_field"
    assert c.field_status("necessary_reagent")=="native_empty_string"
    assert c.native_value().unique("references")==("r","r")
