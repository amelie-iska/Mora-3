"""Full PaRoutes route/stock candidate contracts; synthetic edges are not source coverage."""
import dataclasses,hashlib,importlib.util,json,os,sys
from pathlib import Path
import pytest
ROOT=Path(os.environ.get("MORA3_REPO_ROOT",Path.cwd()))
PACKAGE=ROOT/"outputs/implementation/aux-native-readers-20261101/PaRoutes-full-native-v1"
def load():
 path=Path(os.environ.get("MORA3_PAROUTES_FULL_MODULE",ROOT/"src/mora3/data/paroutes_full.py"))
 try:
  spec=importlib.util.spec_from_file_location("mora3.data.paroutes_full_test",path);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m
 except FileNotFoundError:pytest.fail("source-bound full PaRoutes route reader is missing")
def test_full_route_entrypoint_rejects_boolean_occurrence():
 api=load()
 with pytest.raises(ValueError):api.PaRouteSelector("ref_routes_n1.json",True,"a"*64,"b"*64)
@pytest.fixture(scope="module")
def api():return load()
def canonical(value):return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def tree():
 return dict(type="mol",smiles="[13CH3:7][C@@H:8](F)Cl",children=[
  dict(type="reaction",metadata=dict(rsmi="TARGET>>GOLD",reaction_hash="AUDIT"),children=[
   dict(type="mol",smiles="CC",node_id="same",children=[]),dict(type="mol",smiles="CC",node_id="same",children=[])]),
  dict(type="reaction",metadata=dict(template="GOLD_TEMPLATE",yield_value=99),children=[dict(type="mol",smiles="N",children=[])])])
def candidate(api,payload=None,**changes):
 selector=api.PaRouteSelector("ref_routes_n1.json",0,"a"*64,"b"*64)
 row=dict(source_file_sha256="a"*64,source_row=0,source_role="EVALUATION_EXPOSURE_CONTROL_ONLY",
  payload_json=canonical(payload if payload is not None else tree()).decode(),quality_reasons_json="[]",
  source_reference_hash_exposure=True,task_admitted=False,adapter_supported=False)
 row.update(changes)
 binding=dict(schema="mora3.PaRoutesFullCandidateBinding/v1",handoff_sha256=api.HANDOFF_SHA256,manifest_sha256=api.MANIFEST_SHA256,
  serialization_role="source_prepared_canonical_JSON_not_original_raw_serialization")
 return api.PaRouteCandidate(selector,canonical(row),canonical(binding))
def stock(api,**changes):
 binding=dict(schema="mora3.PaRoutesFullStockBinding/v1",handoff_sha256=api.HANDOFF_SHA256,manifest_sha256=api.MANIFEST_SHA256,
  source_role="EVALUATION_EXPOSURE_CONTROL_ONLY",source_rows=3)
 binding.update(changes)
 return api.PaRoutesStockSnapshot("stock_n1.txt",b"CC\r\nN\nCC\n",canonical(binding))

@pytest.mark.parametrize("field,value",[
 ("source_name","selected_reactions.csv"),("source_name","../ref_routes_n1.json"),("source_row",False),("source_row",-1),
 ("source_row",1.0),("source_file_sha256",True),("typed_file_sha256","0"*63)])
def test_selector_types_source_family_and_extent_do_not_alias(api,field,value):
 values=dict(source_name="ref_routes_n1.json",source_row=0,source_file_sha256="a"*64,typed_file_sha256="b"*64);values[field]=value
 with pytest.raises(ValueError):api.PaRouteSelector(**values)

def test_synthetic_occurrence_pointers_preserve_repeated_SMILES_alternatives_and_AND_order(api):
 c=candidate(api);nodes=c.nodes()
 assert [n.pointer for n in nodes]==["","/children/0","/children/0/children/0","/children/0/children/1","/children/1","/children/1/children/0"]
 assert nodes[2].smiles==nodes[3].smiles=="CC" and nodes[2].pointer!=nodes[3].pointer
 assert nodes[2].parent_pointer==nodes[3].parent_pointer=="/children/0"
 assert c.alternatives("")==("/children/0","/children/1")
 assert c.precursor_dependencies("/children/0")==("/children/0/children/0","/children/0/children/1")
 assert c.native_tree()["children"][0]["metadata"]["rsmi"]=="TARGET>>GOLD"
 assert c.neural_token_count is None and c.actual_purchasability_verified is False
 assert c.source_reference_hash_exposure is True
 assert api._product_query(c)=={"query_role":"product_conditioned_route_query","product":{"syntax":"SMILES","value":"[13CH3][C@@H](F)Cl"}}

def test_synthetic_target_route_and_audit_changes_do_not_enter_default_query(api):
 original=candidate(api);modified=tree();modified["children"]=[dict(type="reaction",metadata={"secret":"GOLD"},children=[])]
 modified["metadata"]={"patent":"AUDIT","target_length":99999,"gold_class":"GOLD"};modified["in_stock"]=True
 mutated=candidate(api,payload=modified)
 assert api._product_query(original)==api._product_query(mutated)
 assert set(api._product_query(mutated))=={"query_role","product"}

@pytest.mark.parametrize("field,value",[
 ("source_row",False),("source_reference_hash_exposure",1),("task_admitted",0),("adapter_supported",0),
 ("source_role","CANDIDATE_REQUIRES_HELDOUT_AND_CHEMICAL_GROUP_EXCLUSION"),("source_file_sha256","0"*64),
 ("quality_reasons_json","{}"),("payload_json",'{"type":"mol","type":"reaction"}')])
def test_synthetic_rehashed_row_binding_type_mask_and_duplicate_key_refusal(api,field,value):
 with pytest.raises(ValueError):candidate(api,**{field:value})

def test_synthetic_source_quarantine_is_not_repaired_or_promoted(api):
 c=candidate(api,quality_reasons_json='["native_reaction_SMILES_parse_failed"]')
 assert c.status=="native_quarantined_unadmitted"
 assert c.quality_reasons==("native_reaction_SMILES_parse_failed",)
 assert len(c.nodes())==6
 with pytest.raises(ValueError):api._product_query(c)
 for purpose in ("training","evaluation","positive_loss"):
  with pytest.raises(ValueError):c.require_model_use(purpose,admission={"approved":True})

@pytest.mark.parametrize("payload",[
 {"type":"mol","smiles":"CC","children":"bad"},
 {"type":"mol","smiles":"CC","children":[{"type":"mol","smiles":"N"}]},
 {"type":"unknown","smiles":"CC"},
])
def test_synthetic_unsupported_source_topology_never_invents_edges(api,payload):
 c=candidate(api,payload=payload,quality_reasons_json='["synthetic_structural_test"]')
 assert c.native_tree()==payload
 with pytest.raises(ValueError):c.nodes()

def test_synthetic_actual_python_cycle_rejected_shared_object_occurrences_not_merged(api):
 cyclic=tree();cyclic["children"][0]["children"].append(cyclic)
 with pytest.raises(ValueError):api._node_occurrences(cyclic)
 shared={"type":"mol","smiles":"CC"};native={"type":"mol","smiles":"N","children":[{"type":"reaction","children":[shared,shared]}]}
 nodes=api._node_occurrences(native)
 assert len(nodes)==4 and nodes[-1].pointer!=nodes[-2].pointer

def test_synthetic_stock_preserves_order_duplicates_line_endings_and_eval_role(api):
 snapshot=stock(api)
 assert snapshot.raw_bytes==b"CC\r\nN\nCC\n" and snapshot.entries()==("CC","N","CC")
 assert snapshot.contains_literal("CC") is True and snapshot.contains_literal("C(C)") is False
 partial=stock(api,source_rows=10,selected_rows=[0,1,2])
 assert partial.coverage=="bounded_positive_subset" and partial.contains_literal("C(C)") is None
 assert snapshot.actual_purchasability_verified is False
 with pytest.raises(ValueError):stock(api,source_role="training")
 with pytest.raises(ValueError):snapshot.require_model_use("training",admission=True)
 request=api.RequestedStockContext(snapshot)
 out=api._product_query(candidate(api),requested_stock=request)
 assert out["requested_stock"]=={"role":"explicit_caller_requested_benchmark_stock","coverage":"declared_entries_complete","entries":[{"syntax":"SMILES","value":"CC"},{"syntax":"SMILES","value":"N"},{"syntax":"SMILES","value":"CC"}]}
 assert "requested_stock" not in api._product_query(candidate(api))
 with pytest.raises(ValueError):api._product_query(candidate(api),requested_stock=snapshot)

@pytest.mark.parametrize("name",("all_loaded_routes.json.gz","ref_routes_n1.json","ref_routes_n5.json"))
def test_synthetic_all_source_roles_and_false_exposure_never_admit_model_use(api,name):
 c=candidate(api)
 row=c.native_record();role="CANDIDATE_REQUIRES_HELDOUT_AND_CHEMICAL_GROUP_EXCLUSION" if name.startswith("all_") else "EVALUATION_EXPOSURE_CONTROL_ONLY"
 row.update(source_role=role,source_reference_hash_exposure=False)
 c=api.PaRouteCandidate(api.PaRouteSelector(name,0,"a"*64,"b"*64),canonical(row),c.binding_bytes)
 assert c.source_reference_hash_exposure is False
 for purpose in ("training","evaluation","positive_loss","generation"):
  with pytest.raises(ValueError):c.require_model_use(purpose,admission={"verified":True,"master":"fake"})

@pytest.mark.parametrize("raw",(b'{"a":NaN}',b'{"a":Infinity}',b'{"a":0,"a":1}'))
def test_strict_source_json_rejects_nonfinite_and_duplicate_aliases(api,raw):
 with pytest.raises(ValueError):api._strict_json(raw)

def test_exact_full_release_pin_rejects_wrong_manifest_and_unsafe_source(api):
 with pytest.raises(ValueError):api.PaRoutesFullNativeSource(root=ROOT,expected_manifest_sha256="0"*64)
 with pytest.raises(ValueError):api.PaRoutesFullNativeSource(root=ROOT,handoff="foreign.pkl")

@pytest.fixture(scope="module")
def actual():
 pointer=PACKAGE/"source-fixture-reference.json"
 if not pointer.exists():pytest.skip("source-owned bounded full PaRoutes fixture unavailable")
 reference=json.loads(pointer.read_text());raw=Path(reference["fixture"]["path"]).read_bytes()
 assert hashlib.sha256(raw).hexdigest()==reference["fixture"]["sha256"]
 guardraw=Path(reference["identity_supplement"]["path"]).read_bytes()
 assert hashlib.sha256(guardraw).hexdigest()==reference["identity_supplement"]["sha256"]
 data=json.loads(raw);data["guards"]=json.loads(guardraw)["guards"];return data
def selectors(api,actual):
 guards={(g["source_name"],g["kind"]):g for g in actual["guards"]}
 return tuple(api.PaRouteSelector(r["source_name"],r["source_row_ordinal"],r["source_whole_sha256"],guards[(r["source_name"],"typed")]["expected_whole_sha256"]) for r in actual["records"])

def test_actual_eight_routes_two_shards_exact_fingerprints_nodes_roles_and_refusals(api,actual):
 source=api.PaRoutesFullNativeSource(root=ROOT);selected=selectors(api,actual)
 assert len(selected)==8 and len({s.source_name for s in selected})==2
 values=source.read_routes(selected);exposed=evalonly=quarantined=0
 for expected,value in zip(actual["records"],values):
  assert hashlib.sha256(value.typed_record_bytes).hexdigest()==expected["typed_row_canonical_json_sha256"]
  assert canonical(value.native_tree())==canonical(expected["native_route"])
  assert hashlib.sha256(value.native_record()["payload_json"].encode()).hexdigest()==expected["parsed_native_route_canonical_json_sha256"]
  assert [n.pointer for n in value.nodes()]==[n["json_pointer"] for n in expected["node_pointer_map"]]
  assert [n.kind for n in value.nodes()]==[n["native_type"] for n in expected["node_pointer_map"]]
  assert value.native_record()==expected["typed_row_json"]
  assert source.verify(value).binding_sha256==value.binding_sha256
  exposed+=value.source_reference_hash_exposure;evalonly+=value.source_role=="EVALUATION_EXPOSURE_CONTROL_ONLY";quarantined+=bool(value.quality_reasons)
  for purpose in ("training","evaluation","positive_loss"):
   with pytest.raises(ValueError):value.require_model_use(purpose,admission={"verified":True})
 assert (exposed,evalonly,quarantined)==(6,4,1)

def test_actual_requested_order_duplicate_limits_wrong_shard_and_closed_table_refusal(api,actual):
 source=api.PaRoutesFullNativeSource(root=ROOT);selected=selectors(api,actual)
 before=source.read_routes(selected);backward=source.read_routes(tuple(reversed(selected)))
 assert [v.binding_sha256 for v in backward]==list(reversed([v.binding_sha256 for v in before]))
 for invalid in ((selected[0],selected[0]),(),tuple([selected[0]]*9)):
  with pytest.raises(ValueError):source.read_routes(invalid)
 with pytest.raises(ValueError):source.read_routes((dataclasses.replace(selected[0],source_file_sha256="0"*64),))
 with pytest.raises(ValueError):source.read_routes((dataclasses.replace(selected[0],typed_file_sha256="0"*64),))
 with pytest.raises(ValueError):source.read_routes((dataclasses.replace(selected[0],source_row=10**12),))
 with pytest.raises(ValueError):source.read_stock_snapshot("targets_n1.txt",(0,))

def test_actual_partial_native_stock_line_and_snapshot_membership_are_not_full_inventory(api,actual):
 source=api.PaRoutesFullNativeSource(root=ROOT);expected=actual["small_stock_auxiliary"]
 snapshot=source.read_stock_snapshot(expected["source_name"],(expected["source_row_ordinal"],))
 assert snapshot.raw_bytes==expected["original_line_utf8"].encode()
 assert hashlib.sha256(snapshot.raw_bytes).hexdigest()==expected["original_line_sha256"]
 native=json.loads(expected["typed_row_json"]["payload_json"])
 assert snapshot.entries()==(native,) and snapshot.contains_literal(native) is True
 assert snapshot.coverage=="bounded_positive_subset" and snapshot.contains_literal("UNOBSERVED_NATIVE_LITERAL") is None
 assert source.verify_stock(snapshot).binding_sha256==snapshot.binding_sha256
 with pytest.raises(ValueError):snapshot.require_model_use("training",admission={"approved":True})
 first=source.read_routes(selectors(api,actual)[:1])[0];default=source.project_product_inputs(first)
 assert set(default)=={"query_role","product"}
 explicit=source.project_product_inputs(first,requested_stock=api.RequestedStockContext(snapshot))
 assert explicit["product"]==default["product"] and explicit["requested_stock"]["coverage"]=="bounded_positive_subset"
 assert explicit["requested_stock"]["entries"]==[{"syntax":"SMILES","value":native}]

def test_actual_rehashed_hidden_tree_and_audit_type_mutations_fail_public_replay(api,actual):
 source=api.PaRoutesFullNativeSource(root=ROOT);value=source.read_routes(selectors(api,actual)[:1])[0]
 row=value.native_record();payload=value.native_tree();payload["metadata"]={"hidden_target_attack":123}
 row["payload_json"]=canonical(payload).decode();mutated=dataclasses.replace(value,typed_record_bytes=canonical(row))
 with pytest.raises(ValueError):source.project_product_inputs(mutated)
 audit=value.binding();audit["model_use_allowed"]=0;mutated=dataclasses.replace(value,binding_bytes=canonical(audit))
 assert mutated.binding_sha256!=value.binding_sha256
 with pytest.raises(ValueError):source.verify(mutated)

def test_actual_raw_route_archives_are_not_opened_by_bounded_reader(api,actual,monkeypatch):
 source=api.PaRoutesFullNativeSource(root=ROOT);rawpaths={Path(r["source_path"]) for r in actual["records"]};open_=Path.open
 def protected(path,*args,**kwargs):
  if path in rawpaths:raise AssertionError("raw route archive unexpectedly opened/rescanned")
  return open_(path,*args,**kwargs)
 monkeypatch.setattr(Path,"open",protected)
 assert len(source.read_routes(selectors(api,actual)))==8

def test_actual_inherited_original_identity_change_refuses_without_raw_rehash(api,actual,monkeypatch):
 source=api.PaRoutesFullNativeSource(root=ROOT);target=Path(actual["records"][0]["source_path"]);other=Path(actual["records"][-1]["source_path"]);stat=Path.stat
 def changed(path,*args,**kwargs):return stat(other,*args,**kwargs) if path==target else stat(path,*args,**kwargs)
 monkeypatch.setattr(Path,"stat",changed)
 with pytest.raises(ValueError):source.read_routes(selectors(api,actual)[:1])

def test_actual_fresh_typed_digest_and_identity_guard_document_tamper_refuse(api,actual,monkeypatch):
 source=api.PaRoutesFullNativeSource(root=ROOT);original=api._stream_sha
 monkeypatch.setattr(api,"_stream_sha",lambda handle:"0"*64)
 with pytest.raises(ValueError):source.read_routes(selectors(api,actual)[:1])
 monkeypatch.setattr(api,"_stream_sha",original)
 target=(ROOT/api.IDENTITY_PATH).resolve();read=Path.read_bytes
 def changed(path):
  raw=read(path);return raw+b" " if path==target else raw
 monkeypatch.setattr(Path,"read_bytes",changed)
 with pytest.raises(ValueError):source.read_routes(selectors(api,actual)[:1])

@pytest.mark.parametrize("where",("selector","stock_snapshot","stock_read"))
def test_exact_source_names_refuse_string_subclass_aliases(api,where):
 class Alias(str):pass
 with pytest.raises(ValueError):
  if where=="selector":api.PaRouteSelector(Alias("ref_routes_n1.json"),0,"a"*64,"b"*64)
  elif where=="stock_snapshot":
   value=stock(api);api.PaRoutesStockSnapshot(Alias("stock_n1.txt"),value.raw_bytes,value.binding_bytes)
  else:api.PaRoutesFullNativeSource(root=ROOT).read_stock_snapshot(Alias("stock_n1.txt"),(0,))

@pytest.mark.parametrize("field,value",(("source_row",False),("source_reference_hash_exposure",0),("task_admitted",0),("adapter_supported",0)))
def test_actual_postdecode_mask_and_coordinate_aliases_are_not_accepted(api,actual,monkeypatch,field,value):
 source=api.PaRoutesFullNativeSource(root=ROOT);read=source._rows
 def changed(*args,**kwargs):
  rows=read(*args,**kwargs)
  for row in rows.values():row[field]=value
  return rows
 monkeypatch.setattr(source,"_rows",changed)
 with pytest.raises(ValueError):source.read_routes(selectors(api,actual)[:1])

def test_actual_third_route_shard_and_large_rowgroup_refuse_without_new_route_reads(api,actual,monkeypatch):
 source=api.PaRoutesFullNativeSource(root=ROOT);selected=selectors(api,actual)
 third=api.PaRouteSelector("ref_routes_n5.json",0,"0"*64,"0"*64)
 with pytest.raises(ValueError):source.read_routes((selected[0],selected[-1],third))
 monkeypatch.setattr(api,"MAX_ROW_GROUP_BYTES",1)
 with pytest.raises(ValueError,match="no crop"):source.read_routes(selected[:1])

def test_actual_public_leaf_membership_keeps_gold_flags_separate_from_partial_stock(api,actual):
 source=api.PaRoutesFullNativeSource(root=ROOT);route=source.read_routes(selectors(api,actual)[:1])[0]
 aux=actual["small_stock_auxiliary"];snapshot=source.read_stock_snapshot(aux["source_name"],(aux["source_row_ordinal"],))
 audit=source.leaf_stock_membership(route,snapshot)
 leaves=[n for n in route.nodes() if n.kind=="mol" and not n.children]
 assert [r["pointer"] for r in audit]==[n.pointer for n in leaves]
 for row,node in zip(audit,leaves):
  assert row["literal_membership"] is snapshot.contains_literal(node.smiles)
  assert row["source_reported_in_stock"]==json.loads(node.native_node_bytes).get("in_stock")
  assert row["current_purchasability_verified"] is False
  if node.smiles not in snapshot.entries():assert row["literal_membership"] is None
