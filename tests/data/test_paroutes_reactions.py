"""PaRoutes native table contracts; synthetic examples are not real-source coverage."""
import dataclasses,hashlib,importlib.util,json,os,sys
from pathlib import Path
import pytest
ROOT=Path(os.environ.get("MORA3_REPO_ROOT",Path.cwd()))
PACKAGE=ROOT/"outputs/implementation/aux-native-readers-20261101/PaRoutes-reaction-tables-native-v1"
def load():
 path=Path(os.environ.get("MORA3_PAROUTES_TABLE_MODULE",ROOT/"src/mora3/data/paroutes_reactions.py"))
 try:
  s=importlib.util.spec_from_file_location("mora3.data.paroutes_tables_test",path);m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);return m
 except FileNotFoundError:pytest.fail("source-bound PaRoutes reaction/template table reader is missing")
def test_native_table_entrypoint_rejects_boolean_occurrence():
 api=load()
 with pytest.raises(ValueError):api.PaTableSelector("selected_reactions.csv",True,"a"*64,"b"*64)
@pytest.fixture(scope="module")
def api():return load()
def canonical(value):return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def candidate(api,name="selected_reactions.csv",cells=None,**changes):
 if cells is None:cells=["[CH3:1]O>>[13CH3:7][C@@H:8](F)Cl","HASH","GOLD_CLASS","0"]
 selector=api.PaTableSelector(name,0,"a"*64,"b"*64);header=api.NativeTSVHeader(api.TABLE_HEADERS[name])
 row=dict(source_file_sha256="a"*64,source_row=0,source_role=api.TABLE_ROLES[name],payload_json=canonical(cells).decode(),
  quality_reasons_json="[]",source_reference_hash_exposure=True,task_admitted=False,adapter_supported=False)
 row.update(changes)
 binding=dict(schema="mora3.PaRoutesNativeTableBinding/v1",handoff_sha256=api.HANDOFF_SHA256,manifest_sha256=api.MANIFEST_SHA256,
  serialization_role="canonical parsed native TSV cell array; not original source row serialization")
 return api.PaTableCandidate(selector,header,canonical(row),canonical(binding))

@pytest.mark.parametrize("field,value",[
 ("source_name","ref_routes_n1.json"),("source_name","../selected_reactions.csv"),("source_row",False),
 ("source_row",-1),("source_row",1.0),("source_file_sha256",0),("typed_file_sha256","bad")])
def test_exact_table_selector_types_and_source_families(api,field,value):
 d=dict(source_name="selected_reactions.csv",source_row=0,source_file_sha256="a"*64,typed_file_sha256="b"*64);d[field]=value
 with pytest.raises(ValueError):api.PaTableSelector(**d)

def test_synthetic_TSV_header_quotes_CRLF_order_and_duplicates(api):
 header=api.parse_tsv_header(b'"with\ttab"\tunknown\r\n')
 assert header.columns==("with\ttab","unknown") and header.native_bytes==b'"with\ttab"\tunknown\r\n'
 for columns in (("a","a"),("",),("a",False),()):
  with pytest.raises(ValueError):api.NativeTSVHeader(columns)
 with pytest.raises(ValueError):api.parse_tsv_header(b"a,b\n",delimiter=",")
 with pytest.raises(ValueError):api.parse_tsv_header(b"a\ta\n")
 with pytest.raises(ValueError):api.parse_tsv_header(b"a\nsecond\n")

def test_synthetic_literal_cells_empty_values_and_source_product_projection(api):
 c=candidate(api,cells=["[CH3:1]O>>[13CH3:7][C@@H:8](F)Cl","","GOLD_CLASS","False"])
 assert c.native_cells()==("[CH3:1]O>>[13CH3:7][C@@H:8](F)Cl","","GOLD_CLASS","False")
 assert c.native_fields()["RingBreaker"]=="False"
 assert c.field_status("PseudoHash")=="native_empty_literal" and c.field_status("missing")=="not_encoded"
 assert c.reaction_evidence()[0].syntax=="REACTION_SMILES"
 assert api._product_query(c)=={"query_role":"product_conditioned_chemical_retrosynthesis","products":[{"syntax":"SMILES","value":"[13CH3][C@@H](F)Cl"}]}
 assert c.neural_token_count is None
 for purpose in ("training","evaluation","positive_loss"):
  with pytest.raises(ValueError):c.require_model_use(purpose,admission={"approved":True})

def test_synthetic_raw_and_processed_reaction_roles_require_explicit_choice(api):
 cells=[""]*len(api.TABLE_HEADERS["selected_reactions_all.csv"])
 cells[api.TABLE_HEADERS["selected_reactions_all.csv"].index("rsmi")]="CC>>C[NH3+:8]"
 cells[api.TABLE_HEADERS["selected_reactions_all.csv"].index("rsmi_processed")]="CC>>[13CH3:2]O"
 c=candidate(api,"selected_reactions_all.csv",cells)
 evidence=c.reaction_evidence()
 assert [(e.column,e.source_role) for e in evidence]==[("rsmi","source_raw_reaction_text"),("rsmi_processed","source_processed_reaction_text")]
 with pytest.raises(ValueError):api._product_query(c)
 assert api._product_query(c,reaction_field="rsmi")["products"]==[{"syntax":"SMILES","value":"C[NH3+]"}]
 assert api._product_query(c,reaction_field="rsmi_processed")["products"]==[{"syntax":"SMILES","value":"[13CH3]O"}]

def test_synthetic_SMARTS_templates_are_not_molecular_product_hints(api):
 c=candidate(api,"uspto_unique_templates.csv.gz",["CODE","[C;H3:1]>>[O:2]","HASH","GOLD","17"],source_reference_hash_exposure=None)
 assert c.source_reference_hash_exposure is None
 t=c.template_evidence()[0]
 assert (t.syntax,t.native_text,t.source_role)==("SMARTS","[C;H3:1]>>[O:2]","source_computed_template_rule")
 assert c.reaction_evidence()==()
 with pytest.raises(ValueError):api._product_query(c)
 with pytest.raises(ValueError):api._product_query(c,reaction_field="retro_template")
 cells=["CC>>C[NH3+:7]","HASH","CLASS","False","[C:1]>>[O:2]","TPL_HASH"]
 library=candidate(api,"uspto_template_library.csv",cells)
 assert library.reaction_evidence()[0].source_role=="source_reaction_reference_in_computed_rule"
 assert api._product_query(library)["products"]==[{"syntax":"SMILES","value":"C[NH3+]"}]
 with pytest.raises(ValueError):api._product_query(library,reaction_field="retro_template")

def test_synthetic_gold_templates_precursors_outcomes_IDs_and_lengths_not_predictors(api):
 a=candidate(api);b=candidate(api,cells=["DIFFERENT_REACTANTS>DIFFERENT_AGENTS>[13CH3:7][C@@H:8](F)Cl","OTHER_HASH","OTHER_CLASS","True"])
 assert api._product_query(a)==api._product_query(b)
 assert set(api._product_query(b))=={"query_role","products"}
 assert "CLASS" not in json.dumps(api._product_query(b))
 c=candidate(api,cells=["CC>>CC.O","HASH","CLASS","0"])
 assert api._product_query(c)["products"]==[{"syntax":"SMILES","value":"CC.O"}]

@pytest.mark.parametrize("reaction",("CC>O","CC>>>O","CC>>","CC>>CC |enhanced_stereo|","CC>> CC"))
def test_synthetic_unsupported_native_reaction_or_extended_product_not_repaired(api,reaction):
 with pytest.raises(ValueError):api._product_query(candidate(api,cells=[reaction,"H","C","0"]))

@pytest.mark.parametrize("field,value",[
 ("source_row",False),("source_reference_hash_exposure",0),("task_admitted",0),("adapter_supported",0),
 ("source_role","training"),("source_file_sha256","0"*64),("quality_reasons_json","{}"),
 ("payload_json",'["CC>>CC",true,"X","0"]')])
def test_synthetic_rehashed_type_mask_source_and_native_cell_alias_refusal(api,field,value):
 with pytest.raises(ValueError):candidate(api,**{field:value})

def test_synthetic_width_quarantine_retains_all_cells_without_padding_or_drop(api):
 c=candidate(api,cells=["CC>>CC","H","C"],quality_reasons_json='["native_TSV_width_mismatch"]')
 assert c.native_cells()==("CC>>CC","H","C") and c.status=="native_quarantined_unadmitted"
 assert c.field_status("RingBreaker")=="unresolved_native_width_mismatch"
 with pytest.raises(ValueError):c.native_fields()
 with pytest.raises(ValueError):api._product_query(c)
 with pytest.raises(ValueError):candidate(api,cells=["CC>>CC","H","C"])

def test_synthetic_header_reordering_and_wrong_delimiter_cannot_rebind_cells(api):
 c=candidate(api)
 wrong=api.NativeTSVHeader(tuple(reversed(c.header.columns)))
 with pytest.raises(ValueError):api.PaTableCandidate(c.selector,wrong,c.typed_record_bytes,c.binding_bytes)
 with pytest.raises(ValueError):api.NativeTSVHeader(c.header.columns,delimiter=",")

def test_exact_source_version_guard_and_template_role_closes_model_admission(api):
 with pytest.raises(ValueError):api.PaRoutesReactionTablesSource(root=ROOT,expected_manifest_sha256="0"*64)
 with pytest.raises(ValueError):api.PaRoutesReactionTablesSource(root=ROOT,handoff="foreign.pkl")

@pytest.fixture(scope="module")
def actual():
 p=PACKAGE/"source-fixture-reference.json"
 if not p.exists():pytest.skip("bounded source-owner table fixture unavailable")
 refs=json.loads(p.read_text());data={}
 for key,ref in refs.items():
  raw=(ROOT/ref["path"]).read_bytes();assert hashlib.sha256(raw).hexdigest()==ref["sha256"];data[key]=json.loads(raw)
 return data
def selectors(api,actual):
 guards={(g["source_name"],g["kind"]):g for g in actual["identity_supplement"]["guards"]}
 return tuple(api.PaTableSelector(r["source_name"],r["source_row_ordinal"],r["source_whole_sha256"],guards[(r["source_name"],"typed")]["expected_whole_sha256"]) for r in actual["fixture"]["records"])

def test_actual_eight_native_table_rows_header_cells_template_and_exposure_fingerprints(api,actual):
 source=api.PaRoutesReactionTablesSource(root=ROOT);selected=selectors(api,actual);values=source.read_rows(selected)
 assert len(values)==8 and len({s.source_name for s in selected})==2
 exposed=quarantined=0
 for r,v in zip(actual["fixture"]["records"],values):
  assert v.native_record()==r["typed_row_json"]
  assert hashlib.sha256(v.typed_record_bytes).hexdigest()==r["typed_row_canonical_json_sha256"]
  assert list(v.header.columns)==r["native_TSV_header"]
  assert list(v.native_cells())==r["ordered_native_cells"]
  assert hashlib.sha256(canonical(list(v.native_cells()))).hexdigest()==r["native_cell_array_canonical_json_sha256"]
  assert source.verify(v).binding_sha256==v.binding_sha256
  exposed+=v.source_reference_hash_exposure is True;quarantined+=bool(v.quality_reasons)
  for purpose in ("training","evaluation","positive_loss"):
   with pytest.raises(ValueError):v.require_model_use(purpose,admission={"approved":True})
 assert (exposed,quarantined)==(4,3)

def test_actual_order_limits_wrong_shard_and_foreign_table_source_refusal(api,actual):
 source=api.PaRoutesReactionTablesSource(root=ROOT);selected=selectors(api,actual)
 before=source.read_rows(selected);back=source.read_rows(tuple(reversed(selected)))
 assert [v.binding_sha256 for v in back]==list(reversed([v.binding_sha256 for v in before]))
 for invalid in ((),(selected[0],selected[0]),tuple([selected[0]]*9)):
  with pytest.raises(ValueError):source.read_rows(invalid)
 with pytest.raises(ValueError):source.read_rows((dataclasses.replace(selected[0],source_file_sha256="0"*64),))
 with pytest.raises(ValueError):source.read_rows((dataclasses.replace(selected[0],typed_file_sha256="0"*64),))
 with pytest.raises(ValueError):source.read_rows((dataclasses.replace(selected[0],source_row=10**12),))

def test_actual_rehashed_target_header_or_audit_mutations_refuse_source_replay(api,actual):
 source=api.PaRoutesReactionTablesSource(root=ROOT);candidate=source.read_rows(selectors(api,actual)[:1])[0]
 row=candidate.native_record();cells=list(candidate.native_cells());cells[-1]="TARGET_ATTACK";row["payload_json"]=canonical(cells).decode()
 mutated=dataclasses.replace(candidate,typed_record_bytes=canonical(row))
 with pytest.raises(ValueError):source.project_product_inputs(mutated)
 audit=candidate.binding();audit["model_use_allowed"]=0;mutated=dataclasses.replace(candidate,binding_bytes=canonical(audit))
 assert mutated.binding_sha256!=candidate.binding_sha256
 with pytest.raises(ValueError):source.verify(mutated)

def test_actual_default_product_queries_never_include_templates_labels_or_source_metadata(api,actual):
 source=api.PaRoutesReactionTablesSource(root=ROOT);values=source.read_rows(selectors(api,actual));previews=0
 for v in values:
  if v.quality_reasons:
   with pytest.raises(ValueError):source.project_product_inputs(v)
  else:
   result=source.project_product_inputs(v);assert set(result)=={"query_role","products"};previews+=1
   assert all(set(x)=={"syntax","value"} for x in result["products"])
 assert previews>0

def test_actual_original_identity_and_fresh_typed_SHA_guard_fail_closed(api,actual,monkeypatch):
 source=api.PaRoutesReactionTablesSource(root=ROOT);selected=selectors(api,actual)
 original=api._stream_sha;monkeypatch.setattr(api,"_stream_sha",lambda handle:"0"*64)
 with pytest.raises(ValueError):source.read_rows(selected[:1])
 monkeypatch.setattr(api,"_stream_sha",original)
 a=Path(actual["fixture"]["records"][0]["source_path"]);b=Path(actual["fixture"]["records"][-1]["source_path"]);stat=Path.stat
 def changed(path,*args,**kwargs):return stat(b,*args,**kwargs) if path==a else stat(path,*args,**kwargs)
 monkeypatch.setattr(Path,"stat",changed)
 with pytest.raises(ValueError):source.read_rows(selected[:1])

@pytest.mark.parametrize("raw",(b"wrong\theader\n",b"RxnSmilesClean,PseudoHash,classification,RingBreaker\n",b""))
def test_declared_native_header_bytes_cannot_disagree_with_columns(api,raw):
 with pytest.raises(ValueError):api.NativeTSVHeader(api.TABLE_HEADERS["selected_reactions.csv"],native_bytes=raw)

@pytest.mark.parametrize("field,value",(("source_row",False),("source_reference_hash_exposure",0),("task_admitted",0),("adapter_supported",0)))
def test_actual_postdecode_coordinate_and_loss_mask_aliases_refuse(api,actual,monkeypatch,field,value):
 source=api.PaRoutesReactionTablesSource(root=ROOT);read=source._rows
 def changed(*args,**kwargs):
  rows=read(*args,**kwargs)
  for row in rows.values():row[field]=value
  return rows
 monkeypatch.setattr(source,"_rows",changed)
 with pytest.raises(ValueError):source.read_rows(selectors(api,actual)[:1])

def test_actual_native_header_decode_substitution_refuses_without_raw_row_scan(api,actual,monkeypatch):
 source=api.PaRoutesReactionTablesSource(root=ROOT)
 monkeypatch.setattr(api,"parse_tsv_header",lambda raw:api.NativeTSVHeader(("wrong","header")))
 with pytest.raises(ValueError):source.read_rows(selectors(api,actual)[:1])

def test_actual_third_shard_and_rowgroup_bounds_refuse_without_new_table_rows(api,actual,monkeypatch):
 source=api.PaRoutesReactionTablesSource(root=ROOT);selected=selectors(api,actual)
 third=api.PaTableSelector("selected_reactions_all.csv",0,"0"*64,"0"*64)
 with pytest.raises(ValueError):source.read_rows((selected[0],selected[-1],third))
 monkeypatch.setattr(api,"MAX_ROW_GROUP_BYTES",1)
 with pytest.raises(ValueError,match="no crop"):source.read_rows(selected[:1])

def test_actual_header_accounting_is_logical_prefix_not_unmeasured_physical_IO(api,actual):
 source=api.PaRoutesReactionTablesSource(root=ROOT);value=source.read_rows(selectors(api,actual)[:1])[0]
 audit=value.binding()["header_read"]
 assert audit["logical_source_prefix_bytes_consumed"]==len(value.header.native_bytes)
 assert "physical_source_bytes_read" not in audit
