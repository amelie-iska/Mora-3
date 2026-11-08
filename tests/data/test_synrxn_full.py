"""Synthetic contract/IO cases; bounded actual-source coverage is separate."""
from dataclasses import replace
from pathlib import Path
import csv,gzip,hashlib,importlib.util,io,json,sys
import pytest
def canonical(v):return json.dumps(v,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def sha(v):return hashlib.sha256(v).hexdigest()
@pytest.fixture
def api():
 path=Path(__file__).resolve().parents[2]/"src/mora3/data/synrxn_full.py"
 name="mora3.data._staged_synrxn_full_test";previous=sys.modules.get(name)
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module
 try:spec.loader.exec_module(module);yield module
 finally:
  if previous is None:sys.modules.pop(name,None)
  else:sys.modules[name]=previous
@pytest.mark.parametrize("name,columns",[
 ("uspto_500.csv.gz",("r_id","rxn","reagent","split")),
 ("uspto_50k.csv.gz",("r_id","aam","split","source")),
 ("uspto_mit.csv.gz",("r_id","aam","split","rc"))])
def test_exact_table_header_and_family_are_not_interchangeable(api,name,columns):
 header=api.parse_csv_header((",".join(columns)+"\r\n").encode())
 assert header.columns==columns and header.native_bytes.endswith(b"\r\n")
 with pytest.raises(ValueError):api.NativeCSVHeader(tuple(reversed(columns)),header.native_bytes)
 for raw in [b"r_id,r_id\n",b"r_id;rxn;split\n",b"r_id,rxn\nx,y\n",b'bad,"unterminated\n']:
  with pytest.raises(ValueError):api.parse_csv_header(raw,expected_columns=columns)

@pytest.fixture
def store(api,tmp_path,monkeypatch):
 import pyarrow as pa,pyarrow.parquet as pq
 pa.set_cpu_count(1);pa.set_io_thread_count(1)
 def make(names=("uspto_500.csv.gz","uspto_mit.csv.gz"),overrides=None,cells_by_name=None):
  entries={};guards={};selectors={}
  for name in names:
   columns=api.HEADERS[name];split="val" if name=="uspto_500.csv.gz" else "valid"
   default=[["id0","[CH3:1]Br>[Na+]Cl>[C@@H:2](F)Cl","HIDDEN","train"],
            ["id1","C>>N","reagent","test"],["id2","N>>O","nan",split]]
   if name!="uspto_500.csv.gz":default=[["id0",default[0][1],"train","GOLD"],["id1","C>>N","test","0"],["id2","N>>O",split,""]]
   cells=(cells_by_name or {}).get(name,default)
   text=io.StringIO(newline="");writer=csv.writer(text);writer.writerow(columns);writer.writerows(cells)
   original=tmp_path/name
   with original.open("wb") as h:h.write(gzip.compress(text.getvalue().encode()))
   source_sha=sha(original.read_bytes());rows=[]
   for i,values in enumerate(cells):
    d=dict(zip(columns,values));native_split=d["split"];role=api.TRAIN if native_split=="train" else api.EVAL
    reasons=["missing_native_row_identifier"] if not d["r_id"] else []
    row=dict(source_file_sha256=source_sha,reaction_id=d["r_id"],native_columns_json=canonical(list(columns)).decode(),
     native_cells_json=canonical(values).decode(),native_split=native_split,source_protocol_role=role,
     gold_target_columns_json=canonical([c for c in columns if c in ("rc","reagent")]).decode(),
     default_predictor_columns_json="[]",quality_reasons_json=canonical(reasons).decode(),source_row=i,
     task_admitted=False,adapter_supported=False,model_use_allowed=False,
     strict_molecule_reaction_parse_pass=True,query_reaction_parse_pass=True)
    if overrides and name in overrides and i in overrides[name]:row.update(overrides[name][i])
    rows.append(row)
   typed=tmp_path/(name+".parquet");pq.write_table(pa.Table.from_pylist(rows,schema=api.arrow_schema()),typed,row_group_size=2)
   asset=dict(typed_path=str(typed),typed_sha256=sha(typed.read_bytes()),typed_bytes=typed.stat().st_size,source_rows=len(rows))
   entry=dict(asset=asset,original=dict(path=str(original),sha256=source_sha,bytes=original.stat().st_size))
   entries[name]=entry
   for kind,p,expected in [("original",original,source_sha),("typed",typed,asset["typed_sha256"])]:
    guards[(name,kind)]=dict(path=str(p),file_identity=api._identity(p.stat()),expected_whole_sha256=expected)
   selectors[name]=tuple(api.SynRXNSelector(name,i,source_sha,asset["typed_sha256"]) for i in range(len(rows)))
  reader=api.SynRXNSource.__new__(api.SynRXNSource);reader.root=tmp_path.resolve()
  reader._snapshot=lambda:(entries,guards,dict(synthetic=True))
  reader._receipt=lambda name,entry:(dict(semantic_rows_digest="d"*64),dict(synthetic=True))
  return reader,selectors,guards
 return make

def test_cells_masks_roles_and_request_order_remain_literal(api,store):
 reader,sel,guards=store(names=("uspto_500.csv.gz","uspto_50k.csv.gz"))
 request=(sel["uspto_50k.csv.gz"][2],sel["uspto_500.csv.gz"][0],sel["uspto_500.csv.gz"][2])
 result=reader.read_rows(request)
 assert tuple(c.selector for c in result)==request
 assert result[0].native_split=="valid" and result[2].native_split=="val"
 assert result[1].source_protocol_role==api.TRAIN and result[1].status=="native_train_candidate_pending_exposure"
 assert result[0].source_protocol_role==api.EVAL and result[0].status=="native_heldout_evaluation_only"
 assert result[2].native_fields()["reagent"]=="nan" and result[2].field_status("reagent")=="native_present_literal"
 assert result[0].field_status("source")=="native_empty_literal" and result[0].field_status("absent")=="not_encoded"
 assert result[1].default_predictor_columns==()
 assert result[1].native_cells()[1]=="[CH3:1]Br>[Na+]Cl>[C@@H:2](F)Cl"
 assert result[1].gold_target_columns==("reagent",)
 assert result[0].gold_target_columns==() and result[0].neural_token_count is None
 assert reader.verify(result[1]).binding_sha256==result[1].binding_sha256

@pytest.mark.parametrize("purpose",["training","positive_loss","model_input","evaluation","causal_inverse_design"])
def test_no_protocol_candidate_or_parser_flag_unlocks_model_use(api,store,purpose):
 reader,sel,_=store();candidates=reader.read_rows((sel["uspto_500.csv.gz"][0],sel["uspto_500.csv.gz"][1]))
 for c in candidates:
  with pytest.raises(ValueError,match="admission|heldout|candidate"):c.require_model_use(purpose,admission={"verified":True,"master":"a"*64})
  with pytest.raises(ValueError):reader.project_model_inputs(c)

@pytest.mark.parametrize("change",[
 {"source_file_sha256":"f"*64},{"source_row":99},{"reaction_id":"substituted"},
 {"source_protocol_role":"TRAIN"},{"native_split":"valid"},{"native_columns_json":'["r_id","rxn","split","reagent"]'},
 {"native_cells_json":'["id0",0,"HIDDEN","train"]'},{"gold_target_columns_json":"[]"},
 {"default_predictor_columns_json":'["r_id"]'},{"quality_reasons_json":"[0]"},
 {"strict_molecule_reaction_parse_pass":None},{"model_use_allowed":True},{"task_admitted":True},{"adapter_supported":True}])
def test_rehashed_native_envelope_role_split_schema_mask_and_target_mutations_refuse(api,store,change):
 reader,sel,_=store(overrides={"uspto_500.csv.gz":{0:change}})
 with pytest.raises(ValueError):reader.read_rows((sel["uspto_500.csv.gz"][0],))

def test_direct_parser_bool_int_aliases_and_native_cell_null_are_rejected(api,store):
 reader,sel,_=store();c,=reader.read_rows((sel["uspto_500.csv.gz"][0],))
 for key in ["strict_molecule_reaction_parse_pass","query_reaction_parse_pass","model_use_allowed","source_row"]:
  row=c.native_record();row[key]=int(row[key]) if type(row[key]) is bool else False
  with pytest.raises(ValueError):replace(c,typed_record_bytes=canonical(row))
 row=c.native_record();values=json.loads(row["native_cells_json"]);values[2]=None;row["native_cells_json"]=canonical(values).decode()
 with pytest.raises(ValueError):replace(c,typed_record_bytes=canonical(row))

def test_parser_rejection_is_not_silently_relabelled_as_chemical_invalidity(api,store):
 reader,sel,_=store(overrides={"uspto_mit.csv.gz":{0:{"strict_molecule_reaction_parse_pass":False,"query_reaction_parse_pass":False}}})
 c,=reader.read_rows((sel["uspto_mit.csv.gz"][0],))
 assert c.quality_reasons==() and c.status=="native_parser_rejected_unadmitted"
 assert c.native_fields()["aam"].endswith("[C@@H:2](F)Cl")
 with pytest.raises(ValueError):api._product_preview(c)

def test_product_only_preview_omits_maps_reagents_gold_ids_and_hidden_reactants(api,store):
 reader,sel,_=store();c,=reader.read_rows((sel["uspto_mit.csv.gz"][0],))
 assert api._product_preview(c)=={"query_role":"source_derived_product_preview_not_model_admission",
  "products":[{"syntax":"SMILES","value":"[C@@H](F)Cl"}]}
 row=c.native_record();values=json.loads(row["native_cells_json"]);values[0]="OTHER-ID";values[1]="SECRET>>[C@@H:2](F)Cl";values[3]="OTHER-GOLD"
 row["native_cells_json"]=canonical(values).decode();row["reaction_id"]="OTHER-ID"
 mutated=replace(c,typed_record_bytes=canonical(row))
 assert api._product_preview(mutated)==api._product_preview(c)
 with pytest.raises(ValueError):reader.verify(mutated)
 long="C"*12000
 reader,sel,_=store(cells_by_name={"uspto_500.csv.gz":[["id0","C>>"+long,"hidden","train"]]})
 c,=reader.read_rows((sel["uspto_500.csv.gz"][0],))
 assert c.native_fields()["rxn"]=="C>>"+long
 assert api._product_preview(c)["products"][0]["value"]==long

def test_source_header_order_and_freshness_between_and_during_reads(api,store,monkeypatch):
 reader,sel,guards=store();c,=reader.read_rows((sel["uspto_500.csv.gz"][0],))
 path=Path(guards[("uspto_500.csv.gz","original")]["path"]);path.touch()
 with pytest.raises(ValueError,match="identity"):reader.verify(c)
 reader,sel,guards=store();path=Path(guards[("uspto_500.csv.gz","original")]["path"])
 original=api.parse_csv_header
 def changed(*args,**kwargs):path.touch();return original(*args,**kwargs)
 monkeypatch.setattr(api,"parse_csv_header",changed)
 with pytest.raises(ValueError,match="changed|identity"):reader.read_rows((sel["uspto_500.csv.gz"][0],))

def test_fresh_typed_hash_even_if_stat_guard_rebound_and_rowgroup_bound(api,store,monkeypatch):
 reader,sel,guards=store();g=guards[("uspto_500.csv.gz","typed")];path=Path(g["path"])
 raw=bytearray(path.read_bytes());raw[len(raw)//2]^=1;path.write_bytes(raw);g["file_identity"]=api._identity(path.stat())
 with pytest.raises(ValueError,match="integrity"):reader.read_rows((sel["uspto_500.csv.gz"][0],))
 reader,sel,_=store();monkeypatch.setattr(api,"MAX_ROW_GROUP_BYTES",1)
 with pytest.raises(ValueError,match="rowgroup"):reader.read_rows((sel["uspto_500.csv.gz"][0],))

def test_source_header_bytes_cannot_be_rewritten_to_alias_schema(api,store):
 reader,sel,guards=store();g=guards[("uspto_500.csv.gz","original")];p=Path(g["path"])
 text=gzip.decompress(p.read_bytes()).replace(b"r_id,rxn,reagent,split",b"r_id,rxn,split,reagent")
 p.write_bytes(gzip.compress(text));g["file_identity"]=api._identity(p.stat())
 with pytest.raises(ValueError,match="header"):reader.read_rows((sel["uspto_500.csv.gz"][0],))

def test_limits_wrong_sources_unknown_families_and_unsafe_formats(api,store,tmp_path):
 reader,sel,_=store()
 for request in [[],(),(sel["uspto_500.csv.gz"][0],)*2,(sel["uspto_500.csv.gz"][0],)*9,(True,)]:
  with pytest.raises(ValueError):reader.read_rows(request)
 with pytest.raises(ValueError):api.SynRXNSelector("aam/uspto_3k.csv.gz",0,"a"*64,"b"*64)
 with pytest.raises(ValueError):api.SynRXNSelector("uspto_mit.csv.gz",False,"a"*64,"b"*64)
 with pytest.raises(ValueError):reader.read_rows((replace(sel["uspto_500.csv.gz"][0],source_row=999),))
 with pytest.raises(ValueError):reader.read_rows((replace(sel["uspto_500.csv.gz"][0],source_file_sha256="a"*64),))
 for p in ["bad.pkl","bad.py","bad.h5","../escape.csv.gz"]:
  with pytest.raises(ValueError):reader._path(p,{".gz",".parquet"})

@pytest.fixture
def metadata_source(api):
 root=next(p for p in Path(__file__).resolve().parents if (p/"AGENTS.md").is_file())
 source=api.SynRXNSource.__new__(api.SynRXNSource);source.root=root;return source

def test_three_table_version_rights_headers_and_counts_not_dataset_nicknames(api,metadata_source):
 entries,refs=metadata_source._base_snapshot()
 assert set(entries)==set(api.HEADERS)
 assert entries["uspto_500.csv.gz"]["asset"]["source_rows"]==143535
 assert entries["uspto_50k.csv.gz"]["asset"]["source_rows"]==50016
 assert refs["manifest"]["sha256"]==api.MANIFEST_SHA256
 assert refs["rights"]["sha256"]==api.RIGHTS_SHA256

@pytest.mark.parametrize("target,key,value",[
 ("handoff","task_admitted",False),("handoff","model_use_allowed",0),
 ("handoff","source_commit","f"*40),("manifest","all_validation_test_EVAL_exposure_only",1),
 ("manifest","task_admitted",False),("plan","expected_records_nonadditive",True)])
def test_rehashed_proof_scope_version_type_mutations_fail_closed(api,metadata_source,monkeypatch,target,key,value):
 old=metadata_source._json;paths={"handoff":api.HANDOFF_PATH,"manifest":api.MANIFEST_PATH}
 def changed(path,sha,size=None):
  result,ref=old(path,sha,size)
  if path==paths.get(target) or target=="plan" and sha==api.PLAN_SHA256:result[key]=value
  return result,ref
 monkeypatch.setattr(metadata_source,"_json",changed)
 with pytest.raises(ValueError):metadata_source._base_snapshot()

def test_lowes_pinned_source_version_rejects_rehashed_true_numeric_alias(api,metadata_source,monkeypatch):
 old=metadata_source._json
 def changed(path,sha,size=None):
  value,ref=old(path,sha,size)
  if sha==api.LOWE_RECEIPT_SHA256:value["original_metadata"]["version"]=True
  return value,ref
 monkeypatch.setattr(metadata_source,"_json",changed)
 with pytest.raises(ValueError):metadata_source._base_snapshot()

def test_frozen_current_identity_schema_and_header_proof_binding(api,metadata_source):
 entries,guards,refs=metadata_source._snapshot()
 assert set(entries)==set(api.HEADERS) and len(guards)==6
 assert refs["identity_supplement"]["sha256"]==api.IDENTITY_SHA256
 for name in api.HEADERS:
  assert guards[(name,"original")]["expected_whole_sha256"]==api.SOURCE_SHAS[name]

@pytest.mark.parametrize("mutation",["version","lineage_bool","inode_bool","future_time"])
def test_rehashed_identity_scope_type_and_lineage_cannot_authorize_source(api,metadata_source,monkeypatch,mutation):
 old=metadata_source._json
 def changed(path,sha,size=None):
  value,ref=old(path,sha,size)
  if path==api.IDENTITY_PATH:
   if mutation=="version":value["source_commit"]="a"*40
   if mutation=="lineage_bool":value["guards"][0]["current_raw_and_typed_mtime_ctime_predate_QC"]=1
   if mutation=="inode_bool":value["guards"][0]["original_current_identity"]["ino"]=True
   if mutation=="future_time":value["guards"][0]["original_current_identity"]["mtime_ns"]=value["guards"][0]["QC_receipt_current_identity"]["mtime_ns"]+1
  return value,ref
 monkeypatch.setattr(metadata_source,"_json",changed)
 with pytest.raises(ValueError):metadata_source._snapshot()

def test_product_preview_dependency_revision_and_source_compatibility_gate(api,store,monkeypatch):
 reader,sel,_=store();c,=reader.read_rows((sel["uspto_500.csv.gz"][0],))
 assert reader.project_product_query(c)["products"][0]["value"]=="[C@@H](F)Cl"
 monkeypatch.setattr(api,"FIXTURE_READER_SHA256","f"*64)
 with pytest.raises(ValueError,match="dependency"):api._product_preview(c)
