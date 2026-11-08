"""Actual bounded source equivalence and per-call CSV cursor regression tests."""
import copy,hashlib,importlib.util,json,os,sys
from pathlib import Path
import pytest
ROOT=Path(os.environ.get("MORA3_REPO_ROOT",Path.cwd()))
PACKAGE=ROOT/"outputs/implementation/aux-native-readers-20261101/ADMET-reader-efficiency-v1"
BASELINE_SHA="eb7273c6b816e6c45de257aa441029e4f06665dea284ae21222aeb2dc7c97a8e"
def load(path,name):
 spec=importlib.util.spec_from_file_location(name,path);api=importlib.util.module_from_spec(spec)
 sys.modules[name]=api;spec.loader.exec_module(api);return api
@pytest.fixture(scope="module")
def api():return load(Path(os.environ.get("MORA3_ADMET_EFF_MODULE","src/mora3/data/admet.py")),"mora3.data.admet_eff_test")
@pytest.fixture(scope="module")
def baseline():
 path=Path(os.environ.get("MORA3_ADMET_EFF_BASELINE",PACKAGE/"baseline/src/mora3/data/admet.py"))
 if not path.is_file():pytest.skip("immutable baseline archive unavailable; no baseline equivalence claim")
 assert hashlib.sha256(path.read_bytes()).hexdigest()==BASELINE_SHA
 return load(path,"mora3.data.admet_eff_reference")
@pytest.fixture(scope="module")
def slices():
 p=PACKAGE/"fixed-slices.json"
 if not p.is_file():pytest.skip("fixed admitted-source selections unavailable")
 data=json.loads(p.read_text())
 if not (ROOT/data["manifest"]["path"]).is_file():pytest.skip("qualified native source projection unavailable")
 return data
def case(data,name):return next(x for x in data["slices"] if x["name"]==name)
def kwargs(data,c):
 return dict(manifest=data["manifest"]["path"],root=ROOT,expected_manifest_sha256=data["manifest"]["sha256"],
  shard_paths=tuple(c["shard_paths"]),limit=c["limit"],observation_ids=tuple(c["observation_ids"]))
def read(api,data,c):return api.read_admet_native_candidates(**kwargs(data,c))
def natives(values):return tuple(x.native_bytes for x in values)
def counted(api,request):
 original=api.csv.DictReader;verified=api._verified_bytes
 counts=dict(parsers=0,rows=0,decodes=0,verified_csv=0)
 class Bytes(bytes):
  def decode(self,*a,**kw):counts["decodes"]+=1;return super().decode(*a,**kw)
 class Reader(original):
  def __init__(self,*a,**kw):counts["parsers"]+=1;super().__init__(*a,**kw)
  def __next__(self):row=super().__next__();counts["rows"]+=1;return row
 def checked(root,ref):
  raw=verified(root,ref)
  if Path(ref["path"]).suffix.lower()==".csv":counts["verified_csv"]+=1;return Bytes(raw)
  return raw
 api.csv.DictReader=Reader;api._verified_bytes=checked
 try:out=request()
 finally:api.csv.DictReader=original;api._verified_bytes=verified
 return out,counts

@pytest.mark.parametrize("name",("early_clustered_32","late_clustered_32","late_spread_32"))
def test_shared_source_is_decoded_and_parsed_once_retaining_only_requested_rows(api,baseline,slices,name):
 c=case(slices,name)
 expected,old=counted(baseline,lambda:read(baseline,slices,c))
 actual,new=counted(api,lambda:read(api,slices,c))
 assert natives(actual)==natives(expected)
 assert old["parsers"]==old["decodes"]==32
 assert new["verified_csv"]==1
 assert new["parsers"]==new["decodes"]==1
 assert new["rows"]==max(c["source_native_row_indices"])+1

def test_backward_shard_order_rewinds_without_redecode_or_output_reorder(api,baseline,slices):
 c=case(slices,"out_of_order_two_shards_32")
 expected=read(baseline,slices,c);actual,counts=counted(api,lambda:read(api,slices,c))
 assert natives(actual)==natives(expected)
 assert counts["parsers"]==2 and counts["decodes"]==1 and counts["verified_csv"]==1
 assert counts["rows"]<10000
 assert actual[0].native()["source_ref"]["native_row_index"]>actual[-1].native()["source_ref"]["native_row_index"]

def test_multiple_verified_sources_and_original_protocol_status_unknown_fields_are_identical(api,baseline,slices):
 c=case(slices,"mixed_sources_protocols")
 expected=read(baseline,slices,c);actual,counts=counted(api,lambda:read(api,slices,c))
 assert natives(actual)==natives(expected)
 assert counts["decodes"]==counts["verified_csv"]==2
 all_cases=[]
 for name in ("protocol_witnesses_0","protocol_witnesses_1"):
  before=read(baseline,slices,case(slices,name));after=read(api,slices,case(slices,name))
  assert natives(before)==natives(after);all_cases.extend(after)
 assert {x.cell.status for x in all_cases}=={"exact","censored_bound","missing"}
 parents={x.native()["source_split"]["parent_protocol"] for x in all_cases}
 assert {"native_train","native_test","raw_only_unallocated","Train"}<=parents
 assert {x.native()["source_ref"]["native_entity_namespace"] for x in all_cases}=={"ExpansionRx","ASAP"}

def test_requested_ID_order_and_shard_native_order_match_baseline(api,baseline,slices):
 c=copy.deepcopy(case(slices,"late_spread_32"));c["observation_ids"].reverse()
 assert natives(read(api,slices,c))==natives(read(baseline,slices,c))

def test_no_eager_CSV_tail_parse_when_only_first_native_row_requested(api,slices,monkeypatch):
 c=case(slices,"early_clustered_32");request=kwargs(slices,c);request.update(limit=1,observation_ids=(c["observation_ids"][0],))
 original=api.csv.DictReader
 class Reader(original):
  def __init__(self,*a,**kw):super().__init__(*a,**kw);self.visited=0
  def __next__(self):
   if self.visited:raise api.csv.Error("unrequested source tail must not be parsed")
   self.visited+=1;return super().__next__()
 monkeypatch.setattr(api.csv,"DictReader",Reader)
 assert len(api.read_admet_native_candidates(**request))==1

def test_limit_without_requested_IDs_does_not_read_later_shard_or_change_first_rows(api,baseline,slices,monkeypatch):
 first=case(slices,"early_clustered_32");late=case(slices,"late_spread_32")
 request=kwargs(slices,first);request.update(shard_paths=tuple(first["shard_paths"]+late["shard_paths"]),limit=3,observation_ids=None)
 expected=baseline.read_admet_native_candidates(**request)
 target=(ROOT/late["shard_paths"][0]).resolve();original=Path.read_bytes
 def guard(path):
  assert path.resolve()!=target,"unvisited shard must not be loaded after early return"
  return original(path)
 monkeypatch.setattr(Path,"read_bytes",guard)
 assert natives(api.read_admet_native_candidates(**request))==natives(expected)

def test_fresh_call_revalidates_source_bytes_and_never_uses_global_rows(api,slices,monkeypatch):
 c=case(slices,"early_clustered_32");out,count=counted(api,lambda:read(api,slices,c))
 source=out[0].native()["source_ref"]["source"];target=(ROOT/source["path"]).resolve()
 again,count2=counted(api,lambda:read(api,slices,c))
 assert natives(out)==natives(again) and count["verified_csv"]==count2["verified_csv"]==1
 original=Path.read_bytes;touched=[]
 def tamper(p):
  raw=original(p)
  if p.resolve()==target:touched.append(True);return raw+b" "
  return raw
 monkeypatch.setattr(Path,"read_bytes",tamper)
 with pytest.raises(ValueError,match="integrity"):read(api,slices,c)
 assert touched

@pytest.mark.parametrize("mutation",("limit0","limit_bool","limit33","duplicate_shard","unknown_shard","duplicate_ID","absent_ID"))
def test_limits_absence_allocation_and_error_messages_match_baseline(api,baseline,slices,mutation):
 request=kwargs(slices,case(slices,"early_clustered_32"))
 if mutation=="limit0":request["limit"]=0
 elif mutation=="limit_bool":request["limit"]=True
 elif mutation=="limit33":request["limit"]=33
 elif mutation=="duplicate_shard":request["shard_paths"]*=2
 elif mutation=="unknown_shard":request["shard_paths"]=("unrelated.parquet",)
 elif mutation=="duplicate_ID":request["observation_ids"]=(request["observation_ids"][0],)*2
 else:request["observation_ids"]=("0"*64,)
 outcomes=[]
 for module in (baseline,api):
  try:module.read_admet_native_candidates(**request)
  except Exception as exc:outcomes.append((type(exc).__name__,str(exc)))
  else:pytest.fail("invalid reader request accepted")
 assert outcomes[0]==outcomes[1]

@pytest.mark.parametrize("mutation",("unknown_native_field","rehashed_native_value","native_mask_type","outer_type","split","index_type"))
def test_rehashed_native_envelope_unknown_field_value_mask_split_and_type_rejections_match(api,baseline,slices,monkeypatch,mutation):
 import pyarrow.parquet as pq
 original=pq.read_table
 def table(*a,**kw):
  actual=original(*a,**kw);rows=actual.to_pylist()
  if "native_observation" not in actual.column_names:return actual
  row=rows[0];d=json.loads(row["native_observation"])
  if mutation=="outer_type":row["task_admitted"]=0
  elif mutation=="split":row["source_parent_protocol"]="validation"
  else:
   if mutation=="unknown_native_field":d["native_source_record"]["injected_unknown"]="not source"
   elif mutation=="rehashed_native_value":
    d["cell"]=api.NativeCell.parse("123456.000").as_mapping()
    d["native_source_record"][d["endpoint"]["source_endpoint"]]="123456.000"
   elif mutation=="native_mask_type":d["cell"]["observation_present"]=1
   else:d["source_ref"]["native_row_index"]=False
   raw=api.canonical(d);row["native_observation"]=raw;row["native_observation_sha256"]=api.digest(raw)
  class Changed:
   num_rows=actual.num_rows
   def __getitem__(self,key):return actual[key]
   def to_pylist(self):return rows
  return Changed()
 monkeypatch.setattr(pq,"read_table",table)
 c=case(slices,"early_clustered_32");request=kwargs(slices,c);request.update(limit=1,observation_ids=(c["observation_ids"][0],))
 outcomes=[]
 for module in (baseline,api):
  with pytest.raises(ValueError) as caught:module.read_admet_native_candidates(**request)
  outcomes.append(str(caught.value))
 assert outcomes[0]==outcomes[1]

def test_actual_prediction_target_exclusion_and_source_supervision_gates_remain_identical(api,baseline,slices):
 c=case(slices,"mixed_sources_protocols");before=read(baseline,slices,c);after=read(api,slices,c)
 for old,new in zip(before,after):
  assert old.native_bytes==new.native_bytes
  for module,obs in ((baseline,old),(api,new)):
   definition=module.EndpointDefinition.from_native(obs.native()["endpoint"])
   registry=module.EndpointRegistry((definition,))
   view=module.admet_prediction_view(obs,registry)
   assert set(view.inputs())=={"role","native_entity","endpoint","assay_context"}
   assert "native_measurement" not in view.inputs()
   assert view.provenance()["task_admitted"] is False
   assert view.provenance()["model_supervision_supported"] is False
   with pytest.raises(ValueError):obs.require_source_admission(proof={"verified":True})


def test_native_source_row_past_EOF_preserves_exact_absence_error(api,baseline,slices,monkeypatch):
 import pyarrow.parquet as pq
 original=pq.read_table
 def table(*a,**kw):
  actual=original(*a,**kw)
  if "native_observation" not in actual.column_names:return actual
  rows=actual.to_pylist();row=rows[0];d=json.loads(row["native_observation"])
  d["source_ref"]["native_row_index"]=1_000_000
  row["source_native_row_index"]=1_000_000
  row["observation_id"]=api.digest(api.canonical([d["source_ref"]["source"]["sha256"],1_000_000,d["endpoint"]["source_endpoint"]]))
  row["native_observation"]=api.canonical(d);row["native_observation_sha256"]=api.digest(row["native_observation"])
  class Changed:
   num_rows=actual.num_rows
   def __getitem__(self,key):return actual[key]
   def to_pylist(self):return rows
  return Changed()
 monkeypatch.setattr(pq,"read_table",table)
 request=kwargs(slices,case(slices,"early_clustered_32"));request.update(limit=1,observation_ids=None)
 for module in (baseline,api):
  with pytest.raises(ValueError,match="^native source row absent$"):module.read_admet_native_candidates(**request)

def test_parser_failure_on_next_required_source_row_is_not_hidden_or_eagerly_triggered(api,baseline,slices,monkeypatch):
 original=api.csv.DictReader
 class Reader(original):
  def __init__(self,*a,**kw):super().__init__(*a,**kw);self.visited=0
  def __next__(self):
   if self.visited:raise api.csv.Error("required source parse failure")
   self.visited+=1;return super().__next__()
 monkeypatch.setattr(api.csv,"DictReader",Reader)
 request=kwargs(slices,case(slices,"early_clustered_32"))
 for module in (baseline,api):
  with pytest.raises(module.csv.Error,match="^required source parse failure$"):
   module.read_admet_native_candidates(**request)
