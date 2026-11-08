"""Bounded actual ProCyon byte-equivalence and repeated integrity regression tests."""
import hashlib,importlib.util,io,json,os,sys
from pathlib import Path
import pytest
ROOT=Path(os.environ.get("MORA3_REPO_ROOT",Path.cwd()))
PACKAGE=ROOT/"outputs/implementation/aux-native-readers-20261101/ProCyon-reader-efficiency-v1"
BASELINE_SHA="e4d199e5bd0059a52f50cb11e195b1378ad67c14a449af2495865ab537959e6f"
def load(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
@pytest.fixture(scope="module")
def api():return load(Path(os.environ.get("MORA3_PROCYON_EFF_MODULE","src/mora3/data/procyon_native.py")),"mora3.data.procyon_eff_test")
@pytest.fixture(scope="module")
def baseline():
 path=Path(os.environ.get("MORA3_PROCYON_EFF_BASELINE",PACKAGE/"baseline/src/mora3/data/procyon_native.py"))
 if not path.is_file():pytest.skip("immutable baseline archive unavailable")
 assert hashlib.sha256(path.read_bytes()).hexdigest()==BASELINE_SHA
 return load(path,"mora3.data.procyon_eff_reference")
@pytest.fixture(scope="module")
def data():
 path=PACKAGE/"fixed-slices.json"
 if not path.is_file():pytest.skip("fixed native candidate selections unavailable")
 value=json.loads(path.read_text())
 if not (ROOT/"outputs/data-sources/20261101-ProCyon-source-v1/handoff-ProCyon-open-native-components-v1.json").is_file():pytest.skip("pinned source unavailable")
 return value
def case(data,name):return next(c for c in data["cases"] if c["name"]==name)
def query(module,c,source=None):
 source=source or module.ProCyonNativeSource(root=ROOT)
 return source.read_sequences(module.SequenceSelector(c["entity_kind"],tuple(c["indices"]))) if c["mode"]=="sequence" else source.read_relations(module.RelationSelector(c["family"],tuple(c["indices"])))
def bindings(values):return tuple(x.binding_sha256 for x in values)
def counted(module,c):
 import pyarrow.parquet as pq
 source=module.ProCyonNativeSource(root=ROOT);_,qc,_=source._snapshot()
 payloads={x["payload"]["sha256"] for x in qc["FASTA_files"]};payload_paths={x["payload"]["path"] for x in qc["FASTA_files"]}
 checked=module.ProCyonNativeSource._bytes;reader=pq.read_table
 count=dict(sequence_decodes=0,checked_paths=[],checked_bytes=[],FASTA=0,payload=0)
 def read(self,ref,suffixes):
  raw=checked(self,ref,suffixes);p=ref.path if type(ref) is module.NativeFileRef else ref["path"]
  count["checked_paths"].append(p);count["checked_bytes"].append(len(raw))
  if Path(p).suffix.lower() in (".fa",".fasta"):count["FASTA"]+=1
  if p in payload_paths:count["payload"]+=1
  return raw
 def table(*a,**kw):
  result=reader(*a,**kw)
  if isinstance(a[0],io.BytesIO) and hashlib.sha256(a[0].getbuffer()).hexdigest() in payloads:count["sequence_decodes"]+=1
  return result
 module.ProCyonNativeSource._bytes=read;pq.read_table=table
 try:values=query(module,c,source)
 finally:module.ProCyonNativeSource._bytes=checked;pq.read_table=reader
 return values,count

@pytest.mark.parametrize("name",("sequence_protein_distinct","sequence_domain_distinct","relation_protein_GO_combined","relation_domain_GO_combined"))
def test_one_decoded_sequence_table_per_call_preserves_all_per_selected_byte_checks(api,baseline,data,name):
 c=case(data,name);before,old=counted(baseline,c);after,new=counted(api,c)
 assert bindings(before)==bindings(after)
 assert old["sequence_decodes"]==len(c["indices"])==8
 assert new["sequence_decodes"]==1
 assert old["FASTA"]==new["FASTA"]==old["payload"]==new["payload"]==8
 assert old["checked_paths"]==new["checked_paths"]
 assert old["checked_bytes"]==new["checked_bytes"]

@pytest.mark.parametrize("name",("sequence_protein_backward","sequence_domain_backward","relation_protein_GO_backward","relation_domain_GO_backward"))
def test_native_backward_order_full_axes_headers_records_and_relation_multiplicity_are_identical(api,baseline,data,name):
 c=case(data,name);before=query(baseline,c);after=query(api,c)
 assert bindings(before)==bindings(after)
 for a,b in zip(before,after):
  old=a if c["mode"]=="sequence" else a.sequence;new=b if c["mode"]=="sequence" else b.sequence
  assert old.sequence_bytes==new.sequence_bytes and old.raw_record_bytes==new.raw_record_bytes
  assert old.native_header_bytes==new.native_header_bytes
  assert new.native_observed_mask()==(True,)*len(new.sequence_bytes)
  if c["mode"]=="relation":
   assert a.raw_csv_header==b.raw_csv_header and a.raw_csv_row==b.raw_csv_row
   assert a.native_fields_bytes==b.native_fields_bytes and a.indexed_fields_bytes==b.indexed_fields_bytes
   assert a.native_split==b.native_split and a.relation_assertions==b.relation_assertions

@pytest.mark.parametrize("kind",("protein","domain"))
@pytest.mark.parametrize("artifact",("source","payload"))
@pytest.mark.parametrize("mode",("tamper","replace"))
def test_during_call_tamper_or_atomic_replacement_is_caught_before_cached_data(api,baseline,data,kind,artifact,mode,monkeypatch):
 c=case(data,"sequence_"+kind+"_distinct");target=(ROOT/c[artifact]["path"]).resolve()
 other=case(data,"sequence_"+("domain" if kind=="protein" else "protein")+"_distinct")
 replacement=(ROOT/other[artifact]["path"]).read_bytes();original=Path.read_bytes;state={"n":0}
 def changed(path):
  raw=original(path)
  if path.resolve()==target:
   state["n"]+=1
   if state["n"]==2:return raw+b" " if mode=="tamper" else replacement
  return raw
 monkeypatch.setattr(Path,"read_bytes",changed)
 errors=[]
 for module in (baseline,api):
  state["n"]=0
  with pytest.raises(ValueError) as e:query(module,c)
  assert state["n"]==2
  errors.append(str(e.value))
 assert errors[0]==errors[1]=="source byte integrity mismatch"

@pytest.mark.parametrize("artifact",("FASTA","sequence_payload","QC","transformation","license","mapping","source_csv","handoff"))
def test_between_calls_reused_source_object_revalidates_all_source_and_proof_artifacts(api,baseline,data,artifact,monkeypatch):
 c=case(data,"relation_protein_GO_early")
 sources=[m.ProCyonNativeSource(root=ROOT) for m in (baseline,api)]
 refs=sources[0].source_references("protein_GO");target=(ROOT/refs[artifact]["path"]).resolve()
 for m,s in zip((baseline,api),sources):query(m,c,s)
 original=Path.read_bytes;touched=[]
 def changed(path):
  raw=original(path)
  if path.resolve()==target:touched.append(True);return raw+b" "
  return raw
 monkeypatch.setattr(Path,"read_bytes",changed)
 errors=[]
 for m,s in zip((baseline,api),sources):
  with pytest.raises(ValueError) as caught:query(m,c,s)
  errors.append(str(caught.value))
 assert touched and errors[0]==errors[1]

@pytest.mark.parametrize("indices",((True,),(-1,),(0,0),(),(1.0,),tuple(range(33))))
def test_selector_type_duplicate_limit_error_semantics_unchanged(api,baseline,indices):
 errors=[]
 for m in (baseline,api):
  with pytest.raises(ValueError) as caught:m.SequenceSelector("protein",indices)
  errors.append(str(caught.value))
 assert errors[0]==errors[1]

def test_absent_native_ordinal_and_opaque_relation_family_keep_error_order(api,baseline,data):
 c=case(data,"sequence_protein_distinct")
 errors=[]
 for m in (baseline,api):
  s=m.ProCyonNativeSource(root=ROOT)
  with pytest.raises(ValueError) as e:s.read_sequences(m.SequenceSelector("protein",(999999,)))
  errors.append(str(e.value))
 assert errors[0]==errors[1]
 for m in (baseline,api):
  with pytest.raises(ValueError):m.ProCyonNativeSource(root=ROOT).read_relations(m.RelationSelector("indexed_only_UniProt",(0,)))

@pytest.mark.parametrize("field",("native_ordinal","source_sha256","source_asset","task_admitted","adapter_support","sequence_length"))
def test_native_payload_typed_source_axis_and_mask_validation_remains_per_selected_record(api,baseline,data,field,monkeypatch):
 import pyarrow.parquet as pq
 original=pq.read_table
 def changed(*a,**kw):
  actual=original(*a,**kw)
  class Slice:
   def __init__(self,ordinal,length):self.ordinal=ordinal;self.length=length
   def to_pylist(self):
    rows=actual.slice(self.ordinal,self.length).to_pylist();r=rows[0]
    r[field]={"native_ordinal":False,"source_sha256":"0"*64,"source_asset":"wrong-source",
     "task_admitted":1,"adapter_support":0,"sequence_length":True}[field]
    return rows
  class Table:
   def __len__(self):return len(actual)
   def slice(self,ordinal,length):return Slice(ordinal,length)
  return Table()
 monkeypatch.setattr(pq,"read_table",changed)
 c=case(data,"sequence_protein_early")
 errors=[]
 for m in (baseline,api):
  with pytest.raises(ValueError) as e:query(m,c)
  errors.append(str(e.value))
 assert errors[0]==errors[1]

def test_exact_old_pins_target_exclusion_heldout_text_domain_identity_and_admission_gates_unchanged(api,baseline,data):
 old=baseline.ProCyonNativeSource(root=ROOT);new=api.ProCyonNativeSource(root=ROOT)
 assert old.quarantines()==new.quarantines()
 for family in ("protein_GO","domain_GO"):
  c=case(data,"relation_"+family+"_combined")
  before=query(baseline,c,old);after=query(api,c,new)
  for a,b in zip(before,after):
   assert a.prediction_inputs()==b.prediction_inputs()
   assert set(b.prediction_inputs())=={"entity_kind","native_sequence","query_role"}
   assert b.audit()["current_UniProt_qualification_verified"] is False
   assert b.audit()["instruction_parent"] is None
   assert b.audit()["phenotype_description_available"] is False
   if family=="domain_GO":
    assert b.sequence.parent_protein_id is None and b.sequence.parent_coordinates is None
   for purpose in ("training","evaluation","text_generation","retrieval_loss","heldout_use"):
    with pytest.raises(ValueError):b.require_model_use(purpose,admission={"verified":True})
 for s in (old,new):
  with pytest.raises(ValueError):s.require_text_capability("QA")
