"""Stage-two bounded actual parity and explicitly synthetic CSV/map edge cases."""
import hashlib,importlib.util,io,json,os,sys
from pathlib import Path
import pytest
ROOT=Path(os.environ.get("MORA3_REPO_ROOT",Path.cwd()))
P=ROOT/"outputs/implementation/aux-native-readers-20261101/ProCyon-relation-efficiency-v1"
def load(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
@pytest.fixture(scope="module")
def api():return load(Path(os.environ.get("MORA3_PROCYON_REL_MODULE","src/mora3/data/procyon_native.py")),"mora3.data.procyon_rel_test")
@pytest.fixture(scope="module")
def baseline():
 p=P/"baseline/src/mora3/data/procyon_native.py"
 if not p.is_file():pytest.skip("frozen stacked baseline unavailable")
 assert hashlib.sha256(p.read_bytes()).hexdigest()=="10afdf0d157884724bee3ee0afe4c6d6ecb7001517902a228d07fee449122d0e"
 return load(p,"mora3.data.procyon_rel_reference")
@pytest.fixture(scope="module")
def cases():
 if not (P/"fixed-slices.json").exists():pytest.skip("fixed bounded source slice unavailable")
 return [c for c in json.loads((P/"fixed-slices.json").read_text())["cases"] if c["mode"]=="relation"]
def query(m,c,source=None):
 s=source or m.ProCyonNativeSource(root=ROOT)
 return s.read_relations(m.RelationSelector(c["family"],tuple(c["indices"])))
def fingerprint(values):return tuple(x.binding_sha256 for x in values)
def instrument(m,c):
 import pyarrow.parquet as pq
 s=m.ProCyonNativeSource(root=ROOT);_,qc,_=s._snapshot();mapping_sha=qc["native_ID_index_map"]["sha256"]
 checked=m.ProCyonNativeSource._bytes;read=pq.read_table;find=m.re.finditer
 count=dict(checked_paths=[],checked_bytes=[],newline_matches=0,mapping_row_dicts=0,mapping_rows=0)
 def bytes_(self,ref,suffixes):
  raw=checked(self,ref,suffixes);count["checked_paths"].append(ref.path if type(ref) is m.NativeFileRef else ref["path"]);count["checked_bytes"].append(len(raw));return raw
 def table(*a,**kw):
  actual=read(*a,**kw)
  if not (isinstance(a[0],io.BytesIO) and hashlib.sha256(a[0].getbuffer()).hexdigest()==mapping_sha):return actual
  count["mapping_rows"]=len(actual)
  class MapTable:
   def __getattr__(self,name):return getattr(actual,name)
   def __getitem__(self,key):return actual[key]
   def to_pylist(self,*args,**kwargs):
    result=actual.to_pylist(*args,**kwargs);count["mapping_row_dicts"]+=len(result);return result
  return MapTable()
 def matches(pattern,*args,**kwargs):
  result=find(pattern,*args,**kwargs)
  if pattern!=b"\n":return result
  def iterator():
   for match in result:count["newline_matches"]+=1;yield match
  return iterator()
 m.ProCyonNativeSource._bytes=bytes_;pq.read_table=table;m.re.finditer=matches
 try:values=query(m,c,s)
 finally:m.ProCyonNativeSource._bytes=checked;pq.read_table=read;m.re.finditer=find
 return values,count

@pytest.mark.parametrize("family",("protein_GO","domain_GO"))
def test_early_bounded_relation_read_avoids_eager_suffix_offsets_and_mapping_row_dictionaries(api,baseline,cases,family):
 c=next(x for x in cases if x["name"]=="relation_"+family+"_early")
 before,old=instrument(baseline,c);after,new=instrument(api,c)
 assert fingerprint(before)==fingerprint(after)
 assert old["checked_paths"]==new["checked_paths"] and old["checked_bytes"]==new["checked_bytes"]
 assert old["mapping_row_dicts"]==old["mapping_rows"]>10000
 assert new["mapping_row_dicts"]==0
 assert new["mapping_rows"]==old["mapping_rows"]
 assert old["newline_matches"]>10000
 assert new["newline_matches"]<=2*(max(c["indices"])+2)

@pytest.mark.parametrize("name",("relation_protein_GO_early","relation_protein_GO_late","relation_protein_GO_combined","relation_protein_GO_backward","relation_domain_GO_early","relation_domain_GO_late","relation_domain_GO_combined","relation_domain_GO_backward"))
def test_actual_original_relations_complete_source_bytes_native_fields_order_and_refusals_match(api,baseline,cases,name):
 c=next(x for x in cases if x["name"]==name)
 before=query(baseline,c);after=query(api,c)
 assert fingerprint(before)==fingerprint(after)
 for old,new in zip(before,after):
  assert (old.raw_csv_header,old.raw_csv_row,old.native_fields_bytes,old.indexed_fields_bytes)==(new.raw_csv_header,new.raw_csv_row,new.native_fields_bytes,new.indexed_fields_bytes)
  assert old.sequence.sequence_bytes==new.sequence.sequence_bytes and old.sequence.raw_record_bytes==new.sequence.raw_record_bytes
  assert new.sequence.native_observed_mask()==(True,)*len(new.sequence.sequence_bytes)
  assert old.native_split==new.native_split and old.relation_assertions==new.relation_assertions
  assert old.prediction_inputs()==new.prediction_inputs()
  assert set(new.prediction_inputs())=={"entity_kind","native_sequence","query_role"}
  for purpose in ("training","evaluation","loss"):
   with pytest.raises(ValueError):new.require_model_use(purpose,admission={"admitted":True})

def direct_csv(m,tmp_path,raw,indices,field_order,rows):
 path=tmp_path/"synthetic.csv";path.write_bytes(raw)
 s=m.ProCyonNativeSource.__new__(m.ProCyonNativeSource);s.root=tmp_path
 entry=dict(source=dict(path=path.name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()),rows=rows,native_field_order=field_order)
 return s._csv(entry,indices)
@pytest.mark.parametrize("raw,indices,fields,nrows,expected",[
 (b'a,b,unknown\r\n"x,1","line1\r\nline2",\xce\xb2\r\nlast,z,""\r\n',(1,0),["a","b","unknown"],2,
  {0:({"a":"x,1","b":"line1\r\nline2","unknown":"\u03b2"},b"a,b,unknown\r\n",b'"x,1","line1\r\nline2",\xce\xb2\r\n'),1:({"a":"last","b":"z","unknown":""},b"a,b,unknown\r\n",b'last,z,""\r\n')}),
 (b"a,b\nx,y",(0,),["a","b"],1,{0:({"a":"x","b":"y"},b"a,b\n",b"x,y")}),
 (b'a,b\n\nx,y\n\n"z","escaped ""quote"""\n',(1,0),["a","b"],2,
  {0:({"a":"x","b":"y"},b"a,b\n",b"\nx,y\n"),1:({"a":"z","b":'escaped "quote"'},b"a,b\n",b'\n"z","escaped ""quote"""\n')})
])
def test_synthetic_quoted_CRLF_multiline_UTF8_unknown_empty_blank_and_unterminated_native_rows(api,baseline,tmp_path,raw,indices,fields,nrows,expected):
 assert direct_csv(baseline,tmp_path,raw,indices,fields,nrows)==expected
 assert direct_csv(api,tmp_path,raw,indices,fields,nrows)==expected

@pytest.mark.parametrize("raw,indices,fields,rows",[
 (b'a,b\n"broken\nx,y\n',(1,),["a","b"],2),
 (b"a,b\nx\nz,q\n",(0,),["a","b"],2),
 (b"a,b\nx,y,extra\n",(0,),["a","b"],1),
 (b"a,b\nx,y\n",(2,),["a","b"],3),
 (b"a,b\nx,y\n",(2,),["a","b"],2),
 (b"bad,b\nx,y\n",(0,),["a","b"],1),
 (b"a,b\nx,y\n\xff",(0,),["a","b"],2),
 (b"a,b\rx,y\r",(0,),["a","b"],1),
 (b"a,b\nx,y\n",(True,),["a","b"],1)
])
def test_synthetic_invalid_prefix_suffix_extent_schema_and_baseline_error_order_are_preserved(api,baseline,tmp_path,raw,indices,fields,rows):
 errors=[]
 for m in (baseline,api):
  try:result=direct_csv(m,tmp_path,raw,indices,fields,rows)
  except Exception as e:errors.append((type(e).__name__,str(e)))
  else:errors.append(("value",result))
 assert errors[0]==errors[1]

@pytest.mark.parametrize("mode",("duplicate_same_value","duplicate_conflicting_value","unselected_unresolved_value"))
def test_entire_native_mapping_remains_traversed_in_original_order_even_for_unselected_keys(api,baseline,cases,monkeypatch,mode):
 import pyarrow as pa,pyarrow.parquet as pq
 c=next(x for x in cases if x["name"]=="relation_domain_GO_early")
 s=baseline.ProCyonNativeSource(root=ROOT);_,qc,_=s._snapshot()
 mapping_sha=qc["native_ID_index_map"]["sha256"];read=pq.read_table
 def changed(*a,**kw):
  t=read(*a,**kw)
  if isinstance(a[0],io.BytesIO) and hashlib.sha256(a[0].getbuffer()).hexdigest()==mapping_sha:
   row=t.slice(len(t)-1,1).to_pylist()[0]
   if mode=="unselected_unresolved_value":row["native_original_ID"]="UNSELECTED_SYNTHETIC_ABSENT";row["native_internal_index"]="unresolved"
   elif mode=="duplicate_conflicting_value":row["native_internal_index"]="999999999"
   return pa.concat_tables([t,pa.Table.from_pylist([row],schema=t.schema)])
  return t
 monkeypatch.setattr(pq,"read_table",changed)
 results=[]
 for m in (baseline,api):
  try:values=query(m,c)
  except ValueError as e:results.append(("error",str(e)))
  else:results.append(("bindings",fingerprint(values)))
 assert results[0]==results[1]
 if mode.startswith("duplicate"):assert results[0]==("error","quarantined duplicate native ID/index allocation")
 else:assert results[0][0]=="bindings"

@pytest.mark.parametrize("which",("QC","transformation"))
def test_global_unselected_mapping_conflict_proof_is_not_bypassed(api,baseline,cases,monkeypatch,which):
 c=next(x for x in cases if x["name"]=="relation_domain_GO_early")
 results=[]
 for m in (baseline,api):
  s=m.ProCyonNativeSource(root=ROOT);hand,qc,trans=s._snapshot();original=m.json.loads
  def changed(raw,*a,**kw):
   value=original(raw,*a,**kw)
   if which=="QC" and isinstance(value,dict) and "native_mapping_conflicts" in value:value["native_mapping_conflicts"]=[{"unselected":"unresolved"}]
   if which=="transformation" and isinstance(value,dict) and "paired_component_audits" in value:
    for audit in value["paired_component_audits"]:audit["stable_ID_to_index_mapping_conflicts"]=[{"unselected":"unresolved"}]
   return value
  with monkeypatch.context() as patch:
   patch.setattr(m.json,"loads",changed)
   with pytest.raises(ValueError) as caught:query(m,c,s)
  results.append(str(caught.value))
 assert results[0]==results[1]

@pytest.mark.parametrize("kind",("source_csv","mapping","FASTA","sequence_payload","handoff","QC"))
def test_between_calls_source_or_proof_tamper_still_fails_fresh_replay(api,baseline,cases,monkeypatch,kind):
 c=next(x for x in cases if x["name"]=="relation_protein_GO_early")
 sources=[m.ProCyonNativeSource(root=ROOT) for m in (baseline,api)]
 refs=sources[0].source_references("protein_GO");target=(ROOT/refs[kind]["path"]).resolve()
 for m,s in zip((baseline,api),sources):query(m,c,s)
 read=Path.read_bytes
 def changed(path):
  raw=read(path);return raw+b" " if path.resolve()==target else raw
 monkeypatch.setattr(Path,"read_bytes",changed)
 errors=[]
 for m,s in zip((baseline,api),sources):
  with pytest.raises(ValueError) as e:query(m,c,s)
  errors.append(str(e.value))
 assert errors[0]==errors[1]

@pytest.mark.parametrize("indices",((0,0),(),(False,),(-1,),tuple(range(33))))
def test_public_limits_types_duplicates_are_unchanged(api,baseline,indices):
 errors=[]
 for m in (baseline,api):
  with pytest.raises(ValueError) as e:m.RelationSelector("protein_GO",indices)
  errors.append(str(e.value))
 assert errors[0]==errors[1]

@pytest.mark.parametrize("missing",("source_namespace","native_original_ID","native_internal_index"))
def test_synthetic_malformed_mapping_schema_keeps_first_native_KeyError(api,baseline,cases,monkeypatch,missing):
 import pyarrow.parquet as pq
 c=next(x for x in cases if x["name"]=="relation_domain_GO_early")
 s=baseline.ProCyonNativeSource(root=ROOT);_,qc,_=s._snapshot();mapping_sha=qc["native_ID_index_map"]["sha256"];read=pq.read_table
 def changed(*a,**kw):
  table=read(*a,**kw)
  if isinstance(a[0],io.BytesIO) and hashlib.sha256(a[0].getbuffer()).hexdigest()==mapping_sha:
   return table.drop_columns([missing])
  return table
 monkeypatch.setattr(pq,"read_table",changed)
 errors=[]
 for m in (baseline,api):
  with pytest.raises(KeyError) as caught:query(m,c)
  errors.append(str(caught.value))
 assert errors[0]==errors[1]==repr(missing)

@pytest.mark.parametrize("duplicate",("source_namespace","native_original_ID","native_internal_index"))
def test_synthetic_duplicate_column_mapping_retains_baseline_materialization_semantics(api,baseline,cases,monkeypatch,duplicate):
 import pyarrow.parquet as pq
 c=next(x for x in cases if x["name"]=="relation_domain_GO_early")
 s=baseline.ProCyonNativeSource(root=ROOT);_,qc,_=s._snapshot();mapping_sha=qc["native_ID_index_map"]["sha256"];read=pq.read_table
 def changed(*a,**kw):
  table=read(*a,**kw)
  if isinstance(a[0],io.BytesIO) and hashlib.sha256(a[0].getbuffer()).hexdigest()==mapping_sha:
   return table.append_column(duplicate,table[duplicate])
  return table
 monkeypatch.setattr(pq,"read_table",changed)
 assert fingerprint(query(baseline,c))==fingerprint(query(api,c))
