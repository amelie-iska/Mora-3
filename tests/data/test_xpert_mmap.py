"""Synthetic mmap/NPY/math oracles; actual source fixtures tested separately."""
from dataclasses import replace
import copy,hashlib,importlib.util,json,os,struct,sys
from pathlib import Path
import numpy as np
import pytest
import mora3.data.xpert_mmap as collected_mmap
ROOT=Path(os.environ.get("MORA3_REPO_ROOT",Path.cwd()))
def canonical(x):return json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def sha(x):return hashlib.sha256(x).hexdigest()
def identity(p):
 s=p.stat();return dict(dev=s.st_dev,ino=s.st_ino,size=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
@pytest.fixture(scope="module")
def api():
 override=os.environ.get("MORA3_XPERT_MMAP_MODULE")
 if override is None:
  yield collected_mmap
  return
 p=Path(override)
 if not p.is_file():pytest.fail("native bounded mmap reader absent")
 # Alternate implementations have private identities; never replace live classes.
 name="mora3.data._test_xpert_mmap_candidate"
 spec=importlib.util.spec_from_file_location(name,p)
 m=importlib.util.module_from_spec(spec)
 with pytest.MonkeyPatch.context() as patch:
  patch.setitem(sys.modules,name,m)
  spec.loader.exec_module(m)
  yield m

def save_pair(root,a,name="dense",chunk_rows=2):
 # Independent standard NumPy writer; original bits are the oracle, never JSON floats.
 path=root/(name+".npy");mask=root/(name+"-finite.npy")
 with path.open("xb") as f:np.save(f,a,allow_pickle=False)
 with path.open("rb") as f:
  np.lib.format.read_magic(f);np.lib.format.read_array_header_1_0(f);offset=f.tell()
 chunks=[];packed=[];numeric_raw=a.tobytes(order="C");vpos=0
 width=int(np.prod(a.shape[1:])) if a.ndim>1 else 1
 for start in range(0,a.shape[0],chunk_rows):
  stop=min(start+chunk_rows,a.shape[0]);body=a[start:stop].tobytes(order="C")
  bits=np.packbits(np.isfinite(a[start:stop]).reshape(-1),bitorder="little").tobytes()
  chunks.append(dict(axis0_start=start,axis0_stop=stop,native_elements=(stop-start)*width,
   numeric_offset=offset+start*width*a.dtype.itemsize,numeric_bytes=len(body),numeric_sha256=sha(body),
   validity_offset=vpos,validity_bytes=len(bits),validity_sha256=sha(bits)))
  packed.append(bits);vpos+=len(bits)
 packed_raw=b"".join(packed)
 with mask.open("xb") as f:np.save(f,np.frombuffer(packed_raw,dtype="u1"),allow_pickle=False)
 with mask.open("rb") as f:
  np.lib.format.read_magic(f);np.lib.format.read_array_header_1_0(f);moffset=f.tell()
 for c in chunks:c["validity_offset"]+=moffset
 return dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path.read_bytes()),native_data_offset=offset,
  validity_path=str(mask),validity_bytes=mask.stat().st_size,validity_sha256=sha(mask.read_bytes()),validity_data_offset=moffset,
  native_value_sha256=sha(numeric_raw),native_validity_sha256=sha(packed_raw),numeric_header_sha256=sha(path.read_bytes()[:offset]),
  validity_header_sha256=sha(mask.read_bytes()[:moffset]),numeric_file_identity=identity(path),validity_file_identity=identity(mask),
  native_dtype=a.dtype.str,native_shape=list(a.shape),all_native_elements=a.size,chunks=chunks)
def native_matrix():
 bits=np.arange(35,dtype="<u4")
 bits[:7]=[0,0x80000000,0x7fc01234,0x7f800000,0xff800000,0xc0000000,0x3f800000]
 return bits.view("<f4").reshape(5,7)
def array(api,tmp_path,a=None,name="dense",chunk_rows=2):
 a=native_matrix() if a is None else a;d=save_pair(tmp_path,a,name,chunk_rows)
 return api.NativeMmapArray(root=tmp_path,descriptor_bytes=canonical(d)),d,a

def test_native_bits_NaN_payload_infinity_negativezero_and_finite_masks_are_lossless(api,tmp_path):
 store,d,a=array(api,tmp_path);result=store.read_elements(0,7)
 assert result.values_bytes==a[0].tobytes()
 assert np.frombuffer(result.values_bytes,dtype="<u4").tolist()==a[0].view("<u4").tolist()
 assert result.finite_mask==np.isfinite(a[0]).tobytes()
 assert result.observed_mask==b"\1"*7
 assert result.native_dtype=="<f4" and result.native_values().flags.writeable is False
 assert result.io()["whole_file_hash_recomputed"] is False
 assert result.io()["numeric_payload_bytes"]==2*7*4
 assert result.io()["numeric_payload_bytes"]<a.nbytes

def test_full_dense_native_gene_axis_and_context_overflow_no_crop(api,tmp_path):
 store,d,a=array(api,tmp_path)
 dense=api.DenseMmapMatrix(store)
 rows=dense.read_rows((0,4))
 assert [r.native_row_index for r in rows]==[0,4]
 for r in rows:
  assert r.axis_length==7 and len(r.values_bytes)==28
  assert r.profile().values_bytes==a[r.native_row_index].tobytes()
  assert r.profile().explicit_storage_mask==b"\1"*7
  assert r.profile().source_positions()==tuple(range(r.native_row_index*7,(r.native_row_index+1)*7))
 with pytest.raises(ValueError,match="context|crop|capacity"):rows[0].check_context_capacity(max_positions=6)
 assert rows[0].values_bytes==a[0].tobytes()

@pytest.mark.parametrize("indices",((),(0,0),(True,),(-1,),(5,),[0],tuple(range(17))))
def test_dense_selector_exact_bounds_and_duplicate_allocation_refuse(api,tmp_path,indices):
 store,_,_=array(api,tmp_path)
 with pytest.raises(ValueError):api.DenseMmapMatrix(store).read_rows(indices)

@pytest.mark.parametrize("start,stop",((-1,1),(0,36),(True,2),(0,1.0),(2,1)))
def test_flat_storage_span_types_extents_and_allocation_bounds_refuse(api,tmp_path,start,stop):
 store,_,_=array(api,tmp_path)
 with pytest.raises(ValueError):store.read_elements(start,stop)

def test_empty_native_CSC_storage_span_has_no_numeric_reads(api,tmp_path):
 store,_,_=array(api,tmp_path)
 result=store.read_elements(3,3)
 assert result.values_bytes==b"" and result.finite_mask==b""
 assert result.io()["numeric_payload_bytes"]==0

@pytest.mark.parametrize("mutation",("truncated","trailing","object","fortran","shape_bool","wrong_dtype","oversized_header","unsafe_literal"))
def test_NPY_preflight_refuses_unsafe_headers_objects_shapes_extents_before_mmap(api,tmp_path,mutation,monkeypatch):
 store,d,a=array(api,tmp_path);path=Path(d["path"]);raw=path.read_bytes()
 if mutation=="truncated":raw=raw[:-1]
 elif mutation=="trailing":raw+=b"opaque"
 elif mutation in ("object","fortran","shape_bool","wrong_dtype","unsafe_literal"):
  text=raw[10:d["native_data_offset"]].decode()
  if mutation=="object":text=text.replace("'<f4'","'|O'")
  elif mutation=="fortran":text=text.replace("False","True ")
  elif mutation=="shape_bool":text=text.replace("(5, 7)","(True,7)")
  elif mutation=="wrong_dtype":text=text.replace("'<f4'","'<i4'")
  else:text="__import__('os').system('false')"
  header=text.strip().encode()+b"\n";raw=b"\x93NUMPY\x01\x00"+struct.pack("<H",len(header))+header+raw[d["native_data_offset"]:]
  d["native_data_offset"]=10+len(header);d["numeric_header_sha256"]=sha(raw[:d["native_data_offset"]])
 elif mutation=="oversized_header":raw=b"\x93NUMPY\x01\x00"+struct.pack("<H",65535)+b" "*64
 path.write_bytes(raw);d["bytes"]=len(raw);d["numeric_file_identity"]=identity(path)
 def forbidden(*a,**k):raise AssertionError("mmap must not execute before safe preflight")
 monkeypatch.setattr(api,"_mapped_bytes",forbidden)
 with pytest.raises(ValueError):api.NativeMmapArray(root=tmp_path,descriptor_bytes=canonical(d))

@pytest.mark.parametrize("mutation",("numeric_stat","validity_stat","header_hash","chunk_value_hash","chunk_mask_hash","gap","overlap","chunk_count_bool","validity_gap","dtype_alias","shape_alias"))
def test_numeric_descriptor_native_chunk_header_stat_and_type_mutations_refuse(api,tmp_path,mutation):
 store,d,a=array(api,tmp_path);d=copy.deepcopy(d)
 if mutation=="numeric_stat":d["numeric_file_identity"]["ino"]+=1
 elif mutation=="validity_stat":d["validity_file_identity"]["mtime_ns"]+=1
 elif mutation=="header_hash":d["numeric_header_sha256"]="0"*64
 elif mutation=="chunk_value_hash":d["chunks"][0]["numeric_sha256"]="0"*64
 elif mutation=="chunk_mask_hash":d["chunks"][0]["validity_sha256"]="0"*64
 elif mutation=="gap":d["chunks"][1]["axis0_start"]+=1
 elif mutation=="overlap":d["chunks"][1]["axis0_start"]-=1
 elif mutation=="chunk_count_bool":d["chunks"][0]["native_elements"]=True
 elif mutation=="validity_gap":d["chunks"][1]["validity_offset"]+=1
 elif mutation=="dtype_alias":d["native_dtype"]="=f4"
 else:d["native_shape"][0]=5.0
 with pytest.raises(ValueError):
  s=api.NativeMmapArray(root=tmp_path,descriptor_bytes=canonical(d));s.read_elements(0,1)

def test_changed_native_bytes_cannot_hide_behind_same_size_or_refreshed_stat(api,tmp_path):
 store,d,a=array(api,tmp_path);p=Path(d["path"])
 raw=bytearray(p.read_bytes());raw[d["native_data_offset"]]^=1;p.write_bytes(raw)
 with pytest.raises(ValueError):store.read_elements(0,1)
 d["numeric_file_identity"]=identity(p)
 with pytest.raises(ValueError):api.NativeMmapArray(root=tmp_path,descriptor_bytes=canonical(d)).read_elements(0,1)

@pytest.mark.parametrize("mutation",("nonfinite_mask","padding"))
def test_rehashed_numeric_validity_mask_and_padding_must_match_native_bits(api,tmp_path,mutation):
 store,d,a=array(api,tmp_path);p=Path(d["validity_path"]);raw=bytearray(p.read_bytes());c=d["chunks"][0]
 raw[c["validity_offset"]]^=4 if mutation=="nonfinite_mask" else 128
 p.write_bytes(raw);d["validity_file_identity"]=identity(p)
 c["validity_sha256"]=sha(bytes(raw[c["validity_offset"]:c["validity_offset"]+c["validity_bytes"]]))
 with pytest.raises(ValueError):api.NativeMmapArray(root=tmp_path,descriptor_bytes=canonical(d)).read_elements(0,1)

def csc(api,tmp_path,indices=(4,1,3),unique=False,ascending=False):
 data=np.array([0.0,-0.0,2.0],dtype="<f4");idx=np.array(indices,dtype="<i4");ptr=np.array([0,2,2,3],dtype="<i4")
 ds,_,_=array(api,tmp_path,data,"data",2);ix,_,_=array(api,tmp_path,idx,"indices",2);ps,_,_=array(api,tmp_path,ptr,"indptr",2)
 return api.CSCMmapMatrix(ds,ix,ps,logical_shape=(6,3),coordinate_unique=unique,strictly_ascending=ascending)

def test_CSC_column_native_row_and_storage_order_explicit_negativezero_vs_implicit_observedzero(api,tmp_path):
 sparse=csc(api,tmp_path)
 col,empty=sparse.read_columns((0,1))
 assert col.row_indices()==(4,1) and col.storage_positions()==(0,1)
 assert col.values_bytes==np.array([0.0,-0.0],dtype="<f4").tobytes()
 assert col.explicit_stored_values==2 and col.implicit_observed_zero_count==4
 assert col.implicit_zero_is_missing is False
 assert empty.values_bytes==b"" and empty.implicit_observed_zero_count==6
 assert col.row_order=="source_native_storage_order_not_sorted"
 with pytest.raises(ValueError,match="index|row|unavailable"):sparse.read_rows((0,))

@pytest.mark.parametrize("indices,unique,ascending",(((1,1,3),True,False),((4,1,3),True,True),((6,1,3),False,False),((-1,1,3),False,False)))
def test_CSC_verified_unique_ascending_and_axis_bounds_refuse_without_repair(api,tmp_path,indices,unique,ascending):
 with pytest.raises(ValueError):csc(api,tmp_path,indices,unique,ascending).read_columns((0,))

def test_CSC_unknown_duplicate_order_is_retained_not_coalesced_or_promoted(api,tmp_path):
 sparse=csc(api,tmp_path,(1,1,3))
 col,=sparse.read_columns((0,))
 assert col.row_indices()==(1,1) and col.values_bytes==np.array([0.0,-0.0],dtype="<f4").tobytes()
 assert col.duplicate_coordinates_present is True
 assert col.implicit_observed_zero_count==5

@pytest.mark.parametrize("purpose",("training","evaluation","loss","split_use","counterfactual","DE_statistics","guide_efficacy"))
def test_numeric_backing_never_admits_tasks_or_roles(api,tmp_path,purpose):
 store,_,_=array(api,tmp_path)
 result=store.read_elements(0,7)
 for proof in (None,True,{"verified":True,"master_version":23}):
  with pytest.raises(ValueError):result.require_model_use(purpose,admission=proof)

def test_query_budget_refuses_large_read_before_mmap_or_output_allocation(api,tmp_path,monkeypatch):
 store,_,_=array(api,tmp_path)
 def forbidden(*a,**k):raise AssertionError("no map for rejected read")
 monkeypatch.setattr(api,"_mapped_bytes",forbidden)
 with pytest.raises(ValueError):store.read_elements(0,35,max_elements=10)


FIXTURE=Path("outputs/data-sources/20261101-XPert-source-v1/curated/numeric-mmap-reader-compatible-fixture-v1/immutable-compatible-numeric-store-fixture-manifest-v1.json")
FIXTURE_SHA="f58eee1f2c68a1b33233ace33b3ac0587b935d54188069f18a63fb127df4b2a7"
@pytest.fixture(scope="module")
def source(api):
 if not (ROOT/FIXTURE).is_file():pytest.skip("source-owner compatible immutable fixture unavailable")
 return api.XPertMmapSource(root=ROOT,manifest=FIXTURE,expected_manifest_sha256=FIXTURE_SHA)
@pytest.fixture(scope="module")
def actual(source):return source.read_dense_rows((0,1,590))

def test_actual_three_L1000_rows_preserve_full_native978axis_bits_masks_positions_context_and_old_roles(source,actual):
 assert [v.native_row_index for v in actual]==[0,1,590]
 for v in actual:
  assert v.profile.axis_length==978 and v.profile.values_bytes==v.reference.profile.values_bytes
  assert v.profile.finite_mask==v.reference.profile.finite_mask
  assert v.profile.source_positions()==tuple(range(v.native_row_index*978,(v.native_row_index+1)*978))
  assert v.prediction_inputs()==v.reference.prediction_inputs()
  assert v.official_split is None
  assert source.verify(v)==v
  assert v.audit()["source_evidence_role"]=="actual_bounded_L1000_native_fixture_not_full_matrix"
  assert v.audit()["whole_numeric_verifier_inherited"] is True
  assert v.audit()["full_source_export_available"] is False
  assert v.numeric_row.values.io()["numeric_payload_bytes"]==978*4
  assert v.numeric_row.values.io()["whole_file_hash_recomputed"] is False
  with pytest.raises(ValueError):v.numeric_row.check_context_capacity(max_positions=977)

def test_actual_metadata_QC_audit_target_values_and_controls_never_enter_predictors(actual):
 a,b,control=actual
 assert a.prediction_inputs()==b.prediction_inputs()
 assert control.reference.control_semantics.same_cell_before_after is False
 text=canonical(a.prediction_inputs()).decode()
 assert all(x not in text for x in ("inst_id","rna_plate","rna_well","plate_id","pvals","DE_statistics"))
 assert control.prediction_inputs()["native_dose_coordinate"]["coordinate_known"] is False

def test_source_owner_synthetic_CSC_fixture_remains_explicitly_synthetic_not_Norman_native_coverage(source):
 values=source.read_columns("SYNTHETIC-CSC-CONTRACT-TEST",(0,1,2))
 assert any(c.explicit_stored_values==0 for c in values)
 assert any(not np.isfinite(c.values.native_values()).all() for c in values)
 for c in values:
  assert c.audit()["source_evidence_role"]=="SYNTHETIC_TEST_ONLY"
  assert c.audit()["original_Norman_CSC_source_coverage"] is False
  assert source.verify(c)==c
 with pytest.raises(ValueError):source.read_dense_rows((32221275,))
 with pytest.raises(ValueError):source.read_CSC_rows((0,))

@pytest.mark.parametrize("artifact",("fixture","source_QC","axis","roles"))
def test_real_source_axis_role_and_manifest_substitutions_rejected(api,source,artifact,monkeypatch):
 target=(ROOT/source.source_references()[artifact]["path"]).resolve();old=Path.read_bytes;touched=[]
 def tamper(p):
  raw=old(p)
  if p.resolve()==target:touched.append(True);return raw+b" "
  return raw
 monkeypatch.setattr(Path,"read_bytes",tamper)
 with pytest.raises(ValueError):
  s=api.XPertMmapSource(root=ROOT);s.read_dense_rows((0,))
 assert touched

def test_actual_mmap_payload_mutation_rejected_without_touching_immutable_sources(api,source,monkeypatch):
 original=api._mapped_bytes;calls=[]
 def tamper(fd,offset,count):
  raw,mapped=original(fd,offset,count);calls.append(count)
  return bytes([raw[0]^1])+raw[1:],mapped
 monkeypatch.setattr(api,"_mapped_bytes",tamper)
 with pytest.raises(ValueError):source.read_dense_rows((0,))
 assert calls

@pytest.mark.parametrize("field",("fixture_sha256","native_row_index","source_role"))
def test_actual_hidden_source_binding_aliases_leave_predictors_but_fresh_replay_refuses(source,actual,field):
 value=actual[0];p=value.audit()
 if field=="fixture_sha256":p[field]="0"*64
 elif field=="native_row_index":p[field]=False
 else:p["source_evidence_role"]="full_source_measured_truth"
 changed=replace(value,provenance_bytes=canonical(p))
 assert changed.prediction_inputs()==value.prediction_inputs()
 with pytest.raises(ValueError):source.verify(changed)

def test_store_availability_no_model_split_guide_DE_or_positive_loss_admission(source,actual):
 assert source.gates()["full_export_available"] is False
 assert source.gates()["Norman_full_CSC_backing_available"] is False
 assert source.gates()["model_tokens"] is None
 for v in actual:
  for purpose in ("training","evaluation","loss","split_use","counterfactual","DE_statistics","guide_efficacy"):
   with pytest.raises(ValueError):v.require_model_use(purpose,admission={"verified":True,"sha256":FIXTURE_SHA})

def test_bounded_numeric_mmap_never_whole_reads_NPY_or_rawH5_pickle_ZIP(api,source,monkeypatch):
 old=Path.read_bytes
 def safe(p):
  assert p.suffix.lower() not in (".h5ad",".h5",".zip",".pkl")
  # Legacy six-row source verifier may read its fixed tiny NPY fixtures.
  assert not (p.suffix==".npy" and "numeric-mmap-reader-compatible-fixture-v1" in str(p))
  return old(p)
 monkeypatch.setattr(Path,"read_bytes",safe)
 rows=source.read_dense_rows((590,))
 assert rows[0].numeric_row.values.io()["selected_chunk_indices"]==[2]

def test_fake_future_full_export_manifest_or_unknown_dense_row_refuses(api,source,tmp_path):
 p=tmp_path/"full.json";p.write_text('{"full_source_export":true,"model_use_allowed":true}')
 with pytest.raises(ValueError):api.XPertMmapSource(root=tmp_path,manifest="full.json")
 with pytest.raises(ValueError):api.XPertMmapSource(root=ROOT,expected_manifest_sha256=sha(p.read_bytes()))
 with pytest.raises(ValueError):source.read_dense_rows((2,))


@pytest.mark.parametrize("field,value",(("native_row_index",False),("axis_length",7.0)))
def test_dense_row_native_integer_fields_cannot_alias_bool_or_float(api,tmp_path,field,value):
 store,_,_=array(api,tmp_path);row,=api.DenseMmapMatrix(store).read_rows((0,))
 with pytest.raises(ValueError):replace(row,**{field:value})

def test_CSC_native_column_integer_cannot_alias_boolean(api,tmp_path):
 col,=csc(api,tmp_path).read_columns((0,))
 with pytest.raises(ValueError):replace(col,native_column_index=False)

def test_file_identity_is_checked_after_consumed_mmap_chunks(api,tmp_path,monkeypatch):
 store,d,a=array(api,tmp_path);original=api._mapped_bytes;changed=[]
 def mutate(fd,offset,count):
  out=original(fd,offset,count)
  if not changed:
   p=Path(d["path"]);s=p.stat();os.utime(p,ns=(s.st_atime_ns,s.st_mtime_ns+1));changed.append(True)
  return out
 monkeypatch.setattr(api,"_mapped_bytes",mutate)
 with pytest.raises(ValueError):store.read_elements(0,1)
 assert changed

def test_nonregular_numeric_substitution_cannot_block_open_before_preflight(api,tmp_path,monkeypatch):
 store,d,a=array(api,tmp_path);p=Path(d["path"]);p.unlink();os.mkfifo(p)
 original=api.os.open
 def guarded(path,flags,*args,**kwargs):
  assert flags&os.O_NONBLOCK,"numeric preflight must not block on a substituted FIFO"
  return original(path,flags,*args,**kwargs)
 monkeypatch.setattr(api.os,"open",guarded)
 with pytest.raises(ValueError):api.NativeMmapArray(root=tmp_path,descriptor_bytes=canonical(d))

@pytest.mark.parametrize("pointers",((0,3,2,3),(0,4,4,3),(1,2,2,3),(0,2,2,4)))
def test_CSC_terminal_and_selected_pointer_bounds_order_refuse_without_repair(api,tmp_path,pointers):
 data,_,_=array(api,tmp_path,np.array([1,2,3],dtype="<f4"),"d")
 indices,_,_=array(api,tmp_path,np.array([0,1,2],dtype="<i4"),"i")
 ptr,_,_=array(api,tmp_path,np.array(pointers,dtype="<i4"),"p")
 with pytest.raises(ValueError):
  matrix=api.CSCMmapMatrix(data,indices,ptr,logical_shape=(3,3),coordinate_unique=True,strictly_ascending=True)
  matrix.read_columns((1,))


def test_default_fixture_preserves_canonical_module_identity(api):
 import mora3.data.xpert_mmap as canonical_module
 assert sys.modules['mora3.data.xpert_mmap'] is collected_mmap
 assert canonical_module is collected_mmap
 if not os.environ.get('MORA3_XPERT_MMAP_MODULE'):
  assert api is canonical_module
