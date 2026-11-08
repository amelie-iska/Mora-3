"""Bounded native numeric NPY mmap access; no source/task/model admission.

Whole export verification is inherited from pinned source-owner receipts.
Runtime checks headers/stat identity and every consumed value/validity chunk.
No H5AD/ZIP/pickle, transpose, reindex, coalesce, normalization or dtype conversion.
"""
from dataclasses import dataclass,replace
import ast
import hashlib
import json
import math
import mmap
import os
from pathlib import Path
import re
import stat
import struct
import numpy as np
from .xpert_native import (Float32Profile,XPertContext,ControlSemantics,require_DE_supervision,
 require_XPert_model_use,XPertCandidate,XPertNativeSource,XPertSelector,FIXTURE_REF as ROLE_REF)

MAX_ELEMENTS=2_000_000
MAX_CHUNK_BYTES=32*1024*1024
MAX_QUERY_BYTES=256*1024*1024
def _canonical(x):return json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def _sha(raw):return hashlib.sha256(raw).hexdigest()
def _is_sha(x):return type(x) is str and re.fullmatch("[0-9a-f]{64}",x) is not None
def _int(x):return type(x) is int and x>=0
def _json(raw):
 if type(raw) is not bytes:raise ValueError("immutable native descriptor bytes required")
 def pairs(items):
  d={}
  for k,v in items:
   if k in d:raise ValueError("duplicate descriptor key")
   d[k]=v
  return d
 def invalid(v):raise ValueError("nonfinite JSON descriptor")
 d=json.loads(raw,object_pairs_hook=pairs,parse_constant=invalid)
 if _canonical(d)!=raw:raise ValueError("canonical typed descriptor required")
 return d
def _same(a,b,message):
 if _canonical(a)!=_canonical(b):raise ValueError(message)
def _identity(s):
 if not stat.S_ISREG(s.st_mode):raise ValueError("numeric regular file required")
 return dict(dev=s.st_dev,ino=s.st_ino,size=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
def _check_identity(actual,expected):
 if (type(expected) is not dict or set(expected)!={"dev","ino","size","mtime_ns","ctime_ns"}
      or any(not _int(v) for v in expected.values())):raise ValueError("exact typed immutable numeric file identity required")
 _same(actual,expected,"numeric file stat identity changed; no source replacement")
def _path(root,path):
 if type(path) is not str or not path:raise ValueError("exact numeric source path required")
 p=root/path
 if p.is_symlink():raise ValueError("numeric symlink substitution refused")
 p=p.resolve()
 if not p.is_relative_to(root) or p.suffix.lower()!=".npy":raise ValueError("source-bound numeric NPY only")
 return p
def _open(root,path):
 p=_path(root,path)
 try:return p,os.fdopen(os.open(p,os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW|os.O_NONBLOCK),"rb")
 except OSError as e:raise ValueError("safe readonly numeric file unavailable") from e
def _header(f,*,dtype,shape,file_bytes,offset,header_sha,identity):
 _check_identity(_identity(os.fstat(f.fileno())),identity)
 if not _int(file_bytes) or file_bytes!=identity["size"] or not _int(offset) or offset<10 or offset>4096:
  raise ValueError("exact bounded NPY extent/data offset required")
 if dtype not in ("<f4","<i4","|u1","<u4","<u8") or type(shape) is not tuple or not 1<=len(shape)<=2 or any(not _int(n) or n<1 for n in shape):
  raise ValueError("plain exact native dtype/shape required; no object/alias/conversion")
 f.seek(0);prefix=f.read(10)
 if len(prefix)!=10 or prefix[:8]!=b"\x93NUMPY\x01\x00":raise ValueError("exact numeric NPY v1 magic/version required")
 length=struct.unpack("<H",prefix[8:])[0]
 if length>4086 or length+10!=offset:raise ValueError("bounded exact NPY header length required")
 body=f.read(length);raw=prefix+body
 if len(body)!=length or not body.endswith(b"\n") or not _is_sha(header_sha) or _sha(raw)!=header_sha:
  raise ValueError("native NPY header integrity mismatch")
 try:header=ast.literal_eval(body.decode("latin1").strip())
 except (ValueError,SyntaxError,TypeError,MemoryError,RecursionError) as e:raise ValueError("unsafe NPY header literal") from e
 if (type(header) is not dict or set(header)!={"descr","fortran_order","shape"}
      or type(header.get("descr")) is not str or header["descr"]!=dtype or header["fortran_order"] is not False
      or type(header["shape"]) is not tuple or any(type(x) is not int for x in header["shape"])
      or header["shape"]!=shape or file_bytes!=offset+math.prod(shape)*np.dtype(dtype).itemsize):
  raise ValueError("native numeric NPY dtype/shape/Corder/extent mismatch; no pickle or repair")
 return len(raw)
def _mapped_bytes(fd,offset,count):
 """Map only one declared native chunk, copy its bytes and release pages."""
 if count<1:raise ValueError("nonempty bounded mmap chunk required")
 delta=offset%mmap.ALLOCATIONGRANULARITY
 with mmap.mmap(fd,delta+count,access=mmap.ACCESS_READ,offset=offset-delta) as mm:
  raw=mm[delta:delta+count]
  if hasattr(mm,"madvise") and hasattr(mmap,"MADV_DONTNEED"):mm.madvise(mmap.MADV_DONTNEED)
 return raw,delta+count

@dataclass(frozen=True)
class NativeNumericSlice:
 values_bytes: bytes
 finite_mask: bytes
 native_dtype: str
 start: int
 stop: int
 provenance_bytes: bytes
 def __post_init__(self):
  if (self.native_dtype not in ("<f4","<i4","|u1","<u4","<u8") or not _int(self.start) or not _int(self.stop) or self.stop<self.start
      or type(self.values_bytes) is not bytes or len(self.values_bytes)!=(self.stop-self.start)*np.dtype(self.native_dtype).itemsize
      or type(self.finite_mask) is not bytes or len(self.finite_mask)!=self.stop-self.start or set(self.finite_mask)-{0,1}):
   raise ValueError("exact immutable numeric slice/mask/extent required")
  if np.isfinite(self.native_values()).tobytes()!=self.finite_mask:raise ValueError("finite mask differs from native bit values")
  _json(self.provenance_bytes)
 @property
 def observed_mask(self):return b"\1"*(self.stop-self.start)
 def native_values(self):return np.frombuffer(self.values_bytes,dtype=self.native_dtype)
 def io(self):return _json(self.provenance_bytes)["io"]
 def audit(self):return _json(self.provenance_bytes)
 def require_model_use(self,purpose,*,admission=None):
  if purpose=="DE_statistics":return require_DE_supervision(purpose)
  return require_XPert_model_use(purpose,admission=admission)

class NativeMmapArray:
 """Format/integrity core. Arbitrary descriptors are NOT source qualification."""
 def __init__(self,*,root,descriptor_bytes):
  self.root=Path(root).resolve();self.descriptor_bytes=descriptor_bytes
  d=_json(descriptor_bytes)
  self._validate(d)
  with self._opened(d) as files:
   pass
 def descriptor(self):return _json(self.descriptor_bytes)
 @staticmethod
 def _validate(d):
  dtype=d["native_dtype"];shape=d["native_shape"]
  if (dtype not in ("<f4","<i4","|u1","<u4","<u8") or type(shape) is not list or not 1<=len(shape)<=2
      or any(not _int(n) or n<1 for n in shape) or type(d["chunks"]) is not list or not d["chunks"]):
   raise ValueError("exact native numeric dtype/full shape/chunks required")
  for k in ("sha256","validity_sha256","native_value_sha256","native_validity_sha256","numeric_header_sha256","validity_header_sha256"):
   if not _is_sha(d.get(k)):raise ValueError("exact native file/header/value/validity SHA required")
  for k in ("bytes","validity_bytes","native_data_offset","validity_data_offset"):
   if not _int(d.get(k)) or d[k]<1:raise ValueError("typed numeric extents required")
  step=math.prod(shape[1:]);item=np.dtype(dtype).itemsize;pos=0;vpos=d["validity_data_offset"]
  for c in d["chunks"]:
   for key in ("axis0_start","axis0_stop","native_elements","numeric_offset","numeric_bytes","validity_offset","validity_bytes"):
    if not _int(c.get(key)):raise ValueError("typed original chunk bounds required")
   count=(c["axis0_stop"]-c["axis0_start"])*step
   if (c["axis0_start"]!=pos or not pos<c["axis0_stop"]<=shape[0] or c["native_elements"]!=count
       or c["numeric_offset"]!=d["native_data_offset"]+pos*step*item or c["numeric_bytes"]!=count*item
       or c["validity_offset"]!=vpos or c["validity_bytes"]!=(count+7)//8
       or c["numeric_bytes"]>MAX_CHUNK_BYTES or not _is_sha(c.get("numeric_sha256")) or not _is_sha(c.get("validity_sha256"))):
    raise ValueError("contiguous original numeric/chunk/packed-validity partition required")
   pos=c["axis0_stop"];vpos+=c["validity_bytes"]
  if pos!=shape[0] or vpos!=d["validity_bytes"] or d["bytes"]!=d["native_data_offset"]+math.prod(shape)*item:
   raise ValueError("complete native chunk/NPY extent conservation required")
 def _opened(self,d):
  # Local context manager keeps exact checked file descriptors through both mmap reads.
  from contextlib import contextmanager
  @contextmanager
  def opened():
   p,f=_open(self.root,d["path"]);g=None
   try:
    q,g=_open(self.root,d["validity_path"])
    if p==q:raise ValueError("numeric and validity files must remain distinct")
    headers=_header(f,dtype=d["native_dtype"],shape=tuple(d["native_shape"]),file_bytes=d["bytes"],
     offset=d["native_data_offset"],header_sha=d["numeric_header_sha256"],identity=d["numeric_file_identity"])
    packed=d["validity_bytes"]-d["validity_data_offset"]
    headers+=_header(g,dtype="|u1",shape=(packed,),file_bytes=d["validity_bytes"],
     offset=d["validity_data_offset"],header_sha=d["validity_header_sha256"],identity=d["validity_file_identity"])
    yield f,g,headers
    _check_identity(_identity(os.fstat(f.fileno())),d["numeric_file_identity"])
    _check_identity(_identity(os.fstat(g.fileno())),d["validity_file_identity"])
    _check_identity(_identity(os.stat(p,follow_symlinks=False)),d["numeric_file_identity"])
    _check_identity(_identity(os.stat(q,follow_symlinks=False)),d["validity_file_identity"])
   finally:
    f.close()
    if g is not None:g.close()
  return opened()
 def read_elements(self,start,stop,*,max_elements=MAX_ELEMENTS):
  d=self.descriptor();total=math.prod(d["native_shape"])
  if (not _int(start) or not _int(stop) or not start<=stop<=total or type(max_elements) is not int
      or not 1<=max_elements<=MAX_ELEMENTS or stop-start>max_elements):
   raise ValueError("explicit bounded exact native storage span required")
  step=math.prod(d["native_shape"][1:]);item=np.dtype(d["native_dtype"]).itemsize
  selected=[(i,c) for i,c in enumerate(d["chunks"]) if start<c["axis0_stop"]*step and stop>c["axis0_start"]*step] if start<stop else []
  planned=sum(c["numeric_bytes"]+c["validity_bytes"] for _,c in selected)
  if planned>MAX_QUERY_BYTES:raise ValueError("selected original chunks exceed bounded query IO budget")
  values=[];finite=[];mapped=0;numeric_bytes=validity_bytes=0
  with self._opened(d) as (f,g,headers):
   for index,c in selected:
    body,nmapped=_mapped_bytes(f.fileno(),c["numeric_offset"],c["numeric_bytes"])
    bits,vmapped=_mapped_bytes(g.fileno(),c["validity_offset"],c["validity_bytes"])
    mapped+=nmapped+vmapped;numeric_bytes+=len(body);validity_bytes+=len(bits)
    if _sha(body)!=c["numeric_sha256"] or _sha(bits)!=c["validity_sha256"]:
     raise ValueError("consumed numeric/validity chunk integrity mismatch")
    a=np.frombuffer(body,dtype=d["native_dtype"]);valid=np.isfinite(a)
    if len(a)!=c["native_elements"] or np.packbits(valid,bitorder="little").tobytes()!=bits:
     raise ValueError("native numeric validity/padding differs from exact per-source chunk")
    origin=c["axis0_start"]*step;left=max(start,origin)-origin;right=min(stop,c["axis0_stop"]*step)-origin
    values.append(body[left*item:right*item]);finite.append(valid[left:right].tobytes())
  raw=b"".join(values);mask=b"".join(finite)
  p=dict(descriptor_sha256=_sha(self.descriptor_bytes),native_array_file_sha256_inherited=d["sha256"],
   native_value_sha256_inherited=d["native_value_sha256"],source_value_role="native_expression_target_only",
   source_identity_verified=False,task_admitted=False,official_split=None,model_tokens=None,
   io=dict(header_bytes_read=headers,numeric_payload_bytes=numeric_bytes,validity_payload_bytes=validity_bytes,
    mapped_page_bytes=mapped,selected_chunk_indices=[i for i,_ in selected],output_value_bytes=len(raw),
    output_validity_bytes=len(mask),whole_file_hash_recomputed=False,
    scope="application bytes/pages mapped; OS physical disk/cache IO not measured"))
  return NativeNumericSlice(raw,mask,d["native_dtype"],start,stop,_canonical(p))

@dataclass(frozen=True)
class DenseNativeRow:
 native_row_index: int
 values: NativeNumericSlice
 axis_length: int
 def __post_init__(self):
  if (not _int(self.native_row_index) or not _int(self.axis_length) or self.axis_length<1
      or type(self.values) is not NativeNumericSlice or self.values.native_dtype!="<f4"
      or self.values.stop-self.values.start!=self.axis_length):
   raise ValueError("exact dense native source row/full-axis integer types required")
 @property
 def values_bytes(self):return self.values.values_bytes
 def profile(self):
  n=self.axis_length
  return Float32Profile(self.values.values_bytes,b"\1"*n,self.values.finite_mask,b"\1"*n,
   b"".join(struct.pack("<q",i) for i in range(self.native_row_index*n,(self.native_row_index+1)*n)),"dense")
 def check_context_capacity(self,*,max_positions):
  if type(max_positions) is not int or max_positions<1 or self.axis_length>max_positions:
   raise ValueError("complete native axis exceeds explicit context capacity; no crop")
 def require_model_use(self,purpose,*,admission=None):return self.values.require_model_use(purpose,admission=admission)

def _selectors(indices,maximum):
 if (type(indices) is not tuple or not 1<=len(indices)<=16 or any(not _int(i) or i>=maximum for i in indices)
     or len(set(indices))!=len(indices)):raise ValueError("1..16unique exact native selectors required")

class DenseMmapMatrix:
 def __init__(self,array):
  if type(array) is not NativeMmapArray or array.descriptor()["native_dtype"]!="<f4" or len(array.descriptor()["native_shape"])!=2:
   raise ValueError("complete native dense float32 matrix required")
  self.array=array
 def read_rows(self,rows):
  shape=self.array.descriptor()["native_shape"];_selectors(rows,shape[0]);width=shape[1]
  return tuple(DenseNativeRow(i,self.array.read_elements(i*width,(i+1)*width),width) for i in rows)

@dataclass(frozen=True)
class CSCNativeColumn:
 native_column_index: int
 values: NativeNumericSlice
 indices: NativeNumericSlice
 logical_rows: int
 coordinate_unique: bool
 strictly_ascending: bool
 provenance_bytes: bytes = b"{}"
 def __post_init__(self):
  _json(self.provenance_bytes)
  rows=self.row_indices()
  if (not _int(self.native_column_index) or self.values.native_dtype!="<f4" or self.indices.native_dtype!="<i4" or self.values.start!=self.indices.start
      or self.values.stop!=self.indices.stop or not _int(self.logical_rows) or self.logical_rows<1
      or type(self.coordinate_unique) is not bool or type(self.strictly_ascending) is not bool
      or any(not 0<=i<self.logical_rows for i in rows)):
   raise ValueError("exact native CSC value/index/storage/axis binding required")
  if self.coordinate_unique and len(set(rows))!=len(rows):raise ValueError("source-proven unique CSC coordinates violated")
  if self.strictly_ascending and any(a>=b for a,b in zip(rows,rows[1:])):raise ValueError("source-proven native CSC row order violated")
 @property
 def values_bytes(self):return self.values.values_bytes
 def audit(self):return _json(self.provenance_bytes)
 def row_indices(self):return tuple(int(i) for i in self.indices.native_values())
 def storage_positions(self):return tuple(range(self.values.start,self.values.stop))
 @property
 def explicit_stored_values(self):return self.values.stop-self.values.start
 @property
 def implicit_observed_zero_count(self):return self.logical_rows-len(set(self.row_indices()))
 @property
 def implicit_zero_is_missing(self):return False
 @property
 def duplicate_coordinates_present(self):return len(set(self.row_indices()))!=self.explicit_stored_values
 @property
 def row_order(self):return "source_native_storage_order_not_sorted"
 def require_model_use(self,purpose,*,admission=None):return self.values.require_model_use(purpose,admission=admission)

class CSCMmapMatrix:
 def __init__(self,data,indices,indptr,*,logical_shape,coordinate_unique,strictly_ascending):
  if (any(type(a) is not NativeMmapArray for a in (data,indices,indptr)) or type(logical_shape) is not tuple
      or len(logical_shape)!=2 or any(not _int(i) or i<1 for i in logical_shape)
      or type(coordinate_unique) is not bool or type(strictly_ascending) is not bool):
   raise ValueError("exact CSC backing arrays/logical axes/proof roles required")
  ds=data.descriptor();ix=indices.descriptor();ptr=indptr.descriptor()
  if (ds["native_dtype"]!="<f4" or ix["native_dtype"]!="<i4" or ptr["native_dtype"]!="<i4"
      or len(ds["native_shape"])!=1 or ix["native_shape"]!=ds["native_shape"] or ptr["native_shape"]!=[logical_shape[1]+1]):
   raise ValueError("native CSC data/int32indices/int32indptr extents required")
  first=int(indptr.read_elements(0,1).native_values()[0]);last=int(indptr.read_elements(logical_shape[1],logical_shape[1]+1).native_values()[0])
  if first!=0 or last!=ds["native_shape"][0]:raise ValueError("full native CSC terminal pointer mismatch")
  self.data=data;self.indices=indices;self.indptr=indptr;self.logical_shape=logical_shape
  self.coordinate_unique=coordinate_unique;self.strictly_ascending=strictly_ascending
 def read_columns(self,columns):
  _selectors(columns,self.logical_shape[1]);out=[];total=self.data.descriptor()["native_shape"][0]
  for column in columns:
   pair=self.indptr.read_elements(column,column+2).native_values();start,stop=map(int,pair)
   if not 0<=start<=stop<=total:raise ValueError("native CSC column pointer bounds/order mismatch")
   out.append(CSCNativeColumn(column,self.data.read_elements(start,stop),self.indices.read_elements(start,stop),
    self.logical_shape[0],self.coordinate_unique,self.strictly_ascending))
  return tuple(out)
 def read_storage(self,start,stop):
  values=self.data.read_elements(start,stop);indices=self.indices.read_elements(start,stop)
  if any(not 0<=int(i)<self.logical_shape[0] for i in indices.native_values()):raise ValueError("native CSC storage row index outside source axis")
  return values,indices
 def read_rows(self,rows):
  raise ValueError("efficient arbitrary CSC row lookup unavailable without separately reviewed derived access index")

FIXTURE_REF=dict(path="outputs/data-sources/20261101-XPert-source-v1/curated/numeric-mmap-reader-compatible-fixture-v1/immutable-compatible-numeric-store-fixture-manifest-v1.json",
 bytes=40773,sha256="f58eee1f2c68a1b33233ace33b3ac0587b935d54188069f18a63fb127df4b2a7")

def transport_descriptor(record):
 """Bind source writer/complete numeric verifier receipts; never run verifier."""
 expected=record["expected_native"];transport=record["transport"];verified=record["verified_store"]
 for key in ("native_value_sha256","native_validity_sha256","numeric_header_sha256","validity_header_sha256",
  "native_data_offset","validity_data_offset","chunks","numeric_file_identity","validity_file_identity","all_native_elements"):
  _same(transport[key],verified[key],"transport/complete-verifier proof mismatch")
 if (verified["allow_pickle"] is not False or verified["source_h5py_runtime_dependency"] is not False
      or verified["numeric_file_sha256"]!=transport["sha256"] or verified["validity_file_sha256"]!=transport["validity_sha256"]
      or expected["native_value_SHA256"]!=transport["native_value_sha256"]
      or expected["native_validity_SHA256"]!=transport["native_validity_sha256"]):
  raise ValueError("completed native numeric/full-file/validity verification required")
 _same(verified["native_shape"],expected["native_shape"],"verified native shape mismatch")
 _same(verified["native_dtype"],expected["native_dtype"],"verified native dtype mismatch")
 _same([{k:c[k] for k in ("axis0_start","axis0_stop","native_elements")} for c in transport["chunks"]],
  expected["chunks"],"original declared source chunk partition mismatch")
 d=dict(transport,native_shape=expected["native_shape"],native_dtype=expected["native_dtype"])
 if type(d["all_native_elements"]) is not int or d["all_native_elements"]!=math.prod(d["native_shape"]):
  raise ValueError("complete native element count mismatch")
 return _canonical(d)

@dataclass(frozen=True)
class MmapXPertDenseCandidate:
 reference: XPertCandidate
 numeric_row: DenseNativeRow
 provenance_bytes: bytes
 def __post_init__(self):
  if (type(self.reference) is not XPertCandidate or type(self.numeric_row) is not DenseNativeRow
      or self.numeric_row.native_row_index!=self.reference.native_row_index
      or self.numeric_row.profile().binding_sha256!=self.reference.profile.binding_sha256):
   raise ValueError("mmap values/masks/native offsets must match existing source-bound candidate")
  p=_json(self.provenance_bytes)
  if p.get("task_admitted") is not False or p.get("model_use_allowed") is not False or p.get("full_source_export_available") is not False:
   raise ValueError("numeric fixture is not full export or model admission")
 @property
 def native_row_index(self):return self.reference.native_row_index
 @property
 def profile(self):return self.numeric_row.profile()
 @property
 def official_split(self):return self.reference.official_split
 def prediction_inputs(self,*,known_fields=None):return self.reference.prediction_inputs(known_fields=known_fields)
 def audit(self):return _json(self.provenance_bytes)
 def require_model_use(self,purpose,*,admission=None):return self.reference.require_model_use(purpose,admission=admission)

class XPertMmapSource:
 """Exact compatible numeric fixture; full-source export is not yet available."""
 def __init__(self,*,root,manifest=FIXTURE_REF["path"],expected_manifest_sha256=FIXTURE_REF["sha256"]):
  self.root=Path(root).resolve();self.manifest=Path(manifest)
  if expected_manifest_sha256!=FIXTURE_REF["sha256"]:raise ValueError("exact immutable source-owner mmap fixture required")
  self._snapshot()
 def _verified(self,ref):
  if (type(ref) is not dict or type(ref.get("path")) is not str or not ref["path"] or not _int(ref.get("bytes"))
      or not _is_sha(ref.get("sha256"))):raise ValueError("exact source/axis reference required")
  path=(self.root/ref["path"]).resolve()
  if not path.is_relative_to(self.root) or path.suffix.lower() not in (".json",".gz"):
   raise ValueError("small source proof/axis only; no raw H5AD/ZIP/pickle")
  raw=path.read_bytes()
  if len(raw)!=ref["bytes"] or _sha(raw)!=ref["sha256"]:raise ValueError("source-owner manifest/axis proof integrity mismatch")
  return raw
 def _snapshot(self):
  raw=self._verified(dict(FIXTURE_REF,path=str(self.manifest)))
  m=json.loads(raw)
  if (m.get("schema")!="mora3.xpert.numeric-array-store.v1" or m.get("fixture_only") is not True
      or m.get("full_source_export") is not False or type(m.get("task_admitted_records")) is not int
      or m["task_admitted_records"]!=0 or m.get("model_use_allowed") is not False
      or m.get("trained_usage") is not False or m.get("official_saved_split") is not None):
   raise ValueError("exact pending source-array fixture scope required; no future full-store/master guesses")
  if len(m["datasets"])!=2 or len(m["array_transports"])!=4:raise ValueError("exact actual-dense/synthetic-CSC fixture allocation required")
  for record in m["array_transports"].values():transport_descriptor(record)
  return m
 def _array(self,m,key):
  if key not in m["array_transports"]:raise ValueError("native array absent from pinned store")
  return NativeMmapArray(root=self.root,descriptor_bytes=transport_descriptor(m["array_transports"][key]))
 def read_dense_rows(self,rows):
  m=self._snapshot();dataset=next(d for d in m["datasets"] if d["matrix_encoding"]=="dense")
  mapping=dataset["axis0_native_source_indices"]
  if (type(mapping) is not list or len(mapping)!=dataset["logical_shape"][0] or any(not _int(i) for i in mapping)
      or len(set(mapping))!=len(mapping)):raise ValueError("exact compact-fixture/original-source row map required")
  if (type(rows) is not tuple or not 1<=len(rows)<=16 or any(not _int(i) or i not in mapping for i in rows)
      or len(set(rows))!=len(rows)):raise ValueError("requested native row unavailable; no full-source fallback")
  array=self._array(m,dataset["arrays"]["X"])
  _same(array.descriptor()["native_shape"],dataset["logical_shape"],"complete numeric fixture axes mismatch")
  _same(dataset["complete_gene_axis"]["native_rows"],dataset["logical_shape"][1],"complete original gene axis required")
  # Reuse all existing source/metadata/control/dose/placeholder/gene-axis gates.
  reference=XPertNativeSource(root=self.root).read_profiles(XPertSelector(dataset["source_article_ID"],rows))
  by_native={r.native_row_index:r for r in reference}
  out=[]
  for native in rows:
   physical=mapping.index(native);rawrow,=DenseMmapMatrix(array).read_rows((physical,))
   row=DenseNativeRow(native,rawrow.values,rawrow.axis_length);old=by_native[native]
   metadata=dataset["native_selected_row_metadata"][physical]
   if (type(metadata["native_source_row_index"]) is not int or metadata["native_source_row_index"]!=native
       or metadata["fixture_row_axis"]!=physical or metadata["task_admitted"] is not False
       or metadata["native_role"]!=old.native_role or metadata["source_native_observation"]["native_row_index"]!=native
       or dataset["source_original_H5AD_sha256"]!=old.audit()["source_H5AD_SHA256"]):
    raise ValueError("original native observation/source/role/row mapping mismatch")
   _same(metadata["source_native_observation"]["native_fields"],old.context.native_fields(),"exact existing observation/context changed")
   axis=dataset["complete_gene_axis"];expected=dict(path=axis["payload_path"],bytes=axis["payload_bytes"],sha256=axis["payload_SHA256"])
   _same(expected,json.loads(old.gene_axis.source_ref_bytes),"complete source gene-axis reference mismatch")
   p=dict(schema="mora3.XPertMmapDenseFixtureCandidate/v1",fixture_sha256=FIXTURE_REF["sha256"],native_row_index=native,
    physical_fixture_row=physical,dataset_id=dataset["dataset_id"],source_evidence_role="actual_bounded_L1000_native_fixture_not_full_matrix",
    source_full_QC=dataset["source_full_QC"],complete_gene_axis=axis,source_original_H5AD_sha256=dataset["source_original_H5AD_sha256"],
    reference_candidate_sha256=old.binding_sha256,numeric_descriptor_sha256=_sha(array.descriptor_bytes),
    whole_numeric_verifier_inherited=True,selected_numeric_chunks_verified=True,
    full_source_export_available=False,task_admitted=False,model_use_allowed=False,model_tokens=None,official_split=None,
    IO=rawrow.values.io(),metadata_role_logic="existing xpert_native.py; no duplicate source-role implementation")
   out.append(MmapXPertDenseCandidate(old,row,_canonical(p)))
  return tuple(out)
 def read_columns(self,dataset_id,columns):
  m=self._snapshot()
  found=[d for d in m["datasets"] if d["dataset_id"]==dataset_id and d["matrix_encoding"]=="csc_matrix"]
  if len(found)!=1:raise ValueError("native Norman CSC full backing unavailable; only explicit synthetic contract fixture")
  d=found[0]
  if d["source_evidence_role"]!="SYNTHETIC_TEST_ONLY; no biological measurements or source rows" or d["native_source_observations"]!=0:
   raise ValueError("synthetic CSC fixture cannot be relabeled as original Norman source")
  self._verified(d["synthetic_axes"])
  data,indices,ptr=(self._array(m,d["arrays"][k]) for k in ("data","indices","indptr"))
  flag=d["csc_indices_unique_and_ascending_within_columns"]
  if type(flag) is not bool:raise ValueError("exact declared CSC proof status required")
  matrix=CSCMmapMatrix(data,indices,ptr,logical_shape=tuple(d["logical_shape"]),coordinate_unique=flag,strictly_ascending=flag)
  return tuple(replace(c,provenance_bytes=_canonical(dict(dataset_id=dataset_id,fixture_sha256=FIXTURE_REF["sha256"],
   source_evidence_role="SYNTHETIC_TEST_ONLY",original_Norman_CSC_source_coverage=False,task_admitted=False,
   full_source_export_available=False,native_column_index=c.native_column_index))) for c in matrix.read_columns(columns))
 def read_CSC_rows(self,rows):
  raise ValueError("native CSC efficient row lookup unavailable; separately reviewed NEW derived index required")
 def verify(self,value):
  p=value.audit()
  if p.get("fixture_sha256")!=FIXTURE_REF["sha256"]:raise ValueError("exact source fixture binding required")
  if type(value) is MmapXPertDenseCandidate:
   if not _int(p.get("native_row_index")):raise ValueError("exact typed native source row required")
   checked,=self.read_dense_rows((p["native_row_index"],))
  elif type(value) is CSCNativeColumn:
   if not _int(p.get("native_column_index")):raise ValueError("exact native column required")
   checked,=self.read_columns(p["dataset_id"],(p["native_column_index"],))
  else:raise ValueError("typed source-bound numeric candidate required")
  if checked!=value:raise ValueError("fresh numeric/source/axis/role/native byte binding mismatch")
  return checked
 def source_references(self):
  m=self._snapshot();d=next(d for d in m["datasets"] if d["matrix_encoding"]=="dense");axis=d["complete_gene_axis"]
  return dict(fixture=dict(FIXTURE_REF,path=str(self.manifest)),source_QC=d["source_full_QC"],roles=d["source_role_masks"],
   axis=dict(path=axis["payload_path"],bytes=axis["payload_bytes"],sha256=axis["payload_SHA256"]))
 def gates(self):
  self._snapshot()
  return dict(full_export_available=False,Norman_full_CSC_backing_available=False,efficient_CSC_rows_available=False,
   actual_dense_source_rows=(0,1,590),actual_dense_gene_axis=978,CSC_real_source_coverage=False,
   task_admitted=False,official_split=None,model_use_allowed=False,positive_loss_supported=False,model_tokens=None,
   source_role_logic_reused=True,full_matrix_bytes_rehashed_per_query=0,full_dataset_ready=False)
