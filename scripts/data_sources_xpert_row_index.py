"""NEW derived cell-row access to ORIGINAL CSC element ordinals, never source edits."""
from __future__ import annotations
import hashlib, mmap, os, re, sys
from pathlib import Path
import numpy as np
import data_sources_xpert_arrays as native

NAMES=("row_indptr","row_source_element_ordinal")
NOHUGE_HEADER=Path("/usr/include/asm-generic/mman-common.h")
def nohugepage_option():
    # Python may omit the constant even though mmap.madvise accepts the Linux ABI.
    # Require installed primary ABI evidence AND an actual successful mapping call.
    if not sys.platform.startswith("linux") or not NOHUGE_HEADER.is_file():
        raise ValueError("Linux NOHUGEPAGE ABI header unavailable")
    m=re.search(r"^#define\s+MADV_NOHUGEPAGE\s+(\d+)\b",NOHUGE_HEADER.read_text(),re.M)
    if m is None or int(m.group(1))!=15:
        raise ValueError("Linux NOHUGEPAGE ABI header changed")
    return int(m.group(1))

def require_page_release(a):
    if (not hasattr(a,"_mmap") or not hasattr(a._mmap,"madvise")
        or any(not hasattr(mmap,k) for k in ("MADV_DONTNEED","MADV_RANDOM"))):
        raise ValueError("safe bounded mmap page release/THP controls unavailable")
    try:
        a._mmap.madvise(mmap.MADV_RANDOM)
        a._mmap.madvise(nohugepage_option())
    except (OSError,ValueError) as e:raise ValueError("safe mmap page policy refused") from e
    return {"page_bytes":mmap.PAGESIZE,"RANDOM":True,"NOHUGEPAGE":True,"DONTNEED":True}
def release(a):
    a._mmap.madvise(mmap.MADV_DONTNEED)
def validate_sizes(shape,nnz):
    if (len(shape)!=2 or any(not isinstance(x,int) or isinstance(x,bool) or x<0 for x in shape)
        or not isinstance(nnz,int) or nnz<0):raise ValueError("exact nonnegative CSC shape/count required")
    if nnz>=2**32:raise ValueError("original ordinal exceeds uint32 capacity")
def validate_csc_layout(indices,indptr,shape,check_rows=True):
    nr,nc=shape;validate_sizes(shape,len(indices))
    if indices.dtype.str!="<i4" or indptr.dtype.str!="<i4" or indices.ndim!=1 or indptr.shape!=(nc+1,):
        raise ValueError("native int32 CSC coordinate shape/dtype required")
    if indptr[0]!=0 or indptr[-1]!=len(indices) or np.any(indptr[1:]<indptr[:-1]):
        raise ValueError("original CSC pointer bounds/order mismatch")
    if check_rows:
        for i in range(0,len(indices),65536):
            z=indices[i:i+65536]
            if np.any(z<0) or np.any(z>=nr):raise ValueError("original CSC row bounds")
            del z
            if hasattr(indices,"_mmap"):release(indices)
def _fresh(paths):
    if len(set(map(str,paths)))!=len(paths) or any(p.exists() or p.is_symlink() for p in paths):
        raise FileExistsError("fresh exclusive derived payload paths required")
def _new_map(path,shape,dtype):
    with Path(path).open("xb") as f:
        off=native._write_header(f,shape,dtype);f.truncate(off+int(np.prod(shape))*np.dtype(dtype).itemsize)
        f.flush();os.fsync(f.fileno())
    a=np.load(path,mmap_mode="r+",allow_pickle=False,max_header_size=65536);require_page_release(a)
    return a

class Source:
    def __init__(self,spec):self.spec=spec;self.maps={};self.identities={}
    def __enter__(self):
        try:
            validate_sizes(self.spec["logical_shape"],self.spec["arrays"]["indices"]["expected_native"]["native_shape"][0])
            for key in ("data","indices","indptr"):
                item=self.spec["arrays"][key];e=item["expected_native"];t=item["transport"];v=item["verified_store"]
                dtype,shape,packed,_=native.validate_expected(e)
                for pk,ik,hk,sz,dt in [
                    ("path","numeric_file_identity","numeric_header_sha256",shape,dtype),
                    ("validity_path","validity_file_identity","validity_header_sha256",(packed,),np.dtype("u1"))]:
                    p=Path(t[pk]);identity=native.file_identity(p)
                    if identity!=t[ik] or identity!=v[ik]:raise ValueError("source verified stat identity mismatch")
                    with p.open("rb") as f:off=native._header(f,sz,dt)
                    if native.header_sha(p,off)!=t[hk] or t[hk]!=v[hk]:raise ValueError("source numeric header SHA mismatch")
                    self.identities[str(p)]=identity
                if t["native_value_sha256"]!=e["native_value_SHA256"] or v["native_value_sha256"]!=e["native_value_SHA256"]:
                    raise ValueError("source native digest proof mismatch")
                if key!="data":
                    a=np.load(t["path"],mmap_mode="r",allow_pickle=False,max_header_size=65536)
                    self.maps[key]=a;require_page_release(a)
            nr,nc=self.spec["logical_shape"];n=len(self.maps["indices"])
            if (self.maps["indices"].dtype.str!="<i4" or self.maps["indptr"].dtype.str!="<i4"
                or self.maps["indptr"].shape!=(nc+1,)
                or self.spec["arrays"]["data"]["expected_native"]["native_shape"]!=[n]
                or self.spec["arrays"]["data"]["expected_native"]["native_dtype"]!="<f4"):
                raise ValueError("complete native CSC data/coordinate schema mismatch")
            p=self.maps["indptr"]
            if p[0]!=0 or p[-1]!=n or np.any(p[1:]<p[:-1]):raise ValueError("original CSC pointer bounds/order mismatch")
            return self
        except BaseException:self.close();raise
    def check(self):
        for p,ident in self.identities.items():
            if native.file_identity(p)!=ident:raise ValueError("source stat identity changed")
    def iter_verified(self,key,checkpoint=None):
        item=self.spec["arrays"][key];e=item["expected_native"];t=item["transport"];a=self.maps[key]
        hv=hashlib.sha256();hm=hashlib.sha256()
        with Path(t["path"]).open("rb") as f:fh=hashlib.sha256(f.read(t["native_data_offset"]))
        with Path(t["validity_path"]).open("rb") as f:
            mh=hashlib.sha256(f.read(t["validity_data_offset"]))
            for c,expected in zip(e["chunks"],t["chunks"],strict=True):
                if checkpoint:checkpoint()
                self.check();z=a[c["axis0_start"]:c["axis0_stop"]];body=z.tobytes()
                bits=np.packbits(np.isfinite(z),bitorder="little").tobytes();offset=f.tell();recorded=f.read(len(bits))
                rec=native._chunk_record(c,t["native_data_offset"]+c["axis0_start"]*a.dtype.itemsize,body,offset,recorded)
                if rec!=expected or recorded!=bits:raise ValueError("source coordinate chunk/hash/validity mismatch")
                hv.update(body);hm.update(recorded);fh.update(body);mh.update(recorded)
                yield z
                del z,body,bits,recorded
                release(a)
            if f.read(1):raise ValueError("source trailing finite-mask bytes")
        if (hv.hexdigest()!=e["native_value_SHA256"] or hm.hexdigest()!=e["native_validity_SHA256"]
            or fh.hexdigest()!=t["sha256"] or mh.hexdigest()!=t["validity_sha256"]):
            raise ValueError("source coordinate native/file digest mismatch")
        self.check()
    def close(self):
        for a in self.maps.values():a._mmap.close()
        self.maps={}
    def __exit__(self,*args):self.close()

def seal_existing(path,mask_path,name,block_elements=65536,checkpoint=None):
    """Stream completed NEW derived values, write validity, pin all native/header/chunk/file bytes."""
    a=np.load(path,mmap_mode="r",allow_pickle=False,max_header_size=65536);require_page_release(a)
    blocks=[{"axis0_start":i,"axis0_stop":min(i+block_elements,len(a)),"native_elements":min(block_elements,len(a)-i)}
            for i in range(0,len(a),block_elements)]
    packed=sum((c["native_elements"]+7)//8 for c in blocks)
    hv=hashlib.sha256();hm=hashlib.sha256();chunks=[]
    with Path(path).open("rb") as f:off=native._header(f,a.shape,a.dtype);f.seek(0);fh=hashlib.sha256(f.read(off))
    try:
        with Path(mask_path).open("xb") as f:
            moff=native._write_header(f,(packed,),"u1")
            # Header SHA is computed from exact file bytes after close.
            for c in blocks:
                if checkpoint:checkpoint()
                z=a[c["axis0_start"]:c["axis0_stop"]];body=z.tobytes();bits=np.packbits(np.isfinite(z),bitorder="little").tobytes()
                chunks.append(native._chunk_record(c,off+c["axis0_start"]*a.dtype.itemsize,body,f.tell(),bits))
                f.write(bits);hv.update(body);hm.update(bits);fh.update(body);del z,body,bits;release(a)
            f.flush();os.fsync(f.fileno())
    finally:a._mmap.close()
    e={"array_id":name,"native_dataset_path":"/derived-original-ordinal-index/"+name,"native_dtype":np.dtype(a.dtype).str,
       "native_shape":list(a.shape),"chunks":blocks,"native_value_SHA256":hv.hexdigest(),"native_validity_SHA256":hm.hexdigest()}
    t={"path":str(path),"bytes":Path(path).stat().st_size,"sha256":fh.hexdigest(),"native_data_offset":off,
       "validity_path":str(mask_path),"validity_bytes":Path(mask_path).stat().st_size,"validity_sha256":native._sha(mask_path),
       "validity_data_offset":moff,"native_value_sha256":hv.hexdigest(),"native_validity_sha256":hm.hexdigest(),
       "all_native_elements":int(np.prod(a.shape)),"numeric_header_sha256":native.header_sha(path,off),
       "validity_header_sha256":native.header_sha(mask_path,moff),"numeric_file_identity":native.file_identity(path),
       "validity_file_identity":native.file_identity(mask_path),"chunks":chunks}
    return {"expected_native":e,"transport":t}

def build_index(spec,out,block_elements=65536,checkpoint=None):
    if not isinstance(block_elements,int) or not 0<block_elements<=65536:raise ValueError("bounded scatter block required")
    out=Path(out);paths=[out/(n+s) for n in NAMES for s in (".native.npy",".finite-validity-packed.npy")];_fresh(paths)
    nr,nc=spec["logical_shape"];n=spec["arrays"]["indices"]["expected_native"]["native_shape"][0];validate_sizes([nr,nc],n)
    counts=np.zeros(nr,dtype="<u8");releases=0
    with Source(spec) as src:
        for z in src.iter_verified("indices",checkpoint):
            if np.any(z<0) or np.any(z>=nr):raise ValueError("original CSC row bounds")
            counts+=np.bincount(z,minlength=nr).astype("<u8",copy=False)
        for z in src.iter_verified("indptr",checkpoint):pass
        ptr=_new_map(out/"row_indptr.native.npy",(nr+1,),"<u8")
        ordinals=None
        try:
            ptr[0]=0;np.cumsum(counts,out=ptr[1:]);ptr.flush();release(ptr)
            if int(ptr[-1])!=n:raise ValueError("source row count conservation failed")
            ordinals=_new_map(out/"row_source_element_ordinal.native.npy",(n,),"<u4")
            cursor=np.zeros(nr,dtype="<u8")
            for lo in range(0,n,block_elements):
                if checkpoint:checkpoint()
                src.check();hi=min(lo+block_elements,n);z=src.maps["indices"][lo:hi]
                order=np.argsort(z,kind="stable");rows=np.asarray(z[order])
                first=np.r_[0,np.flatnonzero(rows[1:]!=rows[:-1])+1]
                sizes=np.diff(np.r_[first,len(rows)])
                rank=(np.arange(len(rows))-np.repeat(first,sizes)).astype("<u8")
                pos=ptr[rows]+cursor[rows]+rank
                ordinals[pos]=np.arange(lo,hi,dtype="<u4")[order]
                cursor+=np.bincount(z,minlength=nr).astype("<u8",copy=False)
                del z,order,rows,first,sizes,rank,pos
                release(src.maps["indices"]);release(ordinals);releases+=1
            if not np.array_equal(cursor,counts):raise ValueError("scatter cursor/source count mismatch")
            ordinals.flush();ptr.flush();src.check()
        finally:
            if ordinals is not None:ordinals._mmap.close()
            ptr._mmap.close()
    arrays={name:seal_existing(out/(name+".native.npy"),out/(name+".finite-validity-packed.npy"),name,block_elements,checkpoint) for name in NAMES}
    return {"schema":"mora3.xpert.original-ordinal-index-writer.v1","logical_shape":[nr,nc],"original_stored_elements":n,
            "array_transports":arrays,"source_spec":spec,"bounded_scatter_page_release_calls":releases,"block_elements":block_elements,
            "source_task_admitted":0,"model_use":False}

def audit_ordinal_bijection(indices,indptr,shape,rowptr,ordinals,block_elements=65536,checkpoint=None):
    validate_csc_layout(indices,indptr,shape,check_rows=False);nr,nc=shape;n=len(indices)
    if rowptr.dtype.str!="<u8" or ordinals.dtype.str!="<u4" or rowptr.shape!=(nr+1,) or ordinals.shape!=(n,):
        raise ValueError("derived index exact coordinate shape/dtype")
    if rowptr[0]!=0 or rowptr[-1]!=n or np.any(rowptr[1:]<rowptr[:-1]):raise ValueError("derived row pointer bounds/order")
    seen=np.zeros((n+7)//8,dtype="u1");duplicates=0;previous=None;genesha=hashlib.sha256();releases=0
    for lo in range(0,n,block_elements):
        if checkpoint:checkpoint()
        hi=min(lo+block_elements,n);o=np.array(ordinals[lo:hi],copy=True)
        if np.any(o>=n):raise ValueError("original ordinal bounds")
        if len(np.unique(o))!=len(o):raise ValueError("duplicate original ordinal within vectorized chunk")
        byte=o>>3;bits=np.left_shift(np.uint8(1),(o&7).astype("u1"))
        if np.any(seen[byte]&bits):raise ValueError("duplicate original ordinal across chunks")
        np.bitwise_or.at(seen,byte,bits)
        positions=np.arange(lo,hi,dtype="<u8");rows=np.searchsorted(rowptr,positions,side="right")-1
        actual=np.asarray(indices[o])
        if np.any(actual!=rows):raise ValueError("original row mapping mismatch")
        genes=np.searchsorted(indptr,o,side="right")-1
        if (np.any(genes<0) or np.any(genes>=nc) or np.any(indptr[genes]>o) or np.any(o>=indptr[genes+1])):
            raise ValueError("original gene interval mapping mismatch")
        if np.any((rows[1:]==rows[:-1])&(o[1:]<=o[:-1])):raise ValueError("original within-row storage order changed")
        duplicates+=int(np.count_nonzero((rows[1:]==rows[:-1])&(genes[1:]==genes[:-1])))
        if previous is not None and previous[0]==int(rows[0]):
            if previous[2]>=int(o[0]):raise ValueError("original within-row storage order changed")
            duplicates+=int(previous[1]==int(genes[0]))
        previous=(int(rows[-1]),int(genes[-1]),int(o[-1]));genesha.update(genes.astype("<u4").tobytes())
        for a in (indices,ordinals):
            if hasattr(a,"_mmap"):release(a);releases+=1
    if n:
        if np.any(seen[:-1]!=255) or seen[-1]!=(255 if n%8==0 else (1<<(n%8))-1):
            raise ValueError("missing original ordinal conservation or bitset padding")
    return {"all_original_offsets_seen_once":True,"original_offsets":n,"native_rows":nr,"native_gene_axis":nc,
            "same_coordinate_occurrences_preserved":duplicates,"implicit_unstored_zero_coordinates":nr*nc-(n-duplicates),
            "derived_gene_coordinate_stream_SHA256":genesha.hexdigest(),"coverage_bitset_bytes":len(seen),
            "bounded_semantic_page_release_calls":releases}

def verify_index(spec,out,writer,block_elements=65536,checkpoint=None):
    out=Path(out)
    if writer["source_spec"]!=spec or set(writer["array_transports"])!=set(NAMES):raise ValueError("exact source/index writer binding mismatch")
    expected_shapes={"row_indptr":([spec["logical_shape"][0]+1],"<u8"),"row_source_element_ordinal":([spec["arrays"]["indices"]["expected_native"]["native_shape"][0]],"<u4")}
    verified={}
    for name,item in writer["array_transports"].items():
        e=item["expected_native"];t=item["transport"];shape,dtype=expected_shapes[name]
        if (e["native_shape"]!=shape or e["native_dtype"]!=dtype or t["path"]!=str(out/(name+".native.npy"))
            or t["validity_path"]!=str(out/(name+".finite-validity-packed.npy"))):raise ValueError("exact derived shape/dtype/path required")
        verified[name]={**item,"verified_store":native.verify_numeric_pair(t["path"],t["validity_path"],e,checkpoint,t)}
    with Source(spec) as src:
        for key in ("indices","indptr"):
            for z in src.iter_verified(key,checkpoint):pass
        ptr=np.load(out/"row_indptr.native.npy",mmap_mode="r",allow_pickle=False);ordinal=np.load(out/"row_source_element_ordinal.native.npy",mmap_mode="r",allow_pickle=False)
        try:
            require_page_release(ptr);require_page_release(ordinal)
            audit=audit_ordinal_bijection(src.maps["indices"],src.maps["indptr"],spec["logical_shape"],ptr,ordinal,block_elements,checkpoint)
            src.check()
            for item in verified.values():
                t=item["transport"];v=item["verified_store"]
                for pk,ik in (("path","numeric_file_identity"),("validity_path","validity_file_identity")):
                    if native.file_identity(t[pk])!=v[ik]:
                        raise ValueError("derived verified stat identity changed during semantic audit")
        finally:ptr._mmap.close();ordinal._mmap.close()
    return {"schema":"mora3.xpert.original-ordinal-row-index.v1","array_transports":verified,"semantic_audit":audit,"source_spec":spec,
            "source_task_admitted":0,"model_use":False,"efficient_rows_are_original_occurrence_offsets":True,
            "native_values_copied_normalized_coalesced_or_reordered":False}
