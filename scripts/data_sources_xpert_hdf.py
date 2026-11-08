"""Source-only HDF5 metadata gate for native numeric transport.

h5py is imported lazily by the approved existing inspection interpreter only.
Mora runtime readers consume numeric NPY/JSON and never import this module.
"""
from __future__ import annotations
import hashlib,os,stat
from pathlib import Path
import numpy as np

ALLOWED_BUILTIN_FILTERS={1,2,3,4,5,6,32000}

def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
    return h.hexdigest()

def source_identity(path):
    s=os.stat(path,follow_symlinks=False)
    if not stat.S_ISREG(s.st_mode):raise ValueError("regular immutable source file required")
    return {"dev":s.st_dev,"ino":s.st_ino,"size":s.st_size,"mtime_ns":s.st_mtime_ns,"ctime_ns":s.st_ctime_ns}

def _safe_hdf_tree(f,h5py):
    visited=set();nodes=0;filters=set()
    def visit(group):
        nonlocal nodes
        for name in group:
            link=group.get(name,getlink=True)
            if not isinstance(link,h5py.HardLink):raise ValueError("source HDF5 external/soft link forbidden")
            obj=group[name];address=h5py.h5o.get_info(obj.id).addr
            if address in visited:continue
            visited.add(address);nodes+=1
            if isinstance(obj,h5py.Group):visit(obj)
            elif isinstance(obj,h5py.Dataset):
                if obj.is_virtual:raise ValueError("external virtual HDF5 storage forbidden")
                dcpl=obj.id.get_create_plist()
                if dcpl.get_external_count():raise ValueError("external raw HDF5 storage forbidden")
                for i in range(dcpl.get_nfilters()):
                    filter_id=dcpl.get_filter(i)[0];filters.add(filter_id)
                    if filter_id not in ALLOWED_BUILTIN_FILTERS or not h5py.h5z.filter_avail(filter_id):
                        raise ValueError("unreviewed or unavailable HDF5 filter forbidden")
                    if not h5py.h5z.get_filter_info(filter_id)&h5py.h5z.FILTER_CONFIG_DECODE_ENABLED:
                        raise ValueError("HDF5 filter has no approved decoder")
            else:raise ValueError("unknown HDF5 native object")
    visited.add(h5py.h5o.get_info(f.id).addr);visit(f)
    return {"native_HDF5_nodes_inspected":nodes,"filter_ids":sorted(filters),"HardLinks_only":True,"no_external_or_virtual_storage":True}

def inspect_source(spec):
    import h5py
    path=Path(spec["source_path"]);identity=source_identity(path)
    if identity["size"]!=spec["source_bytes"] or sha_file(path)!=spec["source_sha256"]:
        raise ValueError("pinned source SHA/size mismatch")
    shape=list(spec["native_logical_shape"]);datasets=[]
    with h5py.File(path,"r") as f:
        gates=_safe_hdf_tree(f,h5py)
        for e in spec["datasets"]:
            name=e["native_dataset_path"];d=f[name]
            dtype=np.dtype(e["native_dtype"])
            if not isinstance(d,h5py.Dataset) or d.dtype.hasobject or d.dtype.kind not in "iufb" or d.dtype.str!=dtype.str:
                raise ValueError("native matrix numeric dtype mismatch; no object/casting/repair")
            if list(d.shape)!=e["native_shape"]:raise ValueError("native matrix dataset shape mismatch")
            datasets.append({"native_dataset_path":name,"native_shape":list(d.shape),"native_dtype":d.dtype.str})
        for axis,count in [("obs",shape[0]),("var",shape[1])]:
            if f[axis+"/_index"].shape!=(count,):raise ValueError("native full observation/gene axis length mismatch")
        if spec["native_encoding"]=="dense":
            if not isinstance(f["X"],h5py.Dataset) or list(f["X"].shape)!=shape:raise ValueError("dense native logical shape mismatch")
        elif spec["native_encoding"]=="csc_matrix":
            x=f["X"];encoding=x.attrs.get("encoding-type")
            if isinstance(encoding,bytes):encoding=encoding.decode()
            if not isinstance(x,h5py.Group) or encoding!="csc_matrix" or list(x.attrs["shape"])!=shape:
                raise ValueError("native CSC encoding/logical shape changed")
            if x["data"].shape!=x["indices"].shape or x["indptr"].shape!=(shape[1]+1,):
                raise ValueError("native CSC coordinate/pointer axes mismatch")
        else:raise ValueError("unreviewed source matrix encoding")
    if source_identity(path)!=identity:raise ValueError("source stat identity changed during readonly metadata gate")
    return {"source_path":str(path),"source_sha256":spec["source_sha256"],"source_identity":identity,
            "native_encoding":spec["native_encoding"],"native_logical_shape":shape,"datasets":datasets,**gates,
            "read_mode":"r","source_data_or_environment_changes":False}

def export_source(spec,output_dir,checkpoint=None):
    """Stream all declared native arrays; retain source CSC order, including duplicates."""
    import h5py
    from data_sources_xpert_arrays import write_numeric_stream
    gate=inspect_source(spec);out=Path(output_dir)
    if not out.is_dir() or out.is_symlink():raise ValueError("fresh regular payload directory required")
    arrays={};shape=spec["native_logical_shape"];csc=spec["native_encoding"]=="csc_matrix"
    with h5py.File(spec["source_path"],"r") as f:
        _safe_hdf_tree(f,h5py)
        if csc:
            for name in ["indices","indptr"]:
                if f["X/"+name].dtype.kind not in "iu":raise ValueError("native CSC index/pointer integer dtype required")
            nnz=f["X/data"].shape[0]
        for e in spec["datasets"]:
            key=e["array_id"]
            if not isinstance(key,str) or not key or any(x not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for x in key):
                raise ValueError("explicit safe numeric array key required")
            dataset=f[e["native_dataset_path"]];previous_pointer=None
            def read(start,stop):
                nonlocal previous_pointer
                a=dataset[start:stop]
                if csc and e["native_dataset_path"]=="/X/indices":
                    if np.any(a<0) or np.any(a>=shape[0]):raise ValueError("native CSC row index out of bounds; no repair")
                if csc and e["native_dataset_path"]=="/X/indptr":
                    if start==0 and (not len(a) or a[0]!=0):raise ValueError("native CSC pointer must start at zero")
                    if np.any(a<0) or np.any(a>nnz) or np.any(a[1:]<a[:-1]) or (previous_pointer is not None and len(a) and a[0]<previous_pointer):
                        raise ValueError("native CSC pointer monotonicity/bounds mismatch")
                    if len(a):previous_pointer=int(a[-1])
                    if stop==dataset.shape[0] and previous_pointer!=nnz:raise ValueError("native CSC final pointer differs from stored count")
                return a
            transport=write_numeric_stream(out/(key+".native.npy"),out/(key+".finite-validity-packed.npy"),e,read,checkpoint=checkpoint)
            arrays[key]={"expected_native":e,"transport":transport}
    if source_identity(spec["source_path"])!=gate["source_identity"] or sha_file(spec["source_path"])!=spec["source_sha256"]:
        raise ValueError("original HDF5 source SHA/stat changed during export")
    return {"source_gate":gate,"array_transports":arrays,"source_whole_SHA_and_identity_after_export_passed":True,
            "native_CSC_order_and_source_element_ordinals_unchanged":True,"no_coalesce_sort_transpose_densify_or_normalization":True,
            "task_admitted":0,"model_use_allowed":False}
