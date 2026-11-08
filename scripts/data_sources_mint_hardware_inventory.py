"""Stream complete retained-file hashes after source SID quiescence; no file count cap."""
import hashlib,json,os,pathlib,time
from data_sources_hardware_storage import hardware_snapshot

def stream_inventory(root,allowed_sources,target,deadline,minimum_free_bytes=64*1024**3,on_poll=None):
    root=pathlib.Path(root).resolve();target=pathlib.Path(target).resolve();allowed={pathlib.Path(p).resolve(strict=True) for p in allowed_sources};total=regular=links=payload_bytes=link_bytes=0;inventory_sha=hashlib.sha256()
    if not target.is_relative_to(root):raise ValueError('inventory must be a new owned-stage file')
    def check(incoming=0):
        if deadline is not None and time.monotonic()>deadline:raise ValueError('finite whole-wrapper deadline; partial inventory retained, no stable receipt')
        if hardware_snapshot(root,minimum_free_bytes)['available_bytes']-incoming<=minimum_free_bytes:raise ValueError('hardware free-space buffer before inventory write')
        if on_poll:on_poll()
    def sha_file(p):
        h=hashlib.sha256()
        with p.open('rb') as f:
            os.fsync(f.fileno())
            for b in iter(lambda:f.read(8*1024**2),b''):check();h.update(b)
        return h.hexdigest()
    check()
    with target.open('xb') as out:
        def emit(p):
            nonlocal total,regular,links,payload_bytes,link_bytes
            check()
            if p==target or p in [root/'outer-live.json',root/'worker-live.json']:return
            if p.is_symlink():
                dest=p.resolve(strict=False)
                if dest not in allowed and not dest.is_relative_to(root):raise ValueError('unapproved external source alias; no source traversal/copy')
                size=p.lstat().st_size;row=dict(path=str(p),symlink_raw_target=str(p.readlink()),symlink_target=str(dest),symlink_target_exists=dest.exists(),link_bytes=size,external_immutable_source=dest in allowed);links+=1;link_bytes+=size
            elif p.is_file():
                size=p.stat().st_size;row=dict(path=str(p),bytes=size,sha256=sha_file(p));regular+=1;payload_bytes+=size
            else:return
            data=(json.dumps(row,sort_keys=True,separators=(',',':'))+'\n').encode();check(len(data));out.write(data);inventory_sha.update(data);total+=1
        for base,dirs,files in os.walk(root,followlinks=False):
            check();dirs.sort()
            for name in list(dirs):
                p=pathlib.Path(base)/name
                if p.is_symlink():emit(p);dirs.remove(name)
            for name in sorted(files):emit(pathlib.Path(base)/name)
        out.flush();os.fsync(out.fileno())
    check()
    return dict(path=str(target),bytes=target.stat().st_size,sha256=inventory_sha.hexdigest(),format='streamed_JSONL_one_retained_regular_file_or_symlink_per_line',records=total,regular_files=regular,symlinks=links,regular_payload_bytes=payload_bytes,symlink_lstat_bytes=link_bytes,inventory_self_reference_via_this_SHA=True,excluded_mutable_top_level_files=['outer-live.json','worker-live.json'],fixed_file_count_or_stage_byte_cap=None)
