"""Frozen-original RNAcentral partial/control cloning and no-reset resume guards."""
import hashlib,json,os,pathlib,subprocess

def sha(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def validate_checkpoint(checkpoint,asset):
    if pathlib.Path('/proc/'+str(checkpoint['original_worker_pid'])).exists():raise ValueError('old RNAcentral worker still exists; no new writer')
    if asset['expected_bytes']!=checkpoint['expected_full_active_bytes'] or asset['expected_etag']!=checkpoint['source_strong_ETag'] or str(asset['expected_etag']).startswith('W/'):raise ValueError('source entity differs from frozen partial')
    refs=checkpoint['immutable_original_files']
    if len(refs)!=2 or sum(str(r['path']).endswith('.aria2') for r in refs)!=1:raise ValueError('exact original data/control pair required')
    for r in refs:
        p=pathlib.Path(r['path'])
        if not p.is_file() or p.stat().st_size!=r['bytes'] or sha(p)!=r['sha256']:raise ValueError('frozen original data/control integrity mismatch')
    return refs

def clone_partial(checkpoint,asset,out,filename):
    refs=validate_checkpoint(checkpoint,asset);out=pathlib.Path(out).resolve()
    if pathlib.Path(filename).name!=filename or filename in ['.','..']:raise ValueError('new destination basename required')
    out.mkdir(parents=True,exist_ok=False);dest=out/filename;copied=[]
    for r in refs:
        target=pathlib.Path(str(dest)+'.aria2') if str(r['path']).endswith('.aria2') else dest
        subprocess.run(['/usr/bin/cp','--reflink=auto','--sparse=always','--no-clobber','--',r['path'],str(target)],check=True)
        if target.stat().st_size!=r['bytes'] or sha(target)!=r['sha256']:raise ValueError('new clone differs from frozen original')
        with target.open('rb') as f:os.fsync(f.fileno())
        copied.append(dict(original=r['path'],destination=str(target),bytes=r['bytes'],sha256=r['sha256']))
    validate_checkpoint(checkpoint,asset)
    return dict(destination=str(dest),copied_files=copied,original_files_unmodified=True,new_output_directory=str(out))

def checked_new_bytes(previous_downloaded,progress,maximum_new_bytes):
    if progress is None or progress.get('downloaded_bytes') is None:return None
    current=progress['downloaded_bytes']
    if not isinstance(current,int) or isinstance(current,bool) or current<previous_downloaded:raise ValueError('source progress reset; refuse fresh-download fallback')
    delta=current-previous_downloaded
    if delta>maximum_new_bytes:raise ValueError('new source-byte reserve exceeded; preserve partial and stop new child')
    return delta
