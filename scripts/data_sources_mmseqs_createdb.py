"""Bounded native MMseqs createdb proof and exact ownership guards. No network."""
import contextlib
import gzip
import os
import pathlib
import signal
import struct
import time


def shuffle_key(source_ordinal,total,splits=32):
    if splits not in (1,32) or not 0<=source_ordinal<total:raise ValueError('invalid pinned shuffle axis')
    q,r=divmod(total,splits);bucket=source_ordinal%splits
    return bucket*q+min(bucket,r)+source_ordinal//splits


def tree_bytes(root):
    total=0
    for p in pathlib.Path(root).rglob('*'):
        if p.is_symlink():raise ValueError('new stage must not contain external/source symlinks')
        if p.is_file():total+=p.stat().st_size
    return total


def guard_decision(used,free,elapsed,plan):
    if used>=plan['stage_cap_bytes']-plan['stop_margin_bytes']:return 'stage_byte_highwater'
    if free<plan['minimum_free_bytes']+plan['stop_margin_bytes']:return 'free_reserve_highwater'
    if elapsed>=plan['max_seconds']:return 'finite_walltime'
    return None


def verify_approval(approval,plan_sha256,code_sha256):
    if approval.get('approved') is not True or approval.get('action')!='MMseqs_StageB_full_createdb_v1' or approval.get('plan_sha256')!=plan_sha256 or approval.get('code_sha256')!=code_sha256:raise ValueError('explicit exact plan/code execution review required')


def process_identity(pid):
    try:
        p=pathlib.Path('/proc')/str(pid);fields=p.joinpath('stat').read_text().rsplit(')',1)[1].split()
        if fields[0]=='Z':return None
        return dict(pid=pid,start_ticks=int(fields[19]),pgid=int(fields[2]),argv=[v.decode() for v in p.joinpath('cmdline').read_bytes().split(b'\0') if v])
    except (FileNotFoundError,ProcessLookupError):return None


def wait_process_identity(pid,expected_argv,timeout=5,poll=.005,_read=None,_clock=None,_sleep=None):
    """Bind only two stable, fully ready snapshots of this exact new child."""
    if not expected_argv or not 0<timeout<=10 or not .001<=poll<=.1:raise ValueError('bounded identity readiness and nonempty expected command required')
    read=_read or process_identity;clock=_clock or time.monotonic;sleep=_sleep or time.sleep
    deadline=clock()+timeout;start=None;previous=None;stable=0;expected_argv=list(expected_argv)
    while clock()<deadline:
        actual=read(pid)
        if actual is None:raise ProcessLookupError('new child exited before stable identity readiness')
        if actual.get('pid')!=pid:raise ValueError('PID identity changed during readiness')
        if start is None:start=actual['start_ticks']
        elif actual['start_ticks']!=start:raise ValueError('PID/start identity changed during readiness')
        if actual.get('argv')==expected_argv and actual.get('pgid')==pid:
            fingerprint=(actual['pid'],actual['start_ticks'],actual['pgid'],tuple(actual['argv']))
            stable=stable+1 if fingerprint==previous else 1;previous=fingerprint
            if stable>=2:return dict(actual,argv=list(actual['argv']))
        else:previous=None;stable=0
        sleep(poll)
    raise ValueError('stable expected PID/start/command/exclusivePGID not ready within bounded timeout')


def stop_owned_group(pid,expected,before_kill):
    if not expected.get('argv'):raise ValueError('nonempty verified expected command required; no signals sent')
    actual=process_identity(pid)
    if actual is None:return dict(owner_already_exited=True,owned_group_stopped=False)
    if actual!=expected or expected['pid']!=pid or expected['pgid']!=pid:raise ValueError('PID/start/command/exclusive group identity mismatch; no signals sent')
    os.killpg(pid,signal.SIGSTOP)
    try:
        deadline=time.monotonic()+5
        while time.monotonic()<deadline:
            try:state=pathlib.Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()[0]
            except FileNotFoundError:break
            if state in ('T','t','Z'):break
            time.sleep(.005)
        else:raise ValueError('new owned group did not reach stable stopped state')
        checkpoint=before_kill()
    finally:
        again=process_identity(pid)
        if again is not None:
            if again!=expected:raise ValueError('owned identity changed while frozen; refuse signal')
            os.killpg(pid,signal.SIGKILL)
    return dict(owner_already_exited=False,owned_group_stopped=True,checkpoint=checkpoint,signals=['SIGSTOP','SIGKILL'])


def _index(line):
    try:key,offset,length=map(int,line.rstrip(b'\n').split(b'\t'))
    except (ValueError,TypeError):raise ValueError('native index requires three integer fields')
    if min(key,offset)<0 or length<2:raise ValueError('invalid native index span')
    return key,offset,length


def _fasta(source):
    header=None;parts=[]
    with gzip.open(source,'rb') as f:
        for line in f:
            line=line.rstrip(b'\r\n')
            if line.startswith(b'>'):
                if header is not None:yield header,b''.join(parts)
                header=line[1:];parts=[]
            else:
                if header is None:raise ValueError('FASTA sequence before header')
                parts.append(line)
        if header is not None:yield header,b''.join(parts)


def _read_native_record(handle,logical_length,compressed):
    if not 2<=logical_length<=1048576:raise ValueError('native record exceeds bounded source format')
    if not compressed:
        data=handle.read(logical_length)
        if len(data)!=logical_length:raise ValueError('truncated native record')
        return data
    import pyarrow as pa
    header=handle.read(4)
    if len(header)!=4:raise ValueError('truncated compressed record header')
    length=struct.unpack('<I',header)[0]
    if length>2*logical_length+128:raise ValueError('compressed frame size exceeds native logical bound')
    payload=handle.read(length);marker=handle.read(1)
    if len(payload)!=length:raise ValueError('truncated compressed frame')
    if marker==b'\xff':decoded=payload
    elif marker==b'\0':
        try:decoded=pa.decompress(payload,decompressed_size=logical_length-1,codec='zstd').to_pybytes()
        except (ValueError,pa.ArrowInvalid) as e:raise ValueError('invalid nativezstd frame') from e
    else:raise ValueError('unknown native compression marker')
    if len(decoded)!=logical_length-1:raise ValueError('native decoded logical length mismatch')
    return decoded+b'\0'


def verify_native_database(source,prefix,total,splits=32,on_progress=None):
    prefix=pathlib.Path(prefix);base=str(prefix);data=prefix;heads=pathlib.Path(base+'_h')
    dtype=struct.unpack('<I',pathlib.Path(base+'.dbtype').read_bytes())[0];htype=struct.unpack('<I',pathlib.Path(base+'_h.dbtype').read_bytes())[0];compressed=bool(dtype&0x80000000)
    if dtype&0x7fffffff!=0 or htype&0x7fffffff!=12 or bool(htype&0x80000000)!=compressed:raise ValueError('expected matching aminoacid/header native storage flags')
    q,r=divmod(total,splits);bands=[dict(bucket=b,start=b*q+min(b,r),rows=q+(b<r)) for b in range(splits) if q+(b<r)>0];starts={v['start']:v for v in bands};index_paths=[pathlib.Path(base+'.index'),pathlib.Path(base+'_h.index'),pathlib.Path(base+'.lookup')];seq_end=head_end=count=0;previous_seq=previous_header=-1
    if on_progress:on_progress(0,total,'native_index_scan')
    with contextlib.ExitStack() as stack:
        files=[stack.enter_context(p.open('rb')) for p in index_paths]
        while True:
            positions=[f.tell() for f in files];lines=[f.readline() for f in files]
            if not any(lines):break
            if not all(lines):raise ValueError('native seq/header/lookup index row counts differ')
            sk,so,sl=_index(lines[0]);hk,ho,hl=_index(lines[1]);lookup=lines[2].rstrip(b'\n').split(b'\t')
            if sk!=count or hk!=count or len(lookup)!=3 or int(lookup[0])!=count or lookup[2]!=b'0':raise ValueError('native contiguous keys/single-source lookup mismatch')
            if compressed:
                if so<=previous_seq or ho<=previous_header or (count==0 and (so!=0 or ho!=0)):raise ValueError('compressed native physical offsets not increasing fromzero')
                previous_seq,previous_header=so,ho
            elif so!=seq_end or ho!=head_end:raise ValueError('native data index has gap, overlap or unexpected physical order')
            if count in starts:starts[count].update(index_offsets=positions,seq_offset=so,header_offset=ho)
            seq_end+=sl;head_end+=hl;count+=1
            if on_progress and count%1000000==0:on_progress(count,total,'native_index_scan')
    if count!=total or (not compressed and (seq_end!=data.stat().st_size or head_end!=heads.stat().st_size)):raise ValueError('native whole-index count/data extent mismatch')
    if on_progress:on_progress(count,total,'native_index_scan')
    for i,band in enumerate(bands):
        band['seq_end']=bands[i+1]['seq_offset'] if i+1<len(bands) else data.stat().st_size;band['header_end']=bands[i+1]['header_offset'] if i+1<len(bands) else heads.stat().st_size
    if on_progress:on_progress(0,total,'pointwise_source_replay')
    with contextlib.ExitStack() as stack:
        readers={}
        for band in bands:
            fs=[stack.enter_context(p.open('rb',buffering=65536)) for p in index_paths]
            for f,offset in zip(fs,band['index_offsets']):f.seek(offset)
            seq=stack.enter_context(data.open('rb',buffering=262144));seq.seek(band['seq_offset']);head=stack.enter_context(heads.open('rb',buffering=65536));head.seek(band['header_offset']);readers[band['bucket']]=(fs,seq,head)
        compared=0
        for ordinal,(native_header,native_sequence) in enumerate(_fasta(source)):
            if ordinal>=total:raise ValueError('more native FASTA records than declared source')
            bucket=ordinal%splits;key=shuffle_key(ordinal,total,splits);fs,seq,head=readers[bucket];sk,so,sl=_index(fs[0].readline());hk,ho,hl=_index(fs[1].readline());lookup=fs[2].readline().rstrip(b'\n').split(b'\t')
            if sk!=key or hk!=key or len(lookup)!=3 or int(lookup[0])!=key or lookup[1]!=native_header.split()[0] or lookup[2]!=b'0':raise ValueError('native source-ordinal/shuffled-key/provider mapping mismatch')
            if seq.tell()!=so or head.tell()!=ho:raise ValueError('native bucket data continuity mismatch')
            if _read_native_record(seq,sl,compressed)!=native_sequence+b'\n\0' or _read_native_record(head,hl,compressed)!=native_header+b'\n\0':raise ValueError('full native sequence/header bytes changed in database')
            compared+=1
            if on_progress and compared%1000000==0:on_progress(compared,total,'pointwise_source_replay')
        if compared!=total:raise ValueError('fewer native FASTA records than database')
        for band in bands:
            _,seq,head=readers[band['bucket']]
            if seq.tell()!=band['seq_end'] or head.tell()!=band['header_end']:raise ValueError('unread or overlapping native bucket data')
    if on_progress:on_progress(compared,total,'pointwise_source_replay')
    return dict(native_compressed_storage_losslessly_decoded=compressed,native_records_compared=compared,all_native_headers_sequences_and_provider_lookup_exact=True,whole_source_gzip_CRC_read_to_EOF=True,all_DB_data_index_header_lookup_spans_covered=True,source_ordinal_to_DB_key_formula='bucket=ordinal%32;key=bucket*(N//32)+min(bucket,N%32)+(ordinal//32)' if splits==32 else 'identity',database_keys_are_shuffled_not_source_ordinals=splits==32 and total>32,bucket_count=len(bands),source_indices_not_replaced_by_DB_keys=True)


def createdb_command(plan):
    if plan.get("threads")!=2:raise ValueError("only reviewed2CPU stage allowed")
    return [plan["binary"],"createdb",plan["source_path"],str(pathlib.Path(plan["new_output"])/"DB100"),"--shuffle","1","--createdb-mode","0","--compressed","1","--threads","2"]
