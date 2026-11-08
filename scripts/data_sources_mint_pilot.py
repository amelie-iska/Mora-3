"""Streaming source-only MINT resource-pilot selection and native coverage gates."""
import gzip,hashlib,json,pathlib
from data_sources_mmseqs_createdb import _fasta,shuffle_key

def sample_ordinals(total,count):
    if not isinstance(total,int) or not isinstance(count,int) or not 0<count<=total:raise ValueError('positive bounded original source cohort required')
    for i in range(count):yield i*total//count

def _sha(data):return hashlib.sha256(data).hexdigest()

def _refs(path):
    with pathlib.Path(path).open('rb') as f:
        for line in f:
            if len(line)>2**20:raise ValueError('source reference exceeds bounded format')
            yield json.loads(line)

def write_sample(source,out,total,count,prewrite):
    source=pathlib.Path(source);out=pathlib.Path(out)
    gzpath=out/'selected-source.fasta.gz';refpath=out/'selected-source-occurrence-refs.jsonl'
    if gzpath.exists() or refpath.exists():raise FileExistsError('immutable sample paths already exist')
    desired=iter(sample_ordinals(total,count));target=next(desired,None);selected=scanned=letters=0;ids=set()
    with gzpath.open('xb') as raw,refpath.open('xb') as refs:
        with gzip.GzipFile(filename='',mode='wb',fileobj=raw,compresslevel=3,mtime=0) as gz:
            for ordinal,(header,seq) in enumerate(_fasta(source)):
                scanned+=1
                if ordinal!=target:continue
                provider=header.split()[0].decode('ascii')
                if not provider or provider in ids:raise ValueError('selected provider ID missing/ambiguous; no renaming or duplicate drop')
                ids.add(provider);frame=b'>'+header+b'\n'+seq+b'\n'
                rec=dict(selected_ordinal=selected,original_source_ordinal=ordinal,provider_id=provider,
                    native_header_utf8=header.decode('utf-8'),native_header_sha256=_sha(header),
                    native_sequence_length=len(seq),native_sequence_sha256=_sha(seq),
                    full_provider_DB_key=shuffle_key(ordinal,total,32),
                    record_role='unqualified_provider_source_candidate',task_admitted=False)
                line=(json.dumps(rec,separators=(',',':'))+'\n').encode();prewrite(len(frame)+len(line)+128)
                gz.write(frame);refs.write(line);selected+=1;letters+=len(seq);target=next(desired,None)
    if scanned!=total or selected!=count or target is not None:raise ValueError('whole source/selected count mismatch; all partials retained')
    return dict(source_records_scanned=scanned,selected_source_records=selected,native_sequence_letters=letters,
        selected_provider_IDs_distinct=len(ids),native_FA_mode='full original headers and concatenated full AA bytes; one-line derived FASTA, no truncation',
        sample_path=str(gzpath),source_ref_path=str(refpath),selection='floor(i*N/M), deterministic original source order',RNG_seed=None)

def audit_sample(source,out,total,count):
    out=pathlib.Path(out);refs=iter(_refs(out/'selected-source-occurrence-refs.jsonl'));sample=iter(_fasta(out/'selected-source.fasta.gz'))
    expected=iter(sample_ordinals(total,count));target=next(expected,None);selected=scanned=letters=0;ids=set()
    for ordinal,(header,seq) in enumerate(_fasta(source)):
        scanned+=1
        if ordinal!=target:continue
        row=next(refs,None);payload=next(sample,None)
        if row is None or payload!=(header,seq):raise ValueError('full selected native header/AA mismatch')
        checks=[row['selected_ordinal']==selected,row['original_source_ordinal']==ordinal,
            row['provider_id']==header.split()[0].decode('ascii'),row['native_header_utf8'].encode()==header,
            row['native_header_sha256']==_sha(header),row['native_sequence_sha256']==_sha(seq),
            row['native_sequence_length']==len(seq),row['full_provider_DB_key']==shuffle_key(ordinal,total,32),
            row['task_admitted'] is False,row['record_role']=='unqualified_provider_source_candidate']
        if not all(checks) or row['provider_id'] in ids:raise ValueError('source reference/index/identity conservation failed')
        ids.add(row['provider_id']);selected+=1;letters+=len(seq);target=next(expected,None)
    if scanned!=total or selected!=count or target is not None or next(refs,None) is not None or next(sample,None) is not None:raise ValueError('whole original and selected source census mismatch')
    return dict(full_native_header_sequence_and_source_ordinal_pass=True,source_records_scanned=scanned,
        selected_source_records=selected,native_sequence_letters=letters,all_selected_provider_IDs_unique=True,
        original_and_selected_whole_gzip_CRC_read_to_EOF=True)

def audit_cluster_tsv(path,source_ids):
    source_ids=set(source_ids);seen=set();representatives=set();self_members=set();rows=0
    with pathlib.Path(path).open('rt') as f:
        for line in f:
            fields=line.rstrip('\r\n').split('\t')
            if len(fields)!=2:raise ValueError('native cluster TSV requires representative and member')
            rep,member=fields
            if rep not in source_ids or member not in source_ids or member in seen:raise ValueError('unknown or repeated source cluster member')
            seen.add(member);representatives.add(rep);rows+=1
            if rep==member:self_members.add(rep)
    if seen!=source_ids or self_members!=representatives:raise ValueError('whole source member/representative conservation failed')
    return dict(source_members_once=rows,clusters=len(representatives),all_source_members_once=True,all_representatives_self_members=True,
        scientific_role='newly computed heuristic cluster candidate, not original MINT clu50 or all-pair homology truth')

def pilot_commands(binary,out):
    out=pathlib.Path(out);db=str(out/'DB100');clu=str(out/'clu50');threads=['--threads','2']
    return [('createdb',[binary,'createdb',str(out/'selected-source.fasta.gz'),db,'--shuffle','1','--createdb-mode','0','--compressed','1']+threads),
        ('cluster',[binary,'cluster',db,clu,str(out/'tmp'),'--min-seq-id','0.50','-c','0.8','--cov-mode','0','--split-memory-limit','4G','--remove-tmp-files','0','--compressed','1']+threads),
        ('createtsv',[binary,'createtsv',db,db,clu,str(out/'clu50.tsv')]+threads)]
