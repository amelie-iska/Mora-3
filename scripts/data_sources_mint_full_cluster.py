"""Bounded NEW SQLite source-key/member proof; no full job launched by import.
Only cluster+createtsv, no interaction permutation/filter/split or annotation inheritance.
"""
import functools,os,pathlib,sqlite3
from data_sources_mmseqs_createdb import shuffle_key

def original_ordinal(key,total):
    if not isinstance(key,int) or not 0<=key<total:raise ValueError('native key outside source axis')
    q,r=divmod(total,32);boundary=r*(q+1)
    if key<boundary:bucket,rank=divmod(key,q+1)
    else:bucket,rank=divmod(key-boundary,q);bucket+=r
    ordinal=rank*32+bucket
    if shuffle_key(ordinal,total,32)!=key:raise ValueError('source/native key inverse failed')
    return ordinal

def _settings(conn):
    conn.execute('PRAGMA cache_size=-65536');conn.execute('PRAGMA mmap_size=0');conn.execute('PRAGMA temp_store=FILE');conn.execute('PRAGMA journal_mode=DELETE');conn.execute('PRAGMA synchronous=FULL');conn.execute('PRAGMA threads=1')

def build_source_keys(lookup,database,total,check):
    database=pathlib.Path(database);fd=os.open(database,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.close(fd)
    conn=sqlite3.connect(database)
    try:
        _settings(conn);conn.execute('CREATE TABLE source(native_key INTEGER PRIMARY KEY,provider_id TEXT NOT NULL UNIQUE,original_source_ordinal INTEGER NOT NULL,representative_key INTEGER)');conn.execute('CREATE TABLE reps(native_key INTEGER PRIMARY KEY,self_member INTEGER NOT NULL)');conn.commit();count=0
        with pathlib.Path(lookup).open('rb') as f:
            for line in f:
                if len(line)>2**20 or not line.endswith(b'\n'):raise ValueError('bounded exact native lookup format required')
                v=line[:-1].split(b'\t')
                if len(v)!=3 or int(v[0])!=count or v[2]!=b'0' or not v[1]:raise ValueError('native contiguous key/single-source provider lookup mismatch')
                try:provider=v[1].decode('ascii')
                except UnicodeDecodeError as exc:raise ValueError('unreviewed provider namespace encoding') from exc
                try:conn.execute('INSERT INTO source VALUES(?,?,?,NULL)',(count,provider,original_ordinal(count,total)))
                except sqlite3.IntegrityError as exc:raise ValueError('ambiguous duplicate provider ID; no renamed or dropped source record') from exc
                count+=1
                if count%10000==0:conn.commit();check()
        conn.commit();check()
        if count!=total or conn.execute('SELECT COUNT(*),MIN(native_key),MAX(native_key) FROM source').fetchone()!=(total,0,total-1):raise ValueError('whole source key census mismatch')
        return dict(source_records=count,source_provider_ids_unique=True,all_native_keys_contiguous=True,all_original_source_ordinals_bijective=True,source_file_index=0,source_ordinal_formula='inverse of prior verified32-way shuffle; native DB key not original source ordinal',sqlite_cache_bytes=64*1024**2,sqlite_mmap_bytes=0,proof_index_is_new_not_reindex_of_existing_data=True)
    finally:conn.close()

def audit_membership(tsv,database,total,check):
    database=pathlib.Path(database)
    if not database.is_file():raise ValueError('complete source-key proof required')
    conn=sqlite3.connect('file:'+str(database)+'?mode=rw',uri=True)
    try:
        _settings(conn)
        if conn.execute('SELECT COUNT(*) FROM source').fetchone()[0]!=total or conn.execute('SELECT COUNT(*) FROM source WHERE representative_key IS NOT NULL').fetchone()[0] or conn.execute('SELECT COUNT(*) FROM reps').fetchone()[0]:raise ValueError('fresh completed source-key proof required; no retry/reset')
        @functools.lru_cache(maxsize=200000)
        def rep_key(provider):
            row=conn.execute('SELECT native_key FROM source WHERE provider_id=?',(provider,)).fetchone()
            if row is None:raise ValueError('unknown representative provider ID')
            return row[0]
        count=0
        with pathlib.Path(tsv).open('rb') as f:
            for line in f:
                if len(line)>2**20 or not line.endswith(b'\n'):raise ValueError('bounded native cluster TSV requires newline')
                values=line[:-1].split(b'\t')
                if len(values)!=2:raise ValueError('exact representative/member TSV axis required')
                try:rep,member=[v.decode('ascii') for v in values]
                except UnicodeDecodeError as exc:raise ValueError('unreviewed provider namespace encoding') from exc
                key=rep_key(rep);cur=conn.execute('UPDATE source SET representative_key=? WHERE provider_id=? AND representative_key IS NULL',(key,member))
                if cur.rowcount!=1:raise ValueError('unknown or duplicate member; every source member must occur once')
                conn.execute('INSERT INTO reps VALUES(?,?) ON CONFLICT(native_key) DO UPDATE SET self_member=MAX(self_member,excluded.self_member)',(key,int(rep==member)));count+=1
                if count%10000==0:conn.commit();check()
        conn.commit();check();n=conn.execute('SELECT COUNT(*) FROM source WHERE representative_key IS NOT NULL').fetchone()[0];clusters,self_members=conn.execute('SELECT COUNT(*),SUM(self_member) FROM reps').fetchone()
        if count!=total or n!=total:raise ValueError('missing whole source members')
        if clusters!=self_members:raise ValueError('representative missing own self membership')
        if conn.execute('PRAGMA integrity_check').fetchone()!=('ok',):raise ValueError('new source/member proof database integrity failed')
        return dict(members_once=count,clusters=clusters,all_representatives_self_members=True,all_members_exact_provider_source_key_links=True,source_ordinal_and_32way_DB_key_preserved=True,full_original_AA_header_lineage='prior all-record native DB proof plus exact unchanged allowlisted components, not annotation or cluster-label transfer',scientific_role='newly computed heuristic50percent cluster candidate; original MINT clu50 identity not asserted',task_admitted_records=0)
    finally:conn.close()

def cluster_commands(binary,out):
    out=pathlib.Path(out);db=str(out/'input/DB100');clu=str(out/'clu50');threads=['--threads','4']
    return [('cluster',[binary,'cluster',db,clu,str(out/'tmp'),'--min-seq-id','0.50','-c','0.8','--cov-mode','0','--split-memory-limit','4G','--remove-tmp-files','0','--compressed','1']+threads),('createtsv',[binary,'createtsv',db,db,clu,str(out/'clu50.tsv')]+threads)]
