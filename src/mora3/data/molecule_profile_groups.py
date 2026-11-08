"""Read-only complete native Rosetta observation groups, with used-file integrity.

The reviewed producer/independent proof establishes complete grouping. This
consumer hashes only its used SQLite artifact once, then pins fresh full9 file
identity around every read. It neither refits/reindexes source nor authorizes
training. Repeated observations stay in the exact source order with weight1/n.
"""
from pathlib import Path
import copy,hashlib,json,sqlite3,time

_CHECKS=('every_artifact_bytes_SHA_and_currentidentity_stable',
 'all_source_condition_bytes_exact_SQLite_BLOB_primary_keys','all_target_source_values_preserved',
 'grouped_JSONL_and_immutable_database_complete_both_directions',
 'native_order_and_duplicate_observation_multisets_exact','uniform_observation_1_over_n_weights_exact')
_FAMILIES={'Rosetta-CDRP-molecule-profile-CellPainting','Rosetta-CDRP-molecule-profile-L1000'}


def _full9(path):
    s=path.stat();return(s.st_dev,s.st_ino,s.st_mode,s.st_nlink,s.st_uid,s.st_gid,s.st_size,s.st_mtime_ns,s.st_ctime_ns)


class ImmutableRosettaSourceGroups:
    def __init__(self,manifest,expected_sha256,*,independent_manifest,independent_sha256,root):
        self.root=Path(root).resolve();self._connection=None;started=time.monotonic()
        def resolve(name):
            p=Path(name);p=(p if p.is_absolute()else self.root/p).resolve()
            if not p.is_relative_to(self.root):raise ValueError('source group artifact outside explicit source root')
            return p
        def small(name,sha):
            p=resolve(name);raw=p.read_bytes()
            if hashlib.sha256(raw).hexdigest()!=sha:raise ValueError('exact source group manifest SHA required')
            return p,raw,json.loads(raw)
        mp,mraw,m=small(manifest,expected_sha256);ip,iraw,proof=small(independent_manifest,independent_sha256)
        count=sum(r['rows']for r in m.get('complete_retained_inverse_rows',[])if r['split']=='train')
        if (m.get('schema')!='Mora3.CompleteNativeRosettaConditionGroups/v1'
                or proof.get('schema')!='Mora3.IndependentCompleteNativeRosettaConditionGroups/v1'
                or proof.get('producer_manifest',{}).get('sha256')!=expected_sha256
                or proof['producer_manifest'].get('bytes')!=len(mraw)
                or any(proof.get(k)is not True for k in _CHECKS)
                or proof.get('complete_native_source_observations')!=count
                or proof.get('complete_exact_full_native_condition_groups')!=m.get('train_condition_groups')
                or type(count)is not int or count<1 or m.get('encoding_failures')):
            raise ValueError('complete same-source independent group verification required')
        refs=[r for r in m['artifacts']if Path(r['path']).name=='source-groups.sqlite']
        if len(refs)!=1:raise ValueError('one exact used source group database required')
        ref=refs[0];db=resolve(ref['path']);before=_full9(db)
        if before[6]!=ref['bytes']:raise ValueError('source group database bytes mismatch')
        sha=hashlib.sha256();hashed=0
        with db.open('rb')as f:
            while True:
                chunk=f.read(16*1024*1024)
                if not chunk:break
                sha.update(chunk);hashed+=len(chunk)
                if time.monotonic()-started>120:raise ValueError('source group used-file hash finite120s budget exceeded')
        after=_full9(db)
        if before!=after:raise ValueError('source group database identity changed during one hash')
        if hashed!=ref['bytes']or sha.hexdigest()!=ref['sha256']:raise ValueError('used source group database SHA mismatch')
        self._path=db;self._identity=after;self.manifest_sha256=expected_sha256
        conn=sqlite3.connect(db.as_uri()+'?mode=ro&immutable=1',uri=True)
        try:
            schema=conn.execute("SELECT type FROM sqlite_master WHERE name='source_groups'").fetchone()
            columns=tuple((r[1],r[2])for r in conn.execute('PRAGMA table_info(source_groups)'))
            if schema!=('table',)or columns!=(('condition','BLOB'),('condition_sha','TEXT'),('observations','TEXT')):
                raise ValueError('exact inert native source-group table schema required')
            self._connection=conn;self._guard()
        except BaseException:
            conn.close();self._connection=None;raise
        self._binding=dict(schema='Mora3.ImmutableRosettaSourceGroups/v1',
            producer_manifest=dict(path=str(mp),bytes=len(mraw),sha256=expected_sha256),
            independent_verification=dict(path=str(ip),bytes=len(iraw),sha256=independent_sha256),
            used_database=dict(path=str(db),bytes=ref['bytes'],sha256=ref['sha256'],full9=self._identity),
            source_observations=count,condition_groups=m['train_condition_groups'],
            database_hash_bytes=hashed,weights='uniform1/n;all-source-native-order-duplicates-preserved',
            training_authorization=False)
        self._construction_receipt=dict(elapsed_seconds=time.monotonic()-started,database_hash_bytes=hashed,
            database_full9_before=before,database_full9_after=after,database_sha256=sha.hexdigest(),
            large_group_JSONL_bytes_read=0,original_source_rows_requalified=0)

    def _guard(self):
        if self._connection is None:raise ValueError('source group reader is closed')
        if _full9(self._path)!=self._identity:raise ValueError('source group database identity drift')

    def binding(self):
        self._guard();return copy.deepcopy(self._binding)

    def construction_receipt(self):
        self._guard();return copy.deepcopy(self._construction_receipt)

    def molecules_for_condition(self,condition_bytes):
        if type(condition_bytes)is not bytes or not condition_bytes:raise ValueError('exact full native condition bytes required')
        self._guard();rows=self._connection.execute('SELECT condition_sha,observations FROM source_groups WHERE condition=?',(condition_bytes,)).fetchall();self._guard()
        if len(rows)!=1:raise ValueError('condition not in complete source training groups')
        key,raw=rows[0];expected=hashlib.sha256(condition_bytes).hexdigest();observations=json.loads(raw)
        if key!=expected or type(observations)is not list or not observations:raise ValueError('complete native source group key/observations mismatch')
        for row in observations:
            if (row.get('condition_sha256')!=expected or row.get('split')!='train' or row.get('source_family')not in _FAMILIES
                    or type(row.get('canonical_isomeric_smiles'))is not str or not row['canonical_isomeric_smiles']):
                raise ValueError('complete same-condition native training observation required')
        return tuple(row['canonical_isomeric_smiles']for row in observations)

    def close(self):
        if self._connection is not None:self._connection.close();self._connection=None
