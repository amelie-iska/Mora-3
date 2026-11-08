"""Compact NEW-only RNA identity/provenance index; old source indices stay read-only.

Native identity is literal bytes. Uppercase and derived T/U+reverse-complement
hashes are grouping guards only, never molecular identity or inherited labels.
"""
from pathlib import Path
import hashlib,os,sqlite3
from data_intake_corpus import canonical
from data_intake_nucleotide_prior import sequence_groups
from data_intake_rnacentral_prior import ALPHABET

_DOMAINS={'new':'RNA_reference_native_symbols',
          'RNAcentral_pilot_existing':'RNA_reference_native_symbols',
          'Rfam_existing':'RNA_family_reference_native_symbols'}
_SELECT='native_sha,first_file,first_row,reuse_kind,uppercase_sha,rc_sha,occurrences'


def task_id_from_native_digest(digest,reuse_kind):
    if not isinstance(digest,bytes) or len(digest)!=32 or reuse_kind not in _DOMAINS:
        raise ValueError('32-byte literal native identity and explicit reuse domain required')
    return hashlib.sha256(canonical([_DOMAINS[reuse_kind],None,digest.hex()])).hexdigest()


def old_identity_resolver(rfam_db,pilot_db):
    """Caller opens both old databases mode=ro; only exact recipes are queried."""
    def resolve(sequence):
        digest=hashlib.sha256(sequence.encode()).digest();pilot=pilot_db.execute(
            'SELECT task_id,reused_prior FROM sequences WHERE native_sha=?',(digest.hex(),)).fetchone()
        rfam_id=task_id_from_native_digest(digest,'Rfam_existing')
        rfam=rfam_db.execute('SELECT source_family FROM tasks WHERE task_id=?',(rfam_id,)).fetchone()
        if rfam is not None:
            if not rfam[0].startswith('Rfam'):
                raise ValueError('old identity matches an unreviewed source domain')
            if pilot is not None and (pilot[0]!=rfam_id or not pilot[1]):
                raise ValueError('old pilot/Rfam exact identity conservation mismatch')
            return 'Rfam_existing',rfam_id
        if pilot is not None:
            expected=task_id_from_native_digest(digest,'RNAcentral_pilot_existing')
            if pilot[1] or pilot[0]!=expected:
                raise ValueError('old pilot identity recipe mismatch or missing Rfam task')
            return 'RNAcentral_pilot_existing',expected
        return None
    return resolve


class NativeIdentityIndex:
    def __init__(self,path,db):self.path=Path(path);self.db=db
    @classmethod
    def create(cls,path):
        path=Path(path);descriptor=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600);os.close(descriptor)
        db=sqlite3.connect(path)
        db.executescript('''
            PRAGMA cache_size=-524288;
            PRAGMA temp_store=FILE;
            CREATE TABLE sequences(
              native_sha BLOB PRIMARY KEY CHECK(length(native_sha)=32),
              first_file INTEGER NOT NULL, first_row INTEGER NOT NULL,
              reuse_kind TEXT NOT NULL, uppercase_sha BLOB NOT NULL,
              rc_sha BLOB NOT NULL, occurrences INTEGER NOT NULL
            ) WITHOUT ROWID;
            CREATE INDEX first_source ON sequences(first_file,first_row);
            CREATE TABLE identifiers(identifier TEXT NOT NULL,native_sha BLOB NOT NULL,
              PRIMARY KEY(identifier,native_sha)) WITHOUT ROWID;
            CREATE INDEX native_identifier ON identifiers(native_sha,identifier);
        ''');db.commit();return cls(path,db)
    @classmethod
    def open_readonly(cls,path):
        path=Path(path).resolve()
        return cls(path,sqlite3.connect('file:'+str(path)+'?mode=ro&immutable=1',uri=True))
    @staticmethod
    def _row(row):
        if row is None:return None
        d=dict(zip(_SELECT.split(','),row));d['task_id']=task_id_from_native_digest(d['native_sha'],d['reuse_kind']);return d
    def lookup(self,digest):
        return self._row(self.db.execute('SELECT '+_SELECT+' FROM sequences WHERE native_sha=?',(digest,)).fetchone())
    def observe(self,sequence,identifier,file_ordinal,row_ordinal,*,resolver=None):
        if not isinstance(sequence,str) or not sequence or set(sequence)-ALPHABET:
            raise ValueError('nonempty source-native RNA alphabet required; no repair')
        if not isinstance(identifier,str) or not identifier or any(x.isspace() for x in identifier):
            raise ValueError('literal native identifier required; no alias/namespace merge')
        if any(not isinstance(x,int) or isinstance(x,bool) or x<0 for x in (file_ordinal,row_ordinal)):
            raise ValueError('nonnegative literal source row/file ordinals required')
        g=sequence_groups(sequence);digest=bytes.fromhex(g['native_sha256']);upper=bytes.fromhex(g['uppercase_sha256']);rc=bytes.fromhex(g['reverse_complement_base_order_group']);row=self.lookup(digest);first=row is None
        if first:
            existing=resolver(sequence) if resolver is not None else None
            kind,expected=existing if existing is not None else ('new',None)
            identity=task_id_from_native_digest(digest,kind)
            if expected is not None and identity!=expected:
                raise ValueError('old source identity is not the exact native prior recipe')
            self.db.execute('INSERT INTO sequences VALUES (?,?,?,?,?,?,?)',(digest,file_ordinal,row_ordinal,kind,upper,rc,1))
        else:
            if row['uppercase_sha']!=upper or row['rc_sha']!=rc:
                raise ValueError('native digest collision or inconsistent derived guard hashes')
            self.db.execute('UPDATE sequences SET occurrences=occurrences+1 WHERE native_sha=?',(digest,))
        self.db.execute('INSERT OR IGNORE INTO identifiers VALUES (?,?)',(identifier,digest))
        result=self.lookup(digest);result['first_occurrence']=first;return result
    def identifiers(self,digest):
        return [r[0] for r in self.db.execute('SELECT identifier FROM identifiers WHERE native_sha=? ORDER BY identifier',(digest,))]
    def identifier_conflicts(self):
        return self.db.execute('SELECT identifier,count(*) FROM identifiers GROUP BY identifier HAVING count(*)>1 ORDER BY identifier')
    def source_rows(self,file_ordinal):
        for row in self.db.execute('SELECT '+_SELECT+' FROM sequences WHERE first_file=? ORDER BY first_row',(file_ordinal,)):
            yield self._row(row)
    def summary(self):
        n,occurrences=self.db.execute('SELECT count(*),coalesce(sum(occurrences),0) FROM sequences').fetchone()
        kinds=dict(self.db.execute('SELECT reuse_kind,count(*) FROM sequences GROUP BY reuse_kind'))
        return dict(native_sequences=n,source_occurrences=occurrences,
            native_identifier_edges=self.db.execute('SELECT count(*) FROM identifiers').fetchone()[0],
            reused_Rfam_sequences=kinds.get('Rfam_existing',0),
            reused_pilot_sequences=kinds.get('RNAcentral_pilot_existing',0),new_sequences=kinds.get('new',0))
    def commit(self):self.db.commit()
    def close(self):self.db.close()
