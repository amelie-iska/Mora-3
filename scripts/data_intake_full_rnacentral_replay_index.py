"""New independent literal-source multiplicity index for full RNA proof.

Only the newly created proof database is writable. The completed census index
is attached explicitly read-only for exact key/count/first-pointer comparisons.
"""
from pathlib import Path
import os,sqlite3

class NativeReplayIndex:
    def __init__(self,path):
        self.path=Path(path);fd=os.open(self.path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600);os.close(fd);self.db=sqlite3.connect(self.path,uri=True);self.db.executescript('''PRAGMA cache_size=-262144;PRAGMA temp_store=FILE;
        CREATE TABLE native_counts(native_sha BLOB PRIMARY KEY,occurrences INTEGER NOT NULL,first_file INTEGER NOT NULL,first_row INTEGER NOT NULL) WITHOUT ROWID;
        CREATE TABLE identifiers(identifier TEXT NOT NULL,native_sha BLOB NOT NULL,PRIMARY KEY(identifier,native_sha)) WITHOUT ROWID;
        CREATE TABLE holdout_blocked(native_sha BLOB NOT NULL,group_id TEXT NOT NULL,PRIMARY KEY(native_sha,group_id)) WITHOUT ROWID;''');self.db.commit()
    def observe(self,native_sha,identifier,file,row,heldout_group_matches):
        if not isinstance(native_sha,bytes) or len(native_sha)!=32 or not identifier or file<0 or row<0:raise ValueError('literal validated native digest and source pointer required')
        first=self.db.execute('INSERT OR IGNORE INTO native_counts VALUES (?,0,?,?)',(native_sha,file,row)).rowcount==1
        self.db.execute('UPDATE native_counts SET occurrences=occurrences+1 WHERE native_sha=?',(native_sha,));self.db.execute('INSERT OR IGNORE INTO identifiers VALUES (?,?)',(identifier,native_sha))
        self.db.executemany('INSERT OR IGNORE INTO holdout_blocked VALUES (?,?)',[(native_sha,g) for g in heldout_group_matches]);return first
    def commit(self):self.db.commit()
    def compare_readonly_census(self,path):
        self.db.commit();uri='file:'+str(Path(path).resolve())+'?mode=ro&immutable=1';self.db.execute('ATTACH DATABASE ? AS source_state',(uri,))
        q={
          'native_count_or_first_pointer_mismatches':'SELECT count(*) FROM native_counts p LEFT JOIN source_state.sequences s ON s.native_sha=p.native_sha WHERE s.native_sha IS NULL OR p.occurrences!=s.occurrences OR p.first_file!=s.first_file OR p.first_row!=s.first_row',
          'census_native_keys_absent_from_source':'SELECT count(*) FROM source_state.sequences s LEFT JOIN native_counts p ON p.native_sha=s.native_sha WHERE p.native_sha IS NULL',
          'source_URS_edges_absent_from_census':'SELECT count(*) FROM (SELECT identifier,native_sha FROM main.identifiers EXCEPT SELECT identifier,native_sha FROM source_state.identifiers)',
          'phantom_census_URS_edges':'SELECT count(*) FROM (SELECT identifier,native_sha FROM source_state.identifiers EXCEPT SELECT identifier,native_sha FROM main.identifiers)',
          'source_heldout_blocks_absent_from_census':'SELECT count(*) FROM (SELECT native_sha,group_id FROM main.holdout_blocked EXCEPT SELECT native_sha,group_id FROM source_state.holdout_blocked)',
          'phantom_census_heldout_blocks':'SELECT count(*) FROM (SELECT native_sha,group_id FROM source_state.holdout_blocked EXCEPT SELECT native_sha,group_id FROM main.holdout_blocked)'}
        result={k:self.db.execute(sql).fetchone()[0] for k,sql in q.items()};self.db.execute('DETACH DATABASE source_state');return result
    def summary(self):
        n,occ=self.db.execute('SELECT count(*),coalesce(sum(occurrences),0) FROM native_counts').fetchone()
        return dict(native_sequences=n,quality_pass_source_occurrences=occ,native_URS_edges=self.db.execute('SELECT count(*) FROM identifiers').fetchone()[0],known_heldout_block_edges=self.db.execute('SELECT count(*) FROM holdout_blocked').fetchone()[0])
    def close(self):self.db.close()
