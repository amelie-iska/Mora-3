from pathlib import Path
import hashlib,sqlite3,sys
import pytest
sys.path.insert(0,'/home/iska/Documents/amelie/bio/Mora-3/scripts')
from data_intake_full_rnacentral_state import NativeIdentityIndex,old_identity_resolver,task_id_from_native_digest
from data_intake_nucleotide_prior import prior_identity


def test_native_case_T_U_and_RC_groups_do_not_collapse_identity(tmp_path):
    ix=NativeIdentityIndex.create(tmp_path/'new.sqlite')
    rows=[ix.observe(s,'literal:'+str(i),0,i) for i,s in enumerate(('AcgT','ACGT','ACGU'))]
    assert len({r['task_id'] for r in rows})==3
    assert len({r['uppercase_sha'].hex() for r in rows})==2
    assert len({r['rc_sha'].hex() for r in rows})==1
    assert ix.summary()['native_sequences']==3
    ix.close()


def test_all_URS_multiplicity_and_conflicts_survive_without_merging(tmp_path):
    ix=NativeIdentityIndex.create(tmp_path/'new.sqlite')
    a=ix.observe('ACGT','URS001',0,1)
    b=ix.observe('ACGT','URS002',0,2)
    c=ix.observe('ACGA','URS001',1,0)
    assert a['first_occurrence'] and not b['first_occurrence']
    assert a['task_id']==b['task_id']!=c['task_id']
    assert ix.identifiers(a['native_sha'])==['URS001','URS002']
    assert list(ix.identifier_conflicts())==[('URS001',2)]
    assert ix.summary()==dict(native_sequences=2,source_occurrences=3,native_identifier_edges=3,reused_Rfam_sequences=0,reused_pilot_sequences=0,new_sequences=2)
    assert ix.lookup(a['native_sha'])['first_row']==1
    ix.close()


def test_preserved_prior_task_identity_and_no_label_inheritance(tmp_path):
    rf=sqlite3.connect(tmp_path/'rf.sqlite');rf.execute('CREATE TABLE tasks(task_id TEXT PRIMARY KEY,source_family TEXT)')
    pi=sqlite3.connect(tmp_path/'pi.sqlite');pi.execute('CREATE TABLE sequences(native_sha TEXT PRIMARY KEY,task_id TEXT,reused_prior INTEGER)')
    sequence='RYSWKM';tid=prior_identity(sequence,'RNA_family_reference_native_symbols',None)
    rf.execute('INSERT INTO tasks VALUES (?,?)',(tid,'Rfam15.1-native-reference-prior'))
    pi.execute('INSERT INTO sequences VALUES (?,?,?)',(hashlib.sha256(sequence.encode()).hexdigest(),tid,1))
    rf.commit();pi.commit();resolve=old_identity_resolver(rf,pi)
    ix=NativeIdentityIndex.create(tmp_path/'new.sqlite');r=ix.observe(sequence,'URS123',0,0,resolver=resolve)
    assert r['task_id']==tid and r['reuse_kind']=='Rfam_existing'
    assert set(r).isdisjoint({'function','taxon','status','family_label'})
    assert ix.summary()['reused_Rfam_sequences']==1
    ix.close();rf.close();pi.close()


def test_pilot_reuse_is_distinct_from_Rfam_and_new_tasks(tmp_path):
    rf=sqlite3.connect(tmp_path/'rf.sqlite');rf.execute('CREATE TABLE tasks(task_id TEXT PRIMARY KEY,source_family TEXT)')
    pi=sqlite3.connect(tmp_path/'pi.sqlite');pi.execute('CREATE TABLE sequences(native_sha TEXT PRIMARY KEY,task_id TEXT,reused_prior INTEGER)')
    sequence='AUGCN';tid=prior_identity(sequence,'RNA_reference_native_symbols',None)
    pi.execute('INSERT INTO sequences VALUES (?,?,?)',(hashlib.sha256(sequence.encode()).hexdigest(),tid,0))
    ix=NativeIdentityIndex.create(tmp_path/'new.sqlite');r=ix.observe(sequence,'source-id',0,0,resolver=old_identity_resolver(rf,pi))
    assert r['task_id']==tid and r['reuse_kind']=='RNAcentral_pilot_existing'
    assert ix.summary()['reused_pilot_sequences']==1
    ix.close();rf.close();pi.close()


def test_old_identity_recipe_mismatch_cannot_be_silently_reused(tmp_path):
    ix=NativeIdentityIndex.create(tmp_path/'new.sqlite')
    with pytest.raises(ValueError,match='identity'):
        ix.observe('ACGT','URS1',0,0,resolver=lambda s:('RNAcentral_pilot_existing','wrong'))
    assert ix.summary()['native_sequences']==0
    ix.close()


def test_bad_source_symbols_or_offsets_are_not_repaired(tmp_path):
    ix=NativeIdentityIndex.create(tmp_path/'new.sqlite')
    for sequence in ('','AXG','A-1'):
        with pytest.raises(ValueError):ix.observe(sequence,'URS1',0,0)
    with pytest.raises(ValueError):ix.observe('ACGT','URS1',0,-1)
    assert ix.summary()['source_occurrences']==0
    ix.close()


def test_fresh_only_creation_and_readonly_reference(tmp_path):
    p=tmp_path/'new.sqlite';ix=NativeIdentityIndex.create(p);ix.observe('ACGT','URS1',0,0);ix.commit();ix.close();before=hashlib.sha256(p.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):NativeIdentityIndex.create(p)
    ro=NativeIdentityIndex.open_readonly(p)
    assert ro.lookup(hashlib.sha256(b'ACGT').digest())['occurrences']==1
    with pytest.raises(sqlite3.OperationalError):ro.db.execute('DELETE FROM sequences')
    ro.close();assert hashlib.sha256(p.read_bytes()).hexdigest()==before


def test_full_axis_task_identity_recipe_never_truncates():
    a='ACGTN'*20000+'a';b=a[:-1]+'A'
    assert task_id_from_native_digest(hashlib.sha256(a.encode()).digest(),'new')==prior_identity(a,'RNA_reference_native_symbols',None)
    assert task_id_from_native_digest(hashlib.sha256(a.encode()).digest(),'new')!=task_id_from_native_digest(hashlib.sha256(b.encode()).digest(),'new')
