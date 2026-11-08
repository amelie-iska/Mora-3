import hashlib
import json
import pytest
import pyarrow as pa
import pyarrow.parquet as pq
from test_goa_source_packet import fixture
from mora3.data import goa_packets as g


def setup(tmp_path, bad=None):
    row,_=fixture(tmp_path);row['task_id']='task1'
    def save(name,rows):
        p=tmp_path/name;pq.write_table(pa.Table.from_pylist(rows),p);b=p.read_bytes()
        return dict(path=name,bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),rows=len(rows))
    shard=save('tasks.parquet',[row])
    groups=json.loads(row['group_ids_json'])+['qualified:guard']
    if bad=='lost_group':groups=groups[1:]
    if bad=='duplicate_group':groups+=groups[:1]
    side=save('groups.parquet',[dict(task_id='other' if bad=='allocation' else 'task1',effective_groups_json=json.dumps(groups))])
    master=dict(shards=[shard],effective_group_lookup={shard['path']:side})
    if bad=='duplicate_shard':master['shards']*=2
    if bad=='missing_sidecar':master['effective_group_lookup']={}
    p=tmp_path/'master.json';p.write_text(json.dumps(master));digest=hashlib.sha256(p.read_bytes()).hexdigest()
    if bad=='master_hash':digest='0'*64
    if bad=='shard_hash':(tmp_path/'tasks.parquet').write_bytes(b'changed')
    if bad=='sidecar_hash':(tmp_path/'groups.parquet').write_bytes(b'changed')
    return p,digest,shard,groups


def test_group_bound_packets_preserve_predictor_and_targets(tmp_path):
    p,d,s,groups=setup(tmp_path)
    result=list(g.read_group_bound_goa_packets(p,root=tmp_path,master_sha256=d,shard_path=s['path'],limit=1))
    assert len(result)==1
    v=result[0]
    assert v.inputs()=={'protein_sequence':'ACDE'}
    assert len(v.targets()['source_GO_annotations'])==1
    assert v.provenance()['groups']==groups
    assert v.provenance()['authoritative_groups']['master_sha256']==d
    assert v.provenance()['training_eligibility_granted'] is False
    assert v.provenance()['sequence_source_verified'] is False


@pytest.mark.parametrize('bad',['lost_group','duplicate_group','allocation','duplicate_shard','missing_sidecar','master_hash','shard_hash','sidecar_hash'])
def test_group_binding_fails_closed(tmp_path,bad):
    p,d,s,_=setup(tmp_path,bad)
    with pytest.raises(ValueError):
        list(g.read_group_bound_goa_packets(p,root=tmp_path,master_sha256=d,shard_path=s['path'],limit=1))


@pytest.mark.parametrize('limit',[0,True,1501])
def test_explicit_bound(tmp_path,limit):
    p,d,s,_=setup(tmp_path)
    with pytest.raises(ValueError):
        list(g.read_group_bound_goa_packets(p,root=tmp_path,master_sha256=d,shard_path=s['path'],limit=limit))
