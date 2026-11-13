"""Real Parquet metadata fixtures; numeric source payloads are never required."""
from pathlib import Path
from types import SimpleNamespace
from hashlib import sha256
import json
import pytest
import pyarrow as pa
import pyarrow.parquet as pq
from mora3.data.split_exposure import HeldoutGroupExposure
from mora3.research.registered_split_exposure import scan_view_phase


def fixture(tmp_path,*,mixed=False):
    descriptors=[]
    for i,(role,items)in enumerate(((1,[('v',['shared']),('v2',['other'])]),(0,[('train',['shared']),('exclude',['other'])]),(2,[('test',['unique'])]))):
        p=tmp_path/f'groups-{i}.parquet'
        pq.write_table(pa.table({'task_id':[x[0]for x in items],'effective_groups_json':[json.dumps(x[1])for x in items]}),p)
        raw=p.read_bytes();n=len(items);disposition=dict(kind='homogeneous-original-row-disposition',code=role,rows=n)
        if mixed and i==1:
            mask=tmp_path/'mask.bin';mask.write_bytes(bytes([0,9]));m=mask.read_bytes()
            disposition=dict(kind='original-row-ordinal-uint8',rows=n,reference=dict(path=str(mask),bytes=len(m),sha256=sha256(m).hexdigest()))
        descriptors.append(dict(source_shard=dict(path=f'never-open-numeric-source-{i}.parquet',rows=n),
            original_group_sidecar=dict(path=str(p),rows=n,bytes=len(raw),sha256=sha256(raw).hexdigest()),
            disposition=disposition,primary_retained_rows=(1 if mixed and i==1 else n)))
    return SimpleNamespace(descriptors=tuple(descriptors),_ordinal_exclusions=frozenset({('never-open-numeric-source-1.parquet',1)})if not mixed else frozenset())


def audit():return HeldoutGroupExposure(max_heldout_keys=20,max_heldout_rows=20,max_groups_per_row=8,max_examples=10)


@pytest.mark.parametrize('mixed',(False,True))
def test_complete_roles_and_exclusions_without_source_payloads(tmp_path,mixed):
    view=fixture(tmp_path,mixed=mixed);a=audit();journal=[]
    held=scan_view_phase(view,tmp_path,a,phase='heldout',record=journal.append,checkpoint=lambda:None)
    a.freeze_heldout();train=scan_view_phase(view,tmp_path,a,phase='train',record=journal.append,checkpoint=lambda:None)
    result=a.finish()
    assert result['role_rows']=={'train':1,'validation':2,'test':1}
    assert result['cross_role_group_keys']==1 and train['selected_rows']==1 and held['selected_rows']==3
    assert len(journal)==3 and not list(tmp_path.glob('never-open*'))


def test_original_sidecar_byte_corruption_refuses(tmp_path):
    view=fixture(tmp_path);p=Path(view.descriptors[0]['original_group_sidecar']['path']);p.write_bytes(p.read_bytes()+b'x')
    with pytest.raises(ValueError):scan_view_phase(view,tmp_path,audit(),phase='heldout',record=lambda r:None,checkpoint=lambda:None)


def test_row_counts_and_sidecar_size_budget_refuse(tmp_path):
    view=fixture(tmp_path)
    with pytest.raises(ValueError):scan_view_phase(view,tmp_path,audit(),phase='heldout',record=lambda r:None,checkpoint=lambda:None,max_sidecar_bytes=1)
    view.descriptors[0]['source_shard']['rows']+=1
    with pytest.raises(ValueError):scan_view_phase(view,tmp_path,audit(),phase='heldout',record=lambda r:None,checkpoint=lambda:None)


def test_metadata_cannot_point_outside_declared_root(tmp_path):
    view=fixture(tmp_path);sub=tmp_path/'allowed';sub.mkdir()
    with pytest.raises(ValueError,match='root'):scan_view_phase(view,sub,audit(),phase='heldout',record=lambda r:None,checkpoint=lambda:None)


def test_complete_high_alias_native_group_row_is_not_truncated(tmp_path):
    view=fixture(tmp_path);d=view.descriptors[1];p=Path(d['original_group_sidecar']['path'])
    keys=['genbank:'+str(i)for i in range(1084)]+['shared']
    pq.write_table(pa.table({'task_id':['train','exclude'],'effective_groups_json':[json.dumps(keys),json.dumps(['other'])]}),p)
    raw=p.read_bytes();d['original_group_sidecar'].update(bytes=len(raw),sha256=sha256(raw).hexdigest())
    index=HeldoutGroupExposure(max_heldout_keys=20,max_heldout_rows=20,max_groups_per_row=65536,max_examples=10)
    scan_view_phase(view,tmp_path,index,phase='heldout',record=lambda r:None,checkpoint=lambda:None)
    index.freeze_heldout();scan_view_phase(view,tmp_path,index,phase='train',record=lambda r:None,checkpoint=lambda:None)
    assert index.finish()['cross_role_group_keys']==1
