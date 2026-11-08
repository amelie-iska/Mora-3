import copy
import hashlib
import json
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from mora3.model import annotation_multihead as am
from test_annotation_vocabulary import admitted
from mora3.model.annotation_vocabulary import fit_annotation_vocabulary

REAL = Path('outputs/data-intake/20261101-annotation-target-axes-v1/manifest.json')


def reader(path, **kwargs):
    from mora3.data.annotation_axes import read_annotation_axes
    return read_annotation_axes(path, **kwargs)


def fixture(tmp_path):
    m=json.loads(REAL.read_text())
    for entry in m['axis_files']:
        source=Path(entry['path']); dest=tmp_path/source.name
        dest.write_bytes(source.read_bytes());entry['path']=str(dest)
    path=tmp_path/'manifest.json';path.write_text(json.dumps(m))
    return path,m


def test_complete_registry_retains_all_metadata_and_provisional_ec():
    r=reader(REAL)
    assert len(r.categories)==16
    assert len(r.axis('GO'))==28079
    assert len(r.axis('EC'))==5124
    assert '1.1.1.n11' in r.axis('EC')
    rows=pq.read_table(REAL.parent/'GO.parquet').to_pylist()
    assert r.rows('GO')==rows
    altered=r.rows('GO');altered[0]['axis_id']='bad'
    assert r.axis('GO')[0]=='GO:0000001'


@pytest.mark.parametrize('kind',['bytes','category','count','axis','policy','pin'])
def test_fail_closed_registry_mutations(tmp_path,kind):
    path,m=fixture(tmp_path)
    if kind=='bytes':
        p=Path(m['axis_files'][0]['path']);p.write_bytes(p.read_bytes()+b'x')
    elif kind=='category':m['axis_files'][0]['axis']='GO'
    elif kind=='count':m['axis_files'][0]['labels']-=1
    elif kind in ('axis','policy'):
        e=m['axis_files'][0];rows=pq.read_table(e['path']).to_pylist()
        if kind=='axis':rows[0]['axis_id']=rows[1]['axis_id']
        else:rows[0]['absence_is_negative']=True
        pq.write_table(pa.Table.from_pylist(rows),e['path'])
        raw=Path(e['path']).read_bytes();e.update(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError):reader(path,**({'expected_sha256':'0'*64} if kind=='pin' else {}))


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_registry_model_full_axes_and_checkpoint_source_binding(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    r=reader(REAL);v=admitted();vocab=fit_annotation_vocabulary([v],learned_budget=0,max_length=2)
    torch.manual_seed(9701)
    m=am.SharedAnnotationModel.from_registry(vocab,r,width=4,heads=1,layers=1).to(device)
    z=m([v]);assert z['GO'].shape==(1,28079);assert z['EC'].shape==(1,5124)
    sum(x.square().mean() for x in z.values()).backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    state=copy.deepcopy(m.state_dict())
    assert state['_extra_state']['annotation_registry']==r.binding()
    state['_extra_state']['annotation_registry']['manifest_sha256']='0'*64
    with pytest.raises(ValueError,match='configuration'):m.load_state_dict(state)
