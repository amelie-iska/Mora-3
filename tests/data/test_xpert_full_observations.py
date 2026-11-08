import gzip
import json
from pathlib import Path
import pytest
from mora3.data.xpert_full_store import XPertFullNumericStore
ROOT=Path(__file__).resolve().parents[2]

@pytest.mark.parametrize('article',[32221275,32221302])
def test_native_observations_exact_order_roles_and_caller_isolation(article):
    store=XPertFullNumericStore(root=ROOT)
    selection=store.read_observations(article,(2,0))
    manifest=json.loads((ROOT/store.MANIFEST['path']).read_bytes())
    dataset=next(d for d in manifest['datasets'] if d['source_article_ID']==article)
    with gzip.open(dataset['obs_axis']['path'],'rb') as f:
        oracle=[next(f) for _ in range(3)]
    assert selection.native_records==tuple(oracle[i] for i in (2,0))
    record=selection.record(0);record['native_fields'].clear()
    assert selection.record(0)==json.loads(oracle[2])
    assert selection.audit()['source_native_observation_unit']==dataset['source_native_observation_unit']
    assert selection.audit()['model_use_allowed'] is False
    assert selection.audit()['context_bound'] is False
    for purpose in ['training','generation','prediction']:
        with pytest.raises(ValueError): selection.require_model_use(purpose)

@pytest.mark.parametrize('article,rows',[(True,(0,)),(0,(0,)),(32221302,(True,)),(32221302,(-1,)),(32221302,(678401,)),(32221302,()),(32221302,(0,0))])
def test_invalid_native_observation_selection(article,rows):
    with pytest.raises(ValueError): XPertFullNumericStore(root=ROOT).read_observations(article,rows)


def test_observation_asset_freshly_verified(tmp_path):
    cls=XPertFullNumericStore
    for ref in (cls.MANIFEST,cls.HANDOFF):
        p=tmp_path/ref['path'];p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes((ROOT/ref['path']).read_bytes())
    manifest=json.loads((ROOT/cls.MANIFEST['path']).read_bytes())
    ref=manifest['datasets'][1]['obs_axis']
    p=tmp_path/Path(ref['path']).relative_to(ROOT)
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_bytes(Path(ref['path']).read_bytes())
    store=cls(root=tmp_path)
    assert len(store.read_observations(32221302,(0,)).native_records)==1
    p.write_bytes(p.read_bytes()+b' ')
    with pytest.raises(ValueError,match='immutable proof mismatch'):
        store.read_observations(32221302,(0,))
