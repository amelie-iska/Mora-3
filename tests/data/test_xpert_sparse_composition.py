"""Native CSC columns with explicitly bounded observation-context selections."""
import json
import numpy as np
import pytest
from pathlib import Path
from mora3.data.xpert_full_store import XPertFullNumericStore
ROOT=Path(__file__).resolve().parents[2]


def test_sparse_composition_preserves_columns_and_explicit_partial_context():
    store=XPertFullNumericStore(root=ROOT)
    result=store.read_candidate_columns(32221275,(33693,0),(590,0))
    m=json.loads((ROOT/store.MANIFEST['path']).read_bytes())
    def raw(key):
        return np.load(ROOT/m['array_transports'][key]['transport']['path'],mmap_mode='r',allow_pickle=False)
    ptr=raw('32221275-X-indptr')
    for c,ordinal in zip(result.columns,(33693,0)):
        start,stop=map(int,ptr[ordinal:ordinal+2])
        assert c.native_column_index==ordinal
        assert c.values_bytes==raw('32221275-X-data')[start:stop].tobytes()
        assert c.indices.values_bytes==raw('32221275-X-indices')[start:stop].tobytes()
        assert c.values.finite_mask==np.isfinite(raw('32221275-X-data')[start:stop]).tobytes()
        assert c.logical_rows==111445
        assert c.implicit_zero_is_missing is False
    assert len(result.gene_axis.native_records)==33694
    assert result.observations.native_records==store.read_observations(32221275,(590,0)).native_records
    for i,c in enumerate(result.contexts.contexts):
        assert c.native_fields()==result.observations.record(i)['native_fields']
    a=result.audit()
    assert a['context_native_row_indices']==[590,0]
    assert a['native_column_indices']==[33693,0]
    assert a['all_observations_context_bound'] is False
    assert a['task_admitted'] is False and a['model_use_allowed'] is False
    assert a['official_split'] is None
    assert a['rights_verified_as_evidence_only'] is True
    assert a['coordinate_proofs'][0]['coordinate_proof_sha256']
    for p in ('training','prediction','generation','split_use','unknown'):
        with pytest.raises(ValueError): result.require_model_use(p)
    a.clear()
    assert result.audit()['native_column_indices']==[33693,0]


@pytest.mark.parametrize('article,columns,rows',[
    (32221302,(0,),(0,)),(True,(0,),(0,)),(0,(0,),(0,)),
    (32221275,(),(0,)),(32221275,(True,),(0,)),
    (32221275,(33694,),(0,)),(32221275,(0,0),(0,)),
    (32221275,(0,),()),(32221275,(0,),(111445,)),
    (32221275,(0,),(True,)),(32221275,(0,),(0,0))])
def test_invalid_sparse_composition_refused(article,columns,rows):
    with pytest.raises(ValueError):
        XPertFullNumericStore(root=ROOT).read_candidate_columns(article,columns,rows)


def test_sparse_composition_uses_current_mmap_type_after_module_reload(monkeypatch):
    import importlib.util
    import sys
    import mora3.data.xpert_mmap as mmap
    spec=importlib.util.spec_from_file_location(mmap.__name__,mmap.__file__)
    current=importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules,spec.name,current)
    spec.loader.exec_module(current)
    result=XPertFullNumericStore(root=ROOT).read_candidate_columns(32221275,(33693,),(0,))
    assert result.columns[0].logical_rows==111445
