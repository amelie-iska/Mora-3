"""Full-axis sparse row masks preserve stored nonfinite values and implicit zeros."""
import numpy as np
import pytest
from mora3.data import xpert_ordinal_index as mod
from mora3.data.xpert_mmap import _canonical


def stored(genes=(0,2,4), values=(-0.,np.nan,np.inf)):
    o=mod.NormanRowOrdinals(1,np.arange(len(genes),dtype='<u4').tobytes(),
        np.array(genes,dtype='<i4').tobytes(),_canonical({}))
    a=np.array(values,dtype='<f4')
    return mod.NormanStoredRow(o,a.tobytes(),np.isfinite(a).tobytes(),_canonical({}))


def test_full_axis_masks_distinguish_stored_zero_nonfinite_and_implicit_zero():
    p=mod.NormanSparseProfile(stored(),5)
    masks=p.axis_masks()
    assert masks['stored'].tolist()==[True,False,True,False,True]
    assert masks['implicit_zero'].tolist()==[False,True,False,True,False]
    assert masks['finite'].tolist()==[True,True,False,True,False]
    assert p.stored.values_bytes==np.array([-0.,np.nan,np.inf],dtype='<f4').tobytes()
    masks['stored'][:]=False
    assert p.axis_masks()['stored'][0]
    with pytest.raises(ValueError,match='admission'):p.require_model_use('training')

@pytest.mark.parametrize('genes,width', [((0,0),5),((2,1),5),((-1,2),5),((0,5),5),((0,1),True),((0,1),0)])
def test_invalid_sparse_axis_refused(genes,width):
    with pytest.raises(ValueError):mod.NormanSparseProfile(stored(genes,(0.,1.)),width)


def test_empty_stored_row_has_observed_finite_implicit_zeros():
    p=mod.NormanSparseProfile(stored((),()),5)
    assert p.axis_masks()['implicit_zero'].all()
    assert p.axis_masks()['finite'].all()
    assert not p.axis_masks()['stored'].any()


def test_composition_api_exists():
    assert callable(getattr(mod.XPertNormanOrdinalIndex,'read_candidate_profiles',None))


def test_real_original_rows_compose_exact_full_axis_and_context():
    from pathlib import Path
    reader=mod.XPertNormanOrdinalIndex(root=Path(__file__).resolve().parents[2])
    result=reader.read_candidate_profiles((590,0,1),max_payload_bytes=3*1024**3)
    assert len(result.gene_axis.native_records)==33694
    assert result.observations.native_records==reader.store.read_observations(32221275,(590,0,1)).native_records
    for i,p in enumerate(result.profiles):
        assert p.native_row_index==(590,0,1)[i]
        masks=p.axis_masks();genes=p.stored.ordinals.gene_coordinates()
        assert masks['stored'].sum()==len(genes)
        assert np.array_equal(np.flatnonzero(masks['stored']),genes)
        assert masks['finite'][masks['implicit_zero']].all()
        assert np.array_equal(masks['finite'][genes],np.isfinite(p.stored.native_values()))
        assert result.contexts.contexts[i].native_fields()==result.observations.record(i)['native_fields']
    audit=result.audit()
    assert audit['context_bound'] is True and audit['numeric_bound'] is True
    assert audit['complete_sparse_profile_composed'] is True
    assert audit['task_admitted'] is False and audit['model_use_allowed'] is False
    assert audit['official_split'] is None
    for purpose in ('training','prediction','generation','split_use'):
        with pytest.raises(ValueError):result.require_model_use(purpose)

@pytest.mark.parametrize('rows',[(),(0,0),(True,),(111445,),(0,1,2,3)])
def test_invalid_profile_rows_refused(rows):
    from pathlib import Path
    reader=mod.XPertNormanOrdinalIndex(root=Path(__file__).resolve().parents[2])
    with pytest.raises(ValueError):reader.read_candidate_profiles(rows,max_payload_bytes=3*1024**3)
