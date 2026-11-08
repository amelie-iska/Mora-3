"""Synthetic fixtures verify sidecar contracts, not real-data coverage."""
import copy
import pytest
from mora3.data import gperturb_sciplex as source


def row():
    return dict(cell_id='cell-a', source_index=0, intervention='Dex',
                dose_coordinate=.001, dose_units='source_scaled_coordinate',
                counts=[0,2], n_genes=1, n_counts=20)


def test_prediction_masks_do_not_leak_response_or_post_assay_qc():
    a=row(); b=copy.deepcopy(a)
    b.update(counts=[30,40],n_genes=900,n_counts=2000,cell_id='hidden-id')
    x=source.array_task(a,('A','B'),direction='response')
    y=source.array_task(b,('A','B'),direction='response')
    assert x['conditions']==y['conditions']
    assert x['targets']!=y['targets']
    assert x['targets']['expression']['values']==[0,2]
    assert x['targets']['expression']['axis_ids']==['A','B']
    assert x['training_ready'] is False


def test_reverse_mask_and_dose_context_preserve_distinct_observations():
    a=row(); b=row(); b['dose_coordinate']=.5
    assert source.array_task(a,('A','B'),direction='response')['conditions'] != source.array_task(b,('A','B'),direction='response')['conditions']
    inv=source.array_task(a,('A','B'),direction='intervention')
    assert inv['conditions']['expression']['values']==[0,2]
    assert inv['targets']['intervention']['identity']=='Dex'
    a.update(intervention='Vehicle',dose_coordinate=1)
    vehicle=source.array_task(a,('A','B'),direction='response')['conditions']['intervention']
    assert vehicle['is_vehicle'] is True
    assert vehicle['physical_concentration'] is None
    assert vehicle['source_coordinate']==1


@pytest.mark.parametrize('change',[
 {'counts':[1]}, {'counts':[0,-1]}, {'counts':[0,float('nan')]},
 {'counts':[0,1.5]}, {'dose_coordinate':float('inf')},
 {'dose_coordinate':0}, {'intervention':'unknown'},
 {'dose_units':'uM'}, {'intervention':'Vehicle','dose_coordinate':.5},
])
def test_invalid_sidecars_fail_closed(change):
    a=row(); a.update(change)
    with pytest.raises(ValueError): source.array_task(a,('A','B'),direction='response')


def test_no_silent_axis_reordering_or_unknown_tasks():
    with pytest.raises(ValueError): source.array_task(row(),('A',''),direction='response')
    with pytest.raises(ValueError): source.array_task(row(),('A','B'),direction='anything')
    # Duplicated symbols are source positions, not unique stable gene identifiers.
    x=source.array_task(row(),('A','A'),direction='response')
    assert x['targets']['expression']['axis_ids']==['A','A']
    assert x['targets']['expression']['axis_identity']=='source_column_position'
