"""Independent export alignment audit; fixtures are synthetic, not source coverage."""
import copy
import pytest
import numpy as np
from scipy.sparse import csr_matrix
from mora3.data.sciplex_alignment import audit_alignment


def fixture():
    rows = [dict(source_index=i, cell_id=f'cell-{i}', intervention='Dex',
                 dose_coordinate=.001, dose_units='source_scaled_coordinate',
                 counts=v, n_genes=2, n_counts=10) for i,v in enumerate(([1,2],[3,0]))]
    records = [dict(source_row=i, expression_row=i, cell_barcode=f'cell-{i}',
        intervention_alias='Dex', source_scaled_dose=.001, post_assay_n_genes=2,
        post_assay_n_counts=10, cell_line='A549', exposure_hours=24,
        leakage_group='study:GSE139944:GSM4150377', split='unassigned_study_group',
        training_ready=False, physical_dose=None, physical_dose_unit=None,
        baseline_matched_cell=None, vehicle_control=False,
        expression_array='counts-csr.npz', gene_order='genes.json') for i in range(2)]
    return dict(reference_rows=rows, reference_genes=('A','A'),
        counts=csr_matrix([[1,2],[3,0]]), records=records,
        gene_records=[dict(position=i,source_symbol='A') for i in range(2)])


def test_exact_audit_preserves_duplicate_symbols_and_quarantine():
    result=audit_alignment(**fixture())
    assert result['verified_export_alignment'] is True
    assert result['selected_cells']==2 and result['count_values_compared']==4
    assert result['independent_raw_barcode_validation'] is False
    assert result['training_ready'] is False
    assert result['leakage_groups']==['study:GSE139944:GSM4150377']


@pytest.mark.parametrize('field,value',[
    ('source_row',1),('expression_row',1),('cell_barcode','other'),
    ('intervention_alias','SAHA'),('source_scaled_dose',.5),
    ('post_assay_n_counts',11),('post_assay_n_genes',3),('cell_line','K562'),
    ('exposure_hours',48),('leakage_group','random-cell'),('split','test'),
    ('training_ready',True),('physical_dose',1),('physical_dose_unit','uM'),
    ('baseline_matched_cell','invented'),('vehicle_control',True),
    ('expression_array','other.npz'),('gene_order','other.json')])
def test_rejects_misaligned_metadata_or_unsupported_admission(field,value):
    f=fixture();f['records'][0][field]=value
    with pytest.raises(ValueError): audit_alignment(**f)


@pytest.mark.parametrize('mutation',['counts','negative','nan','genes','positions','duplicate','rows','shape'])
def test_rejects_expression_axis_and_row_corruption(mutation):
    f=fixture()
    if mutation=='counts': f['counts']=csr_matrix([[3,0],[1,2]])
    if mutation=='negative': f['counts']=csr_matrix([[-1,2],[3,0]])
    if mutation=='nan': f['counts']=csr_matrix([[np.nan,2],[3,0]])
    if mutation=='genes': f['gene_records'][1]['source_symbol']='B'
    if mutation=='positions': f['gene_records'][1]['position']=0
    if mutation=='duplicate': f['reference_rows'][1]['cell_id']='cell-0'
    if mutation=='rows': f['records'].reverse()
    if mutation=='shape': f['counts']=csr_matrix([[1],[3]])
    with pytest.raises(ValueError): audit_alignment(**f)


def test_inputs_immutable_and_bounded():
    f=fixture();before=copy.deepcopy(f['records']);audit_alignment(**f)
    assert f['records']==before
    f['reference_rows']=[]
    with pytest.raises(ValueError): audit_alignment(**f)


def test_expanded_audit_checks_last_of_512_rows():
    f=fixture(); row=f['reference_rows'][0]; record=f['records'][0]
    f['reference_rows']=[dict(row,source_index=i,cell_id=f'cell-{i}') for i in range(512)]
    f['records']=[dict(record,source_row=i,expression_row=i,cell_barcode=f'cell-{i}') for i in range(512)]
    f['counts']=csr_matrix(np.tile([1,2],(512,1)))
    report=audit_alignment(**f)
    assert report['count_values_compared']==1024 and report['training_ready'] is False
    f['reference_rows'][-1]['counts']=[2,1]
    with pytest.raises(ValueError,match='511'): audit_alignment(**f)
