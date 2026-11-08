import importlib.util
from pathlib import Path
import pytest
root=next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').exists())
spec=importlib.util.spec_from_file_location('sciplex_context',root/'scripts/data_intake_sciplex_context.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_hash_sample_annotation_retains_source_label_and_no_physical_unit():
    r=m.hash_sample('Nutlin_100_AA01\tCGGTCAAGAA\t1\t\n',1)
    assert r['source_sample_label']=='Nutlin_100_AA01'
    assert r['source_culture_plate']=='A' and r['source_culture_well']=='A01'
    assert r['source_treatment_label']=='Nutlin' and r['source_dose_coordinate']==100
    assert r['dose_unit'] is None and r['rna_cell_barcode'] is None
    assert r['hash_oligo_role']=='synthetic_sample_label_not_genomic_sequence'

def test_ctg_values_above_one_are_preserved_and_condition_is_separate_assay():
    raw=dict(CTG='1.2',replicate='2',cell_type='A549',treatment='Dex',dose='100',real_dose='50')
    r=m.ctg_record(raw,1)
    assert r['source_CTG']==1.2 and r['source_real_dose_coordinate']==50
    assert r['dose_unit'] is None and r['readout_unit'] is None
    assert r['replicate_role']=='source_CTG_assay_index_not_RNA_cell_replicate'
    assert r['join_granularity']=='source_condition_coordinate_separate_assays'
    assert r['same_cell_link'] is False and r['admitted_for_task'] is False
    assert r['source_fields_json']==m.canonical(raw).decode()

def test_invalid_or_missing_context_cannot_be_silently_imputed():
    with pytest.raises(ValueError):m.hash_sample('Nutlin_100_AA01\tINVALID\t1\t',1)
    with pytest.raises(ValueError):m.ctg_record(dict(CTG='nan',replicate='2',cell_type='A549',treatment='Dex',dose='100',real_dose='50'),1)
    with pytest.raises(ValueError):m.ctg_record(dict(CTG='1',replicate='2.5',cell_type='A549',treatment='Dex',dose='100',real_dose='50'),1)

def test_real_source_incomplete_well_label_is_retained_without_plate_imputation():
    r=m.hash_sample('SAHA_100_E09\tCGGTCAAGAA\t1\t\n',57)
    assert r['source_sample_label']=='SAHA_100_E09' and r['source_well_oligo']=='E09'
    assert r['source_culture_plate'] is None and r['source_culture_well'] is None
    assert r['source_well_context_status']=='incomplete_or_invalid_source_well_label'
    assert r['source_dose_coordinate']==100 and r['admitted_for_task'] is False
