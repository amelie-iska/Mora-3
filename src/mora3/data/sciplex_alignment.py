"""Independent bounded comparison of the curated CSR with author CSV exports.

This verifies export alignment, not the original raw experimental barcode join.
It deliberately grants neither training eligibility nor a held-out split.
"""
import numpy as np
from scipy.sparse import issparse


def audit_alignment(*, reference_rows, reference_genes, counts, records, gene_records):
    """Compare all selected count values, ordered genes and source context exactly.

    ``reference_rows`` must come from the pinned author CSV reader, independently
    of the CSR/parquet conversion. All rows remain one unassigned study group.
    The caller owns file checksum verification. No data or metadata are mutated.
    """
    rows=tuple(reference_rows)
    genes=tuple(reference_genes)
    if not 1 <= len(rows) <= 512 or not genes or len(genes)>5000:
        raise ValueError('require bounded cells and complete selected-gene axis')
    if (not issparse(counts) or counts.ndim!=2 or counts.shape[1]!=len(genes)
            or counts.shape[0]!=len(records) or len(gene_records)!=len(genes)):
        raise ValueError('curated matrix/record/axis shape mismatch')
    if any(g != dict(position=i, source_symbol=name)
           for i,(g,name) in enumerate(zip(gene_records,genes))):
        raise ValueError('ordered gene position/symbol mismatch')
    seen=set()
    for row in rows:
        i=row['source_index']
        if type(i) is not int or not 0<=i<len(records) or row['cell_id'] in seen:
            raise ValueError('invalid selected row or duplicate barcode')
        seen.add(row['cell_id'])
        expected=dict(source_row=i, expression_row=i, cell_barcode=row['cell_id'],
            intervention_alias=row['intervention'], source_scaled_dose=row['dose_coordinate'],
            post_assay_n_genes=row['n_genes'], post_assay_n_counts=row['n_counts'],
            cell_line='A549', exposure_hours=24, leakage_group='study:GSE139944:GSM4150377',
            split='unassigned_study_group', training_ready=False, physical_dose=None,
            physical_dose_unit=None, baseline_matched_cell=None,
            vehicle_control=row['intervention']=='Vehicle',
            expression_array='counts-csr.npz', gene_order='genes.json')
        record=records[i]
        if any(k not in record or record[k]!=v for k,v in expected.items()):
            raise ValueError(f'source context or row alignment mismatch at {i}')
        actual=counts.getrow(i).toarray().ravel()
        reference=np.asarray(row['counts'])
        if (reference.shape!=(len(genes),) or not np.isfinite(actual).all()
            or not np.isfinite(reference).all() or (actual<0).any()
            or (reference<0).any() or not np.equal(reference,np.floor(reference)).all()
            or not np.array_equal(actual,reference)):
            raise ValueError(f'count alignment mismatch at {i}')
    return dict(version=1, verified_export_alignment=True, selected_cells=len(rows),
        gene_columns=len(genes), count_values_compared=len(rows)*len(genes),
        pairing_rule='Pinned author export positional indices; cell_info barcode by same row order',
        independent_raw_barcode_validation=False, training_ready=False,
        leakage_groups=['study:GSE139944:GSM4150377'],
        split='unassigned_study_group', physical_dose_resolved=False)
