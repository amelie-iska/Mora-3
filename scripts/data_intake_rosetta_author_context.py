"""Source-author nominal context candidates; alias/release admission remains gated."""
from decimal import Decimal,InvalidOperation
def candidate_context(row,modality):
    if modality not in ['CellPainting','L1000']:raise ValueError('explicit assay modality required')
    field='Metadata_mmoles_per_liter2' if modality=='CellPainting' else 'pert_dose';raw=row.get(field);value=None
    try:
        parsed=Decimal(str(raw))
        if parsed.is_finite() and parsed>=0:value=format(parsed.normalize(),'f')
    except InvalidOperation:pass
    return dict(local_source_prefix='CDRPBIO-BBBC036-Bray',author_source_prefix='CDRP-BBBC047-Bray',modality=modality,source_dose_field=field,source_dose_literal=raw,source_raw_first_dose=row.get('Metadata_mmoles_per_liter') if modality=='CellPainting' else None,candidate_nominal_dose_micromolar=value,unit_provenance='author_curated_schema_definition_not_literal_old_CSVunit',candidate_protocol_hours=48 if modality=='CellPainting' else 6,time_provenance='author_curation_protocol_constant_not_perrow_timestamp',candidate_cell_line=row.get('Metadata_cell_id') if modality=='CellPainting' else 'U2OS',cell_line_provenance='old_source_metadata' if modality=='CellPainting' else 'author_assigned_dataset_constant',source_alias_and_release_equivalence_verified=False,admitted_as_condition=False,same_cell_or_matched_replicate=False)
