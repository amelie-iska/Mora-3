from pathlib import Path
import sys
R=next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').exists());sys.path.insert(0,str(R/'scripts'))
from data_intake_rosetta_author_context import candidate_context

def test_author_schema_selects_CP_second_dose_column_without_repairing_original():
 row={'Metadata_mmoles_per_liter':'3.02251611288227','Metadata_mmoles_per_liter2':'6.05'};d=candidate_context(row,'CellPainting');assert d['candidate_nominal_dose_micromolar']=='6.05';assert d['source_raw_first_dose']=='3.02251611288227';assert row['Metadata_mmoles_per_liter']=='3.02251611288227';assert not d['admitted_as_condition']
def test_source_dose_missing_is_not_zero_or_inactive():
 for value in ['', 'nan','-666','Infinity']:
  assert candidate_context({'pert_dose':value},'L1000')['candidate_nominal_dose_micromolar'] is None
 assert candidate_context({'pert_dose':'0'},'L1000')['candidate_nominal_dose_micromolar']=='0'
def test_protocol_timepoints_differ_and_never_imply_matched_cells_or_replicates():
 a=candidate_context({'Metadata_mmoles_per_liter2':'10'},'CellPainting');b=candidate_context({'pert_dose':'10'},'L1000');assert a['candidate_protocol_hours']==48 and b['candidate_protocol_hours']==6;assert a['candidate_nominal_dose_micromolar']==b['candidate_nominal_dose_micromolar'];assert a['same_cell_or_matched_replicate'] is False and b['same_cell_or_matched_replicate'] is False
def test_alias_and_release_gate_remains_closed_despite_valid_candidate_units():
 d=candidate_context({'pert_dose':'10'},'L1000');assert d['unit_provenance']=='author_curated_schema_definition_not_literal_old_CSVunit';assert d['candidate_cell_line']=='U2OS';assert d['local_source_prefix']!=d['author_source_prefix'];assert d['admitted_as_condition'] is False
