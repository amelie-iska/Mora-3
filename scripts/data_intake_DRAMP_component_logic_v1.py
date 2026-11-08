"""Candidate writer logic only; no live source scan, file writes or admission."""
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
import hashlib,json,sys
CODE=Path('/home/iska/Documents/amelie/bio/Mora-3/planning/research/data_intake/20261101-DRAMP-offline-preparation-v1/local-offline-DRAMP-schema-draft-v2')
sys.path.insert(0,str(CODE))
from data_intake_DRAMP_stability_v2 import IDENTITY_FIELDS,REVIEWED_CONTEXT_LITERALS,SCALAR,material_descriptor
from data_intake_DRAMP_stability_candidates_v2 import build_report_candidates
MISSING={'','not mentioned','not found','none','n/a','not reported','unknown'}
def _values(row):return {k:v['decoded_value'] for k,v in json.loads(row['full_source_named_cells_json']).items()}
def _sha(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def _valid_identity(values):return all(isinstance(values.get(k),str) and values[k].strip() for k in IDENTITY_FIELDS)

def project_row(row,source_ref):
    values=_values(row);text=values.get('half_life');has_report=isinstance(text,str) and text.strip().casefold() not in MISSING
    disposition=dict(source_native_ID=row['source_native_DRAMP_ID'],source_row=row['native_source_row'],native_named_cells_json=row['full_source_named_cells_json'],source_report_occurrences=int(has_report),candidate_views=0,source_ref=deepcopy(source_ref),global_task_admitted=False,model_use=False)
    if not has_report:disposition['disposition']='missing_report_evidence';return disposition,[]
    if not _valid_identity(values):disposition['disposition']='native_identity_review_required';return disposition,[]
    source=dict(deepcopy(source_ref),native_source_row=row['native_source_row'],native_DRAMP_ID=row['source_native_DRAMP_ID']);candidates=build_report_candidates(values,source)
    for candidate in candidates:verify_candidate(candidate,row)
    disposition.update(disposition='source_report_candidates_only',candidate_views=len(candidates),candidate_ids=[c['candidate_id'] for c in candidates]);return disposition,candidates

def group_proposal(row):
    values=_values(row);sequence=values.get('Sequence');pmid=values.get('Pubmed_ID');known=isinstance(pmid,str) and pmid.strip().casefold() not in MISSING
    return dict(source_native_ID=row['source_native_DRAMP_ID'],source_row=row['native_source_row'],native_material_descriptor_guard=material_descriptor(values)['native_descriptor_sha256'] if _valid_identity(values) else None,native_sequence_guard=hashlib.sha256(sequence.encode()).hexdigest() if isinstance(sequence,str) and sequence else None,modification_descriptor_component_audit_only={k:values.get(k) for k in IDENTITY_FIELDS if k!='Sequence'},study_literal=pmid,study_group='native-source-reference:'+_sha(pmid) if known else None,study_status='literal_source_reference_not_verified_assay_equivalence' if known else 'unknown_native_reference',same_sequence_proves_same_modified_material=False,parent_or_homology_inferred=False,proposed_split=None,proposal_only=True,applied_to_master=False)

def verify_candidate(candidate,row):
    values=_values(row);text=values.get('half_life')
    if not isinstance(text,str) or text.strip().casefold() in MISSING:raise ValueError('missing report is not a supervised target')
    body={k:v for k,v in candidate.items() if k!='candidate_id'}
    if _sha(body)!=candidate['candidate_id']:raise ValueError('candidate canonical identity checksum differs')
    if candidate['schema']!='mora3.SourceSymbolicPeptideADMETCandidate/v2' or candidate['split'] is not None or candidate['native_entity_kind']!='source_symbolic_modified_peptide_descriptor':raise ValueError('candidate schema or unallocated source role changed')
    if candidate['evidence_role']!='database_curated_report;primary_assay_not_independently_verified':raise ValueError('source report cannot become independent experimental evidence')
    for field in ['physical_ADMET_objective_admitted','global_task_admitted','model_use','standard_protein_sequence_codec_applicable','atomic_graph_validated']:
        if candidate[field] is not False:raise ValueError('candidate-only native chemistry/model/admission gate opened')
    expected_context={k:values.get(k) if isinstance(values.get(k),str) and values[k] in allowed else None for k,allowed in REVIEWED_CONTEXT_LITERALS.items()};expected_mask={k:v is not None for k,v in expected_context.items()};expected_conditions=dict(native_peptide_identity={k:values[k] for k in IDENTITY_FIELDS},source_assay_context=expected_context,source_assay_context_applicability=expected_mask)
    if _sha(candidate['conditions'])!=_sha(expected_conditions):raise ValueError('native identity/context mask or hidden-target predictor mismatch')
    if _sha(candidate['audit']['full_native_fields'])!=_sha(values):raise ValueError('native unknown/audit fields lost or changed')
    source=candidate['audit']['source_ref']
    if source.get('native_source_row')!=row['native_source_row'] or source.get('native_DRAMP_ID')!=row['source_native_DRAMP_ID']:raise ValueError('source row identity mismatched')
    if candidate['task_name']=='source_DRAMP_stability_report_prediction':
        if candidate['targets']!={'source_curated_report':text}:raise ValueError('native source report target changed')
    elif candidate['task_name']=='source_DRAMP_single_half_life_literal_prediction':
        if set(candidate['targets'])!={'source_reported_scalar'}:raise ValueError('unexpected scalar target labels')
        scalar=candidate['targets']['source_reported_scalar'];match=SCALAR.fullmatch(text) if isinstance(text,str) else None
        if match is None:raise ValueError('not a complete single source literal scalar')
        relation=match['relation'] or '=';want=dict(schema='mora3.NativeReportedHalfLifeScalarCandidate/v1',source_evidence_role='database_curated_report;primary_assay_verification_pending',source_literal=text,value_literal=match['value'],unit_literal=match['unit'],relation=relation,exact_value_observed=relation=='=',positive_value_candidate=Decimal(match['value'])>0,normalized_value=None,task_admitted=False,model_use=False)
        if _sha(scalar)!=_sha(want):raise ValueError('native source scalar/censor/unit or admission mask changed')
    else:raise ValueError('unreviewed source report task type')
