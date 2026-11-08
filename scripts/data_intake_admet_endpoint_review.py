"""Source-bound endpoint semantics without inferred physical-unit conversions.

The pinned ExpansionRx card defines eight literal units; LogD and efflux retain
source coordinates without invented pH, log base, physical unit or ratio formula.
This review never supplies chemical identity, source rights or split admission.
"""
from copy import deepcopy
from data_intake_corpus import canonical

_SCHEMA = {
 'LogD': dict(unit=None, line='* LogD', readout='source-native LogD coordinate', context={}, coordinate_only=True),
 'KSOL': dict(unit='uM', line='* Kinetic Solubility KSOL: uM', readout='kinetic solubility', context={}, coordinate_only=False),
 'MLM CLint': dict(unit='mL/min/kg', line='* Mouse Liver Microsomal (MLM) CLint: mL/min/kg', readout='source-reported microsomal intrinsic-clearance coordinate', context=dict(species='Mouse',tissue='Liver',assay_system='microsomal'), coordinate_only=False),
 'HLM CLint': dict(unit='mL/min/kg', line='* Human Liver Microsomal (HLM) Clint: mL/min/kg', readout='source-reported microsomal intrinsic-clearance coordinate', context=dict(species='Human',tissue='Liver',assay_system='microsomal'), coordinate_only=False),
 'RLM CLint': dict(unit='mL/min/kg', line='* An additional ednpoint **Rat Liver Microsomal (RLM) stability CLint (mL/min/kg)**, which was not part of the challenge, is included in the *raw* dataset.', readout='source-reported microsomal intrinsic-clearance coordinate', context=dict(species='Rat',tissue='Liver',assay_system='microsomal'), coordinate_only=False),
 'Caco-2 Permeability Papp A>B': dict(unit='10^-6 cm/s', line='* Caco-2 Papp A>B (10^-6 cm/s)', readout='apparent permeability A>B', context=dict(assay_system='Caco-2',assay_direction='A>B'), coordinate_only=False),
 'Caco-2 Permeability Efflux': dict(unit=None, line='* Caco-2 Efflux Ratio', readout='source-native Caco-2 efflux-ratio coordinate; ratio formula not supplied', context=dict(assay_system='Caco-2'), coordinate_only=True),
 'MPPB': dict(unit='% Unbound', line='* Mouse Plasma Protein Binding (MPPB): % Unbound', readout='percent unbound in mouse plasma-protein binding assay', context=dict(species='Mouse',matrix='Plasma'), coordinate_only=False),
 'MBPB': dict(unit='% Unbound', line='* Mouse Brain Protein Binding (MBPB): % Unbound', readout='percent unbound in mouse brain-protein binding assay', context=dict(species='Mouse',matrix='Brain'), coordinate_only=False),
 'MGMB': dict(unit='% Unbound', line='* Mouse Gastrocnemius Muscle Binding (MGMB): % Unbound', readout='percent unbound in mouse gastrocnemius-muscle binding assay', context=dict(species='Mouse',matrix='Gastrocnemius Muscle'), coordinate_only=False),
}

def review_expansion_endpoint(endpoint, *, card_text, card_ref):
    spec=deepcopy(_SCHEMA[endpoint])
    if spec['line'] not in card_text.splitlines():
        raise ValueError('exact pinned source-schema endpoint line is required')
    return dict(schema='mora3.ADMETSourceEndpointReview/v1', source_family='ExpansionRx',
        source_endpoint=endpoint, source_schema_ref=deepcopy(card_ref),
        exact_source_schema_line=spec['line'], source_native_record_literal_unit=None,
        unit_from_official_schema=spec['unit'], physical_unit_literal_in_source_record=False,
        schema_unit_defined=spec['unit'] is not None,
        source_native_coordinate_objective_supported=True,
        native_coordinate_only=spec['coordinate_only'],
        source_readout=spec['readout'], source_context=spec['context'],
        conversions_performed=[], clinical_or_in_vivo_PK_validation=False,
        pH=None, time=None, dose=None, study_or_replicate_context=None,
        biological_measurement_count=None,
        source_protocol_scope='RLM is raw-only extra endpoint; all parent-source test entities remain reserved' if endpoint=='RLM CLint' else 'native source assay readout; no independent protocol/replicate assertion',
        endpoint_definition_gate=True, task_units_or_declared_native_coordinate_gate=True,
        unresolved=['complete protocol, study and replicate details','dose/time/pH and source material/formulation where absent']
          + (['physical unit/transform not inferred; source native coordinate only'] if spec['coordinate_only'] else []))

def unresolved_asap_endpoint(endpoint, *, card_ref):
    if endpoint not in ('HLM','MLM','KSOL','LogD','MDR1-MDCKII'):
        raise ValueError('not a source ASAP endpoint')
    return dict(schema='mora3.ADMETSourceEndpointReview/v1',source_family='ASAP',source_endpoint=endpoint,
        source_schema_ref=deepcopy(card_ref),source_native_record_literal_unit=None,
        unit_from_official_schema=None,schema_unit_defined=False,
        source_native_coordinate_objective_supported=False,native_coordinate_only=True,
        source_readout=endpoint,source_context={},conversions_performed=[],
        endpoint_definition_gate=False,task_units_or_declared_native_coordinate_gate=False,
        unresolved=['exact original Bienta protocol and source tabular readout transform','source literal units and time/assay context'])

def apply_endpoint_review(record, review):
    """Return a new candidate; unreviewed identity/split/rights gates stay closed.

    Full old endpoint/provenance are retained in audit. Predictor endpoint and
    context contain semantic schema only, never source pointers/IDs/citations.
    """
    if record.get('schema')!='mora3.ADMETNativeObservation/v1':
        raise ValueError('native observation required')
    if record['endpoint']['source_endpoint']!=review['source_endpoint']:
        raise ValueError('endpoint review must match exact native endpoint')
    if record['source_ref']['native_entity_namespace']!=review['source_family']:
        raise ValueError('source-specific review cannot cross source families')
    out=deepcopy(record);out['original_endpoint_projection_audit']=deepcopy(record['endpoint'])
    out['endpoint_source_review_audit']=deepcopy(review)
    out['endpoint']=dict(endpoint_key=record['endpoint']['endpoint_key'],
        source_endpoint=review['source_endpoint'],readout=review['source_readout'],
        unit=review['unit_from_official_schema'],
        unit_basis='official_source_schema' if review['schema_unit_defined'] else 'unresolved',
        measurement_type='source-curated assay readout; not clinical PK or independent validation')
    out['known_context']=deepcopy(record['known_context'])
    for key,value in review['source_context'].items():
        if key in out['known_context'] and out['known_context'][key]!=value:
            raise ValueError('never overwrite contradictory source context')
        out['known_context'][key]=value
    if review['native_coordinate_only'] and review['source_native_coordinate_objective_supported']:
        out['known_context']['source_measurement_coordinate']=review['source_readout']
    out['admission']['endpoint_definition']=review['endpoint_definition_gate']
    out['admission']['task_units_or_declared_native_coordinate']=review['task_units_or_declared_native_coordinate_gate']
    canonical(out)
    return out
