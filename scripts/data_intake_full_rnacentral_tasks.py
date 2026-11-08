"""Whole-source native RNA task envelopes; no old tasks or source bytes change."""
import hashlib,json
from data_intake_full_rnacentral_state import task_id_from_native_digest
from data_intake_nucleotide_prior import sequence_groups
from data_intake_rnacentral_prior import prior_task


def native_effective_groups(state,identifiers):
    if not identifiers or any(not isinstance(x,str) or not x for x in identifiers):
        raise ValueError('complete literal native identifiers required')
    return sorted({'nucleotide_native:'+state['native_sha'].hex(),
        'nucleotide_upper:'+state['uppercase_sha'].hex(),
        'nucleotide_derived_RC_baseorder:'+state['rc_sha'].hex()}
        | {'rnacentral_native_identifier:'+x for x in identifiers})


def full_native_prior_envelope(sequence,state,source_ref,identifiers):
    if state['reuse_kind']!='new':
        raise ValueError('only a new exact native prior may serialize a new task')
    groups=sequence_groups(sequence)
    if bytes.fromhex(groups['native_sha256'])!=state['native_sha'] or bytes.fromhex(groups['uppercase_sha256'])!=state['uppercase_sha'] or bytes.fromhex(groups['reverse_complement_base_order_group'])!=state['rc_sha']:
        raise ValueError('native identity or derived grouping guard mismatch')
    expected=task_id_from_native_digest(state['native_sha'],'new')
    if expected!=state['task_id']:
        raise ValueError('native prior identity recipe mismatch')
    if source_ref['source_identifier'] not in identifiers:
        raise ValueError('original native identifier absent from complete provenance groups')
    effective=native_effective_groups(state,identifiers);task=prior_task(sequence,source_ref=source_ref);payload=task.canonical_bytes()
    receipt=dict(source_ref=source_ref,native_sequence_sha256=groups['native_sha256'],exact_full_axis=True,source_case_T_U_and_ambiguity_preserved=True,source_license='CC0-1.0',source_status_is_not_activity_label=True,new_independent_experiments=0,task_admission='eligible native reference prior; full independent master proof pending',record_ID_scope='literal native sequence identity; original source record ID and all occurrences retained separately')
    return dict(task_id=expected,record_id='RNAcentral28-native-sequence:'+groups['native_sha256'],source_family='RNAcentral-reference-priors',source_revision='officialRNAcentral28.0active+inactive',source_sha256=source_ref['native_payload_sha256'],task_name='source_native_nucleotide_sequence_prior',native_schema='mora3.CorpusTask/v1',native_payload=payload,native_payload_sha256=hashlib.sha256(payload).hexdigest(),split='train',split_policy='referenceprior_trainonly;knownheldoutguards;homology_and_genuine_contextheldout_unresolved',group_ids_json=json.dumps(effective),evidence_class='releasedRNAreference_sequenceprior;notnewindependentexperiment',admitted_for_task=True,model_adapter_supported=False,admission_receipt_json=json.dumps(receipt,sort_keys=True),missing_context_json=json.dumps(json.loads(payload)['missing']),full_axis_values=len(sequence),condition_context_json=json.dumps(dict(source_status_notactivitytruth=True,sourceIDsheadersauditonly=True,all_unlisted_annotations_unknown=True,physical_material_modification_context_known=False,source_symbol_knownmask_rule='biological exact-base objective only where native.upper in ACGTU; source ambiguity and case remain observed native symbols; no imputation')))


def native_symbol_units(sequence):
    """Cheap raw character accounting in the existing pass, never model tokens."""
    upper=sequence.upper();known=sum(upper.count(base) for base in 'ACGTU')
    return dict(new_distinct_target_native_symbols=len(sequence),
        biological_ACGTU_positions=known,ambiguity_positions=len(sequence)-known,
        native_T_symbols=upper.count('T'),native_U_symbols=upper.count('U'),
        new_native_sequences_longer_than8192=int(len(sequence)>8192))
