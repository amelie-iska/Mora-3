"""Independent literal source checks for full RNA task/group master admission."""
import hashlib,json
from data_intake_corpus import CorpusTask,canonical
from data_intake_nucleotide_prior import sequence_groups


def verify_native_task(row,source,source_file,file_ordinal,native_row_index,state,index_row,identifiers,group_row):
    payload=row['native_payload'];digest=hashlib.sha256(payload).hexdigest()
    if digest!=row['native_payload_sha256']:raise ValueError('serialized payload hash mismatch')
    task=CorpusTask.from_bytes(payload)
    if task.canonical_bytes()!=payload:raise ValueError('noncanonical payload')
    d=json.loads(payload);seq=source['native_sequence'];native=hashlib.sha256(seq.encode()).hexdigest();g=sequence_groups(seq)
    if native!=source['native_sequence_sha256'] or len(seq)!=source['native_length'] or not source['sequence_alphabet_pass']:raise ValueError('native source target/hash/axis mismatch')
    expected=hashlib.sha256(canonical(['RNA_reference_native_symbols',None,native])).hexdigest()
    if row['task_id']!=expected or state['task_id']!=expected or state['native_sha'].hex()!=native or state['reuse_kind']!='new':raise ValueError('new prior native identity mismatch')
    if index_row['source_file']!=file_ordinal or index_row['source_row']!=native_row_index or index_row['native_sha'].hex()!=native or index_row['reuse_kind']!='new':raise ValueError('serialization location/native identity mismatch')
    if state['first_file']!=file_ordinal or state['first_row']!=native_row_index:raise ValueError('task is not bound to first native occurrence')
    conditions={'prior_domain':'RNA_reference_native_symbols','source_organism':None}
    if d['task']!='source_native_nucleotide_sequence_prior' or d['conditions']!=conditions or task.input_bytes()!=canonical(dict(task='source_native_nucleotide_sequence_prior',conditions=conditions)):raise ValueError('target or source labels in predictor conditions')
    if d['targets']!={'source_native_sequence':seq} or row['full_axis_values']!=len(seq):raise ValueError('target crop/change or fabricated modality')
    sr=d['audit']['source_ref'];expected_source=dict(native_payload=source_file['path'],native_payload_sha256=source_file['sha256'],native_row_index=native_row_index,original_record_index=source['source_record_index'],original_FASTA=source['source_payload'],source_identifier=source['source_identifier'],native_header=source['native_header'],source_status=source_file['component'],source_release='RNAcentral28.0',source_rights='CC0-1.0',original_source_record_id=source['source_record_id'])
    if sr!=expected_source:raise ValueError('literal original source provenance mismatch')
    expected_groups=sorted({'nucleotide_native:'+native,'nucleotide_upper:'+g['uppercase_sha256'],'nucleotide_derived_RC_baseorder:'+g['reverse_complement_base_order_group']}|{'rnacentral_native_identifier:'+x for x in identifiers})
    if source['source_identifier'] not in identifiers or group_row['task_id']!=expected or json.loads(group_row['effective_groups_json'])!=expected_groups or json.loads(row['group_ids_json'])!=expected_groups:raise ValueError('missing/phantom native identifier or grouping context')
    if row['record_id']!='RNAcentral28-native-sequence:'+native or row['source_family']!='RNAcentral-reference-priors' or row['split']!='train' or not row['admitted_for_task'] or row['model_adapter_supported'] is not False or row['native_schema']!='mora3.CorpusTask/v1':raise ValueError('source role, allocation or readiness metadata mismatch')
    receipt=json.loads(row['admission_receipt_json'])
    if receipt['native_sequence_sha256']!=native or receipt['source_ref']!=expected_source or receipt['new_independent_experiments']!=0 or receipt['source_status_is_not_activity_label'] is not True:raise ValueError('source evidence/count semantics mismatch')
    return dict(task_id=expected,native_sha=bytes.fromhex(native),native_length=len(seq),predictor_only=conditions)


def overlay_literal_groups(old_row,additional):
    """Create a new sidecar row; preserve every original non-group field."""
    result=dict(old_row);result['effective_groups_json']=json.dumps(sorted(set(json.loads(old_row['effective_groups_json']))|set(additional)))
    return result


def overlay_group_table_preserving_columns(table,updates):
    """Replace only the group text array; keep source schema/other arrays intact."""
    import pyarrow as pa
    column=table.schema.get_field_index('effective_groups_json')
    if column<0 or 'task_id' not in table.column_names:raise ValueError('native sidecar schema required')
    ids=table.column('task_id').to_pylist();old=table.column(column).to_pylist();out=[];changed=0
    for task_id,groups in zip(ids,old):
        if task_id in updates:
            out.append(json.dumps(sorted(set(json.loads(groups))|set(updates[task_id]))));changed+=1
        else:out.append(groups)
    result=table.set_column(column,table.schema.field(column),pa.array(out,type=table.schema.field(column).type))
    return result,changed
