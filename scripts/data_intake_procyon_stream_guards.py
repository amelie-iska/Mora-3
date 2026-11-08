"""Streaming ProCyon audit integrity and named hardware guard diagnostics."""
import hashlib,json
from decimal import Decimal


def threshold_failures(sample,limits):
    failures=[]
    for key in ['elapsed_seconds','own_peak_RSS_bytes','aggregate_RSS_with_reserve_bytes']:
        if sample[key]>limits[key]:failures.append(key)
    if sample['free_hardware_bytes']<limits['free_hardware_bytes']:failures.append('free_hardware_bytes')
    return failures


def validate_entity_row(row):
    """Validate completed prefix scalar proof fields without loading full tables."""
    if row['source_namespace']=='protein':
        q=json.loads(row['qualification_json'])
        if q['native_ID']!=row['native_ID'] or q['native_sequence_SHA256']!=row['native_sequence_SHA256'] or q['native_length']!=row['native_length']:
            raise ValueError('completed protein identity prefix metadata mismatch')
        if q['exact_accession_and_full_sequence_qualified']:
            if q['baseline_accession']!=row['native_ID'] or q['baseline_sequence_SHA256']!=row['native_sequence_SHA256'] or q['baseline_reviewed']!='reviewed' or Decimal(q['baseline_annotation_score'])<4:
                raise ValueError('exact current accession/sequence/annotation qualification mismatch')
        return q['status']
    if row['source_namespace']=='domain':
        q=json.loads(row['domain_parent_candidate_json'])
        if q['native_domain_ID']!=row['native_ID'] or q['native_parent_or_coordinate_metadata_verified'] or q['annotation_score_or_function_labels_inherited'] or q['whole_canonical_protein_qualification'] or q['domain_task_admitted']:
            raise ValueError('unproven domain prefix or span promoted to native qualification')
        for start,end in q['derived_literal_substring_spans_1based_closed']:
            if start<1 or end-start+1!=row['native_length']:
                raise ValueError('derived substring span does not preserve native full axis')
        return 'domain_parent_coordinate_qualification_not_admitted'
    raise ValueError('unknown native entity namespace')


def stream_row_digest(batches):
    """All fields, including rare optional columns, enter the same row digest."""
    h=hashlib.sha256();count=0
    for batch in batches:
        for row in batch.to_pylist():
            raw=json.dumps(row,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
            h.update(len(raw).to_bytes(8,'big'));h.update(raw);count+=1
    return count,h.hexdigest()
