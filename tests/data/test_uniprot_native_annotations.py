"""Exact-source typed UniProt adapter checks on synthetic native rows."""
from dataclasses import replace
import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mora3.data.finite_union_native_rows import FiniteUnionNativeView, full9
from mora3.data import uniprot_native_annotations as native


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def inverse():
    sequence = 'ACDEFGHIKLMNPQRSTVWYBOUXZ'
    ref = dict(source_release='2026_03', entry_version=90, sequence_version=3,
               raw_file='source-page', raw_file_sha256='a'*64, results_index=10)
    identity = dict(primary_accession='fixture-accession', sequence_sha256=digest(sequence.encode()),
                    sequence_length=len(sequence), annotation_score=5, entry_version=90, sequence_version=3)
    conditions = {
        'organism_taxon_id': 9606,
        'source_experimentally_supported_function_texts': ['β text\twith original whitespace\n'],
        'source_experimental_associated_GO_ids': ['GO:0000002'],
        'source_explicit_experimental_NOT_GO_ids': [],
        'source_experimentally_supported_EC_ids': ['1.1.1.1', '1.1.1.-', '2.3.1.n6'],
        'source_kinetic_conditions': [
            dict(parameter_type='MichaelisConstant', source_value=0.0, source_unit='µM', substrate_label='x\n',
                 assay_pH=None, assay_temperature=None),
            dict(parameter_type='maximumVelocity', source_value=2100000.0, source_unit='native units',
                 substrate_label=None, assay_pH=None, assay_temperature=None),
        ],
    }
    task = dict(schema='mora3.CorpusTask/v1', task=native.UNIPROT_NATIVE_INVERSE_TASK,
                conditions=conditions, targets={'canonical_sequence': sequence},
                audit=dict(source_ref=ref, identity=identity, hidden_target_length=len(sequence)),
                missing=['unknown context'])
    receipt = dict(source_ref=ref, reviewed=True, annotation_score=5, license='CC-BY-4.0',
                   scope_safe_generation_v2=True, canonical_sequence_exact_baseline=True)
    return row(task, receipt, 'UniProt-full-record-annotations', 'UniProt2026_03;entryVersion90;sequenceVersion3',
               ref['raw_file_sha256'], len(sequence))


def forward():
    sequence = 'ACDEFGHIKLMNPQRSTVWYBOUXZ'
    original = {'Reviewed': 'reviewed', 'Entry': 'fixture-accession', 'Sequence': sequence,
        'Organism (ID)': '9606', 'Function [CC]': '  μ physiology\r\n',
        'Catalytic activity': '', 'Protein names': 'Native β-name\t',
        'Gene Ontology IDs': 'GO:0000002; GO:0000001; GO:0000002'}
    task = dict(schema='mora3.CorpusTask/v1', task=native.UNIPROT_NATIVE_FORWARD_TASK,
        conditions=dict(amino_acid_sequence=sequence,organism_taxon=9606),
        targets=dict(source_curated_function=original['Function [CC]'],
                     source_protein_name=original['Protein names'],
                     source_GO_identifiers=['GO:0000002','GO:0000001','GO:0000002']),
        audit=dict(original_source_row=original),missing=['catalytic_activity_annotation'])
    receipt = dict(reviewed=True, annotation_score=5, release='2026_03', exact_old_sequence_match=True,
                   source_only_annotation_scope=True)
    return row(task,receipt,'UniProt-function-annotations','UniProt2026_03',native._FORWARD_EXPORT_SHA256,len(sequence))


def row(task, receipt, family, revision, source_sha, length):
    payload = canonical(task)
    return dict(task_id='synthetic-native-task', record_id='fixture-accession',
        source_family=family, source_revision=revision, source_sha256=source_sha,
        task_name=task['task'], native_schema=task['schema'], native_payload=payload,
        native_payload_sha256=digest(payload), split='train', admitted_for_task=True,
        model_adapter_supported=False, admission_receipt_json=canonical(receipt).decode(),
        missing_context_json=canonical(task['missing']).decode(), full_axis_values=length)


def revise(original, *, section, field, value):
    task = json.loads(original['native_payload'])
    task[section][field] = value
    payload = canonical(task)
    return {**original,'native_payload':payload,'native_payload_sha256':digest(payload)}


def fixture_view(tmp_path, original, *, guarded=False, attach_scope=True):
    def save(name, content):
        (tmp_path/name).write_bytes(content)
        return dict(path=name,bytes=len(content),sha256=digest(content))

    source_path = tmp_path/'source.parquet'
    pq.write_table(pa.Table.from_pylist([original]),source_path)
    raw = source_path.read_bytes()
    descriptor = dict(source_shard=dict(path='source.parquet',bytes=len(raw),sha256=digest(raw),rows=1),
        source_full9=list(full9(source_path)),original_group_sidecar=save('groups.bin',b'fixture guard'),
        disposition=dict(kind='homogeneous-original-row-disposition',rows=1,code=0),
        primary_retained_rows=1)
    reference = save('descriptors.jsonl',canonical(descriptor)+b'\n')
    manifest = save('manifest.json',canonical(dict(schema='Mora3.ImmutableFiniteEligibleNativeUnion/v1',
        original_immutable_source_views=reference,full_project_training_ready=False,
        canonical_master_pointer_unchanged=True,source_taskviews=1,primary_retained_taskviews=1)))
    view = FiniteUnionNativeView(tmp_path/manifest['path'],manifest['sha256'],root=tmp_path)
    if attach_scope:
        count = 228 if guarded else 227
        view._ordinal_exclusions = frozenset(('fixture-other-file',i) for i in range(count))
        view.effective_filter = dict(combined_excluded_taskviews=count,source_role='train')
        if guarded:
            view.effective_filter.update(schema='Mora3.FiniteNativeInverseLiteralTargetOrdinalFilter/v1',
                parent_effective_filter={'sha256':native._EFFECTIVE_V2_SHA256},
                source_inverse_literal_guard_proposal={'sha256':native._INVERSE_GUARD_PROPOSAL_SHA256})
        else:
            view.effective_filter.update(schema='Mora3.AdditionalFiniteNativeStudyOrdinalFilter/v1',
                source_task_consumption_manifest={'sha256':native._EFFECTIVE_V2_EVIDENCE_SHA256})
    batch = next(view.iter_native_task_batches(split='train'))
    return view,batch


@pytest.mark.parametrize('guarded',[False,True])
def test_inverse_native_conditions_target_and_audit_separation(tmp_path, guarded):
    original = inverse()
    view,batch = fixture_view(tmp_path,original,guarded=guarded)
    result, = native.uniprot_native_inverse_rows(view,batch)
    payload = json.loads(original['native_payload'])
    fields = result.model_condition_fields()
    assert result.canonical_sequence == payload['targets']['canonical_sequence']
    assert set(fields) == set(payload['conditions'])
    assert fields['source_experimentally_supported_function_texts'] == ('β text\twith original whitespace\n',)
    assert fields['source_experimentally_supported_EC_ids'] == ('1.1.1.1','1.1.1.-','2.3.1.n6')
    assert fields['source_explicit_experimental_NOT_GO_ids'] == ()
    assert fields['source_kinetic_conditions'] == tuple(payload['conditions']['source_kinetic_conditions'])
    fields['source_kinetic_conditions'][0]['source_value'] = 9
    assert result.source_kinetic_conditions[0].source_value == 0.0
    assert result.origin.source_row_ordinal == 0
    assert result.origin.native_payload_sha256 == original['native_payload_sha256']
    assert not {'origin','canonical_sequence','hidden_target_length','accession'} & set(fields)


def test_forward_literal_segments_missingness_and_ordered_go_framing(tmp_path):
    view,batch = fixture_view(tmp_path,forward())
    result, = native.uniprot_native_forward_rows(view,batch)
    assert set(result.model_condition_fields()) == {'amino_acid_sequence','organism_taxon'}
    assert result.target_bytes('source_curated_function') == '  μ physiology\r\n'.encode()
    assert result.target_bytes('source_curated_catalytic_activity') is None
    assert result.target_bytes('source_protein_name') == 'Native β-name\t'.encode()
    raw = result.target_bytes('source_GO_identifiers')
    assert tuple(raw[i:i+10].decode('ascii') for i in range(0,len(raw),10)) == result.source_GO_identifiers
    assert result.source_GO_identifiers == ('GO:0000002','GO:0000001','GO:0000002')
    with pytest.raises(ValueError,match='unknown native'):
        result.target_bytes('original_source_row')


def test_inverse_literal_target_fragment_candidate_refused(tmp_path):
    original = {**inverse(),'task_id':native._INVERSE_GUARD_TASK}
    view,batch = fixture_view(tmp_path,original)
    with pytest.raises(ValueError,match='literal-target-fragment'):
        native.uniprot_native_inverse_rows(view,batch)


@pytest.mark.parametrize('field,value',[
    ('organism_taxon_id',True),
    ('source_experimentally_supported_EC_ids',['2.3.n6.1']),
    ('source_experimental_associated_GO_ids',['GO:123']),
    ('source_kinetic_conditions',[dict(parameter_type='maximumVelocity',source_value=True,
        source_unit='unit',substrate_label=None,assay_pH=None,assay_temperature=None)]),
    ('source_kinetic_conditions',[dict(parameter_type='MichaelisConstant',source_value=1.0,
        source_unit='unit',substrate_label='s',assay_pH=7,assay_temperature=None)]),
    ('target_length',23),
])
def test_inverse_unknown_or_target_derived_fields_refused(tmp_path,field,value):
    view,batch = fixture_view(tmp_path,revise(inverse(),section='conditions',field=field,value=value))
    with pytest.raises(ValueError):
        native.uniprot_native_inverse_rows(view,batch)


def test_forward_modified_literal_original_bytes_refused(tmp_path):
    modified = revise(forward(),section='targets',field='source_curated_function',value='μ physiology')
    view,batch = fixture_view(tmp_path,modified)
    with pytest.raises(ValueError,match='original UniProt forward field'):
        native.uniprot_native_forward_rows(view,batch)


def test_native_view_required_and_forged_batch_rejected(tmp_path):
    view,batch = fixture_view(tmp_path,forward())
    with pytest.raises(ValueError,match='original train'):
        native.uniprot_native_forward_rows(view,batch,split='test')
    with pytest.raises(ValueError):
        native.uniprot_native_inverse_rows(view,batch)
    changed = batch.native_rows.set_column(batch.native_rows.schema.get_field_index('native_payload'),
        'native_payload',pa.array([b'{}']))
    with pytest.raises(ValueError):
        native.uniprot_native_forward_rows(view,replace(batch,native_rows=changed))
    view.effective_filter['combined_excluded_taskviews'] = 0
    with pytest.raises(ValueError,match='source exclusions'):
        native.uniprot_native_forward_rows(view,batch)


def test_unfiltered_finite_source_view_cannot_use_typed_routes(tmp_path):
    view,batch = fixture_view(tmp_path,forward(),attach_scope=False)
    with pytest.raises(ValueError,match='native source filter'):
        native.uniprot_native_forward_rows(view,batch)
