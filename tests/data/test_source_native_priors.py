"""Synthetic source-binding tests; fixtures confer no scientific admission."""
from dataclasses import replace
import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mora3.data.finite_union_native_rows import FiniteUnionNativeView, full9
from mora3.data.source_native_priors import source_native_prior_views


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()


CASES = (
    ('DNA-native-reference-prior', 'source_native_nucleotide_sequence_prior',
     '4e6860536e67a2799c206bb66b0ccce4931e78f1', 'dna', 'DNA',
     {'prior_domain': 'source_DNA_sequence', 'source_organism': 'literal organism'},
     'source_native_sequence', 'aCgT UNx\n'),
    ('Rfam15.1-native-reference-prior', 'source_native_nucleotide_sequence_prior',
     'Rfam15.1;January2026;originalCC0FASTA', 'rfam_rna', 'RNA',
     {'prior_domain': 'RNA_family_reference_native_symbols', 'source_organism': None},
     'source_native_sequence', 'aCuGRYN\t'),
    ('RNAcentral-reference-priors', 'source_native_nucleotide_sequence_prior',
     'officialRNAcentral28.0active+inactive', 'rnacentral_rna', 'RNA',
     {'prior_domain': 'RNA_reference_native_symbols', 'source_organism': None},
     'source_native_sequence', 'uTrYN\r\n'),
    ('UniProt', 'protein_sequence_modeling', 'UniProt2026_03', 'protein', 'protein',
     {}, 'amino_acid_sequence', 'MABxZU O\n'),
    ('PubChem-native-SMILES-reference-prior', 'source_native_molecular_SMILES_reference_prior',
     'f83219601635a0a80fc99c13a9ca37f99ef34f0a', 'native_smiles', 'native_SMILES',
     {'prior_domain': 'molecular_reference_native_SMILES'}, 'native_smiles', '[13CH3:4].[Na+]\n'),
    ('PubChem-standardSELFIES-codec', 'computed_standard_SELFIES_smiles_to_selfies',
     'f83219601635a0a80fc99c13a9ca37f99ef34f0a', 'smiles_to_selfies', 'standard_SELFIES',
     {'representation_task': 'standard_SELFIES_codec', 'native_smiles': ' C[C@H](O)Cl\n'},
     'standard_selfies', '[C][C@H1][Branch1][O][Cl]'),
    ('PubChem-standardSELFIES-codec', 'computed_standard_SELFIES_selfies_to_smiles',
     'f83219601635a0a80fc99c13a9ca37f99ef34f0a', 'selfies_to_smiles', 'native_SMILES',
     {'representation_task': 'standard_SELFIES_codec', 'standard_selfies': '[C][O]'},
     'source_canonical_isomeric_smiles', 'CO'),
)


def row_for(case):
    family, task, revision, _, _, conditions, target, text = case
    payload = canonical(dict(schema='mora3.CorpusTask/v1', task=task, conditions=conditions,
        targets={target: text}, audit={'native_reader_supported': False, 'source_ref': {'native': 'fixture'}},
        missing=['paired_assay']))
    return dict(task_id='fixture-task', record_id='fixture-record', source_family=family,
        source_revision=revision, source_sha256='a'*64, task_name=task,
        native_schema='mora3.CorpusTask/v1', native_payload=payload,
        native_payload_sha256=hashlib.sha256(payload).hexdigest(), split='train',
        split_policy='source policy preserved', group_ids_json='["fixture-native-guard"]',
        evidence_class='source evidence preserved', admitted_for_task=True,
        model_adapter_supported=False, admission_receipt_json='{"source_license":"fixture-only"}',
        missing_context_json='["paired_assay"]', full_axis_values=len(text), condition_context_json='{}')


def make_view(root, rows, codes=None):
    codes = bytes([0]*len(rows) if codes is None else codes)
    def put(name, raw):
        (root/name).write_bytes(raw)
        return dict(path=name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    pq.write_table(pa.Table.from_pylist(rows), root/'source.parquet', row_group_size=2)
    source = put('source-reference-copy', (root/'source.parquet').read_bytes())
    source['path'] = 'source.parquet'
    source['rows'] = len(rows)
    d = dict(source_shard=source, source_full9=list(full9(root/'source.parquet')),
        original_group_sidecar=put('guards.bin', b'immutable fixture guards'),
        primary_retained_rows=sum(c <= 2 for c in codes), disposition=dict(
            kind='original-row-ordinal-uint8', rows=len(rows), reference=put('codes.u8', codes)))
    ds = put('descriptors.jsonl', canonical(d)+b'\n')
    m = dict(schema='Mora3.ImmutableFiniteEligibleNativeUnion/v1', full_project_training_ready=False,
        canonical_master_pointer_unchanged=True, original_immutable_source_views=ds,
        source_taskviews=len(rows), primary_retained_taskviews=sum(c <= 2 for c in codes))
    ref = put('manifest.json', canonical(m))
    return FiniteUnionNativeView(root/ref['path'], ref['sha256'], root=root)


def extract(root, rows, codes=None):
    view = make_view(root, rows, codes)
    batch = next(view.iter_native_task_batches(split='train', batch_size=32))
    return view, batch, source_native_prior_views(view, batch, split='train')


@pytest.mark.parametrize('case', CASES, ids=[c[3] for c in CASES])
def test_full_native_strings_conditions_and_semantics(tmp_path, case):
    view, batch, (result,) = extract(tmp_path, [row_for(case)])
    assert result.route == case[3]
    assert result.datatype == case[4]
    assert result.target_field == case[6]
    assert result.target_bytes == case[7].encode('utf8')
    expected = case[5] if case[3] != 'protein' else {'prior_domain': 'protein_reference_native_amino_acids'}
    assert result.condition_inputs() == expected
    assert result.inputs() == expected
    p = result.provenance()
    assert p['original_source_row_ordinal'] == 0
    assert p['original_split'] == 'train'
    assert p['full_project_training_ready'] is False
    assert p['original_envelope']['model_adapter_supported'] is False
    assert p['original_task']['audit']['native_reader_supported'] is False
    assert not {'task_id', 'record_id', 'source_family', 'target_length', 'target_sha256'} & set(result.condition_inputs())
    assert result.target_bytes.decode('utf8') == case[7]


def test_exact_noncontiguous_original_rows_and_omitted_mask(tmp_path):
    rows = [row_for(CASES[0]) for _ in range(3)]
    _, _, results = extract(tmp_path, rows, [0, 6, 0])
    assert [r.provenance()['original_source_row_ordinal'] for r in results] == [0, 2]


def test_mutated_condition_does_not_mutate_frozen_result(tmp_path):
    _, _, (result,) = extract(tmp_path, [row_for(CASES[0])])
    conditions = result.condition_inputs(); conditions['source_organism'] = 'forged'
    assert result.condition_inputs()['source_organism'] == 'literal organism'


def test_forged_native_target_and_source_coordinate_refuse(tmp_path):
    view, batch, _ = extract(tmp_path, [row_for(CASES[0]), row_for(CASES[0])])
    changed = batch.native_rows.set_column(batch.native_rows.schema.get_field_index('native_payload'),
        'native_payload', pa.array([b'{}', b'{}']))
    for b in (replace(batch, native_rows=changed), replace(batch, source_row_ordinals=(0, 0))):
        with pytest.raises(ValueError): source_native_prior_views(view, b, split='train')


def test_projected_batch_cannot_authenticate_full_source_task(tmp_path):
    view = make_view(tmp_path, [row_for(CASES[0])])
    batch = next(view.iter_native_batches(split='train', columns=['native_payload']))
    with pytest.raises(ValueError): source_native_prior_views(view, batch, split='train')


def test_no_other_role_or_unbound_reader_can_promote_prior(tmp_path):
    view, batch, _ = extract(tmp_path, [row_for(CASES[0])])
    for split in ('validation', 'test', 'heldout', None):
        with pytest.raises(ValueError): source_native_prior_views(view, batch, split=split)
    with pytest.raises(ValueError): source_native_prior_views(object(), batch, split='train')


@pytest.mark.parametrize('mutation', ['family', 'task', 'revision', 'schema', 'admission', 'payload_sha',
    'condition_leak', 'target_extra', 'target_empty', 'target_number', 'missing_mismatch', 'duplicate_key',
    'nonfinite', 'organism_number', 'domain', 'codec_condition', 'group_duplicate'])
def test_authentic_native_rows_still_require_explicit_route_semantics(tmp_path, mutation):
    row = row_for(CASES[0]); payload = json.loads(row['native_payload'])
    if mutation == 'family': row['source_family'] = 'invented'
    elif mutation == 'task': row['task_name'] = 'invented'
    elif mutation == 'revision': row['source_revision'] = 'unknown'
    elif mutation == 'schema': payload['schema'] = 'unknown'
    elif mutation == 'admission': row['admitted_for_task'] = False
    elif mutation == 'payload_sha': row['native_payload_sha256'] = 'b'*64
    elif mutation == 'condition_leak': payload['conditions']['target_length'] = 42
    elif mutation == 'target_extra': payload['targets']['gold'] = 'extra'
    elif mutation == 'target_empty': payload['targets']['source_native_sequence'] = ''
    elif mutation == 'target_number': payload['targets']['source_native_sequence'] = 1
    elif mutation == 'missing_mismatch': row['missing_context_json'] = '[]'
    elif mutation == 'organism_number': payload['conditions']['source_organism'] = 3
    elif mutation == 'domain': payload['conditions']['prior_domain'] = 'invented'
    elif mutation == 'codec_condition': payload['conditions']['native_smiles'] = 'CO'
    elif mutation == 'group_duplicate': row['group_ids_json'] = '["a","a"]'
    raw = canonical(payload)
    if mutation == 'duplicate_key': raw = raw[:-1] + b',"conditions":{}}'
    if mutation == 'nonfinite': raw = raw[:-1] + b',"extra":NaN}'
    row['native_payload'] = raw
    if mutation != 'payload_sha': row['native_payload_sha256'] = hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError): extract(tmp_path, [row])


def test_original_source_or_guard_change_refuses_consumption(tmp_path):
    view, batch, _ = extract(tmp_path, [row_for(CASES[0])])
    (tmp_path/'guards.bin').write_bytes(b'changed')
    with pytest.raises(ValueError): source_native_prior_views(view, batch, split='train')
