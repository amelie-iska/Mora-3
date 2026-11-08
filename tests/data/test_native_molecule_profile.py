import copy
import hashlib
import json

import pytest

from mora3.data.molecule_profile import molecule_profile_view


def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def fixture(direction='molecule_to_profile', modality='L1000'):
    n = 977 if modality == 'L1000' else 1783
    array = dict(name='profile', axis_identity='source_order', axis_ids=[str(i) for i in range(n)],
                 values=[.25] * n, status='observed', units='source_normalized_or_image_computed_feature_units')
    conditions = dict(assay_modality=modality, cell_line=None)
    if direction == 'molecule_to_profile':
        conditions.update(canonical_isomeric_smiles='CCO', known_profile_axis=array['axis_ids'])
        targets = dict(source_profile_array=array)
    else:
        conditions['source_profile_array'] = array
        targets = dict(canonical_isomeric_smiles='CCO')
    task = dict(schema='mora3.CorpusTask/v1', task='source_observational_' + direction,
                conditions=conditions, targets=targets, audit={}, missing=['cell_line','physical_dose','exposure_time'])
    row = dict(task_id='task', record_id='record', source_family='Rosetta-CDRP-molecule-profile-' + modality,
               source_revision='pinned', source_sha256='a'*64, task_name=direction,
               native_schema=task['schema'], split='train', admitted_for_task=True,
               full_axis_values=n, original_array_sha256=hashlib.sha256(canonical(array)).hexdigest(),
               admission_receipt_json=json.dumps(dict(source_rights='CPG0003CC0', full_feature_axis=True, no_same_cell_join=True)))
    return seal(row, task), task


def seal(row, task):
    row = copy.deepcopy(row)
    row['native_payload'] = canonical(task)
    row['native_payload_sha256'] = hashlib.sha256(row['native_payload']).hexdigest()
    return row


@pytest.mark.parametrize('direction', ['molecule_to_profile', 'profile_to_molecule'])
@pytest.mark.parametrize('modality', ['CellPainting', 'L1000'])
def test_full_axis_and_target_audit_exclusion(direction, modality):
    row, task = fixture(direction, modality)
    v = molecule_profile_view(row, effective_groups=('group',))
    task['audit'] = {'source_id': 'changed', 'MOA': 'hidden'}
    if direction == 'molecule_to_profile':
        task['targets']['source_profile_array']['values'][0] = 9
        row['original_array_sha256'] = hashlib.sha256(canonical(task['targets']['source_profile_array'])).hexdigest()
    else:
        task['targets']['canonical_isomeric_smiles'] = 'CCN'
    other = molecule_profile_view(seal(row, task), effective_groups=('different',))
    assert v.inputs() == other.inputs()
    assert v.binding_sha256 != other.binding_sha256
    assert len(v.axis) == (1783 if modality == 'CellPainting' else 977)
    assert v.provenance()['heldout_eligible'] is False
    assert v.provenance()['image_available'] is False
    mutable = v.inputs(); mutable['cell_line'] = 'invented'
    assert v.inputs()['cell_line'] is None


@pytest.mark.parametrize('failure', ['hash', 'admission', 'split', 'axis', 'nan', 'leak', 'groups', 'array_hash'])
def test_rejects_unsafe_or_corrupt_tasks(failure):
    row, task = fixture()
    groups = ('group',)
    if failure == 'hash': row['native_payload_sha256'] = 'bad'
    elif failure == 'admission': row['admitted_for_task'] = False
    elif failure == 'split': row['split'] = 'test'
    elif failure == 'axis':
        task['conditions']['known_profile_axis'].reverse(); row = seal(row, task)
    elif failure == 'nan':
        # JSON can encode NaN, but the reader must reject it before tensorization.
        task['targets']['source_profile_array']['values'][0] = float('nan')
        row['native_payload'] = json.dumps(task).encode()
        row['native_payload_sha256'] = hashlib.sha256(row['native_payload']).hexdigest()
    elif failure == 'leak':
        task['conditions']['MOA'] = 'hidden'; row = seal(row, task)
    elif failure == 'groups': groups = ()
    elif failure == 'array_hash': row['original_array_sha256'] = 'bad'
    with pytest.raises(ValueError): molecule_profile_view(row, effective_groups=groups)


def corpus_fixture(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    row, _ = fixture()
    pq.write_table(pa.Table.from_pylist([row]), tmp_path / 'shard.parquet')
    pq.write_table(pa.Table.from_pylist([dict(task_id='task', effective_groups_json='["group"]')]), tmp_path / 'groups.parquet')
    def receipt(name):
        raw = (tmp_path / name).read_bytes()
        return dict(path=name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), rows=1)
    shard = dict(receipt('shard.parquet'), source_family=row['source_family'], split='train')
    manifest = dict(shards=[shard], effective_group_lookup={'shard.parquet': receipt('groups.parquet')})
    raw = canonical(manifest); (tmp_path / 'manifest.json').write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize('failure', [None, 'manifest', 'shard', 'sidecar', 'identity'])
def test_bounded_corpus_reader_verifies_envelopes_and_effective_groups(tmp_path, failure):
    from mora3.data.molecule_profile import read_molecule_profile_views
    digest = corpus_fixture(tmp_path)
    if failure in ('manifest', 'shard', 'sidecar'):
        name = {'manifest':'manifest.json', 'shard':'shard.parquet', 'sidecar':'groups.parquet'}[failure]
        with (tmp_path / name).open('ab') as f: f.write(b'corrupt')
    elif failure == 'identity':
        import pyarrow as pa
        import pyarrow.parquet as pq
        path = tmp_path / 'groups.parquet'
        pq.write_table(pa.Table.from_pylist([dict(task_id='other', effective_groups_json='["group"]')]), path)
        m = json.loads((tmp_path / 'manifest.json').read_bytes());raw=path.read_bytes()
        m['effective_group_lookup']['shard.parquet'].update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        raw=canonical(m);(tmp_path / 'manifest.json').write_bytes(raw);digest=hashlib.sha256(raw).hexdigest()
    kwargs = dict(root=tmp_path, expected_manifest_sha256=digest, modality='L1000', direction='molecule_to_profile', limit=1)
    if failure:
        with pytest.raises(ValueError): read_molecule_profile_views('manifest.json', **kwargs)
    else:
        views = read_molecule_profile_views('manifest.json', **kwargs)
        assert len(views) == 1 and views[0].provenance()['effective_groups'] == ['group']
        assert views[0].provenance()['manifest_sha256'] == digest
        for limit in (0, 257, True):
            with pytest.raises(ValueError): read_molecule_profile_views('manifest.json', **dict(kwargs, limit=limit))
