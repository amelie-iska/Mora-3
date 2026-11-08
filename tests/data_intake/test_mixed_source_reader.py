"""Unknown shard summaries must defer to exact row metadata, not drop records."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'AGENTS.md').exists())
spec = importlib.util.spec_from_file_location('mixed_reader_corpus', ROOT / 'scripts/data_intake_corpus.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


@pytest.mark.parametrize('summary_split', ['train', None])
def test_mixed_shard_filters_rows_and_checks_authoritative_groups(tmp_path, summary_split):
    rows = []
    for i, source in enumerate(['DNA', 'Rfam']):
        payload = m.CorpusTask('native_prior', {'domain': source}, {'sequence': 'ACGT'}, {}, []).canonical_bytes()
        rows.append(dict(task_id=str(i), record_id=str(i), split='train',
                         source_family=source, native_schema='mora3.CorpusTask/v1',
                         model_adapter_supported=False, native_payload=payload,
                         native_payload_sha256=hashlib.sha256(payload).hexdigest(),
                         group_ids_json='["first-parent"]'))
    data = tmp_path / 'data.parquet'
    groups = tmp_path / 'groups.parquet'
    pq.write_table(pa.Table.from_pylist(rows), data)
    pq.write_table(pa.Table.from_pylist([
        dict(task_id=str(i), effective_groups_json=json.dumps(['all-parent-' + str(i)]))
        for i in range(2)]), groups)
    def ref(path, n):
        return dict(path=path.name, rows=n, bytes=path.stat().st_size,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(dict(
        shards=[dict(ref(data, 2), split=summary_split, source_family=None)],
        effective_group_lookup={data.name: ref(groups, 2)})))
    selected = list(m.read_corpus_tasks(manifest, root=tmp_path, source='Rfam', supported_only=False))
    assert len(selected) == 1
    assert selected[0]['native_payload'] == rows[1]['native_payload']
    assert selected[0]['group_ids_json'] == '["all-parent-1"]'
    assert selected[0]['source_group_ids_json'] == '["first-parent"]'
    assert not list(m.read_corpus_tasks(manifest, root=tmp_path, source='Rfam'))
    groups.write_bytes(b'tampered')
    with pytest.raises(ValueError, match='group metadata integrity'):
        list(m.read_corpus_tasks(manifest, root=tmp_path, source='Rfam', supported_only=False))
