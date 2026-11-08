"""Actual full-source composition, independently compared to native assets."""
import gzip
import json
from pathlib import Path
import numpy as np
import pytest
from mora3.data.xpert_full_store import XPertFullNumericStore
ROOT = Path(__file__).resolve().parents[2]


def test_complete_dense_rows_bind_exact_native_context_axis_and_numeric_bits():
    store = XPertFullNumericStore(root=ROOT)
    result = store.read_candidate_profiles(32221302, (590, 0))
    manifest = json.loads((ROOT/store.MANIFEST['path']).read_bytes())
    dataset = next(d for d in manifest['datasets'] if d['source_article_ID'] == 32221302)
    array = np.load(ROOT/manifest['array_transports']['32221302-X']['transport']['path'], mmap_mode='r', allow_pickle=False)
    with gzip.open(dataset['obs_axis']['path'], 'rb') as f:
        obs = [next(f) for _ in range(591)]
    with gzip.open(dataset['var_axis']['path'], 'rb') as f:
        genes = tuple(f)
    assert result.gene_axis.native_records == genes
    assert result.observations.native_records == (obs[590], obs[0])
    for position, ordinal in enumerate((590, 0)):
        row = result.profiles[position]
        assert row.native_row_index == ordinal
        assert row.axis_length == len(genes) == 978
        assert row.values_bytes == array[ordinal].tobytes()
        assert row.values.finite_mask == np.isfinite(array[ordinal]).tobytes()
        assert result.contexts.contexts[position].native_fields() == json.loads(obs[ordinal])['native_fields']
    audit = result.audit()
    assert audit['native_row_indices'] == [590, 0]
    assert audit['numeric_bound'] is True and audit['context_bound'] is True
    assert audit['model_use_allowed'] is False and audit['task_admitted'] is False
    assert audit['official_split'] is None
    assert audit['source_article_id'] == 32221302
    assert audit['gene_axis_sha256'] == dataset['var_axis']['sha256']
    assert audit['observation_axis_sha256'] == dataset['obs_axis']['sha256']
    assert audit['rights_verified_as_evidence_only'] is True
    for purpose in ('training', 'prediction', 'generation', 'split_use', 'unknown'):
        with pytest.raises(ValueError): result.require_model_use(purpose)
    audit.clear()
    assert result.audit()['numeric_bound'] is True


@pytest.mark.parametrize('article,rows', [(32221275,(0,)), (True,(0,)), (0,(0,)),
    (32221302,()), (32221302,(True,)), (32221302,(0,0)),
    (32221302,(678401,)), (32221302,tuple(range(17)))])
def test_unqualified_sparse_rows_and_invalid_selections_refused(article, rows):
    with pytest.raises(ValueError):
        XPertFullNumericStore(root=ROOT).read_candidate_profiles(article, rows)


def test_composition_freshly_revalidates_pinned_store(tmp_path):
    cls = XPertFullNumericStore
    for ref in (cls.MANIFEST, cls.HANDOFF):
        p=tmp_path/ref['path']; p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes((ROOT/ref['path']).read_bytes())
    store=cls(root=tmp_path)
    p=tmp_path/cls.HANDOFF['path'];p.write_bytes(p.read_bytes()+b' ')
    with pytest.raises(ValueError,match='immutable proof mismatch'):
        store.read_candidate_profiles(32221302,(0,))
