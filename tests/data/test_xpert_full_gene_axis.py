"""Real full axes are source metadata, never task admission."""
import gzip
import json
from pathlib import Path
import pytest
from mora3.data.xpert_full_store import XPertFullNumericStore

ROOT = Path(__file__).resolve().parents[2]

@pytest.mark.parametrize('article,width', [(32221275,33694),(32221302,978)])
def test_complete_gene_axis_exact_native_records_and_closed_gates(article,width):
    store=XPertFullNumericStore(root=ROOT)
    axis=store.read_gene_axis(article)
    manifest=json.loads((ROOT/store.MANIFEST['path']).read_bytes())
    dataset=next(d for d in manifest['datasets'] if d['source_article_ID']==article)
    with gzip.open(dataset['var_axis']['path'],'rb') as f:
        oracle=tuple(f)
    assert axis.native_records==oracle
    assert len(axis.native_records)==width
    assert axis.record(width-1)==json.loads(oracle[-1])
    # Caller edits cannot mutate retained native metadata.
    changed=axis.record(0); changed['native_fields']['_index']='changed'
    assert axis.record(0)==json.loads(oracle[0])
    assert axis.audit()['gene_axis_sha256']==dataset['var_axis']['sha256']
    assert axis.audit()['context_bound'] is False
    assert axis.audit()['task_admitted'] is False
    for purpose in ('training','prediction','generation'):
        with pytest.raises(ValueError): axis.require_model_use(purpose)
    for index in (True,-1,width):
        with pytest.raises(ValueError): axis.record(index)

@pytest.mark.parametrize('article',[True,'32221302',0,32221303])
def test_unknown_or_noninteger_source_refused(article):
    with pytest.raises(ValueError): XPertFullNumericStore(root=ROOT).read_gene_axis(article)

def test_gene_axis_fresh_hash_check_on_every_call(tmp_path):
    cls=XPertFullNumericStore
    for ref in (cls.MANIFEST,cls.HANDOFF):
        p=tmp_path/ref['path'];p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes((ROOT/ref['path']).read_bytes())
    store=cls(root=tmp_path)
    manifest=json.loads((ROOT/cls.MANIFEST['path']).read_bytes())
    ref=manifest['datasets'][1]['var_axis']
    p=tmp_path/Path(ref['path']).relative_to(ROOT)
    p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(Path(ref['path']).read_bytes())
    assert len(store.read_gene_axis(32221302).native_records)==978
    p.write_bytes(p.read_bytes()+b' ')
    with pytest.raises(ValueError,match='immutable proof mismatch'):
        store.read_gene_axis(32221302)
