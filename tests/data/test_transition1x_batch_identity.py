"""Per-call batch reuse must preserve source authentication and ordered bindings."""
import shutil
import pytest
from test_transition1x_atom_identity import MANIFEST, PAYLOAD, RECORD, GROUP, AXIS
import mora3.data.transition1x_atom_identity as mod


def selection(**changes):
    s=dict(manifest_path=MANIFEST,payload_index=0,ordinal=13,
           expected_record_id=RECORD,expected_reaction_group=GROUP,
           expected_native_split='val',expected_atomic_numbers=AXIS)
    s.update(changes)
    return s


def batch(rows):
    assert hasattr(mod,'read_retained_atom_identities'), 'missing batch reader'
    return mod.read_retained_atom_identities(rows)


def test_repeats_preserve_order_and_independent_reference():
    rows=[selection(),selection(ordinal=14,expected_record_id='Transition1x_original:v4:C2H2N2O2:rxn3534',expected_reaction_group='Transition1x:v4:C2H2N2O2:rxn3534',expected_native_split='train',expected_atomic_numbers=(8,6,7,8,6,7,1,1)),selection()]
    assert batch(rows)==tuple(mod.read_retained_atom_identity(**r) for r in rows)
    assert [x.ordinal for x in batch(rows)]==[13,14,13]


def test_authentication_once_per_unique_path_per_call(monkeypatch):
    from pathlib import Path
    original=Path.read_bytes;calls=[]
    def read(p):
        calls.append(p.resolve());return original(p)
    monkeypatch.setattr(Path,'read_bytes',read)
    rows=[selection(),selection()]
    batch(rows)
    assert calls.count(MANIFEST.resolve())==1
    assert calls.count(PAYLOAD.resolve())==1
    batch(rows)
    assert calls.count(MANIFEST.resolve())==2
    assert calls.count(PAYLOAD.resolve())==2


@pytest.mark.parametrize('kind',['manifest','payload'])
def test_mutation_between_calls_refused(kind,tmp_path):
    path=tmp_path/kind;shutil.copyfile(MANIFEST if kind=='manifest' else PAYLOAD,path)
    rows=[selection(**{kind+'_path':path})]*2
    batch(rows)
    raw=bytearray(path.read_bytes());raw[len(raw)//2]^=1;path.write_bytes(raw)
    with pytest.raises(ValueError,match=kind):batch(rows)


@pytest.mark.parametrize('rows',[[],[object()],tuple(),[selection(expected_native_split='test')], [selection(),selection(expected_atomic_numbers=AXIS[::-1])]])
def test_invalid_batch_refused(rows):
    with pytest.raises(ValueError):batch(rows)


def test_selected_row_group_decoded_once(monkeypatch):
    import pyarrow.parquet as pq
    original=pq.ParquetFile;calls=[]
    class Observed:
        def __init__(self,*a,**kw):self.inner=original(*a,**kw)
        @property
        def metadata(self):return self.inner.metadata
        def read_row_group(self,group,columns):
            calls.append((group,tuple(columns)))
            return self.inner.read_row_group(group,columns=columns)
    monkeypatch.setattr(pq,'ParquetFile',Observed)
    batch([selection(),selection()])
    assert len(calls)==1 and calls[0][1]==mod._COLUMNS


@pytest.mark.parametrize('rows',[[selection()]*65,[dict(selection(),cache={})],[dict(selection(),positions=[])],object()])
def test_batch_bound_and_unexpected_authority_fields_refused(rows):
    with pytest.raises(ValueError):batch(rows)
