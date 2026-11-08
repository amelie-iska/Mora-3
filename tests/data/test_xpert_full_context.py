import json
from pathlib import Path
import pytest
from mora3.data.xpert_full_store import XPertFullNumericStore
from mora3.data.xpert_native import XPertContext, _canonical
ROOT=Path(__file__).resolve().parents[2]

@pytest.mark.parametrize('article',[32221275,32221302])
def test_full_context_exact_native_projection_and_closed_roles(article):
    store=XPertFullNumericStore(root=ROOT)
    result=store.read_contexts(article,(2,0))
    observations=store.read_observations(article,(2,0))
    for i,context in enumerate(result.contexts):
        native=observations.record(i)['native_fields']
        assert context.native_fields()==native
        assert context.prediction_inputs()==XPertContext(article,_canonical(native)).prediction_inputs()
        with pytest.raises(ValueError):context.prediction_inputs(known_fields=('_index',))
    audit=result.audit()
    assert audit['native_row_indices']==[2,0]
    assert audit['rights_verified_as_evidence_only'] is True
    assert audit['model_use_allowed'] is False
    assert audit['numeric_bound'] is False
    for purpose in ['training','generation','split_use','prediction']:
        with pytest.raises(ValueError):result.require_model_use(purpose)
    audit['proof_refs'].clear()
    assert result.audit()['proof_refs']

def test_context_rights_revalidated_on_every_selection(tmp_path):
    from mora3.data.xpert_native import FULL_REF
    full=json.loads((ROOT/FULL_REF['path']).read_bytes())
    refs=[XPertFullNumericStore.MANIFEST,XPertFullNumericStore.HANDOFF,FULL_REF]
    m=json.loads((ROOT/XPertFullNumericStore.MANIFEST['path']).read_bytes())
    d=m['datasets'][1]
    refs += [d['obs_axis'],d['source_QC'],d['source_preparation_gates'],d['source_role_masks'],full['source_release_rights']]
    for ref in refs:
        source=Path(ref['path']);relative=source.relative_to(ROOT) if source.is_absolute() else source
        target=tmp_path/relative;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((ROOT/relative).read_bytes())
    store=XPertFullNumericStore(root=tmp_path)
    assert store.read_contexts(32221302,(0,)).contexts
    rights=tmp_path/Path(full['source_release_rights']['path']).relative_to(ROOT)
    rights.write_bytes(rights.read_bytes()+b' ')
    with pytest.raises(ValueError,match='immutable proof mismatch'):store.read_contexts(32221302,(0,))
