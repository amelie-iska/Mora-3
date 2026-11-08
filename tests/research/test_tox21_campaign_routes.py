"""Match Tox21 campaign units to the exact existing source losses."""

from dataclasses import replace
from decimal import Decimal
from itertools import product

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from mora3.data.admet import canonical, digest
from mora3.data.finite_union_native_rows import FiniteUnionNativeView, full9
from mora3.data.tox21 import ASSAYS, FORWARD, INVERSE
from mora3.data.tox21_current import Tox21CurrentSource
from mora3.model.annotation_vocabulary import AnnotationVocabulary
from mora3.research.primary_native_routes import select_native_route_targets
from mora3.research.tox21_campaign_routes import Tox21CampaignAdapter
from test_native_tox21 import fixture, seal
from test_primary_native_routes import artifact
from test_tox21_current_model import make


SNAPSHOT = 'a' * 64
VOCAB = AnnotationVocabulary('{"learned_hex":["4343"],"controls":["pad","node","edge"]}')


def source_fixture(root):
    directions = (FORWARD, INVERSE)
    splits = ('train', 'validation', 'test')
    records = []
    for i, (split, direction) in enumerate(product(splits, directions)):
        row, payload = fixture(direction, split)
        row['task_id'] = f'native-{i}'
        values = [1, 0] + [None] * (len(ASSAYS) - 2)
        mask = [True, True] + [False] * (len(ASSAYS) - 2)
        if direction == FORWARD:
            payload['targets'].update(source_call_values=values, source_call_mask=mask)
        else:
            payload['conditions'].update(desired_source_call_values=values, desired_source_call_mask=mask)
        records.append(seal(row, payload))
    pq.write_table(pa.Table.from_pylist(records), root/'native.parquet')
    pq.write_table(pa.Table.from_pylist([
        dict(task_id=r['task_id'], effective_groups_json=r['group_ids_json']) for r in records
    ]), root/'groups.parquet')
    def ref(path):
        raw = (root/path).read_bytes()
        return dict(path=path, sha256=digest(raw), bytes=len(raw), rows=len(records))
    codes = (0, 0, 1, 1, 2, 2)
    descriptor = dict(source_shard=ref('native.parquet'), original_group_sidecar=ref('groups.parquet'),
        primary_retained_rows=len(records), source_full9=full9(root/'native.parquet'),
        disposition=dict(kind='original-row-ordinal-uint8', rows=len(records),
                         reference=artifact(root,'native.mask',bytes(codes))),
        source_task_schema_counts=[dict(source_family=r['source_family'],task_name=r['task_name'],
            native_schema=r['native_schema'],source_revision=r['source_revision'],split=r['split'],
            eligible_view_disposition=code,rows=1) for r,code in zip(records,codes,strict=True)])
    descriptor_ref = artifact(root,'native.jsonl',canonical(descriptor)+b'\n')
    manifest_ref = artifact(root,'native-view.json',canonical(dict(schema='Mora3.ImmutableFiniteEligibleNativeUnion/v1',
        full_project_training_ready=False, canonical_master_pointer_unchanged=True,
        original_immutable_source_views=descriptor_ref,source_taskviews=6,primary_retained_taskviews=6)))
    view = FiniteUnionNativeView(root/manifest_ref['path'],manifest_ref['sha256'],root=root)
    view.effective_filter = {'fixture':'base'}
    base_ref = artifact(root,'base.json',canonical(view.effective_filter))
    census = artifact(root,'tox21-census.json',canonical(dict(
        schema='Mora3.CurrentNativeTox21CallAndMoleculeCensus/v1',full_original_assay_axis=ASSAYS,
        same_original_roles_calls_masks_and_full_smiles_across_directions=True,unknown_never_inactive=True,
        original_role_contexts_preserved=True,effective_filter=base_ref,all_original_taskviews=6,
        exact_mirrored_source_molecule_pairs=3,original_source_paths=['native.parquet'],
        max_full_native_smiles_UTF8_bytes=3,original_role_direction_counts=[
            dict(split=split,direction=direction,rows=1) for split,direction in product(splits,directions)],
    )))
    source = Tox21CurrentSource(view,census['path'],census['sha256'])
    return source,view


@pytest.mark.parametrize(('direction','count'),[(FORWARD,2),(INVERSE,3)])
def test_exact_source_target_counts_and_train_route_before_model(tmp_path,direction,count):
    source,view = source_fixture(tmp_path)
    adapter = Tox21CampaignAdapter(source,view,VOCAB,SNAPSHOT,direction,chunk_size=2)
    binding = adapter.train_binding(batch_size=2)
    prepared = select_native_route_targets(binding,binding.route.initial_cursor,max_windows=1)
    assert prepared.status=='observed' and prepared.targets.observed_positions==count
    assert prepared.window.batch.source_row_ordinals==((0,) if direction==FORWARD else (1,))
    with pytest.raises(ValueError,match='owner'):
        prepared.targets.evaluate(None)
    validation = list(view.iter_native_task_batches(split='validation',source_paths=('native.parquet',),batch_size=1))
    assert adapter.validation_observed_targets(validation[direction==INVERSE])==count
    with pytest.raises(ValueError,match='direction'):
        adapter.validation_observed_targets(validation[direction==FORWARD])
    test = next(view.iter_native_task_batches(split='test',source_paths=('native.parquet',),batch_size=1))
    with pytest.raises(ValueError,match='validation'):
        adapter.validation_observed_targets(test)


def test_bound_identity_changes_with_direction_source_snapshot_codec_and_policy(tmp_path):
    source,view = source_fixture(tmp_path)
    adapter = Tox21CampaignAdapter(source,view,VOCAB,SNAPSHOT,INVERSE,chunk_size=2)
    wire = adapter.executable_binding()
    assert wire['byte_codec']['controls']['EOS']==259
    assert replace(adapter,direction=FORWARD).executable_binding()!=wire
    assert replace(adapter,source_snapshot_sha256='c'*64).executable_binding()!=wire
    assert replace(adapter,chunk_size=3).executable_binding()!=wire
    view.effective_filter={'fixture':'drift'}
    with pytest.raises(ValueError,match='snapshot'):
        adapter.executable_binding()


def test_synthetic_native_forward_validation_reduction_uses_only_observed_positions(tmp_path):
    source,view = source_fixture(tmp_path)
    model = make()
    # Rebind the existing synthetic owner to this exact fixture's source axes.
    model = type(model)(model.tox21,model.decoder,source.axes).double()
    vocab = AnnotationVocabulary(model.decoder.shared.payload.get_extra_state()['vocabulary'])
    adapter = Tox21CampaignAdapter(source,view,vocab,SNAPSHOT,FORWARD,chunk_size=2)
    batch = next(view.iter_native_task_batches(split='validation',source_paths=('native.parquet',),batch_size=1))
    with torch.no_grad():
        result = adapter._validation_reduction(model,batch)
        native = model.prediction_objective(source,batch,split='validation',return_logits=True)
    independent = torch.nn.functional.softplus(native['logits'][0,0].neg())+torch.nn.functional.softplus(native['logits'][0,1])
    assert result.observed_targets==2 and result.missing_reason is None
    assert float(result.loss_sum)==pytest.approx(independent.item(),rel=1e-13)
    assert isinstance(result.loss_sum,Decimal)
    with pytest.raises(ValueError,match='no-grad'):
        adapter._validation_reduction(model,batch)


def test_executable_pin_includes_bound_state_and_distinct_entry_points(tmp_path):
    import inspect
    from pathlib import Path

    from mora3.research.mup_campaign import executable_adapter_sha256

    source,view = source_fixture(tmp_path)
    adapter = Tox21CampaignAdapter(source,view,VOCAB,SNAPSHOT,INVERSE,chunk_size=2)
    root = Path(inspect.getfile(Tox21CampaignAdapter)).resolve().parents[3]
    prepare = executable_adapter_sha256(adapter.prepare,root=root)
    assert len(prepare)==64
    assert prepare!=executable_adapter_sha256(adapter.evaluate_validation,root=root)
    assert prepare!=executable_adapter_sha256(replace(adapter,chunk_size=3).prepare,root=root)
    assert prepare!=executable_adapter_sha256(replace(adapter,direction=FORWARD).prepare,root=root)
    with pytest.raises(ValueError,match='adapter method'):
        executable_adapter_sha256(adapter.train_binding,root=root)


def test_synthetic_native_inverse_validation_uses_full_eos_and_learned_codec(tmp_path):
    source,view = source_fixture(tmp_path)
    model = make()
    model = type(model)(model.tox21,model.decoder,source.axes).double()
    vocab = AnnotationVocabulary(model.decoder.shared.payload.get_extra_state()['vocabulary'])
    adapter = Tox21CampaignAdapter(source,view,vocab,SNAPSHOT,INVERSE,chunk_size=1)
    _,batch = view.iter_native_task_batches(split='validation',source_paths=('native.parquet',),batch_size=1)
    with torch.no_grad():
        result = adapter._validation_reduction(model,batch)
        native = source.views(batch,split='validation',direction=INVERSE)
        context = model.tox21.condition(native)
        ids = torch.tensor(model.decoder.codec.encode(b'CCO'))
        hidden,_ = model.decoder.rnn(model.decoder.embedding(ids[:-1][None]),context[None].contiguous())
        expected = torch.nn.functional.cross_entropy(model.decoder._readout(hidden)[0],ids[1:],reduction='sum')
    assert ids[-1]==model.decoder.codec.eos
    assert result.observed_targets==len(ids)-1
    assert float(result.loss_sum)==pytest.approx(expected.item(),rel=1e-13)
