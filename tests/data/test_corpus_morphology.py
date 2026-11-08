"""Synthetic envelope gates; fixtures do not grant real source admission."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import copy
import hashlib
import json
from dataclasses import asdict, replace
import pytest
import torch
from mora3.data.raster_morphology import RasterMorphologyTask

def task():
    return RasterMorphologyTask(3,4,'>u2','>u2','0001'*12,'000000010001000000000000000000000000000000000000',
        'https://zenodo.org/records/5550933','v1','image.tif','a'*64,'b'*64,'c'*64,
        'CC-BY-4.0','parent','train','Staphylococcus aureus','JE2','brightfield','manual',.08)
from mora3.data import corpus_morphology as cm
from mora3.model.cooperative_training import CooperativeModel


@pytest.fixture(autouse=True)
def deterministic():
    prior=torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(prior)


def row():
    t=replace(task(),source='https://zenodo.org/records/5550933',revision='v1',
        license='CC-BY-4.0',annotation='manual ellipses drawn on Nile Red fluorescence; whole-cell instance labels')
    raw=t.canonical_bytes()
    return dict(task_id='task',record_id=t.source_id,source_family='DeepBacs-Staph',
        source_revision='Zenodo5550933.v1',source_sha256=t.archive_sha256,
        task_name='image_to_mask',native_schema='mora3.RasterMorphologyTask/v1',
        native_payload=raw,native_payload_sha256=hashlib.sha256(raw).hexdigest(),
        split='train',admitted_for_task=True,evidence_class=cm.EVIDENCE,
        split_policy=cm.POLICY,admission_receipt_json=json.dumps(dict(policy=cm.POLICY,
            source_quality='source-specific documented QC',physical_assay_equivalence=False)),
        missing_context_json=json.dumps(['intervention','paired_expression','donor','independent_heldout_status']),
        condition_context_json='{}',full_axis_values=t.height*t.width)


def view(r=None,groups=('parent','pixels')):
    return cm.corpus_morphology_task(row() if r is None else r,effective_groups=groups,
        corpus_binding={'manifest_sha256':'a'*64,'shard_sha256':'b'*64,'groups_sha256':'c'*64})


@pytest.mark.parametrize('change',[
    {'admitted_for_task':False},{'admitted_for_task':1},{'source_family':'Other'},
    {'source_revision':'v0'},{'evidence_class':'model_produced'},
    {'split':'test'},{'record_id':'other'},{'source_sha256':'f'*64},
    {'task_name':'mask_to_image'},{'full_axis_values':999},{'condition_context_json':'{"labels":1}'},
    {'missing_context_json':'[]'},{'split_policy':'blind-holdout'},
    {'admission_receipt_json':'{}'},{'native_payload_sha256':'f'*64}])
def test_envelope_gates(change):
    r=row();r.update(change)
    with pytest.raises(ValueError):view(r)


@pytest.mark.parametrize('groups',[(),('x','x'),('',)])
def test_effective_groups_required(groups):
    with pytest.raises(ValueError):view(groups=groups)


def test_receipt_groups_and_payload_bound_without_target_input():
    a=view();r=row();d=json.loads(r['native_payload']);d['labels_hex']='00'*(len(bytes.fromhex(d['labels_hex'])))
    r['native_payload']=json.dumps(d).encode();r['native_payload_sha256']=hashlib.sha256(r['native_payload']).hexdigest()
    b=view(r)
    assert torch.equal(a.input_image(),b.input_image())
    assert a.input_graph()==b.input_graph()
    assert a.canonical_bytes()!=b.canonical_bytes()
    assert a.canonical_bytes()!=view(groups=('changed',)).canonical_bytes()
    a.require_training()
    assert a.admission()['independent_heldout'] is False
    assert a.admission()['paired_expression'] is False


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_native_training_equation_and_gradients(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('no CUDA')
    a=view(); original=cm.RasterMorphologyTask.from_bytes(row()['native_payload'])
    model=CooperativeModel(('gene',),width=4).to(device)
    actual=model.objective('image',[a]);expected=model.objective('image',[original])
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    params=tuple(model.parameters())
    ga=torch.autograd.grad(actual,params,allow_unused=True)
    ge=torch.autograd.grad(expected,params,allow_unused=True)
    for x,y in zip(ga,ge):
        if x is None:assert y is None
        else:torch.testing.assert_close(x,y,rtol=0,atol=0)


def corpus(tmp_path, *, group_task='task', split='train'):
    import pyarrow as pa
    import pyarrow.parquet as pq
    r=row();r['split']=split
    d=json.loads(r['native_payload']);d['split']=split
    r['native_payload']=json.dumps(d).encode();r['native_payload_sha256']=hashlib.sha256(r['native_payload']).hexdigest()
    pq.write_table(pa.Table.from_pylist([r]),tmp_path/'rows.parquet')
    pq.write_table(pa.Table.from_pylist([dict(task_id=group_task,effective_groups_json='["corrected-pixel-group"]')]),tmp_path/'groups.parquet')
    def receipt(name):
        raw=(tmp_path/name).read_bytes();return dict(path=name,sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),rows=1)
    shard=dict(receipt('rows.parquet'),source_family='DeepBacs-Staph',split=split)
    m=dict(shards=[shard],effective_group_lookup={'rows.parquet':receipt('groups.parquet')})
    raw=json.dumps(m).encode();(tmp_path/'manifest.json').write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def test_reader_pins_real_group_schema_and_evaluation_only_split(tmp_path):
    digest=corpus(tmp_path,split='test')
    t,=cm.read_corpus_morphology('manifest.json',root=tmp_path,expected_manifest_sha256=digest,split='test',limit=1)
    assert t.admission()['effective_groups']==['corrected-pixel-group']
    with pytest.raises(ValueError,match='split'):t.require_training()
    with pytest.raises(ValueError,match='manifest'):cm.read_corpus_morphology('manifest.json',root=tmp_path,expected_manifest_sha256='0'*64,split='test',limit=1)
    (tmp_path/'groups.parquet').write_bytes(b'changed')
    with pytest.raises(ValueError,match='integrity'):cm.read_corpus_morphology('manifest.json',root=tmp_path,expected_manifest_sha256=digest,split='test',limit=1)


def test_reader_group_identity_alignment(tmp_path):
    digest=corpus(tmp_path,group_task='wrong')
    with pytest.raises(ValueError,match='alignment'):
        cm.read_corpus_morphology('manifest.json',root=tmp_path,expected_manifest_sha256=digest,split='train',limit=1)


def test_admission_cannot_survive_payload_or_split_replacement():
    t=view()
    for change in (dict(split='test'),dict(labels_hex='0000'*12),dict(source_id='other')):
        with pytest.raises(ValueError,match='admission'):
            replace(t,**change)
