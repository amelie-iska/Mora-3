"""Native SCRI validation contracts and denominator integrity; zero updates."""
from dataclasses import replace
from pathlib import Path
import inspect
from decimal import Decimal
import pytest
import torch
import pyarrow as pa
import pyarrow.parquet as pq
from test_full_owner import actual_core_assets,cases
from test_goa_objective import view as goa_view
from test_scri_native_routes import envelope,AXIS
from mora3.data.admet import canonical,digest
from mora3.data.finite_union_native_rows import FiniteUnionNativeView,full9
from mora3.model.full_owner import PrimaryConfiguration,construct_primary_owner
from mora3.extensions.scri.native_owner import SpatialBudget
from mora3.extensions.scri.native_routes import SCRICampaignAdapter,decode_envelope
from mora3.research.primary_native_routes import FiniteNativeTrainRoute,finite_view_policy_sha256,NativeRouteWindow
from mora3.research.mup_campaign import executable_adapter_sha256


def source(root):
    rows=[]
    for i,(split,missing)in enumerate((('train',False),('validation',False),('validation',True),('test',False))):
        raw=canonical(envelope(missing=missing,split=split))
        rows.append(dict(task_id=str(i),source_family='SCRI',task_name='spatial',native_schema='Mora3.SCRISpatialEnvelope/v1',
            source_revision='r1',split=split,native_payload=raw,native_payload_sha256=digest(raw)))
    path=root/'source.parquet';pq.write_table(pa.Table.from_pylist(rows),path,row_group_size=1)
    def save(name,raw):
        (root/name).write_bytes(raw);return dict(path=name,bytes=len(raw),sha256=digest(raw))
    shard=save(path.name,path.read_bytes());shard['rows']=4
    descriptor=dict(source_shard=shard,source_full9=full9(path),primary_retained_rows=4,
        original_group_sidecar=save('groups',b'fixture-only'),
        disposition=dict(kind='original-row-ordinal-uint8',rows=4,reference=save('mask',bytes([0,1,1,2]))),
        source_task_schema_counts=[dict(source_family='SCRI',task_name='spatial',native_schema='Mora3.SCRISpatialEnvelope/v1',
            source_revision='r1',split=split,eligible_view_disposition=code,rows=n)
            for split,code,n in(('train',0,1),('validation',1,2),('test',2,1))])
    refs=save('descriptors.jsonl',canonical(descriptor)+b'\n')
    manifest=save('manifest.json',canonical(dict(schema='Mora3.ImmutableFiniteEligibleNativeUnion/v1',
        full_project_training_ready=False,canonical_master_pointer_unchanged=True,
        original_immutable_source_views=refs,source_taskviews=4,primary_retained_taskviews=4)))
    native=FiniteUnionNativeView(root/'manifest.json',manifest['sha256'],root=root)
    route=FiniteNativeTrainRoute(native,task_key='scri-spatial',source_snapshot_sha256='a'*64,
        native_view_policy_sha256=finite_view_policy_sha256(native),source_family='SCRI',task_name='spatial',
        native_schema='Mora3.SCRISpatialEnvelope/v1',source_revisions=('r1',),source_paths=(path.name,),batch_size=1)
    return native,route


@pytest.fixture
def setup(tmp_path,actual_core_assets):
    assets=replace(actual_core_assets,goa_packets=(goa_view(tmp_path),),scri_axis=AXIS,
        scri_spatial_budget=SpatialBudget(16,257,300000,2000000,2))
    owner=construct_primary_owner(PrimaryConfiguration(8,8,'a'*64),assets)
    path=tmp_path/'native';path.mkdir();native,route=source(path)
    adapter=SCRICampaignAdapter(route,AXIS,('source.parquet',))
    batches=tuple(native.iter_native_task_batches(split='validation',source_paths=('source.parquet',),batch_size=1))
    return owner,adapter,batches


def test_validation_fingerprint_and_exact_native_reduction(setup):
    owner,adapter,batches=setup
    root=Path(inspect.getfile(SCRICampaignAdapter)).resolve().parents[4]
    assert len(executable_adapter_sha256(adapter.evaluate_validation,root=root))==64
    assert adapter.validation_observed_targets(batches[0])==1
    with torch.no_grad():
        result=adapter.evaluate_validation(owner,batches[0])
        q,values,mask,_=decode_envelope(batches[0].native_rows['native_payload'][0].as_py(),task_key='scri-spatial',axis=AXIS,split='validation')
        z=owner.scri.forward_spatial((q,)).double()[0,0]
        reference=z.exp()-2*z+torch.lgamma(z.new_tensor(3.))
    assert result.observed_targets==1 and result.missing_reason is None
    assert float(result.loss_sum)==pytest.approx(float(reference),rel=1e-12)


def test_missing_validation_is_explicit_not_observed_zero(setup):
    owner,adapter,batches=setup
    with torch.no_grad():result=adapter.evaluate_validation(owner,batches[1])
    assert result.observed_targets==0 and result.loss_sum==Decimal(0)and result.missing_reason
    with pytest.raises(ValueError,match='evaluation-only'):adapter.evaluate_validation(owner,batches[0])


def test_source_roles_and_payload_budget_refuse_before_model(setup):
    owner,adapter,batches=setup
    with pytest.raises(ValueError,match='split'):
        adapter.prepare(NativeRouteWindow(batches[0],b'cursor','a'*64))
    limited=SCRICampaignAdapter(adapter.route,AXIS,('source.parquet',),max_payload_bytes=16)
    with pytest.raises(ValueError,match='byte budget'):limited.validation_observed_targets(batches[0])
    no_validation=SCRICampaignAdapter(adapter.route,AXIS)
    with pytest.raises(ValueError,match='outside'):no_validation.validation_observed_targets(batches[0])
    with pytest.raises(ValueError):decode_envelope(canonical(envelope(split='test')),task_key='scri-spatial',axis=AXIS,split='test')


def test_complete_axis_binding_changes_fingerprint_and_refuses_wrong_owner(setup):
    owner,adapter,batches=setup
    changed=SCRICampaignAdapter(adapter.route,replace(AXIS,source_sha256='e'*64),('source.parquet',))
    root=Path(inspect.getfile(SCRICampaignAdapter)).resolve().parents[4]
    assert executable_adapter_sha256(adapter.prepare,root=root)!=executable_adapter_sha256(changed.prepare,root=root)
    with torch.no_grad(),pytest.raises(ValueError,match='axis'):changed.evaluate_validation(owner,batches[0])
