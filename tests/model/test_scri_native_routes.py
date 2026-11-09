"""Actual finite native reader binding; fixtures cannot authorize training."""
import json
from dataclasses import replace
from hashlib import sha256
import pytest
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from test_full_owner import actual_core_assets,cases
from test_goa_objective import view
from mora3.model.full_owner import PrimaryConfiguration,construct_primary_owner
from mora3.data.finite_union_native_rows import FiniteUnionNativeView,full9
from mora3.data.admet import canonical,digest
from mora3.research.primary_native_routes import FiniteNativeTrainRoute,finite_view_policy_sha256,select_native_route_targets
from mora3.extensions.scri.native_owner import SpatialBudget
from mora3.extensions.scri.model import SCRITargetAxis
from mora3.extensions.scri.contracts import SourceEvidence,StateRecord,NativeValue
from mora3.extensions.scri.native_routes import decode_envelope,build_native_scri_binding,registration_proposal

AXIS=SCRITargetAxis('d'*64,(('rna','gene','count'),),'poisson')

def envelope(missing=False,split='train'):
    source=SourceEvidence('fixture','r1','c'*64,'fixture','CC0','synthetic_fixture','study','sample','donor','experiment',None)
    records=[StateRecord(str(i),replace(source,source_locator=str(i)),'fixture',None,0,'s','baseline',
        (NativeValue('rna','gene',i,'count',True),),(float(i),0.),'um','fixture-frame',(),split).wire()for i in range(4)]
    return dict(schema='Mora3.SCRISpatialEnvelope/v1',source_snapshot_sha256='a'*64,
        baseline_records=records,query_record_id='0',radius=4.,target_axis=AXIS.features,
        target_values=[None if missing else 2],target_observed=[not missing])


def finite_route(tmp_path,documents):
    rows=pa.Table.from_pylist([dict(task_id=str(i),source_family='SCRI',task_name='spatial',native_schema='Mora3.SCRISpatialEnvelope/v1',
        source_revision='r1',split='train',native_payload=canonical(d),native_payload_sha256=digest(canonical(d)))for i,d in enumerate(documents)])
    p=tmp_path/'native.parquet';pq.write_table(rows,p,row_group_size=1)
    def save(name,raw):
        (tmp_path/name).write_bytes(raw);return dict(path=name,bytes=len(raw),sha256=digest(raw))
    source=save('native.parquet',p.read_bytes());source['rows']=len(documents)
    descriptor=dict(source_shard=source,source_full9=full9(p),primary_retained_rows=len(documents),
        original_group_sidecar=save('native.groups',b'synthetic-group'),
        disposition=dict(kind='original-row-ordinal-uint8',rows=len(documents),reference=save('native.mask',bytes([0]*len(documents)))),
        source_task_schema_counts=[dict(source_family='SCRI',task_name='spatial',native_schema='Mora3.SCRISpatialEnvelope/v1',source_revision='r1',split='train',eligible_view_disposition=0,rows=len(documents))])
    refs=save('descriptors.jsonl',canonical(descriptor)+b'\n')
    manifest=save('manifest.json',canonical(dict(schema='Mora3.ImmutableFiniteEligibleNativeUnion/v1',full_project_training_ready=False,
        canonical_master_pointer_unchanged=True,original_immutable_source_views=refs,source_taskviews=len(documents),primary_retained_taskviews=len(documents))))
    native=FiniteUnionNativeView(tmp_path/'manifest.json',manifest['sha256'],root=tmp_path)
    return FiniteNativeTrainRoute(native,task_key='scri-spatial',source_snapshot_sha256='a'*64,
        native_view_policy_sha256=finite_view_policy_sha256(native),source_family='SCRI',task_name='spatial',
        native_schema='Mora3.SCRISpatialEnvelope/v1',source_revisions=('r1',),source_paths=('native.parquet',),batch_size=1)


def test_decoder_masks_and_official_heldout_exclusion():
    a=decode_envelope(canonical(envelope()),task_key='scri-spatial',axis=AXIS)
    b=envelope();b['target_values']=[900]
    c=decode_envelope(canonical(b),task_key='scri-spatial',axis=AXIS)
    assert a[0].record==c[0].record and a[0].task==c[0].task
    assert a[1]!=c[1]
    with pytest.raises(ValueError,match='heldout'):
        decode_envelope(canonical(envelope(split='test')),task_key='scri-spatial',axis=AXIS)
    for invalid in (True,-1,.5,float('inf')):
        d=envelope();d['target_values']=[invalid]
        with pytest.raises(ValueError):decode_envelope(json.dumps(d),task_key='scri-spatial',axis=AXIS)


def test_actual_finite_binding_prepares_without_model_and_matches_native_callable(tmp_path,actual_core_assets):
    source=replace(actual_core_assets,goa_packets=(view(tmp_path),),scri_axis=AXIS,
        scri_spatial_budget=SpatialBudget(16,257,300000,2000000,2))
    owner=construct_primary_owner(PrimaryConfiguration(8,8,'a'*64),source)
    folder=tmp_path/'route';folder.mkdir();route=finite_route(folder,[envelope(missing=True),envelope()])
    binding=build_native_scri_binding(owner,route)
    selected=select_native_route_targets(binding,route.initial_cursor,max_windows=2)
    assert len(selected.unavailable)==1 and selected.targets.observed_positions==1
    result=selected.targets.evaluate(owner)
    assert result.ndim==0 and torch.isfinite(result) and result.requires_grad
    with pytest.raises(ValueError):selected.targets.evaluate(object())
    with pytest.raises(ValueError,match='not full corpus'):owner.require_training_admission()
    proposal=registration_proposal('a'*64)
    assert proposal['production_launch_enabled'] is False and proposal['newly_admitted_sources']==0
