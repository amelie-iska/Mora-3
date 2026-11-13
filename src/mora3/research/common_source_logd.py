"""Exactly two original ChEMBL37 AZ logD7.4 train-source windows.

Evidence-bound train-role source and cursor conservation only. Source relation
and units are unspecified: a numeric nominal logD7.4 coordinate is NOT a
verified unconstrained experimental effect, calibrated uncertainty or label
for another assay. No network/optimizer/worker/admission call is made.
"""
from hashlib import sha256
from pathlib import Path

from mora3.data.admet import canonical
from mora3.data.source_scalar_native_rows import project_native_scalar_batch
from mora3.research.remaining_eight_registry import load_assets
from mora3.research.native_scalar_campaign_routes import TrainOnlyScalarCampaignAdapter
from mora3.research.primary_campaign_source_rebind import load_inactive_common_source_rebind
from mora3.research.primary_source_factory_preflight import Ref,document
from mora3.research.primary_native_routes import finite_view_policy_sha256
from mora3.research.mup_campaign import executable_adapter_sha256
from mora3.research.common_source_native_replay import (
    _source_fingerprint,_first,_checked_rebound_position,_checked_two_window_transition,
)

_TASK='source_reported_experimental_logD7_4'
_FAMILY='ChEMBL37-AZ-experimental-logD7_4'
_EXPECTED_POLICY='c3812ecfcfea8a01f81769bf5ef6fb3880e04a0ac054a6349e9272db14765393'


def _sha(value):return sha256(value).hexdigest()


def _native_logd_signature(adapter,window):
    """Verify native scalar projection, provider values and distinct evidence classes."""
    projected=project_native_scalar_batch(adapter.original_view,window.batch,split='train')
    if projected.original_split!='train' or len(projected.scalar_rows)!=1:
        raise ValueError('Exactly one genuine original nominal logD source coordinate required')
    row,=projected.scalar_rows
    target=row.target
    if (row.source_family!=_FAMILY or row.task_name!=_TASK or row.original_split!='train'
            or row.disposition!='qualified-native-reported-source-coordinate'
            or target.kind!='reported_nominal_coordinate' or target.reported_value is None
            or not target.numeric_coordinate_available or target.raw_relation is not None
            or target.lower is not None or target.upper is not None
            or row.partition_key[:2]!=(_FAMILY,_TASK)
            or row.partition_key[3] is not None
            or row.partition_key[4]!='shake_flask'
            or row.partition_key[6]!='declared_native_nominal_logD7_4_coordinate_relation_unspecified'):
        raise ValueError('Source-relation-unspecified nominal logD is not a censored or unit-qualified measurement')
    condition=row.condition_inputs()
    if (condition['known_context']!={'pH':7.4,'solvent_pair':['octan-1-ol','water'],'method':'shake_flask'}
            or row.audit()['physical_missingness'] is None):
        raise ValueError('Exact AZ shake-flask pH7.4 source context and missingness required')
    if not (-1.5<target.reported_value<4.5):
        raise ValueError('LogD reported coordinate exceeds protected original validity window')
    prepared=adapter.prepare(window)
    if (prepared.observed_positions!=1 or prepared.unavailable_reason is not None
            or prepared.evaluate is None or not callable(prepared.evaluate)):
        raise ValueError('Original reported nominal numeric coordinate lost during native preparation')
    # Preserve the executable evaluator callable but NEVER execute it; owner
    # admission is closed. Hash the complete native schema/lexemes/audit fields.
    return dict(
        original_target_kind=target.kind,literal_source_unit=row.partition_key[3],
        source_relation=target.raw_relation,
        source_coordinate_scope=row.partition_key[6],
        original_float64_bits=target.reported_value_float64_bits,
        original_numeric_lexeme=target.native_numeric_lexeme,
        provider_lexemes_sha256=_sha(target.provider_lexemes_json),
        native_target_metadata_sha256=_sha(target.native_target_metadata_json),
        condition_sha256=_sha(row.condition_json),source_audit_sha256=_sha(row.source_audit_json),
        native_source_coordinate_sha256=_sha(row.source_coordinate_json),
        partition_sha256=_sha(canonical(row.partition_key)),
        assay_pH=condition['known_context']['pH'],
        exact_numeric_source_projection=True,training_objective_invoked=False,
        source_semantics='source-nominal-logD7.4;relation-and-literal-units-unspecified;not-independently-verified-effect')


def audit_original_logd_two_windows(root,*,task_key=_TASK,max_windows=2):
    """Read two real train windows twice, one per immutable snapshot binding.

    This is bounded evidence, not a validation baseline or whole-corpus scan.
    Do not widen max_windows or infer eligibility from equal source-row hashes.
    """
    if type(task_key)is not str or task_key!=_TASK or type(max_windows)is not int or max_windows!=2:
        raise ValueError('Only the original AZ logD task and exact two-window budget allowed')
    root=Path(root).resolve()
    if not root.is_dir() or Path.cwd().resolve()!=root:
        raise ValueError('Exact original source checkout and current working directory required')
    context=load_inactive_common_source_rebind(root)
    evidence=document(root,context.evidence)
    row,=(x for x in evidence['route_bindings'] if x['task_key']==_TASK)
    if (row['candidate']['max_windows']!=2 or row['common']['max_windows']!=2
            or row['common_view_policy_sha256']!=_EXPECTED_POLICY
            or context.common_snapshot.sha256!=evidence['common_source_snapshot']['sha256']):
        raise ValueError('Exact reviewed AZ logD finite source-view plan differs')
    candidate=Ref(**evidence['inactive_candidate_snapshot']);common=context.common_snapshot
    if candidate.sha256==common.sha256:
        raise ValueError('Independent candidate and common snapshot references required')
    view,original,_,_,axis,_,_,_,_=load_assets(root)
    if finite_view_policy_sha256(view)!=_EXPECTED_POLICY:
        raise ValueError('Original AZ logD source view/group policy changed')
    bindings={}
    for label,snapshot in (('candidate',candidate),('common',common)):
        adapter=TrainOnlyScalarCampaignAdapter(original,view,axis,snapshot.sha256,_FAMILY)
        binding=adapter.train_binding(batch_size=1)
        expected=row[label]
        actual=dict(task_key=binding.route.task_key,
            initial_cursor_sha256=_sha(binding.route.initial_cursor),
            adapter_sha256=executable_adapter_sha256(binding.prepare,root=Path(__file__).resolve().parents[3]),
            max_windows=max_windows)
        if (actual!=expected or binding.route.source_snapshot_sha256!=snapshot.sha256
                or binding.route.native_view_policy_sha256!=_EXPECTED_POLICY):
            raise ValueError('Reviewed AZ logD executable adapter/cursor binding mismatch: '+label)
        bindings[label]=(adapter,binding)

    def read_two(binding):
        route=binding.route;iterator=route.iter_from(route.initial_cursor)
        windows=[];position=route._cursor_position(route.initial_cursor)
        try:
            for _ in range(2):
                try:window=next(iterator)
                except StopIteration as exc:
                    raise ValueError('Source exhausted before frozen AZ logD two-window scan budget') from exc
                if window.original_source_rows!=1 or window.batch.original_split!='train':
                    raise ValueError('Exactly one original AZ train row per source window required')
                route._view.require_native_rows(window.batch,split='train')
                after=route._cursor_position(window.cursor_after)
                _checked_two_window_transition(position,after)
                windows.append(window);position=after
        finally:
            close=getattr(iterator,'close',None)
            if close is not None:close()
        return tuple(windows)

    a=read_two(bindings['candidate'][1]);b=read_two(bindings['common'][1])
    rows=[];identities=set()
    for i,(left,right) in enumerate(zip(a,b,strict=True)):
        if (left.batch.source_shard!=right.batch.source_shard
                or left.batch.source_row_ordinals!=right.batch.source_row_ordinals
                or not left.batch.native_rows.equals(right.batch.native_rows,check_metadata=True)):
            raise ValueError('Changed original Arrow source/role/order under common 29-head snapshot')
        la,lb=bindings['candidate'];ra,rb=bindings['common']
        sig_left=_native_logd_signature(la,left);sig_right=_native_logd_signature(ra,right)
        fingerprint=_source_fingerprint(left)
        if sig_left!=sig_right or fingerprint!=_source_fingerprint(right):
            raise ValueError('Original AZ logD payload/condition/nominal numeric coordinate changed')
        source_digest,source_bytes,payload,ordinal=fingerprint
        identity=(left.batch.source_shard['sha256'],ordinal)
        if identity in identities:raise ValueError('Duplicate AZ source row in two-window audit')
        identities.add(identity)
        cpos=lb.route._cursor_position(left.cursor_after)
        npos=rb.route._cursor_position(right.cursor_after)
        _checked_rebound_position(cpos,npos)
        rows.append(dict(window_index=i,source_shard_sha256=identity[0],source_ordinal=ordinal,
            native_source_fingerprint=source_digest,source_payload_bytes=source_bytes,
            original_native_payload_sha256=payload,candidate_next_position=cpos,
            common_next_position=npos,candidate_cursor_sha256=_sha(left.cursor_after),
            common_cursor_sha256=_sha(right.cursor_after),**sig_left))
    for label,windows in (('candidate',a),('common',b)):
        adapter,binding=bindings[label]
        restarted=_first(binding.route.iter_from(windows[0].cursor_after))
        second=windows[1]
        if (restarted.cursor_after!=second.cursor_after
                or restarted.evidence_sha256!=second.evidence_sha256
                or restarted.batch.source_shard!=second.batch.source_shard
                or restarted.batch.source_row_ordinals!=second.batch.source_row_ordinals
                or not restarted.batch.native_rows.equals(second.batch.native_rows,check_metadata=True)
                or _source_fingerprint(restarted)!=_source_fingerprint(second)
                or _native_logd_signature(adapter,restarted)!=_native_logd_signature(adapter,second)):
            raise ValueError('Original second AZ logD train window changes on independent cursor restart')
    return dict(schema='Mora3.OriginalAZLogDTwoWindowSourceReplay/v1',task_key=_TASK,
        source_family=_FAMILY,split='train',windows_replayed=2,native_source_rows=2,
        native_observed_targets=2,rows=tuple(rows),
        candidate_and_common_source_equal=True,native_conditions_and_targets_equal=True,
        second_window_restart_exact=True,native_view_policy_sha256=_EXPECTED_POLICY,
        candidate_source_snapshot_sha256=candidate.sha256,
        common_source_snapshot_sha256=common.sha256,
        scalar_projection_proof_sha256=axis.proof_sha256,
        full_source_corpus_exhausted=None,complete62_route_replay=False,
        validation_baseline_measured=False,global_split_admitted=False,
        training_admission=False,optimizer_updates=0,actual_campaign_runs=0,
        source_confidence='original-source-native-row-and-nominal-projection-not-independent-assay-validation')
