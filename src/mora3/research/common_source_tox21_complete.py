"""Exhaust original Tox21 train roles under both frozen N29 source bindings.

Genuine immutable Arrow payloads, 12 original assay-call axes, observed masks,
inverse molecular targets, group sidecars and source cursors are conserved.
This is source-only qualification. Nothing here admits a global evaluation split,
qualifies a biological assay or evaluates/updates a model.
"""
from collections import Counter
from contextlib import ExitStack
from hashlib import sha256
from pathlib import Path
import argparse
import json
import math
import re

from mora3.data.admet import canonical
from mora3.data.finite_union_native_rows import full9, read_exact
from mora3.data.tox21 import ASSAYS, FORWARD, INVERSE, SOURCE
from mora3.research.common_source_native_replay import (
    _first, _source_fingerprint, _checked_two_window_transition,
    _checked_rebound_position,
)
from mora3.research.finite_assay_registry import load_original_assay_assets
from mora3.research.tox21_campaign_routes import Tox21CampaignAdapter
from mora3.research.primary_campaign_source_rebind import load_inactive_common_source_rebind
from mora3.research.primary_source_factory_preflight import Ref, document
from mora3.research.mup_campaign import executable_adapter_sha256
from mora3.research.primary_native_routes import finite_view_policy_sha256

_ORIGINAL=Path('/home/iska/Documents/amelie/bio/Mora-3')
_CODE_ROOT=Path(__file__).resolve().parents[3]
_REV='originalNIH2014challenge;contentpinned20261031'
_POLICY='c3812ecfcfea8a01f81769bf5ef6fb3880e04a0ac054a6349e9272db14765393'
_ROLE=(FORWARD,INVERSE)
_SHA=re.compile(r'[0-9a-f]{64}\Z')


def _sha(raw):
    return sha256(raw).hexdigest()


def _unique(items):
    d={}
    for k,v in items:
        if k in d:
            raise ValueError('Duplicate native Tox21 JSON key')
        d[k]=v
    return d


def _bad_constant(_):
    raise ValueError('Nonfinite original source Tox21 value')


def reference_original_tox21(raw,direction):
    """Separate strict JSON/mask/role oracle, without the Tox21View projector."""
    if type(raw) is not bytes or not raw or direction not in _ROLE:
        raise ValueError('Exact source bytes and direction required')
    try:
        p=json.loads(raw,object_pairs_hook=_unique,parse_constant=_bad_constant)
        if (type(p) is not dict or set(p)!={
            'schema','task','conditions','targets','audit','missing'
        } or p['schema']!='mora3.CorpusTask/v1' or p['task']!=direction
                or type(p['conditions']) is not dict or type(p['targets']) is not dict
                or type(p['audit']) is not dict or type(p['missing']) is not list):
            raise ValueError('Original Tox21 task/split/role schema changed')
        conditions,targets,audit=p['conditions'],p['targets'],p['audit']
        forward=direction==FORWARD
        if forward:
            if (set(conditions)!={'canonical_isomeric_smiles','known_assay_axis'}
                    or set(targets)!={'source_call_values','source_call_mask'}):
                raise ValueError('Forward target/condition boundaries changed')
            calls=targets;values=calls['source_call_values'];mask=calls['source_call_mask']
            smi=conditions['canonical_isomeric_smiles']
        else:
            if (set(conditions)!={'known_assay_axis','desired_source_call_values','desired_source_call_mask'}
                    or set(targets)!={'canonical_isomeric_smiles'}):
                raise ValueError('Inverse target/condition boundaries changed')
            values=conditions['desired_source_call_values']
            mask=conditions['desired_source_call_mask']
            smi=targets['canonical_isomeric_smiles']
        axis=conditions['known_assay_axis']
        if (type(axis) is not list or tuple(axis)!=ASSAYS
                or type(values) is not list or type(mask) is not list
                or len(values)!=len(ASSAYS) or len(mask)!=len(ASSAYS)
                or not any(mask) or type(smi) is not str or not smi.strip()
                or len(smi.encode('utf-8'))>332):
            raise ValueError('Full original twelve-assay axis, mask, SMILES required')
        positives=negatives=unknowns=0
        for value,observed in zip(values,mask,strict=True):
            if type(observed) is not bool:
                raise ValueError('Mask must be exact Boolean')
            if observed:
                if type(value) is not int or value not in (0,1):
                    raise ValueError('Observed source-call label must be binary')
                positives+=int(value==1);negatives+=int(value==0)
            else:
                if value is not None:
                    raise ValueError('Unknown Tox21 is not inactive or numeric')
                unknowns+=1
        ref=audit.get('source_ref')
        if (type(ref) is not dict or not _SHA.fullmatch(str(ref.get('native_jsonl_sha256','')))
                or audit.get('source_protocol_role')!='source_training'
                or audit.get('source_missing_not_negative') is not True
                or audit.get('physical_dose') is not None or audit.get('exposure_time') is not None
                or 'perassayphysicaldose_exposuretime_cellline' not in p['missing']):
            raise ValueError('Original experimental source/ref/physical-context discipline changed')
        return dict(
            smi_sha256=_sha(smi.encode('utf-8')),
            smi_utf8_bytes=len(smi.encode('utf-8')),
            observed_positive=positives,observed_negative=negatives,missing_unknown=unknowns,
            native_mask_sha256=_sha(canonical(mask)),
            native_call_signature_sha256=_sha(canonical((ASSAYS,values,mask))),
            source_ref_sha256=_sha(canonical(ref)),
            raw_conditions_sha256=_sha(canonical(conditions)),
            raw_targets_sha256=_sha(canonical(targets)),
            evidence_class='original-source-reported-binary-call-or-desired-pattern',
            source_not_independent_biological_validation=True,
        )
    except (KeyError,TypeError,OverflowError,UnicodeError,json.JSONDecodeError) as exc:
        raise ValueError('Malformed original native Tox21 JSON') from exc


def _source_train_metadata(root,view,paths):
    """Original-only verified raw Arrow and matching group records, never heldout targets."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    descriptors={d['source_shard']['path']:d for d in view.descriptors
                 if d['source_shard']['path'] in paths}
    if len(paths)!=51 or set(paths)!=set(descriptors):
        raise ValueError('Exactly 51 original Tox21 training source shards required')
    refs=[];meta={};ordered={d:[] for d in _ROLE}
    for shard_index,path in enumerate(paths):
        d=descriptors[path];shard=d['source_shard'];side=d['original_group_sidecar']
        if (full9(root/path)!=tuple(d['source_full9'])
                or shard['rows']!=side['rows']):
            raise ValueError('Original training Arrow/sidecar identity changed')
        original=pq.read_table(pa.BufferReader(read_exact(shard,root=root)),
            columns=['task_id','record_id','split','source_family','source_revision',
                     'task_name','native_schema','native_payload_sha256'],use_threads=False)
        groups=pq.read_table(pa.BufferReader(read_exact(side,root=root)),use_threads=False)
        if (original.num_rows!=shard['rows'] or groups.num_rows!=shard['rows']
                or groups.column_names!=['task_id','effective_groups_json']):
            raise ValueError('Original Tox21 source/group row count or schema changed')
        for ordinal,(row,group) in enumerate(zip(original.to_pylist(),groups.to_pylist(),strict=True)):
            if (row['split']!='train' or row['source_family']!=SOURCE
                    or row['source_revision']!=_REV or row['task_name'] not in _ROLE
                    or row['native_schema']!='mora3.CorpusTask/v1'
                    or row['task_id']!=group['task_id']):
                raise ValueError('Original Tox21 training row identity/role changed')
            keys=json.loads(group['effective_groups_json'],object_pairs_hook=_unique)
            if (type(keys) is not list or not keys or len(set(keys))!=len(keys)
                    or not any(x.startswith('molecule:') for x in keys)
                    or not any(x.startswith('scaffold:') for x in keys)):
                raise ValueError('Original Tox21 chemical group sidecar changed')
            record=dict(task_id=row['task_id'],record_id=row['record_id'],
                payload_sha256=row['native_payload_sha256'],
                group_sha256=_sha(canonical(keys)),group_keys=tuple(keys),
                shard_index=shard_index,ordinal=ordinal)
            meta[(path,ordinal)]=record
            ordered[row['task_name']].append((path,ordinal))
        refs.append(dict(source_shard=shard,source_full9=list(d['source_full9']),
                         original_group_sidecar=side))
    if (len(meta)!=12892 or len(ordered[FORWARD])!=6446 or len(ordered[INVERSE])!=6446
            or sum(x['source_shard']['rows'] for x in refs)!=12892):
        raise ValueError('Original train shards/full mirrored-role denominator changed')
    for f,i in zip(ordered[FORWARD],ordered[INVERSE],strict=True):
        if f[0]!=i[0] or i[1]!=f[1]+1 or f[1]%2!=0:
            raise ValueError('Adjacent original Tox21 pair census changed')
        left,right=meta[f],meta[i]
        if left['record_id']!=right['record_id'] or left['group_keys']!=right['group_keys']:
            raise ValueError('Mirrored original molecule groups/record identity changed')
    return meta,ordered,refs


def audit_original_tox21_complete_train(root):
    """Exhaust real source train rows in each direction and bound source snapshot."""
    root=Path(root).resolve()
    if root!=_ORIGINAL or Path.cwd().resolve()!=root:
        raise ValueError('Exact original source checkout working directory required')
    context=load_inactive_common_source_rebind(root)
    proof=document(root,context.evidence)
    candidate=Ref(**proof['inactive_candidate_snapshot'])
    common=context.common_snapshot
    view,sources,vocabulary,_,_=load_original_assay_assets(root)
    source=sources['tox21']
    if (candidate.sha256==common.sha256 or
            finite_view_policy_sha256(view)!=_POLICY or
            source.axes.native_view_policy_sha256!=_POLICY or
            tuple(source.axes.binding()['full_original_assays'])!=ASSAYS):
        raise ValueError('Original Tox21 exact source/axis snapshots differ')
    route_plans={p['task_key']:p for p in proof['route_bindings'] if p['task_key'] in _ROLE}
    if set(route_plans)!=set(_ROLE):
        raise ValueError('Two frozen original Tox21 native route plans required')
    bound={}
    for direction in _ROLE:
        for label,snapshot in (('candidate',candidate),('common',common)):
            adapter=Tox21CampaignAdapter(source,view,vocabulary,snapshot.sha256,direction,128)
            binding=adapter.train_binding(batch_size=1)
            actual=dict(task_key=binding.route.task_key,
                initial_cursor_sha256=_sha(binding.route.initial_cursor),
                adapter_sha256=executable_adapter_sha256(binding.prepare,root=_CODE_ROOT),
                max_windows=2)
            if (actual!=route_plans[direction][label]
                    or binding.route.source_snapshot_sha256!=snapshot.sha256
                    or binding.route.native_view_policy_sha256!=_POLICY):
                raise ValueError('Frozen N29 train adapter, codec, batch, or cursor drift: '+direction)
            bound[(direction,label)]=(adapter,binding.route)
    paths=bound[(FORWARD,'candidate')][1]._paths
    if any(route._paths!=paths for _,route in bound.values()):
        raise ValueError('Candidate/common Tox21 frozen original shard scope differs')
    meta,ordered,files=_source_train_metadata(root,view,paths)
    results={}
    boundary={}
    for direction in _ROLE:
        prepared={}
        positions={(label):route._cursor_position(route.initial_cursor)
                   for (name,label),(_,route) in bound.items() if name==direction}
        with ExitStack() as stack:
            streams={}
            for label in ('candidate','common'):
                route=bound[(direction,label)][1]
                it=route.iter_from(route.initial_cursor)
                stack.callback(it.close)
                streams[label]=it
            rows=[]
            for index,coord in enumerate(ordered[direction]):
                windows={};signatures={}
                for label in ('candidate','common'):
                    adapter,route=bound[(direction,label)]
                    try:
                        window=next(streams[label])
                    except StopIteration as exc:
                        raise ValueError('Original Tox21 source exhausted before complete census') from exc
                    if (window.original_source_rows!=1 or window.batch.original_split!='train'
                            or (window.batch.source_shard['path'],
                                window.batch.source_row_ordinals[0])!=coord):
                        raise ValueError('Original Tox21 source row/role/ordinal drift')
                    after=route._cursor_position(window.cursor_after)
                    _checked_two_window_transition(positions[label],after)
                    if (index+1<len(ordered[direction])
                            and ordered[direction][index+1][0]!=coord[0]):
                        # Forward rows precede inverse rows: their last yielded
                        # cursor still points at the unselected inverse twin.
                        next_shard=meta[ordered[direction][index+1]]['shard_index']
                        boundary[(direction,label,next_shard)]=window.cursor_after
                    positions[label]=after
                    window.batch.native_rows.num_rows
                    raw,=window.batch.native_rows.to_pylist()
                    ref=reference_original_tox21(raw['native_payload'],direction)
                    v,=source.views(window.batch,split='train',direction=direction)
                    if (ref['native_mask_sha256']!=_sha(canonical(list(v.observed_mask)))
                            or ref['native_call_signature_sha256']!=
                                _sha(canonical((ASSAYS,list(v.values),list(v.observed_mask))))
                            or ref['smi_sha256']!=_sha(
                                (v.inputs() if direction==FORWARD else v.targets())
                                ['canonical_isomeric_smiles'].encode('utf-8'))
                            or raw['native_payload_sha256']!=_sha(raw['native_payload'])):
                        raise ValueError('Independent native Tox21 assay mask, role or SMILES differs')
                    objective=adapter.prepare(window)
                    if (objective.unavailable_reason is not None
                            or objective.evaluate is None
                            or not callable(objective.evaluate)):
                        raise ValueError('Original Tox21 observed native objective gate changed')
                    expected_positions=(ref['observed_positive']+ref['observed_negative']
                        if direction==FORWARD else
                        len(adapter._codec.encode(
                            (v.targets()['canonical_isomeric_smiles']).encode('utf-8')))-1)
                    if objective.observed_positions!=expected_positions:
                        raise ValueError('Tox21 source-only native observed positions changed')
                    windows[label]=window
                    signatures[label]=(ref,expected_positions,_source_fingerprint(window))
                left,right=windows['candidate'],windows['common']
                ref,positions_count,fingerprint=signatures['candidate']
                _checked_rebound_position(positions['candidate'],positions['common'])
                if (left.batch.source_shard!=right.batch.source_shard
                        or left.batch.source_row_ordinals!=right.batch.source_row_ordinals
                        or not left.batch.native_rows.equals(right.batch.native_rows,check_metadata=True)
                        or signatures['candidate']!=signatures['common']
                        or left.cursor_after==right.cursor_after):
                    raise ValueError('Original Tox21 payload/target differs after snapshot rebind')
                record=meta[coord]
                raw,=left.batch.native_rows.to_pylist()
                if (raw['task_id']!=record['task_id'] or raw['record_id']!=record['record_id']
                        or raw['native_payload_sha256']!=record['payload_sha256']):
                    raise ValueError('Original Arrow row disagrees with independent sidecar')
                rows.append(dict(index=index,shard_index=record['shard_index'],
                    source_shard_sha256=left.batch.source_shard['sha256'],
                    source_ordinal=coord[1],task_id=record['task_id'],
                    record_id=record['record_id'],
                    source_payload_sha256=record['payload_sha256'],
                    native_source_fingerprint=fingerprint[0],
                    group_sha256=record['group_sha256'],
                    original_mask_sha256=ref['native_mask_sha256'],
                    original_call_signature_sha256=ref['native_call_signature_sha256'],
                    original_smiles_sha256=ref['smi_sha256'],
                    source_ref_sha256=ref['source_ref_sha256'],
                    observed_positive=ref['observed_positive'],
                    observed_negative=ref['observed_negative'],
                    missing_unknown=ref['missing_unknown'],
                    native_observed_positions=positions_count,
                    candidate_cursor_sha256=_sha(left.cursor_after),
                    common_cursor_sha256=_sha(right.cursor_after),
                    next_position=list(positions['candidate'])))
            for it in streams.values():
                try:next(it)
                except StopIteration:pass
                else:raise ValueError('Original Tox21 route contains unaccounted extra rows')
        terminal=((50,files[-1]['source_shard']['rows']-1)
                  if direction==FORWARD else (51,0))
        if positions['candidate']!=terminal or positions['common']!=terminal:
            raise ValueError('Original mirrored Tox21 route did not exhaust its eligible train roles')
        results[direction]=rows
    # Verify independent fresh cursor construction at every original shard boundary.
    for (direction,label,shard_index),cursor in boundary.items():
        adapter,route=bound[(direction,label)]
        w=_first(route.iter_from(cursor))
        expected=next(row for row in results[direction] if row['shard_index']==shard_index)
        raw,=w.batch.native_rows.to_pylist()
        if (_source_fingerprint(w)[0]!=expected['native_source_fingerprint']
                or _sha(w.cursor_after)!=expected[label+'_cursor_sha256']
                or reference_original_tox21(raw['native_payload'],direction)
                    ['native_call_signature_sha256']!=expected['original_call_signature_sha256']):
            raise ValueError('Tox21 source shard fresh cursor restart changed native role or target')
    if len(boundary)!=200:
        raise ValueError('Both directions and N29 snapshots need every cross-shard restart')
    pairs=[];total=Counter()
    for forward,inverse in zip(results[FORWARD],results[INVERSE],strict=True):
        if (forward['record_id']!=inverse['record_id']
                or forward['source_shard_sha256']!=inverse['source_shard_sha256']
                or inverse['source_ordinal']!=forward['source_ordinal']+1
                or forward['group_sha256']!=inverse['group_sha256']
                or forward['original_smiles_sha256']!=inverse['original_smiles_sha256']
                or forward['original_mask_sha256']!=inverse['original_mask_sha256']
                or forward['original_call_signature_sha256']!=inverse['original_call_signature_sha256']
                or forward['source_ref_sha256']!=inverse['source_ref_sha256']):
            raise ValueError('Original Tox21 mirrored molecule/assay source evidence mismatched')
        for name in ('observed_positive','observed_negative','missing_unknown'):
            if forward[name]!=inverse[name]:
                raise ValueError('Tox21 mirrored source masks changed')
            total[name]+=forward[name]
        pairs.append(dict(
            record_id=forward['record_id'],
            source_shard_sha256=forward['source_shard_sha256'],
            source_ordinals=[forward['source_ordinal'],inverse['source_ordinal']],
            forward_task_id=forward['task_id'],inverse_task_id=inverse['task_id'],
            group_sha256=forward['group_sha256'],
            original_mask_sha256=forward['original_mask_sha256'],
            call_signature_sha256=forward['original_call_signature_sha256'],
            source_smiles_sha256=forward['original_smiles_sha256'],
            source_ref_sha256=forward['source_ref_sha256'],
            forward_observed_positions=forward['native_observed_positions'],
            inverse_byte_token_positions=inverse['native_observed_positions']))
    if len(pairs)!=6446 or sum(total.values())!=len(pairs)*12:
        raise ValueError('Complete mirrored 12-assay count changed')
    # Original SHA/full9 proof again after completion, including independent group sidecars.
    for f in files:
        if full9(root/f['source_shard']['path'])!=tuple(f['source_full9']):
            raise ValueError('Original Tox21 source changed during audit')
        read_exact(f['source_shard'],root=root)
        read_exact(f['original_group_sidecar'],root=root)
    return dict(schema='Mora3.Tox21CompleteOriginalTrainReplay/v1',
        source_family=SOURCE,original_source_revision=_REV,
        candidate_source_snapshot_sha256=candidate.sha256,
        common_source_snapshot_sha256=common.sha256,
        source_view_policy_sha256=_POLICY,assay_axis=list(ASSAYS),
        original_train_role_counts={direction:len(results[direction]) for direction in _ROLE},
        original_train_task_rows=12892,source_train_shards=51,
        independent_mirrored_original_pairs=len(pairs),
        total_qualified_train_source_records=12892,
        original_source_files=files,pairs=pairs,
        pair_sha256=_sha(canonical(pairs)),
        original_task_rows_sha256={direction:_sha(canonical(results[direction])) for direction in _ROLE},
        call_state_counts=dict(total),
        forward_observed_target_positions=sum(r['native_observed_positions'] for r in results[FORWARD]),
        inverse_source_target_tokens=sum(r['native_observed_positions'] for r in results[INVERSE]),
        candidate_common_raw_native_equal=True,
        independent_raw_role_mask_target_equal=True,
        both_snapshot_iterators_exhausted=True,source_pair_mismatch_count=0,
        independent_cross_shard_restart_count=len(boundary),
        complete_original_train_source_readback=True,
        source_license_scope='US-government-original-Tox21-public-domain-provider-claim;no-biology-guarantee',
        unknown_is_never_inactive=True,
        no_dose_time_or_cell_context=True,
        heldout_target_payload_decoded=False,
        complete_global_group_split_admitted=False,
        source_study_independence_admitted=False,
        full_expanded_corpus_ready=False,
        same_owner_matched_validation_baseline=False,
        training_admission=False,optimizer_updates=0,actual_campaign_trials=0,
        target_hyperparameter_transfer_complete=False,
        scientific_scope='original train native mask and generative target parity;not experimental validation')


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(argv)
    output=args.output.resolve()
    if output.exists() or not output.parent.is_dir() or not output.is_relative_to(_CODE_ROOT/'outputs'):
        raise ValueError('Only a new isolated continuation-owned output path is allowed')
    proof=audit_original_tox21_complete_train(args.root)
    with output.open('xb') as stream:
        stream.write(json.dumps(proof,indent=2,sort_keys=True).encode()+b'\n')
    print(json.dumps({k:proof[k] for k in (
        'original_train_task_rows','source_train_shards','independent_mirrored_original_pairs',
        'call_state_counts','forward_observed_target_positions',
        'inverse_source_target_tokens','independent_cross_shard_restart_count',
        'training_admission','actual_campaign_trials')},sort_keys=True))


if __name__=='__main__':
    main()
