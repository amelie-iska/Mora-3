"""Exhaustive original AZ logD training-source conservation under both frozen N29 bindings.

This is a source-only audit, not a campaign training run or a validation reference.
Original source study/assay associations are not independent held-out groups.
"""
from collections import Counter
from contextlib import ExitStack
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from pathlib import Path
import json
import math
import re
import struct

from mora3.data.admet import canonical
from mora3.data.finite_union_native_rows import full9, read_exact
from mora3.research.common_source_logd import _TASK, _FAMILY, _EXPECTED_POLICY, _native_logd_signature
from mora3.research.common_source_native_replay import (
    _source_fingerprint, _first, _checked_two_window_transition, _checked_rebound_position,
)
from mora3.research.remaining_eight_registry import load_assets
from mora3.research.native_scalar_campaign_routes import TrainOnlyScalarCampaignAdapter
from mora3.research.primary_campaign_source_rebind import load_inactive_common_source_rebind
from mora3.research.primary_source_factory_preflight import Ref, document
from mora3.research.primary_native_routes import finite_view_policy_sha256
from mora3.research.mup_campaign import executable_adapter_sha256

_ORIGINAL = Path('/home/iska/Documents/amelie/bio/Mora-3')
_REV = 'ChEMBL_37:2026-05-01;original_dataset_CHEMBL_20;content_snapshot20261031'
_KEYS = ('assay', 'chembl_molecule', 'chembl_parent', 'molecule', 'scaffold', 'source_document')
_META_COLUMNS = ('task_id', 'record_id', 'source_family', 'source_revision',
                 'split', 'task_name', 'native_schema')


def _hash(value):
    return sha256(value).hexdigest()


def _unique_pairs(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError('Duplicate original JSON key')
        out[key] = value
    return out


def _reject_nonfinite(_):
    raise ValueError('Nonfinite original JSON coordinate')


def reference_logd_original(payload):
    """Independent original JSON numeric/string reference; no projection library."""
    if type(payload) is not bytes or not payload:
        raise ValueError('Original native payload bytes required')
    try:
        data = json.loads(payload, object_pairs_hook=_unique_pairs, parse_constant=_reject_nonfinite)
        literal = json.loads(payload, object_pairs_hook=_unique_pairs, parse_int=str,
                             parse_float=str, parse_constant=_reject_nonfinite)
        if (type(data) is not dict or set(data) != {'schema','task','conditions','targets','audit','missing'}
                or data['schema'] != 'mora3.CorpusTask/v1' or data['task'] != _TASK
                or type(data['conditions']) is not dict
                or set(data['conditions']) != {'canonical_isomeric_smiles','method','pH','solvent_pair'}
                or type(data['conditions']['canonical_isomeric_smiles']) is not str
                or not data['conditions']['canonical_isomeric_smiles']
                or data['conditions']['pH'] != 7.4
                or data['conditions']['method'] != 'shake_flask'
                or data['conditions']['solvent_pair'] != ['octan-1-ol','water']
                or set(data['targets']) != {'source_reported_logD7_4'}
                or type(data['targets']['source_reported_logD7_4']) not in (int,float)
                or not math.isfinite(data['targets']['source_reported_logD7_4'])
                or not -1.5 < data['targets']['source_reported_logD7_4'] < 4.5):
            raise ValueError('Exact source-native nominal AZ logD7.4 numeric envelope required')
        a = data['audit']
        source = a['source_activity']
        if (a['source_relation'] is not None or a['source_units'] is not None
                or source['relation'] is not None or source['standard_relation'] is not None
                or source['units'] is not None or source['standard_units'] is not None
                or source['type'] != 'LogD7.4' or source['standard_type'] != 'LogD7.4'
                or type(source['standard_value']) is not str or type(source['value']) is not str
                or 'source_relation_unspecified' not in data['missing']):
            raise ValueError('Do not invent source relation, physical unit or evidence role')
        coordinate = data['targets']['source_reported_logD7_4']
        lexeme = literal['targets']['source_reported_logD7_4']
        if (type(lexeme) is not str or
                Decimal(lexeme) != Decimal(source['value']) or
                Decimal(lexeme) != Decimal(source['standard_value'])):
            raise ValueError('Original source string does not match native numeric target')
        return dict(float64_bits=struct.pack('>d', coordinate).hex(),
                    numeric_lexeme=lexeme, source_value_text=source['value'],
                    condition_sha256=_hash(canonical(data['conditions'])),
                    source_relation=None, literal_source_unit=None)
    except (KeyError, TypeError, OverflowError, UnicodeError, InvalidOperation, json.JSONDecodeError) as exc:
        raise ValueError('Incomplete or inconsistent original AZ native payload') from exc


def declared_logd_group_census(records):
    """Transitive closure of exact original sidecar keys; no chemical similarity inference."""
    if type(records) is not tuple or not records:
        raise ValueError('Original nonempty immutable group tuples required')
    parents = list(range(len(records)))
    owners = {}
    seen_rows, seen_tasks = set(), set()
    counts = Counter()
    group_rows = Counter()

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    for i, record in enumerate(records):
        if (type(record) is not dict or set(record) != {'record_id','task_id','groups'}
                or type(record['record_id']) is not str or
                not re.fullmatch(r'chembl37:activity:[0-9]+',record['record_id'])
                or type(record['task_id']) is not str or
                not re.fullmatch('[0-9a-f]{64}',record['task_id'])
                or type(record['groups']) is not tuple or not record['groups']
                or any(type(g) is not str or ':' not in g or
                       g.split(':',1)[0] not in _KEYS or not g.split(':',1)[1]
                       for g in record['groups'])
                or len(record['groups']) != len(set(record['groups']))):
            raise ValueError('Original AZ task identity and declared sidecar group keys required')
        if record['record_id'] in seen_rows or record['task_id'] in seen_tasks:
            raise ValueError('Duplicate original source identity must not be silently collapsed')
        seen_rows.add(record['record_id']);seen_tasks.add(record['task_id'])
        for group in record['groups']:
            kind = group.split(':',1)[0]
            counts[kind] += 1;group_rows[group] += 1
            if group in owners:
                parents[find(i)] = find(owners[group])
            else:
                owners[group] = i
    components = Counter(find(i) for i in range(len(records)))
    return dict(records=len(records),component_count=len(components),
        largest_component=max(components.values()),
        group_count_by_axis={k:len({g for g in owners if g.startswith(k+':')}) for k in _KEYS},
        shared_assay_groups=sum(1 for g,n in group_rows.items() if g.startswith('assay:') and n>1),
        shared_document_groups=sum(1 for g,n in group_rows.items() if g.startswith('source_document:') and n>1),
        all_declared_study_grouped_together=len(components)==1,
        scope='source-local declared original study/assay/molecule/scaffold keys; no global cross-source equivalence',
        global_chemical_split_admitted=False)


def _original_metadata(root, view):
    import pyarrow as pa
    import pyarrow.parquet as pq
    descriptors = sorted((d for d in view.descriptors if any(
        r['source_family']==_FAMILY for r in d['source_task_schema_counts'])),
        key=lambda d:d['source_shard']['path'])
    if len(descriptors)!=17:
        raise ValueError('Exactly 17 original AZ source shards required')
    records, locations, files = [], {}, []
    for index, d in enumerate(descriptors):
        shard=d['source_shard']
        if not shard['path'].endswith(f'/shards/train/{index:05d}.parquet'):
            raise ValueError('Original ordered ChEMBL train shard path changed')
        if full9(root/shard['path']) != tuple(d['source_full9']):
            raise ValueError('Original AZ full9 shard fingerprint changed')
        raw=read_exact(shard,root=root)
        side_raw=read_exact(d['original_group_sidecar'],root=root)
        table=pq.read_table(pa.BufferReader(raw),columns=list(_META_COLUMNS),use_threads=False)
        side=pq.read_table(pa.BufferReader(side_raw),use_threads=False)
        if (table.num_rows!=shard['rows'] or side.num_rows!=shard['rows']
                or d['original_metadata_role_counts']!={'train':shard['rows']}
                or side.column_names!=['task_id','effective_groups_json']
                or any(p==shard['path'] for p,_ in view._ordinal_exclusions)):
            raise ValueError('Original AZ role, group cardinality or exclusions changed')
        for ordinal,(meta,group) in enumerate(zip(table.to_pylist(),side.to_pylist(),strict=True)):
            if (meta['source_family']!=_FAMILY or meta['source_revision']!=_REV
                    or meta['split']!='train' or meta['task_name']!=_TASK
                    or meta['native_schema']!='mora3.CorpusTask/v1'
                    or group['task_id']!=meta['task_id']):
                raise ValueError('Original AZ source/train/group-role identity changed')
            groups=json.loads(group['effective_groups_json'])
            if (type(groups) is not list or len(groups)!=6
                    or {g.split(':',1)[0] for g in groups}!=set(_KEYS)):
                raise ValueError('All six original declared grouping axes required')
            record=dict(record_id=meta['record_id'],task_id=meta['task_id'],groups=tuple(groups))
            records.append(record)
            location=(shard['path'],ordinal)
            if location in locations:
                raise ValueError('Duplicate original shard coordinate')
            locations[location]=record
        files.append(dict(source_shard=dict(shard),source_full9=list(d['source_full9']),
                          original_group_sidecar=dict(d['original_group_sidecar'])))
    if len(records)!=4184 or sum(f['source_shard']['rows'] for f in files)!=4184:
        raise ValueError('Original complete AZ denominator changed')
    return locations,files,declared_logd_group_census(tuple(records))


def audit_original_logd_complete_train(root):
    """Conserve all 4184 original train rows across both source bindings and restarts."""
    root=Path(root).resolve()
    if root!=_ORIGINAL or Path.cwd().resolve()!=root:
        raise ValueError('Exact installed original source working-directory authority required')
    context=load_inactive_common_source_rebind(root)
    source=document(root,context.evidence)
    plans=[r for r in source['route_bindings'] if r['task_key']==_TASK]
    if len(plans)!=1:
        raise ValueError('One original reviewed route binding required')
    plan=plans[0]
    candidate=Ref(**source['inactive_candidate_snapshot'])
    common=context.common_snapshot
    if candidate.sha256==common.sha256:
        raise ValueError('Distinct candidate/common source snapshots required')
    view,original,_,_,axis,*_=load_assets(root)
    if (finite_view_policy_sha256(view)!=_EXPECTED_POLICY
            or plan['common_view_policy_sha256']!=_EXPECTED_POLICY):
        raise ValueError('Original AZ effective view policy differs from reviewed source')
    locations,files,census=_original_metadata(root,view)
    bound={}
    for label,snapshot in (('candidate',candidate),('common',common)):
        adapter=TrainOnlyScalarCampaignAdapter(original,view,axis,snapshot.sha256,_FAMILY)
        binding=adapter.train_binding(batch_size=1)
        actual=dict(task_key=binding.route.task_key,
                    initial_cursor_sha256=_hash(binding.route.initial_cursor),
                    adapter_sha256=executable_adapter_sha256(binding.prepare,root=Path(__file__).resolve().parents[3]),
                    max_windows=2)
        if (actual!=plan[label] or
                binding.route.native_view_policy_sha256!=_EXPECTED_POLICY or
                binding.route.source_snapshot_sha256!=snapshot.sha256):
            raise ValueError('AZ adapter/cursor/finite-budget pin drift')
        bound[label]=(adapter,binding.route)
    expected=[(f['source_shard']['path'],i) for f in files for i in range(f['source_shard']['rows'])]
    if tuple(expected)!=tuple((p,i) for p in bound['candidate'][1]._paths for i in range(
            next(f['source_shard']['rows'] for f in files if f['source_shard']['path']==p))):
        raise ValueError('Original ordered train shard view changed')
    positions={label:route._cursor_position(route.initial_cursor) for label,(_,route) in bound.items()}
    rows=[];boundary={}
    with ExitStack() as stack:
        streams={}
        for label,(_,route) in bound.items():
            streams[label]=route.iter_from(route.initial_cursor)
            stack.callback(streams[label].close)
        for index,coordinate in enumerate(expected):
            windows={};signatures={}
            for label,(adapter,route) in bound.items():
                try:window=next(streams[label])
                except StopIteration as exc:
                    raise ValueError('Original AZ source exhausted before row 4184') from exc
                if (window.original_source_rows!=1 or window.batch.original_split!='train' or
                        (window.batch.source_shard['path'],window.batch.source_row_ordinals[0])!=coordinate):
                    raise ValueError('Original AZ row/order/source-coordinate changed')
                after=route._cursor_position(window.cursor_after)
                _checked_two_window_transition(positions[label],after)
                if after[0]!=positions[label][0] and after[0]<17:
                    boundary[(label,after[0])]=window.cursor_after
                positions[label]=after
                windows[label]=window;signatures[label]=_native_logd_signature(adapter,window)
            a,b=windows['candidate'],windows['common']
            sa,sb=signatures['candidate'],signatures['common']
            _checked_rebound_position(positions['candidate'],positions['common'])
            if (a.batch.source_shard!=b.batch.source_shard or
                    a.batch.source_row_ordinals!=b.batch.source_row_ordinals or
                    not a.batch.native_rows.equals(b.batch.native_rows,check_metadata=True) or
                    _source_fingerprint(a)!=_source_fingerprint(b) or sa!=sb or
                    a.cursor_after==b.cursor_after):
                raise ValueError('Original AZ payload/typed scalar differs across N29 source binding')
            meta=locations[coordinate]
            raw_row,=a.batch.native_rows.to_pylist()
            if (raw_row['record_id']!=meta['record_id'] or
                    raw_row['task_id']!=meta['task_id'] or raw_row['source_revision']!=_REV):
                raise ValueError('Original AZ Arrow row identity differs from sidecar')
            ref=reference_logd_original(raw_row['native_payload'])
            if (ref['float64_bits']!=sa['original_float64_bits'] or
                    ref['numeric_lexeme']!=sa['original_numeric_lexeme']):
                raise ValueError('Independent original source numeric reference differs')
            rows.append(dict(index=index,source_shard_sha256=a.batch.source_shard['sha256'],
                source_ordinal=coordinate[1],record_id=meta['record_id'],
                native_source_fingerprint=_source_fingerprint(a)[0],
                native_payload_sha256=raw_row['native_payload_sha256'],
                source_relation=sa['source_relation'],literal_source_unit=sa['literal_source_unit'],
                original_float64_bits=sa['original_float64_bits'],independent_float64_bits=ref['float64_bits'],
                original_numeric_lexeme=sa['original_numeric_lexeme'],
                independent_numeric_lexeme=ref['numeric_lexeme'],
                source_group_sha256=_hash(canonical(meta['groups'])),
                original_raw_condition_sha256=ref['condition_sha256'],
                source_audit_sha256=sa['source_audit_sha256'],
                condition_sha256=sa['condition_sha256'],
                candidate_cursor_sha256=_hash(a.cursor_after),
                common_cursor_sha256=_hash(b.cursor_after),
                next_position=list(positions['candidate'])))
        for iterator in streams.values():
            try:next(iterator)
            except StopIteration:pass
            else:raise ValueError('Unaccounted extra original AZ train source row')
    if (positions['candidate']!=(17,0) or positions['common']!=(17,0)
            or len(boundary)!=32):
        raise ValueError('Original AZ source exhaustion/cursor boundary incomplete')
    for (label,shard_index),cursor in boundary.items():
        adapter,route=bound[label]
        window=_first(route.iter_from(cursor))
        row=rows[shard_index*256]
        if (_source_fingerprint(window)[0]!=row['native_source_fingerprint']
                or _hash(window.cursor_after)!=row[label+'_cursor_sha256']
                or _native_logd_signature(adapter,window)['original_float64_bits']!=row['original_float64_bits']):
            raise ValueError('Independent source shard restart altered original AZ row')
    for file in files:
        if full9(root/file['source_shard']['path'])!=tuple(file['source_full9']):
            raise ValueError('Original AZ complete source changed during readback')
        read_exact(file['source_shard'],root=root)
        read_exact(file['original_group_sidecar'],root=root)
    return dict(schema='Mora3.AZLogDCompleteOriginalTrainReplay/v1',
        source_family=_FAMILY,original_train_rows=4184,original_train_shards=17,
        original_native_observed_targets=4184,original_source_files=files,
        original_declared_group_census=census,rows=rows,
        ordered_original_row_digest=_hash(canonical(rows)),
        candidate_source_snapshot_sha256=candidate.sha256,
        common_source_snapshot_sha256=common.sha256,
        native_view_policy_sha256=_EXPECTED_POLICY,
        all_train_candidate_common_equal=True,all_train_independent_numeric_equal=True,
        both_native_iterators_exhausted=True,final_cursor_position=[17,0],
        independent_shard_restart_count=len(boundary),
        no_heldout_role_shards_for_this_route=True,heldout_target_payload_decoded=False,
        source_nominal_relation_unspecified=True,literal_units_unspecified=True,
        full_expanded_corpus_ready=False,global_chemical_split_admitted=False,
        complete62_route_replay=False,same_owner_matched_validation_baseline=False,
        independent_construction_model_attestations=False,training_admission=False,
        frozen_training_window_limit_unchanged=2,
        optimizer_updates=0,actual_proxy_runs=0,actual_target_runs=0,
        evidence_scope='complete original AZ train-source only; no optimizer or model evaluation')
