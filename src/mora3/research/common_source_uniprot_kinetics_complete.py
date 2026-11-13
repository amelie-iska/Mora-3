"""Exhaust ORIGINAL eligible UniProt kinetics train source under both N29 bindings.

This source-only qualification preserves all unretained original ordinals, study
and homology-group sidecars and typed Km/Vmax units. No loss, optimizer, global
split, validated reference, model admission, or counted campaign is executed.
"""
from collections import Counter
from contextlib import ExitStack
from hashlib import sha256
from pathlib import Path
import json
import math
import struct

from mora3.data.admet import canonical
from mora3.data.finite_union_native_rows import decode_disposition, full9, read_exact
from mora3.research.common_source_native_replay import (
    _first,_source_fingerprint,_checked_two_window_transition,_checked_rebound_position,
)
from mora3.research.common_source_uniprot_kinetics import (
    _TASK,_FAMILY,_EXPECTED_POLICY,_native_kinetics_signature,
)
from mora3.research.remaining_eight_registry import load_assets
from mora3.research.native_scalar_campaign_routes import TrainOnlyScalarCampaignAdapter
from mora3.research.primary_campaign_source_rebind import load_inactive_common_source_rebind
from mora3.research.primary_source_factory_preflight import Ref, document
from mora3.research.primary_native_routes import finite_view_policy_sha256
from mora3.research.mup_campaign import executable_adapter_sha256


def _sha(raw): return sha256(raw).hexdigest()


def _unique_pairs(items):
    d = {}
    for key, value in items:
        if key in d: raise ValueError('Duplicate native kinetic JSON key')
        d[key] = value
    return d


def _bad_constant(_):
    raise ValueError('Nonfinite native kinetic value')


def _bits(value):
    if type(value) not in (int,float) or not math.isfinite(value):
        raise ValueError('Finite numeric source-reported kinetic coordinate required')
    return struct.pack('>d',value).hex()


def independent_kinetic_reference(raw):
    """Independent original JSON reference, without native scalar projector.

    Evidence is source-reported, not clinical proof. Native literal units are
    never converted; Km, Vmax and the absent assay context are not conflated.
    """
    if type(raw) is not bytes or not raw:raise ValueError('Original native kinetic bytes required')
    try:
        p=json.loads(raw,object_pairs_hook=_unique_pairs,parse_constant=_bad_constant)
        if (type(p) is not dict or set(p)!=set(('schema','task','conditions','targets','audit','missing'))
                or p['schema']!='mora3.CorpusTask/v1' or p['task']!=_TASK
                or type(p['conditions']) is not dict
                or set(p['conditions'])!=set(('amino_acid_sequence','measurement_type','substrate_alias','unit'))
                or type(p['targets']) is not dict or set(p['targets'])!=set(('value',))
                or type(p['audit']) is not dict or type(p['missing']) is not list):
            raise ValueError('Original source kinetic envelope, roles and target exclusion required')
        c=p['conditions']; a=p['audit']; m=a['original_measurement']
        kind=c['measurement_type']
        if kind not in ('Km','Vmax') or type(kind) is not str:
            raise ValueError('Exact original Km/Vmax distinction required')
        key='constant' if kind=='Km' else 'velocity'
        alias='substrate' if kind=='Km' else 'enzyme'
        if (type(c['amino_acid_sequence']) is not str or not c['amino_acid_sequence']
                or type(c['unit']) is not str or not c['unit']
                or type(c['substrate_alias']) is not str or not c['substrate_alias']
                or type(m) is not dict or not all(k in m for k in (key,alias,'unit'))
                or type(m['unit']) is not str or m['unit']!=c['unit']
                or m[alias]!=c['substrate_alias']
                or any(k in m for k in ('pH','temperature','method','assay_method','molecule'))
                or not all(type(k) is str for k in p['missing'])):
            raise ValueError('Original kinetic aliases, literal unit and missing assay context required')
        value=_bits(p['targets']['value'])
        if value!=_bits(m[key]):
            raise ValueError('Original measurement differs from target numeric coordinate')
        lexical=json.loads(raw,object_pairs_hook=_unique_pairs,
                           parse_int=str,parse_float=str,parse_constant=_bad_constant)
        lexeme=lexical['targets']['value']
        if type(lexeme) is not str:
            raise ValueError('Native numeric JSON lexeme required')
        return dict(measurement_type=kind,literal_unit=c['unit'],value_float64_bits=value,
            numeric_lexeme=lexeme,
            original_measurement_sha256=_sha(canonical(m)),
            protein_sequence_sha256=_sha(c['amino_acid_sequence'].encode()),
            original_alias_sha256=_sha(c['substrate_alias'].encode()))
    except (KeyError,TypeError,UnicodeError,OverflowError) as exc:
        raise ValueError('Incomplete or malformed original native kinetic envelope') from exc


def kinetic_declared_group_components(records):
    """Transitive connected components of DECLARED original train-only groups.

    A shared accession, paper, sequence or detected homology guard can bridge
    records. This is not an independent global biological split certification.
    """
    if type(records) is not tuple or not records:raise ValueError('Immutable nonempty group census required')
    parents=list(range(len(records))); owner={}; seen=set(); n_groups=set()
    def find(i):
        while parents[i]!=i:
            parents[i]=parents[parents[i]]
            i=parents[i]
        return i
    for i,row in enumerate(records):
        if (type(row) is not dict or set(row)!=set(('task_id','groups'))
                or type(row['task_id']) is not str or len(row['task_id'])!=64
                or type(row['groups']) is not tuple or not row['groups']
                or any(type(g) is not str or ':' not in g for g in row['groups'])):
            raise ValueError('Complete literal kinetic group row required')
        if row['task_id'] in seen:raise ValueError('Duplicate source task ID must not be coalesced')
        seen.add(row['task_id'])
        for g in row['groups']:
            n_groups.add(g)
            if g in owner:
                parents[find(i)]=find(owner[g])
            else:owner[g]=i
    components={}
    for i,row in enumerate(records):
        components.setdefault(find(i),[]).append(row['task_id'])
    ordered=sorted([sorted(v) for v in components.values()],key=lambda v:_sha(canonical(v)))
    return dict(source_rows=len(records),declared_groups=len(n_groups),
        connected_components=len(ordered),largest_component_rows=max(map(len,ordered)),
        component_size_histogram=dict(sorted(Counter(map(len,ordered)).items())),
        declared_group_component_commitment=_sha(canonical(ordered)),
        global_split_admitted=False,
        scope='exact original UniProt kinetics effective train-only sidecars; no cross-source closure')


def _original_role_and_groups(root,view,paths):
    import pyarrow as pa
    import pyarrow.parquet as pq
    desc={d['source_shard']['path']:d for d in view.descriptors if d['source_shard']['path'] in paths}
    if set(desc)!=set(paths) or len(desc)!=3:raise ValueError('Exactly three original kinetic source shards required')
    records={}; excluded=[]; ordered=[]; ref_files=[]
    for path in paths:
        d=desc[path]; shard=d['source_shard']
        if full9(root/path)!=tuple(d['source_full9']):
            raise ValueError('Original UniProt kinetics full9 source identity changed')
        # Read exact immutable source and sidecar as authenticated bytes.
        raw=read_exact(shard,root=root)
        original=pq.read_table(pa.BufferReader(raw),columns=['task_id','record_id','split',
            'source_family','source_revision','task_name','native_schema','group_ids_json'])
        if original.num_rows!=shard['rows']:
            raise ValueError('Original UniProt kinetic shard length changed')
        side=d['original_group_sidecar']
        groups=pq.read_table(pa.BufferReader(read_exact(side,root=root)))
        if groups.num_rows!=shard['rows'] or set(groups.schema.names)!=set((
                'task_id','source_file','source_group_ids_json','effective_groups_json','evidence_scope')):
            raise ValueError('Original native protein group sidecar schema/length changed')
        dispositions=decode_disposition(d,root=root)
        if len(dispositions)!=shard['rows']:
            raise ValueError('Original ordinal disposition length changed')
        source=original.to_pylist(); grouped=groups.to_pylist()
        for ordinal,(row,g,code) in enumerate(zip(source,grouped,dispositions,strict=True)):
            if (row['task_id']!=g['task_id'] or g['source_file']!=path
                    or g['evidence_scope']!='identity/leakagegroupsonly;notpredictorfeaturesorsamecell/effectproof'
                    or row['split']!='train' or row['source_family']!=_FAMILY
                    or row['task_name']!=_TASK or row['native_schema']!='mora3.CorpusTask/v1'
                    or row['source_revision']!='UniProt2026_03-nonrandom-first100'):
                raise ValueError('Original kinetic provenance, source family, role or sidecar changed')
            source_groups=json.loads(g['source_group_ids_json'],object_pairs_hook=_unique_pairs)
            effective_groups=json.loads(g['effective_groups_json'],object_pairs_hook=_unique_pairs)
            row_groups=json.loads(row['group_ids_json'],object_pairs_hook=_unique_pairs)
            if (type(source_groups) is not list or not source_groups
                    or source_groups!=row_groups or type(effective_groups) is not list
                    or len(effective_groups)!=len(set(effective_groups))
                    or not set(source_groups).issubset(effective_groups)
                    or any(type(x) is not str or ':' not in x for x in effective_groups)):
                raise ValueError('Original and effective UniProt group identities disagree')
            if code!=0 or (path,ordinal) in view._ordinal_exclusions:
                excluded.append(dict(source_shard_sha256=shard['sha256'],ordinal=ordinal,code=code,
                    task_id_sha256=_sha(row['task_id'].encode())))
                continue
            coordinate=(path,ordinal)
            ordered.append(coordinate)
            records[coordinate]=dict(task_id=row['task_id'],record_id=row['record_id'],
                source_groups=tuple(source_groups),effective_groups=tuple(effective_groups),
                source_group_sha256=_sha(canonical(source_groups)),
                effective_group_sha256=_sha(canonical(effective_groups)))
        ref_files.append(dict(source_shard=shard,original_group_sidecar=side,
            disposition=d['disposition'],source_full9=d['source_full9']))
    if len(ordered)!=618 or len(excluded)!=4:
        raise ValueError('Expected exactly 618 retained and four quarantined original kinetic source rows')
    return records,ordered,excluded,ref_files


def audit_complete_uniprot_kinetics_train(root):
    """Read every eligible original train coordinate under BOTH frozen snapshots."""
    root=Path(root).resolve()
    if not root.is_dir() or Path.cwd().resolve()!=root:
        raise ValueError('Original source checkout working directory required')
    context=load_inactive_common_source_rebind(root)
    evidence=document(root,context.evidence)
    binding_record,=(x for x in evidence['route_bindings'] if x['task_key']==_TASK)
    if (binding_record['candidate']['max_windows']!=2 or binding_record['common']['max_windows']!=2
            or binding_record['common_view_policy_sha256']!=_EXPECTED_POLICY
            or context.common_snapshot.sha256!=evidence['common_source_snapshot']['sha256']):
        raise ValueError('Original reviewed source bindings or frozen per-step window limit changed')
    candidate=Ref(**evidence['inactive_candidate_snapshot']);common=context.common_snapshot
    if candidate.sha256==common.sha256:raise ValueError('Distinct N29 source snapshot pins required')
    view,original,_,_,axis,_,_,_,_=load_assets(root)
    if finite_view_policy_sha256(view)!=_EXPECTED_POLICY:raise ValueError('Original view policy changed')
    bound={}
    for name,snapshot in (('candidate',candidate),('common',common)):
        adapter=TrainOnlyScalarCampaignAdapter(original,view,axis,snapshot.sha256,_FAMILY)
        b=adapter.train_binding(batch_size=1)
        actual=dict(task_key=b.route.task_key,initial_cursor_sha256=_sha(b.route.initial_cursor),
            adapter_sha256=executable_adapter_sha256(b.prepare,root=Path(__file__).resolve().parents[3]),
            max_windows=2)
        if (actual!=binding_record[name] or b.route.source_snapshot_sha256!=snapshot.sha256
                or b.route.native_view_policy_sha256!=_EXPECTED_POLICY):
            raise ValueError('Original executable scalar adapter/cursor pin changed')
        bound[name]=(adapter,b.route)
    paths=bound['candidate'][1]._paths
    if paths!=bound['common'][1]._paths:
        raise ValueError('Candidate and common original kinetic shards differ')
    metadata,expected,excluded,ref_files=_original_role_and_groups(root,view,paths)
    groups=kinetic_declared_group_components(tuple(dict(task_id=metadata[c]['task_id'],
        groups=metadata[c]['effective_groups']) for c in expected))
    positions={name:route._cursor_position(route.initial_cursor) for name,(_,route) in bound.items()}
    rows=[];counts=Counter();units=Counter();boundaries={}
    prior_coordinate=None;prior_cursors={}
    with ExitStack() as stack:
        streams={}
        for name,(_,route) in bound.items():
            streams[name]=route.iter_from(route.initial_cursor)
            stack.callback(streams[name].close)
        for index,coordinate in enumerate(expected):
            windows={};signatures={}
            for name,(adapter,route) in bound.items():
                try:w=next(streams[name])
                except StopIteration as exc:
                    raise ValueError('Original kinetic iterator exhausted before all eligible rows') from exc
                if (w.original_source_rows!=1 or w.batch.original_split!='train'
                        or (w.batch.source_shard['path'],w.batch.source_row_ordinals[0])!=coordinate):
                    raise ValueError('Complete original kinetic ordered row/ordinal differs')
                pos=route._cursor_position(w.cursor_after)
                _checked_two_window_transition(positions[name],pos)
                positions[name]=pos
                windows[name]=w
                signatures[name]=_native_kinetics_signature(adapter,w)
            left,right=windows['candidate'],windows['common']
            sig=signatures['candidate']
            _checked_rebound_position(positions['candidate'],positions['common'])
            if (left.batch.source_shard!=right.batch.source_shard
                    or left.batch.source_row_ordinals!=right.batch.source_row_ordinals
                    or not left.batch.native_rows.equals(right.batch.native_rows,check_metadata=True)
                    or sig!=signatures['common'] or _source_fingerprint(left)!=_source_fingerprint(right)):
                raise ValueError('Original kinetic native rows, typed values or cursor replay changed')
            native,=left.batch.native_rows.to_pylist()
            ref=independent_kinetic_reference(native['native_payload'])
            if (ref['value_float64_bits']!=sig['original_float64_bits']
                    or ref['numeric_lexeme']!=sig['original_numeric_lexeme']
                    or ref['measurement_type']!=sig['measurement_type']
                    or ref['literal_unit']!=sig['literal_source_unit']
                    or ref['protein_sequence_sha256']!=_sha(
                        json.loads(native['native_payload'])['conditions']['amino_acid_sequence'].encode())):
                raise ValueError('Independent native protein/kinetics reference disagrees with scalar projector')
            meta=metadata[coordinate]
            if native['task_id']!=meta['task_id'] or native['record_id']!=meta['record_id']:
                raise ValueError('Original kinetic record/sidecar identity differs')
            if prior_coordinate is not None and coordinate[0]!=prior_coordinate[0]:
                for name in bound:
                    boundaries[(name,index)]=prior_cursors[name]
            counts[sig['measurement_type']]+=1
            units[sig['measurement_type']+':'+sig['literal_source_unit']]+=1
            rows.append(dict(index=index,source_shard_sha256=left.batch.source_shard['sha256'],
                source_ordinal=coordinate[1],source_payload_sha256=native['native_payload_sha256'],
                source_fingerprint_sha256=_source_fingerprint(left)[0],
                original_numeric_lexeme=sig['original_numeric_lexeme'],
                independent_lexeme=ref['numeric_lexeme'],
                original_float64_bits=sig['original_float64_bits'],
                independent_float64_bits=ref['value_float64_bits'],
                measurement_type=sig['measurement_type'],literal_source_unit=sig['literal_source_unit'],
                condition_sha256=sig['condition_sha256'],native_target_metadata_sha256=sig['native_target_metadata_sha256'],
                source_group_sha256=meta['source_group_sha256'],
                effective_group_sha256=meta['effective_group_sha256'],
                physical_missingness=sig['physical_missingness'],
                candidate_cursor_sha256=_sha(left.cursor_after),
                common_cursor_sha256=_sha(right.cursor_after),
                next_position=list(positions['candidate'])))
            prior_coordinate=coordinate
            prior_cursors={name:w.cursor_after for name,w in windows.items()}
        for stream in streams.values():
            try:next(stream)
            except StopIteration:pass
            else:raise ValueError('Unaccounted original kinetic iterator tail rows')
    if (len(rows)!=618 or positions['candidate']!=(3,0)
            or positions['common']!=(3,0) or len(boundaries)!=4):
        raise ValueError('Complete original three-shard kinetic cursor/exhaustion proof differs')
    for (name,index),cursor in boundaries.items():
        adapter,route=bound[name]
        restarted=_first(route.iter_from(cursor))
        row=rows[index]
        if (restarted.batch.source_shard['sha256']!=row['source_shard_sha256']
                or restarted.batch.source_row_ordinals!=(row['source_ordinal'],)
                or _source_fingerprint(restarted)[0]!=row['source_fingerprint_sha256']
                or _sha(restarted.cursor_after)!=row[name+'_cursor_sha256']
                or _native_kinetics_signature(adapter,restarted)['original_float64_bits']!=row['original_float64_bits']):
            raise ValueError('Original source-shard boundary restart changed kinetic source row')
    for file in ref_files:
        shard=file['source_shard']; p=root/shard['path']
        if full9(p)!=tuple(file['source_full9']):
            raise ValueError('Kinetic original source changed after complete replay')
        read_exact(shard,root=root)
        read_exact(file['original_group_sidecar'],root=root)
    return dict(schema='Mora3.UniProtKineticsCompleteOriginalTrainReplay/v1',
        source_family=_FAMILY,source_revision='UniProt2026_03-nonrandom-first100',
        candidate_source_snapshot_sha256=candidate.sha256,common_source_snapshot_sha256=common.sha256,
        original_view_policy_sha256=_EXPECTED_POLICY,source_files=ref_files,
        original_shards=3,original_source_rows=622,excluded_source_ordinals=len(excluded),
        excluded_original_coordinates=excluded,
        native_train_rows=618,observed_source_coordinates=len(expected),rows=rows,
        ordered_rows_sha256=_sha(canonical(rows)),
        measurement_type_counts=dict(sorted(counts.items())),
        literal_quantity_unit_counts=dict(sorted(units.items())),
        effective_group_census=groups,source_boundary_restarts_checked=len(boundaries),
        final_cursor_position=list(positions['candidate']),
        both_original_train_iterators_exhausted=True,candidate_common_all_rows_equal=True,
        independent_native_scalar_references_equal=True,frozen_per_trial_window_limit_unchanged=2,
        no_heldout_targets_decoded=True,full_expanded_corpus_ready=False,all62_route_replay=False,
        global_split_admitted=False,training_admission=False,matched_owner_validation=False,
        optimizer_updates=0,actual_proxy_trials=0,actual_target_runs=0,
        source_scope='read-only entire eligible train route; source-reported not independent experimental validation')
