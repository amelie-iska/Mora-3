"""Original FreeSolv native train-source replay across frozen 29-head source IDs.

The retained two-window check and complete 556-row qualification are separate
scopes. Neither changes the campaign scan limits or source roles.

One bounded source-fidelity operation; no learned model, optimization, validation
baseline, release signature, clinical truth, or source eligibility is granted.
The original reported experimental point is retained in kcal/mol; source-listed
uncertainty is audit metadata, not a fabricated learned Gaussian noise scale.
"""
from hashlib import sha256
from pathlib import Path

from mora3.data.admet import canonical
from mora3.research.remaining_eight_registry import load_assets
from mora3.research.assay_campaign_routes import FreeSolvCampaignAdapter
from mora3.research.primary_campaign_source_rebind import load_inactive_common_source_rebind
from mora3.research.primary_source_factory_preflight import Ref,document
from mora3.research.primary_native_routes import finite_view_policy_sha256
from mora3.research.mup_campaign import executable_adapter_sha256
from mora3.research.common_source_native_replay import (
    _source_fingerprint,_first,_checked_rebound_position,_checked_two_window_transition,
)

_TASK='experimental_hydration_free_energy'


def _sha(x):return sha256(x).hexdigest()


def _native_scalar_signature(adapter,window):
    """Authenticate one complete source scalar without invoking a model."""
    from mora3.data.source_scalar_native_rows import project_native_scalar_batch
    source=adapter._targets(window.batch,'train')
    separately=project_native_scalar_batch(adapter.original_view,window.batch,split='train')
    if source!=separately or source.original_split!='train' or len(source.scalar_rows)!=1:
        raise ValueError('Exact original FreeSolv scalar train target required')
    item,=source.scalar_rows
    target=item.target
    if (item.source_family!='FreeSolv' or item.task_name!=_TASK or item.original_split!='train'
            or item.disposition!='qualified-native-reported-source-coordinate' or target.kind!='reported_point'
            or target.reported_value is None or not target.numeric_coordinate_available
            or item.partition_key[:2]!=('FreeSolv',_TASK)
            or item.partition_key[3]!='kcal/mol'):
        raise ValueError('Original reported FreeSolv point, units and role are required; no imputed targets')
    prepared=adapter.prepare(window)
    if prepared.observed_positions!=1 or prepared.unavailable_reason is not None:
        raise ValueError('Native training preparation changed original reported coordinate')
    if prepared.evaluate is None or not callable(prepared.evaluate):
        raise ValueError('Native source target preparation requires an admission-guarded objective')
    # The callable is deliberately never executed; it would require an admitted
    # registered owner. Condition, numeric lexeme, uncertainty and provider
    # metadata remain separately hashed to detect changes across snapshots.
    return dict(original_target_kind=target.kind,
        original_source_unit=item.partition_key[3],
        original_value_float64_bits=target.reported_value_float64_bits,
        native_numeric_lexeme=target.native_numeric_lexeme,
        lexeme_scope=target.lexeme_scope,
        condition_sha256=_sha(item.condition_json),
        source_audit_sha256=_sha(item.source_audit_json),
        provider_lexemes_sha256=_sha(target.provider_lexemes_json),
        native_target_metadata_sha256=_sha(target.native_target_metadata_json),
        original_uncertainty_kind=target.uncertainty_kind,
        original_uncertainty_float64_bits=None if target.uncertainty_value is None
            else __import__('struct').pack('>d',target.uncertainty_value).hex(),
        partition_sha256=_sha(canonical(item.partition_key)),
        exact_numeric_source_projection=True,
        source_semantics='reported-experimental-point-not-predicted-or-censored',
        training_objective_invoked=False)


def audit_original_freesolv_two_windows(root,*,task_key=_TASK,max_windows=2):
    """Read exactly two original train rows and verify independent cursor restart.

    The supplied root must be the installed ORIGINAL source checkout. The two
    snapshot identities and executable hashes are loaded only from the immutable
    historical rebind, not rewritten or supplied by the caller. This is not an
    all-corpus scan or validation reference for an approximately80M owner.
    """
    if type(task_key)is not str or task_key!=_TASK or type(max_windows)is not int or max_windows!=2:
        raise ValueError('Only exactly two real FreeSolv original train windows are supported')
    root=Path(root).resolve()
    if not root.is_dir()or Path.cwd().resolve()!=root:
        raise ValueError('Exact original source checkout working directory required')
    context=load_inactive_common_source_rebind(root)
    full=document(root,context.evidence)
    row,=(x for x in full['route_bindings'] if x['task_key']==_TASK)
    if (row['candidate']['max_windows']!=2 or row['common']['max_windows']!=2
            or row['common_view_policy_sha256'] is None
            or context.common_snapshot.sha256!=full['common_source_snapshot']['sha256']):
        raise ValueError('Original FreeSolv source view/finite scan identity changed')
    candidate=Ref(**full['inactive_candidate_snapshot'])
    common=context.common_snapshot
    if candidate.sha256==common.sha256:
        raise ValueError('Distinct candidate/common29-head source identities required')
    current,original,_,_,axis,_,_,_,_=load_assets(root)
    if finite_view_policy_sha256(current)!=row['common_view_policy_sha256']:
        raise ValueError('Original FreeSolv effective source role/split guard changed')
    bindings={}
    for label,snapshot in (('candidate',candidate),('common',common)):
        adapter=FreeSolvCampaignAdapter(original,current,axis,snapshot.sha256)
        binding=adapter.train_binding(batch_size=1)
        expected=row[label]
        actual=dict(task_key=binding.route.task_key,
            initial_cursor_sha256=_sha(binding.route.initial_cursor),
            adapter_sha256=executable_adapter_sha256(binding.prepare,root=Path(__file__).resolve().parents[3]),
            max_windows=max_windows)
        if (actual!=expected or binding.route.source_snapshot_sha256!=snapshot.sha256
                or binding.route.native_view_policy_sha256!=row['common_view_policy_sha256']):
            raise ValueError('Current source-native FreeSolv adapter/cursor differs from reviewed identity: '+label)
        bindings[label]=(adapter,binding)

    def read_two(binding):
        route=binding.route;iterator=route.iter_from(route.initial_cursor)
        found=[];position=route._cursor_position(route.initial_cursor)
        try:
            for _ in range(2):
                try:window=next(iterator)
                except StopIteration as exc:
                    raise ValueError('Original FreeSolv train source exhausted inside frozen two-window budget') from exc
                if window.original_source_rows!=1 or window.batch.original_split!='train':
                    raise ValueError('One complete original scalar train source row per window required')
                route._view.require_native_rows(window.batch,split='train')
                after=route._cursor_position(window.cursor_after)
                _checked_two_window_transition(position,after)
                position=after;found.append(window)
        finally:
            close=getattr(iterator,'close',None)
            if close is not None:close()
        return tuple(found)

    left=read_two(bindings['candidate'][1]);right=read_two(bindings['common'][1])
    rows=[];identities=set()
    for i,(x,y) in enumerate(zip(left,right,strict=True)):
        if (x.batch.source_shard!=y.batch.source_shard
                or x.batch.source_row_ordinals!=y.batch.source_row_ordinals
                or not x.batch.native_rows.equals(y.batch.native_rows,check_metadata=True)):
            raise ValueError('Original scalar Arrow row/role/field order differs after common rebind')
        ca,cb=bindings['candidate'];na,nb=bindings['common']
        l=_native_scalar_signature(ca,x);r=_native_scalar_signature(na,y)
        a=_source_fingerprint(x);b=_source_fingerprint(y)
        if l!=r or a!=b:raise ValueError('Original scalar condition/value/metadata or payload changed after rebind')
        digest,size,payload,ordinal=a
        identity=(x.batch.source_shard['sha256'],ordinal)
        if identity in identities:raise ValueError('Duplicate original FreeSolv source row')
        identities.add(identity)
        ca_position=cb.route._cursor_position(x.cursor_after)
        na_position=nb.route._cursor_position(y.cursor_after)
        _checked_rebound_position(ca_position,na_position)
        rows.append(dict(window_index=i,source_shard_sha256=identity[0],
            source_ordinal=ordinal,native_source_fingerprint=digest,
            source_payload_bytes=size,native_payload_sha256=payload,
            candidate_next_position=ca_position,common_next_position=na_position,
            candidate_cursor_sha256=_sha(x.cursor_after),common_cursor_sha256=_sha(y.cursor_after),**l))

    for label,windows in (('candidate',left),('common',right)):
        adapter,binding=bindings[label]
        restarted=_first(binding.route.iter_from(windows[0].cursor_after))
        expected=windows[1]
        if (restarted.cursor_after!=expected.cursor_after
                or restarted.evidence_sha256!=expected.evidence_sha256
                or restarted.batch.source_shard!=expected.batch.source_shard
                or restarted.batch.source_row_ordinals!=expected.batch.source_row_ordinals
                or not restarted.batch.native_rows.equals(expected.batch.native_rows,check_metadata=True)
                or _source_fingerprint(restarted)!=_source_fingerprint(expected)
                or _native_scalar_signature(adapter,restarted)!=_native_scalar_signature(adapter,expected)):
            raise ValueError('Restart from immutable first cursor altered original second FreeSolv scalar target')

    return dict(schema='Mora3.OriginalFreeSolvTwoWindowSourceReplay/v1',
        task_key=_TASK,source_family='FreeSolv',split='train',windows_replayed=2,
        native_source_rows=2,native_observed_targets=2,rows=tuple(rows),
        candidate_and_common_source_equal=True,native_conditions_and_targets_equal=True,
        second_window_restart_exact=True,
        native_view_policy_sha256=finite_view_policy_sha256(current),
        candidate_source_snapshot_sha256=candidate.sha256,
        common_source_snapshot_sha256=common.sha256,
        scalar_projection_proof_sha256=axis.proof_sha256,
        full_source_corpus_exhausted=None,complete62_route_replay=False,
        validation_baseline_measured=False,global_split_admitted=False,
        training_admission=False,optimizer_updates=0,actual_campaign_runs=0,
        source_confidence='genuine-original-row-and-typed-scalar-projection;not-admitted-biology')


# The complete source qualification below is separate from the frozen two-window
# *per update* campaign scan cap. It neither changes that cap nor calls a loss.
_ROLES = ('train', 'validation', 'test')
_ROLE_ROWS = dict(train=556, validation=42, test=44)
_METADATA_COLUMNS = ('task_id', 'record_id', 'source_family', 'source_revision',
                     'split', 'task_name', 'native_schema')
_REVISION = 'MobleyLab-FreeSolv-source-snapshot-20261031'


def freesolv_declared_group_census(records):
    """Close co-occurring DECLARED groups transitively within this source only.

    This does not discover new chemical similarity, certify global cross-source
    separation, deduplicate measured rows or accept any held-out benchmark.
    """
    import re
    if type(records) is not tuple or not records:
        raise ValueError('Nonempty immutable original role/group records required')
    parents = list(range(len(records)))
    owners, seen_ids, seen_tasks = {}, set(), set()
    roles = {role: [] for role in _ROLES}

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    for i, row in enumerate(records):
        if (type(row) is not dict or set(row) != {'record_id', 'task_id', 'split', 'groups'}
                or type(row['record_id']) is not str or not re.fullmatch(r'mobley_[0-9]+', row['record_id'])
                or type(row['task_id']) is not str or not re.fullmatch('[0-9a-f]{64}', row['task_id'])
                or type(row['split']) is not str or row['split'] not in _ROLES
                or type(row['groups']) is not tuple or len(row['groups']) != 3
                or any(type(g) is not str or not re.fullmatch(r'(component|molecule|scaffold):[0-9a-f]{64}', g)
                       for g in row['groups'])
                or {g.split(':', 1)[0] for g in row['groups']} != {'component', 'molecule', 'scaffold'}):
            raise ValueError('Exact original identity, role and three declared chemical group axes required')
        if row['record_id'] in seen_ids or row['task_id'] in seen_tasks:
            raise ValueError('Duplicate original record or task identity; do not silently discard rows')
        seen_ids.add(row['record_id']); seen_tasks.add(row['task_id'])
        roles[row['split']].append(row)
        for group in row['groups']:
            if group in owners:
                parents[find(i)] = find(owners[group])
            else:
                owners[group] = i
    components = {}
    for i, row in enumerate(records):
        components.setdefault(find(i), []).append(row)
    cross = []
    for rows in components.values():
        occupied = sorted({r['split'] for r in rows})
        if len(occupied) > 1:
            cross.append(dict(roles=occupied, record_count=len(rows),
                record_ids_sha256=_sha(canonical(sorted(r['record_id'] for r in rows)))))
    molecules = {role: {g for r in rows for g in r['groups'] if g.startswith('molecule:')}
                 for role, rows in roles.items()}
    counts = {role: len(rows) for role, rows in roles.items()}
    return dict(role_rows=counts,
        declared_group_counts_by_role={role: {kind: len({g for r in rows for g in r['groups']
            if g.startswith(kind + ':')}) for kind in ('component', 'molecule', 'scaffold')}
            for role, rows in roles.items()},
        distinct_molecule_groups_by_role={role: len(g) for role, g in molecules.items()},
        distinct_molecule_groups_total=len(set().union(*molecules.values())),
        same_molecule_extra_rows_by_role={role: counts[role]-len(molecules[role]) for role in _ROLES},
        connected_components=len(components),
        cross_role_connected_components=sorted(cross, key=lambda x: x['record_ids_sha256']),
        source_local_declared_groups_disjoint=not cross,
        declared_group_scope='complete original FreeSolv sidecars only; not newly computed chemical similarity',
        global_chemical_split_accepted=False)


def _reference_freesolv_numeric(raw):
    """Independent stdlib readback of the immutable native experimental target.

    Does not call the production scalar projector. Uncertainty remains potentially
    default source metadata, not an assumed Gaussian observation standard deviation.
    """
    import json
    import math
    import struct

    def pairs(items):
        result = {}
        for k, v in items:
            if k in result:
                raise ValueError('Duplicate native JSON key')
            result[k] = v
        return result

    def invalid(_):
        raise ValueError('Nonfinite native JSON scalar')

    def bits(value):
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError('Finite original numeric coordinate required, not bool or imputation')
        return struct.pack('>d', value).hex()

    if type(raw) is not bytes or not raw:
        raise ValueError('Original native payload bytes required')
    try:
        p = json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)
        if (type(p) is not dict or set(p) != {'schema', 'task', 'conditions', 'targets', 'audit', 'missing'}
                or p['schema'] != 'mora3.CorpusTask/v1' or p['task'] != _TASK
                or set(p['conditions']) != {'smiles'} or type(p['conditions']['smiles']) is not str
                or not p['conditions']['smiles'] or set(p['targets']) != {'expt', 'unit'}
                or p['targets']['unit'] != 'kcal/mol'):
            raise ValueError('Exact original molecule condition and experimental scalar target required')
        source = p['audit']['source_row']
        labels = p['audit']['label_contract']
        value = bits(p['targets']['expt'])
        if (value != bits(source['expt']) or labels['default_target_fields'] != ['expt']
                or labels['label_evidence']['expt'] != 'experimental'):
            raise ValueError('Original target bits or experimental evidence class changed')
        uncertainty = source.get('d_expt')
        uncertainty_bits = None if uncertainty is None else bits(uncertainty)
        kind = None if uncertainty is None else labels['label_evidence']['d_expt']
        if uncertainty is not None and kind != 'experimental_uncertainty_may_be_default':
            raise ValueError('Original uncertainty evidence must not be relabeled')
        literal = json.loads(raw, parse_int=str, parse_float=str, parse_constant=invalid,
                             object_pairs_hook=pairs)['targets']['expt']
        if type(literal) is not str:
            raise ValueError('Original numeric JSON token required')
        return dict(value_float64_bits=value, numeric_lexeme=literal,
            uncertainty_float64_bits=uncertainty_bits, uncertainty_kind=kind,
            molecule_condition_sha256=_sha(canonical(p['conditions'])))
    except (KeyError, TypeError, UnicodeError, OverflowError) as exc:
        raise ValueError('Incomplete or malformed original FreeSolv numeric envelope') from exc


def _freesolv_role_metadata(root, view):
    """Hash original files as opaque bytes; decode only identity/group columns.

    BufferReader ensures the parsed bytes are the same bytes just authenticated.
    Validation/test target values are never selected from their Parquet columns.
    """
    import json
    import pyarrow as pa
    import pyarrow.parquet as pq
    from mora3.data.finite_union_native_rows import full9, read_exact
    descriptors = sorted((d for d in view.descriptors if any(
        x['source_family'] == 'FreeSolv' for x in d['source_task_schema_counts'])),
        key=lambda d: d['source_shard']['path'])
    if len(descriptors) != 5:
        raise ValueError('Exactly five original FreeSolv role shards required')
    records, by_coordinate, references = [], {}, []
    for d in descriptors:
        shard = d['source_shard']; path = root / shard['path']
        if full9(path) != tuple(d['source_full9']):
            raise ValueError('Original source full9 fingerprint changed')
        raw = read_exact(shard, root=root)
        side_raw = read_exact(d['original_group_sidecar'], root=root)
        table = pq.read_table(pa.BufferReader(raw), columns=list(_METADATA_COLUMNS), use_threads=False)
        side = pq.read_table(pa.BufferReader(side_raw), use_threads=False)
        if (table.num_rows != shard['rows'] or side.num_rows != table.num_rows
                or side.num_rows != d['original_group_sidecar']['rows']):
            raise ValueError('Original metadata/sidecar row conservation differs')
        if any(p == shard['path'] for p, _ in view._ordinal_exclusions):
            raise ValueError('Unreviewed original FreeSolv row exclusion appeared')
        occupied = set()
        for ordinal, (row, group) in enumerate(zip(table.to_pylist(), side.to_pylist(), strict=True)):
            if (row['source_family'] != 'FreeSolv' or row['task_name'] != _TASK
                    or row['native_schema'] != 'mora3.CorpusTask/v1' or row['source_revision'] != _REVISION
                    or row['split'] not in _ROLES or row['task_id'] != group['task_id']
                    or group['source_file'] != shard['path']
                    or group['evidence_scope'] != 'identity/leakagegroupsonly;notpredictorfeaturesorsamecell/effectproof'):
                raise ValueError('Original source/task/group-sidecar identity differs')
            effective = json.loads(group['effective_groups_json'])
            source = json.loads(group['source_group_ids_json'])
            if type(effective) is not list or effective != source:
                raise ValueError('Exact original declared group identities required')
            record = dict(record_id=row['record_id'], task_id=row['task_id'], split=row['split'],
                          groups=tuple(effective))
            records.append(record)
            coordinate = (shard['path'], ordinal)
            if coordinate in by_coordinate:
                raise ValueError('Duplicate original source coordinate')
            by_coordinate[coordinate] = record
            occupied.add(row['split'])
        if len(occupied) != 1 or d['original_metadata_role_counts'] != {next(iter(occupied)): table.num_rows}:
            raise ValueError('Original declared role counts changed')
        references.append(dict(source_shard=dict(shard), group_sidecar=dict(d['original_group_sidecar']),
                               source_full9=list(d['source_full9']),
                               role=next(iter(occupied)), decoded_columns=list(_METADATA_COLUMNS)))
    census = freesolv_declared_group_census(tuple(records))
    if census['role_rows'] != _ROLE_ROWS:
        raise ValueError('Complete original 556/42/44 role denominator changed')
    return by_coordinate, references, census


def audit_original_freesolv_complete_train(root):
    """Exhaust all 556 genuine train scalars under both immutable N29 identities.

    Two original native iterators, independent scalar references, original groups
    and source-boundary restarts are checked. No synthetic rows, model, optimizer,
    numerical held-out target, fresh validation baseline or training slot is used.
    """
    from contextlib import ExitStack
    root = Path(root).resolve()
    if not root.is_dir() or Path.cwd().resolve() != root:
        raise ValueError('Exact original source checkout working directory required')
    context = load_inactive_common_source_rebind(root)
    manifest = document(root, context.evidence)
    plans = [r for r in manifest['route_bindings'] if r['task_key'] == _TASK]
    if len(plans) != 1:
        raise ValueError('One frozen FreeSolv source binding required')
    plan = plans[0]
    candidate = Ref(**manifest['inactive_candidate_snapshot']); common = context.common_snapshot
    if candidate.sha256 == common.sha256:
        raise ValueError('Distinct reviewed candidate/common snapshot identities required')
    view, original, _, _, axis, *_ = load_assets(root)
    policy = finite_view_policy_sha256(view)
    if policy != plan['common_view_policy_sha256']:
        raise ValueError('Original FreeSolv view policy changed')
    metadata, files, census = _freesolv_role_metadata(root, view)
    bound = {}
    for label, snapshot in (('candidate', candidate), ('common', common)):
        adapter = FreeSolvCampaignAdapter(original, view, axis, snapshot.sha256)
        binding = adapter.train_binding(batch_size=1)
        actual = dict(task_key=_TASK, initial_cursor_sha256=_sha(binding.route.initial_cursor),
            adapter_sha256=executable_adapter_sha256(binding.prepare, root=Path(__file__).resolve().parents[3]),
            max_windows=2)
        if actual != plan[label] or binding.route.native_view_policy_sha256 != policy:
            raise ValueError('Frozen native source adapter/cursor/per-update scan policy changed')
        bound[label] = (adapter, binding.route)
    left, right = bound['candidate'][1], bound['common'][1]
    if left._paths != right._paths or len(left._paths) != 3:
        raise ValueError('Exact common ordered three training shards required')
    expected = [(path, ordinal) for path in left._paths for ordinal in range(
        next(f['source_shard']['rows'] for f in files if f['source_shard']['path'] == path))]
    if len(expected) != 556 or any(metadata[c]['split'] != 'train' for c in expected):
        raise ValueError('Complete original train coordinate denominator differs')
    positions = {label: route._cursor_position(route.initial_cursor) for label, (_, route) in bound.items()}
    rows, boundary_cursors = [], {}
    with ExitStack() as stack:
        streams = {}
        for label, (_, route) in bound.items():
            streams[label] = route.iter_from(route.initial_cursor)
            stack.callback(streams[label].close)
        for index, coordinate in enumerate(expected):
            windows, signatures = {}, {}
            for label, (adapter, route) in bound.items():
                try:
                    window = next(streams[label])
                except StopIteration as exc:
                    raise ValueError('Original train iterator exhausted before all 556 rows') from exc
                if (window.original_source_rows != 1 or window.batch.original_split != 'train'
                        or (window.batch.source_shard['path'], window.batch.source_row_ordinals[0]) != coordinate):
                    raise ValueError('Original complete ordered source coordinate differs')
                after = route._cursor_position(window.cursor_after)
                _checked_two_window_transition(positions[label], after)
                if after[0] != positions[label][0] and after[0] < len(route._paths):
                    boundary_cursors[(label, after[0])] = window.cursor_after
                positions[label] = after
                windows[label] = window
                signatures[label] = _native_scalar_signature(adapter, window)
            a, b = windows['candidate'], windows['common']
            sa, sb = signatures['candidate'], signatures['common']
            _checked_rebound_position(positions['candidate'], positions['common'])
            if (a.batch.source_shard != b.batch.source_shard
                    or a.batch.source_row_ordinals != b.batch.source_row_ordinals
                    or not a.batch.native_rows.equals(b.batch.native_rows, check_metadata=True)
                    or sa != sb or _source_fingerprint(a) != _source_fingerprint(b)
                    or a.cursor_after == b.cursor_after):
                raise ValueError('Original native value/condition/metadata differs across common snapshots')
            raw_row, = a.batch.native_rows.to_pylist()
            meta = metadata[coordinate]
            if (raw_row['task_id'] != meta['task_id'] or raw_row['record_id'] != meta['record_id']
                    or raw_row['source_revision'] != _REVISION):
                raise ValueError('Actual original train row differs from authenticated role/group metadata')
            reference = _reference_freesolv_numeric(raw_row['native_payload'])
            if (reference['value_float64_bits'] != sa['original_value_float64_bits']
                    or reference['numeric_lexeme'] != sa['native_numeric_lexeme']
                    or reference['uncertainty_float64_bits'] != sa['original_uncertainty_float64_bits']
                    or reference['uncertainty_kind'] != sa['original_uncertainty_kind']):
                raise ValueError('Independent original JSON scalar differs from native production projection')
            rows.append(dict(index=index, record_id=meta['record_id'], source_ordinal=coordinate[1],
                source_shard_sha256=a.batch.source_shard['sha256'], native_payload_sha256=raw_row['native_payload_sha256'],
                native_source_fingerprint=_source_fingerprint(a)[0],
                next_position=list(positions['candidate']), source_unit=sa['original_source_unit'],
                original_value_float64_bits=sa['original_value_float64_bits'],
                independent_value_float64_bits=reference['value_float64_bits'],
                original_uncertainty_float64_bits=sa['original_uncertainty_float64_bits'],
                independent_uncertainty_float64_bits=reference['uncertainty_float64_bits'],
                uncertainty_kind=reference['uncertainty_kind'], native_numeric_lexeme=reference['numeric_lexeme'],
                condition_sha256=sa['condition_sha256'], source_audit_sha256=sa['source_audit_sha256'],
                native_target_metadata_sha256=sa['native_target_metadata_sha256'],
                original_group_sha256=_sha(canonical(meta['groups'])),
                candidate_cursor_sha256=_sha(a.cursor_after), common_cursor_sha256=_sha(b.cursor_after)))
        for iterator in streams.values():
            try:
                next(iterator)
            except StopIteration:
                pass
            else:
                raise ValueError('Unaccounted extra original FreeSolv train row')
    if positions['candidate'] != (3, 0) or positions['common'] != (3, 0) or len(boundary_cursors) != 4:
        raise ValueError('Incomplete original three-shard exhaustion or source-boundary cursors')
    for (label, shard_index), cursor in boundary_cursors.items():
        adapter, route = bound[label]
        window = _first(route.iter_from(cursor))
        signature = _native_scalar_signature(adapter, window)
        index = next(i for i, c in enumerate(expected) if c == (route._paths[shard_index], 0))
        row = rows[index]
        if (_source_fingerprint(window)[0] != row['native_source_fingerprint']
                or _sha(window.cursor_after) != row[label + '_cursor_sha256']
                or signature['original_value_float64_bits'] != row['original_value_float64_bits']):
            raise ValueError('Independent restart at original shard boundary changed source continuation')
    from mora3.data.finite_union_native_rows import full9, read_exact
    # Postflight reads remain opaque hashes for held-out files, not target parses.
    for entry in files:
        if full9(root / entry['source_shard']['path']) != tuple(entry['source_full9']):
            raise ValueError('Original source changed during complete train qualification')
        read_exact(entry['source_shard'], root=root)
        read_exact(entry['group_sidecar'], root=root)
    return dict(schema='Mora3.FreeSolvCompleteOriginalTrainReplay/v1', source_family='FreeSolv',
        original_role_rows=dict(_ROLE_ROWS), source_files=files, declared_group_census=census,
        candidate_source_snapshot_sha256=candidate.sha256, common_source_snapshot_sha256=common.sha256,
        native_view_policy_sha256=policy, scalar_projection_proof_sha256=axis.proof_sha256,
        native_train_rows=556, native_observed_train_targets=556, rows=rows,
        ordered_native_train_rows_sha256=_sha(canonical(rows)),
        candidate_common_all_train_rows_equal=True, all_train_numeric_references_equal=True,
        both_train_iterators_exhausted=True, train_final_cursor_position=[3, 0],
        source_boundary_restarts_checked=4, frozen_per_trial_window_limit_unchanged=2,
        heldout_metadata_rows=86, heldout_decoded_columns=list(_METADATA_COLUMNS),
        heldout_target_payload_decoded=False, full_expanded_corpus_ready=False,
        complete62_route_replay=False, fresh_owner_validation_reference=False,
        global_split_admitted=False, training_admission=False,
        operation_scope='read-only source qualification; zero run counts describe this operation, not live campaign monitoring',
        optimizer_updates=0, actual_proxy_trials=0, actual_target_runs=0)
