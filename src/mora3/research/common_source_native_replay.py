"""Bounded REAL source-row replay of one train-only route across snapshot IDs.

This is a source-fidelity diagnostic, not campaign, model, or global split admission.
It reconstructs source-native typed route objects using the actual original train
view and fitted vocabulary, checks the explicitly pinned common-source metadata,
reads the SAME ONE original train window under both source-snapshot bindings,
and prepares (but never evaluates) an original observed training target.

All complete original source bytes remain immutable. No 5,000-window campaign,
optimizer/model construction, worker/lease, checkpoint or source role mutation.
"""
from hashlib import sha256
from pathlib import Path

from mora3.data.admet import canonical
from mora3.model.source_assembly_inputs import load_actual_source_assets
from .primary_campaign_source_rebind import load_inactive_common_source_rebind
from .native_prior_campaign import NativePriorCampaignAdapter
from .primary_native_routes import finite_view_policy_sha256
from .mup_campaign import executable_adapter_sha256
from .primary_source_factory_preflight import Ref,document,require

_TASK='native-prior:dna'
_BATCH_SIZE=1
_MAX_BYTES=4096
_CHUNK_SIZE=128


def _digest(raw):return sha256(raw).hexdigest()


def _first(window_iterator):
    # Only one native training-window object may be materialized. The iterator
    # is closed without advancing to a second source batch.
    try:
        window=next(window_iterator)
    except StopIteration as exc:
        raise ValueError('Bound original train route has no genuine eligible window') from exc
    finally:
        close=getattr(window_iterator,'close',None)
        if close is not None:close()
    if window.original_source_rows!=1 or window.batch.original_split!='train':
        raise ValueError('Expected exactly one original train native source row')
    return window



def _source_fingerprint(window):
    """Independently hash one authenticated Arrow-native row including raw bytes."""
    native=window.batch.native_rows
    if (window.batch.original_split!='train' or native.num_rows!=1
            or native['split'].to_pylist()!=['train']):
        raise ValueError('Only one real original train row may be fingerprinted')
    payloads=native['native_payload'].to_pylist()
    ids=native['task_id'].to_pylist()
    declared=native['native_payload_sha256'].to_pylist()
    if (len(payloads)!=1 or type(payloads[0])is not bytes or not payloads[0]
            or len(ids)!=1 or type(ids[0])is not str or not ids[0]
            or len(declared)!=1 or _digest(payloads[0])!=declared[0]):
        raise ValueError('Raw native train payload or source digest is incompatible')
    ordinals=window.batch.source_row_ordinals
    if len(ordinals)!=1 or type(ordinals[0])is not int or ordinals[0]<0:
        raise ValueError('Exact original source ordinal required')
    raw=canonical(dict(shard=window.batch.source_shard,source_ordinals=ordinals,
        task_id_sha256=(_digest(ids[0].encode()),),
        native_payload_sha256=tuple(declared),native_payload_bytes=len(payloads[0])))
    return _digest(raw),len(payloads[0]),declared[0],ordinals[0]

def audit_common_source_first_prior_window(root, *,task_key=_TASK,max_windows=1):
    """Return SHA-only readback for one authentic original train-only row.

    The expected route/snapshot/codec, full policy digests and original byte
    references are taken from the immutable reviewed inactive rebind, not input
    arguments. This does not claim a completed full source replay or an accepted
    source split; a 5,000-step campaign still requires every admitted route.
    """
    if type(task_key)is not str or task_key!=_TASK or type(max_windows)is not int or max_windows!=1:
        raise ValueError('Only one explicitly selected first native-prior:dna train window is in scope')
    root=Path(root).resolve()
    if not root.is_dir():raise ValueError('Original checked-out source root required')
    context=load_inactive_common_source_rebind(root)
    full=document(root,context.evidence)
    rows=[r for r in full['route_bindings']if r['task_key']==task_key]
    if len(rows)!=1:raise ValueError('Exact single previously reviewed original route identity required')
    row=rows[0]
    if (row['common']['max_windows']!=2 or row['candidate']['max_windows']!=2
            or row['common_view_policy_sha256'] is None):
        raise ValueError('Exact original finite train source-view and budget metadata required')
    candidate_snapshot=Ref(**full['inactive_candidate_snapshot'])
    common_snapshot=context.common_snapshot
    if (candidate_snapshot.sha256==common_snapshot.sha256
            or _digest((root/context.evidence.path).read_bytes())!=context.evidence.sha256):
        raise ValueError('Distinct source snapshots and exact inactive rebind reference required')
    # The current native view reads the actual immutable Parquet source rows and
    # exact original group/ordinal sidecars. It does not mint training eligibility.
    assets=load_actual_source_assets(root,source_view_policy='effective-study-v2')
    view=assets.native_view
    policy=finite_view_policy_sha256(view)
    if policy!=row['common_view_policy_sha256']:
        raise ValueError('Full retained native original role/exclusion view differs from reviewed scope')
    # Reconstruct two separate typed native adapters. Only the snapshot binding
    # changes; the original data, fitted codec, batch size and target preparation
    # policy are held fixed. Identical endpoints are not assumed: verified by read.
    def adapter(snapshot):
        obj=NativePriorCampaignAdapter(view,assets.vocabulary,snapshot,'dna',_MAX_BYTES,_CHUNK_SIZE)
        return obj,obj.train_binding(batch_size=_BATCH_SIZE)
    candidate,candidate_binding=adapter(candidate_snapshot.sha256)
    common,common_binding=adapter(common_snapshot.sha256)
    results={}
    for label,obj,binding in (('candidate',candidate,candidate_binding),('common',common,common_binding)):
        expected=row[label]
        actual=dict(task_key=binding.route.task_key,
            initial_cursor_sha256=_digest(binding.route.initial_cursor),
            adapter_sha256=executable_adapter_sha256(binding.prepare,root=Path(__file__).resolve().parents[3]),
            max_windows=expected['max_windows'])
        if actual!=expected:
            raise ValueError('Actual source-built '+label+' native adapter/cursor mismatches reviewed metadata')
        if binding.route.source_snapshot_sha256!= (candidate_snapshot.sha256 if label=='candidate'else common_snapshot.sha256):
            raise ValueError('Original native route source snapshot drift')
        results[label]=(obj,binding)
    # Consume precisely one original train window for each SAME underlying route
    # and compare whole native Arrow tables (including bytes) as well as ordinals.
    cobj,cbinding=results['candidate'];nobj,nbinding=results['common']
    old=_first(cbinding.route.iter_from(cbinding.route.initial_cursor))
    new=_first(nbinding.route.iter_from(nbinding.route.initial_cursor))
    if (old.batch.source_shard!=new.batch.source_shard
            or old.batch.source_row_ordinals!=new.batch.source_row_ordinals
            or not old.batch.native_rows.equals(new.batch.native_rows,check_metadata=True)):
        raise ValueError('Source-rebound first original train row differs in values, types, order or provenance')
    view.require_native_rows(old.batch,split='train')
    view.require_native_rows(new.batch,split='train')
    # Only source-role train values enter the target; never run the returned
    # callable, because that would require a separately admitted live model.
    left=cobj.prepare(old);right=nobj.prepare(new)
    if (left.observed_positions!=right.observed_positions or left.observed_positions<=0):
        raise ValueError('Actual native train target count differs after snapshot rebind')
    old_hash,old_size,old_payload_hash,old_ordinal=_source_fingerprint(old)
    new_hash,new_size,new_payload_hash,new_ordinal=_source_fingerprint(new)
    if (old_hash!=new_hash or old_size!=new_size
            or old_payload_hash!=new_payload_hash or old_ordinal!=new_ordinal):
        raise ValueError('Independent raw native source fingerprints disagree after snapshot rebind')
    return dict(schema='Mora3.CommonSourceFirstNativeWindow/v1',
        task_key=task_key,role='train',windows_replayed=1,actual_native_rows=1,
        native_payload_bytes_read=old_size,
        observed_train_target_tokens=left.observed_positions,
        original_native_source_digest=old_hash,common_native_source_digest=new_hash,
        original_rows_and_payloads_identical=True,
        first_original_source_shard_sha256=old.batch.source_shard['sha256'],
        first_original_source_ordinal=old_ordinal,
        candidate_adapter_sha256=row['candidate']['adapter_sha256'],
        common_adapter_sha256=row['common']['adapter_sha256'],
        candidate_initial_cursor_sha256=row['candidate']['initial_cursor_sha256'],
        common_initial_cursor_sha256=row['common']['initial_cursor_sha256'],
        active_full_source_view_policy_sha256=policy,
        common_source_snapshot_sha256=common_snapshot.sha256,
        native_payload_sha256=old_payload_hash,training_admission=False,
        full_replay_complete=False,source_global_split_accepted=False,
        optimizer_updates=0,actual_campaign_runs=0,
        scope='one exact original train window only; not complete62-route replay or validation baseline')

def _checked_two_window_transition(before, after):
    """Require canonical finite native cursor to advance in shard/ordinal order."""
    for position in (before, after):
        if (type(position) is not tuple or len(position) != 2
                or any(type(x) is not int or x < 0 for x in position)):
            raise ValueError('Exact nonnegative native shard/ordinal cursor required')
    if not before < after:
        raise ValueError('Native train cursor did not advance strictly forward')
    return after


def _checked_rebound_position(candidate, common):
    """The two source bindings use different hashes but the SAME raw position."""
    for position in (candidate, common):
        if (type(position) is not tuple or len(position) != 2
                or any(type(x) is not int or x < 0 for x in position)):
            raise ValueError('Exact native cursor positions required')
    if candidate != common:
        raise ValueError('Rebound native cursors diverged in source coordinates')
    return candidate


def _audit_common_source_two_prior_windows(root, *, task_key, max_windows=2):
    """Replay the entire TWO-WINDOW SCAN BUDGET, not the full source corpus.

    A two-window bound is the maximum attempted windows for this particular
    scheduled update, not an assertion that the entire original source is
    exhausted or that a 5,000-update campaign has been replayed. The original
    immutable source and roles are read through two independently bound
    native routes; a second independent read starts from each saved first
    cursor and checks exact byte-level second-window reproduction.
    """
    allowed=('native-prior:dna','native-prior:native_smiles',
             'native-prior:rfam_rna','native-prior:rnacentral_rna',
             'native-prior:smiles_to_selfies','native-prior:selfies_to_smiles',
             'native-prior:protein')
    # Protein is uniquely bound to its ORIGINAL extra homology quarantine;
    # never share the unquarantined source view from the other six priors.
    if (type(task_key) is not str or task_key not in allowed
            or type(max_windows) is not int or max_windows != 2):
        raise ValueError('Only exact reviewed native prior routes and two-window scan budget supported')
    from mora3.data.source_native_priors import source_native_prior_views
    from .native_prior_campaign import _prior_target_count
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError('Exact existing original source checkout required')
    context = load_inactive_common_source_rebind(root)
    full = document(root, context.evidence)
    matching = [r for r in full['route_bindings'] if r['task_key'] == task_key]
    if len(matching) != 1:
        raise ValueError('One exact reviewed native train route required')
    row = matching[0]
    if (row['candidate']['max_windows'] != 2 or row['common']['max_windows'] != 2
            or row['common_view_policy_sha256'] is None
            or context.common_snapshot.sha256 != full['common_source_snapshot']['sha256']):
        raise ValueError('Reviewed native two-window budget/source/view identity differs')
    candidate_snapshot = Ref(**full['inactive_candidate_snapshot'])
    common_snapshot = context.common_snapshot
    if candidate_snapshot.sha256 == common_snapshot.sha256:
        raise ValueError('Distinct original candidate and full owner snapshots required')
    assets = load_actual_source_assets(root, source_view_policy='effective-study-v2')
    view = assets.native_view
    quarantine=None
    if task_key=='native-prior:protein':
        from .common_source_protein_replay import reconstruct_protein_quarantine_views
        quarantine=reconstruct_protein_quarantine_views(root,base_view=view,context=context,row=row)
        views={'candidate':quarantine.candidate,'common':quarantine.common}
    else:
        views={'candidate':view,'common':view}
    policy = finite_view_policy_sha256(views['common'])
    if policy != row['common_view_policy_sha256']:
        raise ValueError('Current source role/split view does not match the reviewed policy')
    bindings = {}
    for label, source in (('candidate', candidate_snapshot), ('common', common_snapshot)):
        adapter = NativePriorCampaignAdapter(
            view, assets.vocabulary, source.sha256, task_key.removeprefix('native-prior:'), _MAX_BYTES, _CHUNK_SIZE)
        binding = adapter.train_binding(batch_size=_BATCH_SIZE)
        if quarantine is not None:
            # The ORIGINAL protected rebind changes the iterator's effective
            # train view, NOT the already pinned target-preparation callable.
            # Both must remain distinct: a fresh adapter built directly on the
            # quarantine would change the frozen executable adapter SHA256.
            from dataclasses import replace
            protected = NativePriorCampaignAdapter(
                views[label], assets.vocabulary, source.sha256,
                task_key.removeprefix('native-prior:'), _MAX_BYTES, _CHUNK_SIZE)
            filtered = protected.train_binding(batch_size=_BATCH_SIZE)
            if (binding.route._identity != filtered.route._identity
                    or binding.route._paths != filtered.route._paths
                    or binding.route._batch_size != filtered.route._batch_size
                    or binding.route._shard_refs != filtered.route._shard_refs):
                raise ValueError('Protected protein route changed original data, target or iterator axes')
            binding = replace(binding, route=filtered.route)
        expected = row[label]
        actual = dict(
            task_key=binding.route.task_key,
            initial_cursor_sha256=_digest(binding.route.initial_cursor),
            adapter_sha256=executable_adapter_sha256(
                binding.prepare, root=Path(__file__).resolve().parents[3]),
            max_windows=2)
        if (actual != expected or binding.route.source_snapshot_sha256 != source.sha256):
            raise ValueError('Live typed original train route/cursor/adapter mismatches frozen review')
        bindings[label] = (adapter, binding)

    def read_exact_two(binding):
        route = binding.route
        iterator = route.iter_from(route.initial_cursor)
        result = []
        before = route._cursor_position(route.initial_cursor)
        try:
            for index in range(2):
                try:
                    window = next(iterator)
                except StopIteration as exc:
                    raise ValueError('Original source exhausted before both budgeted train windows') from exc
                if (window.original_source_rows != 1
                        or window.batch.original_split != 'train'):
                    raise ValueError('Only original one-row train windows admitted to this diagnostic')
                route._view.require_native_rows(window.batch, split='train')
                position = route._cursor_position(window.cursor_after)
                _checked_two_window_transition(before, position)
                before = position
                result.append(window)
        finally:
            close = getattr(iterator, 'close', None)
            if close is not None:
                close()
        return tuple(result)

    old = read_exact_two(bindings['candidate'][1])
    common = read_exact_two(bindings['common'][1])
    def checked_target_disposition(adapter, window):
        # Source row preservation and configured admissibility are DIFFERENT.
        # A genuine original target longer than the immutable 4096-byte bound
        # remains a refused row. Never truncate or enlarge this frozen budget.
        records = source_native_prior_views(adapter.view, window.batch, split='train')
        raw_bytes = sum(len(record.target_bytes) for record in records)
        # The route decides the exact source-native target alphabet. Nucleotides
        # use the original symbol codec, whereas native SMILES uses the already
        # fitted byte/shortest-path codec. Both counts retain EOS.
        count = _prior_target_count(records, adapter.route, adapter._codec)
        if count < 1 or raw_bytes < 1:
            raise ValueError('Original native prior target has no source-native supervision')
        if raw_bytes > adapter.max_total_target_bytes:
            try:
                adapter.prepare(window)
            except ValueError as exc:
                if str(exc) != 'Complete original native prior targets exceed the declared byte budget':
                    raise
            else:
                raise ValueError('Over-budget original target was silently accepted')
            return dict(preparation_status='budget_refused',original_target_bytes=raw_bytes,
                        source_target_tokens=count,admitted_targets=0)
        prepared = adapter.prepare(window)
        if prepared.observed_positions != count:
            raise ValueError('Independently recounted complete native prior targets differ')
        return dict(preparation_status='observed',original_target_bytes=raw_bytes,
                    source_target_tokens=count,admitted_targets=count)

    reports = []
    source_identities = set()
    exact_counts = 0
    source_counts = 0
    for index, (left, right) in enumerate(zip(old, common, strict=True)):
        if (left.batch.source_shard != right.batch.source_shard
                or left.batch.source_row_ordinals != right.batch.source_row_ordinals
                or not left.batch.native_rows.equals(right.batch.native_rows, check_metadata=True)):
            raise ValueError('Candidate/common original source row content, role or order differs')
        candidate_adapter, candidate_binding = bindings['candidate']
        common_adapter, common_binding = bindings['common']
        left_target = checked_target_disposition(candidate_adapter, left)
        right_target = checked_target_disposition(common_adapter, right)
        lhash, lsize, lsha, lordinal = _source_fingerprint(left)
        rhash, rsize, rsha, rordinal = _source_fingerprint(right)
        if (lhash != rhash or lsize != rsize or lsha != rsha or lordinal != rordinal
                or left_target != right_target):
            raise ValueError('Actual source, target or budget refusal changed on common snapshot rebind')
        logical_id = (left.batch.source_shard['sha256'], lordinal)
        if logical_id in source_identities:
            raise ValueError('Native iterator repeated the same original source row')
        source_identities.add(logical_id)
        old_position = candidate_binding.route._cursor_position(left.cursor_after)
        common_position = common_binding.route._cursor_position(right.cursor_after)
        _checked_rebound_position(old_position, common_position)
        exact_counts += left_target['admitted_targets']
        source_counts += left_target['source_target_tokens']
        reports.append(dict(
            window_index=index, source_shard_sha256=logical_id[0],
            source_ordinal=logical_id[1],
            native_payload_bytes=lsize, native_payload_sha256=lsha,
            native_source_fingerprint=lhash, **left_target,
            candidate_next_position=old_position, common_next_position=common_position,
            candidate_cursor_sha256=_digest(left.cursor_after),
            common_cursor_sha256=_digest(right.cursor_after)))

    # Read again from both immutable first cursors. A stream's second item and
    # an independent restart at cursor_after MUST produce the identical second
    # native row, target and serialized next cursor. This is only a data-cursor
    # check, not the optimizer/RNG/checkpoint resume needed for training.
    for label, stream in (('candidate', old), ('common', common)):
        adapter, binding = bindings[label]
        restarted = _first(binding.route.iter_from(stream[0].cursor_after))
        expected = stream[1]
        if (restarted.cursor_after != expected.cursor_after
                or restarted.evidence_sha256 != expected.evidence_sha256
                or restarted.batch.source_shard != expected.batch.source_shard
                or restarted.batch.source_row_ordinals != expected.batch.source_row_ordinals
                or not restarted.batch.native_rows.equals(
                    expected.batch.native_rows, check_metadata=True)
                or checked_target_disposition(adapter,restarted)
                    != checked_target_disposition(adapter,expected)
                or _source_fingerprint(restarted) != _source_fingerprint(expected)):
            raise ValueError('Fresh original native cursor restart changed the second train window')

    return dict(schema='Mora3.CommonSourceTwoNativeWindows/v1',
        task_key=task_key, source_role='train', windows_replayed=2,
        actual_native_rows=2, rows=tuple(reports),
        observed_train_target_tokens=exact_counts,
        source_target_tokens=source_counts,
        budget_refused_rows=sum(r['preparation_status']=='budget_refused' for r in reports),
        original_rows_and_payloads_identical=True,
        both_rebound_route_cursors_conserved=True,
        resumed_second_window_exact_match=True,
        independent_target_count_conserved=True,
        bounded_original_update_scan_budget=2,
        candidate_source_snapshot_sha256=candidate_snapshot.sha256,
        common_source_snapshot_sha256=common_snapshot.sha256,
        native_source_view_policy_sha256=policy,
        original_source_corpus_exhausted=None,  # not checked past two windows
        complete62_route_replay=False, global_split_accepted=False,
        matched_validation_baseline_measured=False, training_admission=False,
        optimizer_updates=0, actual_campaign_runs=0,
        scope='two real original windows at one update scan bound; not full source replay',
        protein_original_homology_quarantine_applied=quarantine is not None,
        original_quarantined_row_ordinal=None if quarantine is None else quarantine.source_original_row_ordinal,
        protein_quarantine_original_policy_sha256=None if quarantine is None else quarantine.original_policy_sha256)


def audit_common_source_two_prior_windows(root, *, task_key=_TASK, max_windows=2):
    """Backward-compatible, exactly bounded original DNA train-window audit."""
    if type(task_key) is not str or task_key!=_TASK or type(max_windows) is not int or max_windows!=2:
        raise ValueError('Only the exact two-window original native-prior:dna scan budget is supported')
    return _audit_common_source_two_prior_windows(root,task_key=task_key,max_windows=max_windows)


def audit_common_source_two_native_smiles_windows(root, *, task_key='native-prior:native_smiles', max_windows=2):
    """Read exactly two genuine train SMILES windows from both common bindings.

    Byte/token EOS accounting is source-native and uses the original fitted
    codec, not a nucleotide substitution. Returned evidence does not certify
    training eligibility, a completed route, independent splits or transfer.
    """
    if (type(task_key) is not str or task_key!='native-prior:native_smiles'
            or type(max_windows) is not int or max_windows!=2):
        raise ValueError('Only the exactly reviewed native-prior:native_smiles two-window scan is supported')
    return _audit_common_source_two_prior_windows(root,task_key=task_key,max_windows=max_windows)


def audit_common_source_two_remaining_prior_windows(root, *, task_key, max_windows=2):
    """Read two actual source rows for one of four independently pinned routes.

    Includes two original RNA symbol priors and two directional SELFIES/SMILES
    representation tasks. The native adapter chooses nucleotide vs fitted byte
    codec; there is no cross-route relabeling, guessed protein view, or corpus
    admission. The original finite two-window budget is enforced per route.
    """
    supported=frozenset(('native-prior:rfam_rna','native-prior:rnacentral_rna',
                         'native-prior:smiles_to_selfies','native-prior:selfies_to_smiles'))
    if (type(task_key)is not str or task_key not in supported or
            type(max_windows)is not int or max_windows!=2):
        raise ValueError('Only four explicitly scoped native prior routes and two-window budget supported')
    return _audit_common_source_two_prior_windows(root,task_key=task_key,max_windows=max_windows)
