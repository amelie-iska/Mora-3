"""Finite source-bound forest segments through the existing native update engine.

This opt-in route consumes already prepared native supervision. It is not a new
source admission authority, a whole-corpus materializer, or an uncounted runner.
The existing counted launcher must qualify and bind its segment and task cycle.
"""
from dataclasses import fields, is_dataclass
from hashlib import sha256
import json
import torch
from mora3.model.forest_flow import _wire
from mora3.model.forest_training import (ForestTrainingHeads, ForestTrainingSpec,
    ForestTrainingBatch, preflight_forest_batch, forest_joint_loss,
    forest_batch_sha256, _identity)
from mora3.model.full_owner import SourceBoundOwner
from .primary_training import NativeMuUpdateEngine, _rng_state, _restore_rng


def heads_from_owner(owner):
    if type(owner) is not SourceBoundOwner:
        raise ValueError('Exact registered full native source owner required')
    heads = ForestTrainingHeads(owner.forest_flow, owner.forest_decoder,
                                owner.forest_quality, owner.forest_analogy, owner.forest_topology)
    heads.binding()
    return heads


def _clone(value):
    if type(value) is torch.Tensor:
        return value.detach().clone()
    if is_dataclass(value):
        return type(value)(**{f.name: _clone(getattr(value, f.name)) for f in fields(value)})
    if type(value) is tuple:
        return tuple(_clone(x) for x in value)
    if type(value) is list:
        return [_clone(x) for x in value]
    if type(value) is dict:
        return {k: _clone(v) for k, v in value.items()}
    return value


def _storage_bytes(value):
    if type(value) is torch.Tensor:
        return value.numel() * value.element_size()
    if is_dataclass(value):
        return sum(_storage_bytes(getattr(value, f.name)) for f in fields(value))
    if type(value) in (tuple, list):
        return sum(_storage_bytes(x) for x in value)
    if type(value) is dict:
        return sum(_storage_bytes(x) for x in value.values())
    return 0


def _text(value):
    return type(value) is str and 0 < len(value) <= 4096


class ForestTrainingRoute:
    """A bounded immutable in-memory source segment, with explicit repeat count.

    Native source IDs/ordinals and evidence classes are retained. Heldout rows
    refuse. No pickle, repository script, network acquisition or target-derived
    grouping is performed. Distinct segments require distinct source commitments.
    """
    def __init__(self, heads, spec, batches, *, task_key, trial_id, max_passes=1,
                 max_resident_bytes=64 * 1024**2):
        if (type(heads) is not ForestTrainingHeads or type(spec) is not ForestTrainingSpec
                or type(batches) is not tuple or not 1 <= len(batches) <= 1024
                or not _text(task_key) or not _text(trial_id)
                or type(max_passes) is not int or not 1 <= max_passes <= 1_000_000
                or type(max_resident_bytes) is not int or not 1 <= max_resident_bytes <= 512 * 1024**2):
            raise ValueError('Exact bounded source segment, task, trial and repeat policy required')
        ids = set(); ordinals = set(); size = 0
        for batch in batches:
            preflight_forest_batch(heads, spec, batch)
            ident = batch.identity
            for record_id, ordinal in zip(ident.record_ids, ident.original_ordinals):
                id_key = (ident.source_revision, ident.export_sha256, record_id)
                ordinal_key = (ident.source_revision, ident.export_sha256, ordinal)
                if id_key in ids or ordinal_key in ordinals:
                    raise ValueError('Duplicate original forest record/ordinal in source segment')
                ids.add(id_key); ordinals.add(ordinal_key)
            size += _storage_bytes(batch) + len(_wire(_identity(batch)))
            if size > max_resident_bytes:
                raise ValueError('Complete source segment exceeds resident-byte capacity; no truncation')
        self.task_key = task_key; self.trial_id = trial_id; self.spec = spec
        self.max_passes = max_passes; self.max_resident_bytes = max_resident_bytes
        self._batches = tuple(_clone(b) for b in batches)
        self._batch_pins = tuple(forest_batch_sha256(b) for b in self._batches)
        self._frozen_declaration = _wire(self._declaration())
        self.route_sha256 = sha256(self._frozen_declaration).hexdigest()
        self.resident_bytes = size

    @property
    def source_snapshot_sha256(self):
        return self.spec.source_snapshot_sha256

    def _declaration(self):
        return dict(schema='Mora3.FiniteForestTrainingRoute/v2'if self.spec.topology_weight else'Mora3.FiniteForestTrainingRoute/v1', task_key=self.task_key,
            trial_id=self.trial_id, objective_sha256=self.spec.sha256,
            heads_sha256=self.spec.heads_sha256, source_snapshot_sha256=self.spec.source_snapshot_sha256,
            batch_sha256=self._batch_pins, max_passes=self.max_passes,
            max_resident_bytes=self.max_resident_bytes, split='train',
            sampler='engine-owned-gaussian-and-uniform-time-v1',
            skeleton_policy=self.spec.skeleton_policy)

    def _verify(self):
        if (_wire(self._declaration()) != self._frozen_declaration
                or sha256(self._frozen_declaration).hexdigest() != self.route_sha256
                or tuple(forest_batch_sha256(b) for b in self._batches) != self._batch_pins):
            raise ValueError('Frozen forest route/source/objective contents changed')

    def binding(self):
        self._verify()
        return dict(schema='Mora3.NativeForestObjectiveBinding/v2'if self.spec.topology_weight else'Mora3.NativeForestObjectiveBinding/v1', trial_id=self.trial_id,
            task_key=self.task_key, source_snapshot_sha256=self.source_snapshot_sha256,
            route_sha256=self.route_sha256, objective_sha256=self.spec.sha256,
            heads_sha256=self.spec.heads_sha256, total_windows=len(self._batches)*self.max_passes,
            sampling_method='engine-owned-gaussian-and-uniform-time-v1',
            skeleton_policy=self.spec.skeleton_policy)

    def _cursor(self, position):
        return _wire(dict(schema='Mora3.ForestTrainingCursor/v1',
                          route_sha256=self.route_sha256, position=position))

    def initial_cursor(self):
        self._verify()
        return self._cursor(0)

    def _position(self, cursor):
        if type(cursor) is not bytes or len(cursor) > 4096:
            raise ValueError('Bounded exact native forest cursor bytes required')
        try:
            value = json.loads(cursor)
        except (ValueError, UnicodeError, RecursionError) as exc:
            raise ValueError('Malformed native forest cursor') from exc
        if (type(value) is not dict or set(value) != {'schema', 'route_sha256', 'position'}
                or value['schema'] != 'Mora3.ForestTrainingCursor/v1'
                or value['route_sha256'] != self.route_sha256 or type(value['position']) is not int
                or not 0 <= value['position'] <= len(self._batches) * self.max_passes
                or _wire(value) != cursor):
            raise ValueError('Exact canonical forest source cursor and unchanged route required')
        return value['position']

    def select(self, heads, cursor, *, max_windows):
        """Source-only preparation, never committing a cursor or drawing randomness."""
        self._verify()
        if type(max_windows) is not int or not 1 <= max_windows <= 1024:
            raise ValueError('Explicit bounded source-window scan required')
        position = self._position(cursor); end = len(self._batches) * self.max_passes
        examined = 0
        while position < end and examined < max_windows:
            batch = self._batches[position % len(self._batches)]
            counts = preflight_forest_batch(heads, self.spec, batch)
            position += 1; examined += 1
            if any(counts.values()):
                return dict(status='prepared', batch=_clone(batch), counts=counts,
                    batch_sha256=self._batch_pins[(position - 1) % len(self._batches)],
                    cursor_before=cursor, cursor_after=self._cursor(position),
                    unavailable_batches=examined - 1,
                    original_source_rows=len(batch.identity.record_ids))
        return dict(status='source-exhausted' if position == end else 'unavailable-scan-limit',
            batch=None, counts=None, cursor_before=cursor, cursor_after=cursor,
            unavailable_batches=examined, original_source_rows=0)


def native_engine_forest_loss(engine, route, *, trial_id, max_windows=1):
    """Prepare one admitted native objective using checkpoint-owned randomness.

    No optimizer or cursor mutation occurs here. Production admission is checked
    before route access. A pre-update failure restores sampler/global RNG; an
    optimizer-phase failure belongs to counted-run recovery, not this function.
    """
    if type(engine) is not NativeMuUpdateEngine or type(engine.model) is not SourceBoundOwner:
        raise ValueError('Exact native engine and registered source owner required')
    engine.model.require_training_admission(campaign_sha256=engine.binding_sha256, trial_id=trial_id)
    if type(route) is not ForestTrainingRoute or trial_id != route.trial_id:
        raise ValueError('Exact forest route and trial binding required')
    binding = route.binding()
    engine.require_forest_objective_binding(**binding)
    heads = heads_from_owner(engine.model)
    if (engine.next_task != route.task_key or engine.task_cycle.count(route.task_key) < 1
            or engine.model.get_extra_state()['source_snapshot_sha256'] != route.source_snapshot_sha256
            or heads.sha256 != route.spec.heads_sha256):
        raise ValueError('Exact next native forest task, source snapshot and registered heads required')
    sampler_state = engine._sampler_state()
    if sampler_state is None or torch.device(sampler_state['device']) != heads.flow.readout.weight.device:
        raise ValueError('Engine-owned same-device forest sampler generator required')
    before = engine.source_cursors[route.task_key]
    selected = route.select(heads, before, max_windows=max_windows)
    if selected['batch'] is None:
        return dict(selected, loss=None, component_losses={}, training_admission=False)
    batch = selected['batch']
    # Repeat all-objective preflight on the returned independent source copy.
    counts = preflight_forest_batch(heads, route.spec, batch)
    global_rng = _rng_state(); random_before = engine.sampler_generator.get_state().clone()
    try:
        noise = time = None
        if counts['flow']:
            noise = torch.randn(batch.flow.values.shape, generator=engine.sampler_generator,
                device=batch.flow.values.device, dtype=batch.flow.values.dtype)
            time = torch.rand(len(batch.identity.record_ids), generator=engine.sampler_generator,
                device=noise.device, dtype=noise.dtype)
        result = forest_joint_loss(heads, route.spec, batch, noise=noise, time=time)
        engine.require_forest_objective_binding(**route.binding())
        if heads.sha256 != route.spec.heads_sha256 or forest_batch_sha256(batch) != selected['batch_sha256']:
            raise ValueError('Source/objective configuration changed during native forest evaluation')
        if result['loss'] is None or not result['loss'].requires_grad:
            raise ValueError('Prepared forest objective must be differentiable')
    except BaseException:
        engine.sampler_generator.set_state(random_before)
        _restore_rng(global_rng)
        raise
    selected.pop('batch')
    selected['source_evidence_kind']=batch.identity.evidence_kind
    selected.update(result)
    selected['status'] = 'objective-prepared-not-updated'
    selected['optimizer_updates'] = 0
    return selected


def run_native_forest_update(engine, route, *, trial_id, max_windows=1):
    """One existing-engine update after genuine admission; not a standalone runner."""
    prepared = native_engine_forest_loss(engine, route, trial_id=trial_id, max_windows=max_windows)
    if prepared['loss'] is None:
        return prepared
    # Do not catch and pretend rollback once optimizer execution has begun.
    event = engine.update(route.task_key, prepared['loss'], cursor_before=prepared['cursor_before'],
                          cursor_after=prepared['cursor_after'])
    return dict(status='updated', event=event, counts=prepared['counts'],
        component_losses={k: float(v.detach()) for k, v in prepared['component_losses'].items()},
        cursor_before=prepared['cursor_before'], cursor_after=prepared['cursor_after'],
        batch_sha256=prepared['batch_sha256'], objective_sha256=prepared['objective_sha256'],
        original_source_rows=prepared['original_source_rows'],
        unavailable_batches=prepared['unavailable_batches'])
