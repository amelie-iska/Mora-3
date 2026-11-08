"""Finite passes over the reviewed original M-CSA train source role."""

from dataclasses import asdict, dataclass
from pathlib import Path
import re
from decimal import Decimal, localcontext
from types import MappingProxyType

from mora3.data.admet import _json, canonical, digest
from mora3.data.finite_union_native_rows import read_exact
from mora3.data.mcsa_role_list_admission import FAMILY, NATIVE, TASK, MCSARoleListAdmission
from mora3.data.mcsa_weak_sites import (
    WeakRoleVocabulary, WeakSiteBatch, collate_weak_site_candidates, parse_weak_site_candidate, read_weak_site_candidates,
)


_SHA256 = re.compile(r'[a-f0-9]{64}\Z')
_CURSOR_SCHEMA = 'Mora3.MCSAOriginalTrainPassCursor/v1'
_ORDER_POLICY = 'sha256-canonical-pass-seed-and-absolute-original-candidate-ordinal-v1'


@dataclass(frozen=True)
class MCSAFinitePassPolicy:
    """Use every reviewed train record once per pass and retain the short last batch."""

    max_passes: int
    batch_size: int
    order_seed: int

    def __post_init__(self):
        if (
            type(self.max_passes) is not int or self.max_passes < 1
            or type(self.batch_size) is not int or self.batch_size < 1
            or type(self.order_seed) is not int or not 0 <= self.order_seed < 2**64
        ):
            raise ValueError('Explicit finite M-CSA passes, batch size and 64-bit order seed required')

    def order(self, ordinals, pass_index):
        if (
            type(pass_index) is not int or not 0 <= pass_index < self.max_passes
            or type(ordinals) is not tuple or not ordinals
            or any(type(i) is not int or i < 0 for i in ordinals)
            or ordinals != tuple(sorted(set(ordinals)))
        ):
            raise ValueError('Exact pass and complete sorted original candidate ordinals required')
        return tuple(sorted(ordinals,key=lambda ordinal: (
            digest(canonical([_ORDER_POLICY,self.order_seed,pass_index,ordinal])),ordinal,
        )))


@dataclass(frozen=True)
class MCSATrainPassWindow:
    pass_index: int
    original_candidate_row_ordinals: tuple[int, ...]
    batch: WeakSiteBatch
    observed_targets: int
    cursor_after: bytes
    evidence_sha256: str

    @property
    def original_source_rows(self):
        return len(self.original_candidate_row_ordinals)


@dataclass(frozen=True)
class MCSAValidationBatch:
    batch: WeakSiteBatch
    coordinates: tuple
    observed_targets: int

    @property
    def original_source_rows(self):
        return len(self.coordinates)


class MCSACampaignTrainRoute:
    """One reviewed M-CSA source for original train and matched validation."""

    task_key = TASK
    loss_unit = 'ordered native source-role-list CE in nats'
    target_unit = 'observed ordered source role symbols and EOS'

    def __init__(self, review_packet, expected_sha256, *, vocabulary, source_snapshot_sha256, pass_policy,
                 reviewed_payload_max_tokens):
        if (
            type(vocabulary) is not WeakRoleVocabulary or type(pass_policy) is not MCSAFinitePassPolicy
            or type(reviewed_payload_max_tokens) is not int or reviewed_payload_max_tokens < 1
            or any(type(value) is not str or not _SHA256.fullmatch(value)
                   for value in (source_snapshot_sha256,expected_sha256))
        ):
            raise ValueError('Exact trusted M-CSA source review, finite policy and campaign snapshot required')
        review_packet = Path(review_packet)
        packet_ref = dict(path=str(review_packet),bytes=review_packet.stat().st_size,sha256=expected_sha256)
        packet = _json(read_exact(packet_ref,root=Path.cwd()))
        model_binding = packet.get('approved_model_binding') if type(packet) is dict else None
        if (type(model_binding) is not dict or type(model_binding.get('payload_max_tokens')) is not int
                or model_binding['payload_max_tokens']!=reviewed_payload_max_tokens):
            raise ValueError('M-CSA route requires literal reviewed payload owner capacity')
        admission = MCSARoleListAdmission.from_review_packet(review_packet,expected_sha256,vocabulary=vocabulary)
        component_ref = packet['component_manifest']
        component = _json(read_exact(component_ref,root=Path.cwd()))
        candidate = component['source_candidate_file']
        records = read_weak_site_candidates(candidate['path'],candidate['sha256'],vocabulary)
        train = {i:record for i,record in enumerate(records) if record.audit['proposed_split']=='train'}
        symbols = sum(len(site.role_ids)+1 for record in train.values() for site in record.source_lists)
        if (not train or len(records)!=component['source_entry_views']
                or len(train)!=component['source_splits']['train']
                or symbols!=component['role_EOS_symbols_by_split']['train']):
            raise ValueError('Complete reviewed original train source records and EOS counts required')
        self._admission = admission
        self._records = MappingProxyType(train)
        self._ordinals = tuple(train)
        self._vocabulary = vocabulary
        self._model_binding = canonical(model_binding)
        self._policy = pass_policy
        self._symbols = symbols
        self._configuration = canonical(dict(
            schema='Mora3.MCSAOriginalTrainPassRoute/v1',source_snapshot_sha256=source_snapshot_sha256,
            source_family=FAMILY,task_name=TASK,native_schema=NATIVE,role='train',
            review_packet_sha256=expected_sha256,source_component_sha256=component_ref['sha256'],
            source_candidate_sha256=candidate['sha256'],reviewed_model_binding_sha256=digest(self._model_binding),
            reviewed_payload_max_tokens=reviewed_payload_max_tokens,
            native_train_record_count=len(train),native_train_role_and_EOS_symbols=symbols,
            policy=dict(**asdict(pass_policy),order=_ORDER_POLICY,retain_incomplete_batch=True,
                        no_pass_boundary_batch_mixing=True,no_heldout_roles=True),
        ))
        self._route_sha256 = digest(self._configuration)
        self._validation_shards = tuple(canonical(shard) for shard in sorted(
            (shard for shard in component['shards'] if shard['split']=='validation'),key=lambda shard:shard['path'],
        ))
        self._validation_rows = component['source_splits']['validation']
        self._validation_symbols = component['role_EOS_symbols_by_split']['validation']
        self._validation_policy_sha256 = digest(canonical([
            'Mora3.MCSAOriginalValidationSource/v1',source_snapshot_sha256,expected_sha256,
            component_ref['sha256'],tuple(_json(ref) for ref in self._validation_shards),
            self._validation_rows,self._validation_symbols,
        ]))

    @property
    def source_snapshot_sha256(self):
        return self.binding()['source_snapshot_sha256']

    @property
    def native_view_policy_sha256(self):
        return self._route_sha256

    @property
    def validation_policy_sha256(self):
        return self._validation_policy_sha256

    @property
    def admission(self):
        return self._admission

    @property
    def records_per_pass(self):
        return len(self._ordinals)

    @property
    def symbols_per_pass(self):
        return self._symbols

    @property
    def maximum_updates(self):
        batch = self._policy.batch_size
        return self._policy.max_passes*((self.records_per_pass+batch-1)//batch)

    def require_task_cycle_capacity(self, task_cycle, total_updates):
        if (type(task_cycle) is not tuple or self.task_key not in task_cycle
                or type(total_updates) is not int or total_updates<1):
            raise ValueError('Exact finite M-CSA training task cycle and update budget required')
        complete, remainder = divmod(total_updates,len(task_cycle))
        required = complete*task_cycle.count(self.task_key)+task_cycle[:remainder].count(self.task_key)
        if self.maximum_updates<required:
            raise ValueError('Exact M-CSA finite pass policy cannot cover the planned task cycle')

    def binding(self):
        return _json(self._configuration)

    def _cursor(self, pass_index, offset):
        return canonical(dict(schema=_CURSOR_SCHEMA,route_sha256=self._route_sha256,role='train',
                              pass_index=pass_index,next_offset=offset))

    @property
    def initial_cursor(self):
        return self._cursor(0,0)

    def _position(self, cursor):
        if type(cursor) is not bytes:
            raise ValueError('Exact serialized M-CSA source cursor required')
        try:
            value = _json(cursor)
            pass_index,offset = value['pass_index'],value['next_offset']
        except (KeyError,TypeError,ValueError,UnicodeError) as exc:
            raise ValueError('Exact serialized M-CSA source cursor required') from exc
        if (
            type(pass_index) is not int or not 0 <= pass_index <= self._policy.max_passes
            or type(offset) is not int or not 0 <= offset < self.records_per_pass
            or offset%self._policy.batch_size
            or pass_index==self._policy.max_passes and offset!=0
            or cursor!=self._cursor(pass_index,offset)
        ):
            raise ValueError('Exact M-CSA source cursor route, pass and batch boundary required')
        return pass_index,offset

    def iter_from(self, cursor):
        """Yield the exact remaining reviewed train windows without updating the caller's cursor."""
        first,offset = self._position(cursor)
        model_binding = _json(self._model_binding)
        size = self._policy.batch_size
        for pass_index in range(first,self._policy.max_passes):
            order = self._policy.order(self._ordinals,pass_index)
            start = offset if pass_index==first else 0
            for index in range(start,len(order),size):
                stop = min(index+size,len(order))
                ordinals = order[index:stop]
                records = tuple(self._records[i] for i in ordinals)
                batch = collate_weak_site_candidates(records,self._vocabulary)
                self._admission.require_batch(batch,model_binding=model_binding,split='train')
                count = sum(len(row.target_ids) for row in batch.source_list_supervision)
                after = self._cursor(pass_index+1,0) if stop==len(order) else self._cursor(pass_index,stop)
                evidence = digest(canonical([
                    self._route_sha256,pass_index,tuple(zip(ordinals,
                        (record.audit['record_id'] for record in records),strict=True)),count,
                ]))
                yield MCSATrainPassWindow(pass_index,ordinals,batch,count,after,evidence)


    def train_binding(self):
        from .primary_native_routes import NativeRouteBinding

        return NativeRouteBinding(self,self.prepare,self.loss_unit,self.target_unit)

    def _owner_branch(self, model):
        from mora3.model.full_owner import SourceBoundOwner
        from mora3.model.mcsa_role_list import SourceRoleListModel

        if (type(model) is not SourceBoundOwner
                or model.get_extra_state()['source_snapshot_sha256']!=self.source_snapshot_sha256
                or type(model.mcsa) is not SourceRoleListModel
                or canonical(model.mcsa.admission_binding())!=self._model_binding):
            raise ValueError('Exact registered M-CSA source owner, snapshot and literal reviewed capacity required')
        return model.mcsa

    def prepare(self, window):
        from .primary_native_routes import PreparedNativeTargets

        if type(window) is not MCSATrainPassWindow:
            raise ValueError('Exact reviewed M-CSA train window required')
        self.admission.require_batch(window.batch,model_binding=_json(self._model_binding),split='train')
        if sum(len(row.target_ids) for row in window.batch.source_list_supervision)!=window.observed_targets:
            raise ValueError('Exact M-CSA train target denominator required')

        def evaluate(model):
            branch = self._owner_branch(model)
            model.require_training_admission()
            logits = branch(window.batch,purpose='training',admission=self.admission,split='train')
            return branch.objective(logits,window.batch,purpose='training',admission=self.admission,split='train')['total']

        return PreparedNativeTargets(window.observed_targets,None,evaluate)

    def _validation_batches(self, *, read_batch_size, max_selected_bytes):
        import io
        import pyarrow.parquet as pq
        from .matched_validation import ValidationSourceRow

        if (type(read_batch_size) is not int or read_batch_size<1
                or type(max_selected_bytes) is not int or max_selected_bytes<1):
            raise ValueError('Finite M-CSA validation read and selected Arrow byte budget required')
        batches,task_ids = [],set()
        selected_bytes = 0
        for reference in map(_json,self._validation_shards):
            if reference['bytes']>max_selected_bytes:
                raise ValueError('Complete M-CSA validation shard exceeds selected source budget')
            table = pq.read_table(io.BytesIO(read_exact(reference,root=Path.cwd())),use_threads=False)
            selected_bytes += table.nbytes
            if selected_bytes>max_selected_bytes:
                raise ValueError('Complete M-CSA validation selected Arrow byte budget exceeded')
            if table.num_rows!=reference['rows']:
                raise ValueError('Complete original M-CSA validation shard row count differs')
            for start in range(0,table.num_rows,read_batch_size):
                coordinates,records = [],[]
                for ordinal,row in enumerate(table.slice(start,read_batch_size).to_pylist(),start):
                    payload = row['native_payload']
                    payload_sha = digest(payload)
                    task_id = digest(b'Mora3.MCSA.source-role-list/v1\0'+payload)
                    if ((row['source_family'],row['task_name'],row['native_schema'],row['split'])!=(FAMILY,TASK,NATIVE,'validation')
                            or row['native_payload_sha256']!=payload_sha or row['task_id']!=task_id or task_id in task_ids):
                        raise ValueError('Exact complete original M-CSA validation identity and role required')
                    task_ids.add(task_id)
                    record = parse_weak_site_candidate(_json(payload),self._vocabulary)
                    if record.audit['record_id']!=row['record_id']:
                        raise ValueError('Original M-CSA validation record identity differs')
                    records.append(record)
                    # Legacy component schemas without this column are pinned
                    # to their literal immutable component SHA instead.
                    revision = row.get('source_revision','reviewed-component-sha256:'+self.binding()['source_component_sha256'])
                    coordinates.append(ValidationSourceRow(reference['path'],reference['sha256'],ordinal,task_id,payload_sha,revision))
                batch = collate_weak_site_candidates(records,self._vocabulary)
                self.admission.require_batch(batch,model_binding=_json(self._model_binding),split='validation')
                count = sum(len(target.target_ids) for target in batch.source_list_supervision)
                batches.append(MCSAValidationBatch(batch,tuple(coordinates),count))
        if (sum(batch.original_source_rows for batch in batches)!=self._validation_rows
                or sum(batch.observed_targets for batch in batches)!=self._validation_symbols):
            raise ValueError('Complete reviewed original M-CSA validation records and EOS symbols required')
        return tuple(batches)

    def validation_enrollment(self, *, max_selected_bytes):
        batches = self._validation_batches(read_batch_size=64,max_selected_bytes=max_selected_bytes)
        return tuple(coordinate for batch in batches for coordinate in batch.coordinates),self._validation_symbols

    def matched_validation_batches(self, task, *, read_batch_size, max_selected_bytes):
        from .matched_validation import MatchedValidationTask

        if (type(task) is not MatchedValidationTask
                or (task.task_key,task.source_family,task.native_task_name,task.native_schema,
                    task.loss_unit,task.target_unit)!=(TASK,FAMILY,TASK,NATIVE,self.loss_unit,self.target_unit)):
            raise ValueError('Exact matched M-CSA native validation route and units required')
        batches = self._validation_batches(read_batch_size=read_batch_size,max_selected_bytes=max_selected_bytes)
        if (tuple(coordinate for batch in batches for coordinate in batch.coordinates)!=task.coordinates
                or self._validation_symbols!=task.observed_targets):
            raise ValueError('Exact complete reviewed M-CSA validation coordinates and denominator required')
        return batches

    def evaluate_validation(self, model, window):
        import torch
        from .mup_campaign import ValidationBatchReduction

        if torch.is_grad_enabled() or type(window) is not MCSAValidationBatch:
            raise ValueError('Exact reviewed M-CSA validation batch requires no-grad evaluation')
        branch = self._owner_branch(model)
        self.admission.require_batch(window.batch,model_binding=_json(self._model_binding),split='validation')
        if sum(len(row.target_ids) for row in window.batch.source_list_supervision)!=window.observed_targets:
            raise ValueError('Exact M-CSA validation target denominator required')
        logits = branch(window.batch,purpose='evaluation',admission=self.admission,split='validation')
        loss = branch.objective(logits,window.batch,purpose='evaluation',admission=self.admission,split='validation')['total']
        if loss.ndim or not torch.isfinite(loss):
            raise ValueError('Finite scalar M-CSA native validation objective required')
        with localcontext() as ctx:
            ctx.prec = 34
            total = Decimal(str(float(loss)))*window.observed_targets
        return ValidationBatchReduction(total,window.observed_targets,None)
