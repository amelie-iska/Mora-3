"""Role-isolated M-CSA source materialization, not a training admission.

Only the existing train shards, aligned train groups and condition-only fit are
opened. The mixed candidate JSONL and validation/test shards are not opened,
including for hashing. Exact original candidate ordinals come from an explicitly
pinned, separately checked original train context ledger. New corpus authority,
model authority, role changes and target-derived predictor inputs are not created.
"""
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import re

from .mcsa_role_list_admission import (
    FAMILY, TASK, NATIVE, _SEMANTICS, _identity, _json, _wire, _live_code_hashes,
)
from .mcsa_weak_sites import (
    WeakRoleVocabulary, collate_weak_site_candidates, parse_weak_site_candidate,
)

_HEX = re.compile(r'[0-9a-f]{64}\Z')
_CONTEXT_FIELDS = frozenset((
    'task_id', 'source_record_id', 'source_candidate_record_ordinal',
    'current_sequence_provenance', 'observed_source_mapping_views',
    'observed_source_role_EOS_symbols', 'vocabulary_axis_sha256',
    'grouping_keys_are_audit_only',
))
_PROTECTED_NAMES = frozenset(('initial_specs_prompt.md','keys.txt','.env','.git','.ssh','.aws'))


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _path(root, value):
    p = Path(value)
    _require('..' not in p.parts and not any(
        part in _PROTECTED_NAMES or part.startswith('.env.') for part in p.parts),
        'Unapproved source path')
    if not p.is_absolute():
        p = root / p
    _require(p.is_relative_to(root) and not any(q.is_symlink() for q in (p,*p.parents)),
             'Source must be a nonsymlink file inside the explicit root')
    return p


class _Reader:
    def __init__(self, root, budget):
        self.root, self.budget = root, budget
        self.used, self.pins, self.cache = 0, {}, {}

    def read(self, ref):
        _require(type(ref) is dict and type(ref.get('path')) is str
                 and type(ref.get('bytes')) is int and ref['bytes'] > 0
                 and type(ref.get('sha256')) is str and _HEX.fullmatch(ref['sha256']),
                 'Exact source path, bytes and SHA256 required')
        p = _path(self.root, ref['path'])
        key = str(p)
        if key in self.cache:
            digest, raw = self.cache[key]
            _require(digest == ref['sha256'] and len(raw) == ref['bytes']
                     and _identity(p) == self.pins[key][0], 'Source reference changed')
            return raw
        _require(self.used + ref['bytes'] <= self.budget, 'Source materialization byte budget exceeded')
        before = _identity(p)
        _require(before[6] == ref['bytes'], 'Source size changed')
        with p.open('rb') as stream:
            raw = stream.read(ref['bytes'] + 1)
        _require(_identity(p) == before and len(raw) == ref['bytes']
                 and sha256(raw).hexdigest() == ref['sha256'], 'Source artifact changed or digest differs')
        self.used += len(raw)
        self.pins[key] = (before, ref['sha256'])
        self.cache[key] = (ref['sha256'], raw)
        return raw


@dataclass(frozen=True)
class MCSATrainSourceBatch:
    pass_index: int
    original_candidate_row_ordinals: tuple[int, ...]
    batch: object
    observed_targets: int


class MCSATrainMaterialization:
    """Immutable source bytes; each decoded record/batch is a fresh value.

    This class is deliberately NOT MCSARoleListAdmission. It cannot authorize
    optimization or make any unsupported full-corpus/global-split assertion.
    """
    __slots__ = ('_payloads','_pins','_review_sha','_component','_binding','_vocabulary',
                 '_contexts_sha','_bytes_read','_site_count','_symbol_count','_component_sha')

    def __init__(self, *args, **kwargs):
        raise TypeError('Use read_mcsa_train_materialization with explicit source pins and budgets')

    def __setattr__(self, name, value):
        raise AttributeError('Source materialization is immutable')

    def require_unchanged(self):
        _require(all(hasattr(self,k) for k in self.__slots__), 'Complete source materialization required')
        for path, identity, digest in self._pins:
            _require(_identity(Path(path)) == identity, 'Source artifact identity changed')

    @property
    def review_packet_sha256(self): return self._review_sha

    @property
    def original_ordinals(self): return tuple(i for i,_ in self._payloads)

    @property
    def vocabulary(self): return self._vocabulary

    def component(self): return _json(self._component)

    def model_binding(self): return _json(self._binding)

    def records(self):
        self.require_unchanged()
        return tuple(parse_weak_site_candidate(_json(raw),self._vocabulary) for _,raw in self._payloads)

    def iter_batches(self, policy, *, excluded_ordinals=()):
        from mora3.research.mcsa_campaign_route import MCSAFinitePassPolicy
        _require(type(policy) is MCSAFinitePassPolicy, 'Exact native finite-pass policy required')
        policy.__post_init__()
        _require(type(excluded_ordinals) is tuple
                 and all(type(i) is int for i in excluded_ordinals)
                 and excluded_ordinals == tuple(sorted(set(excluded_ordinals)))
                 and set(excluded_ordinals) <= set(self.original_ordinals),
                 'Exact original train exclusion ordinals required')
        records = dict(zip(self.original_ordinals,self.records(),strict=True))
        kept = tuple(i for i in records if i not in excluded_ordinals)
        _require(bool(kept), 'No train rows remain')
        for pass_index in range(policy.max_passes):
            order = policy.order(kept,pass_index)
            for offset in range(0,len(order),policy.batch_size):
                self.require_unchanged()
                ordinals = order[offset:offset+policy.batch_size]
                batch = collate_weak_site_candidates((records[i] for i in ordinals),self._vocabulary)
                count = sum(len(x.target_ids) for x in batch.source_list_supervision)
                yield MCSATrainSourceBatch(pass_index,ordinals,batch,count)

    def summary(self):
        self.require_unchanged()
        return dict(schema='Mora3.MCSATrainOnlyMaterialization/v1',
            review_packet_sha256=self._review_sha,source_context_sha256=self._contexts_sha,
            train_records=len(self._payloads),observed_mapping_views=self._site_count,
            observed_role_eos_symbols=self._symbol_count,source_input_bytes=self._bytes_read,
            original_ordinal_payload_commitment_sha256=sha256(_wire(
                [(i,sha256(raw).hexdigest()) for i,raw in self._payloads])).hexdigest(),
            source_references=[dict(path=p,sha256=d,bytes=identity[6]) for p,identity,d in self._pins],
            mixed_candidate_opened=False,heldout_shards_opened=False,
            full_corpus_admitted=False,global_split_accepted=False,training_authorized=False)


def read_mcsa_train_materialization(root, *, review_path, expected_review_sha256,
                                    contexts_reference, max_input_bytes, max_records):
    """Read and cross-check COMPLETE original training shards, not mixed rows.

    The caller supplies trusted version pins. They establish integrity, not owner
    signatures. Training continues to require the original separately compiled
    MCSARoleListAdmission and complete native owner admission.
    """
    _require(all(type(n) is int and n > 0 for n in (max_input_bytes,max_records)),
             'Explicit positive integer source byte and row budgets required')
    _require(type(expected_review_sha256) is str and _HEX.fullmatch(expected_review_sha256),
             'Exact reviewed source SHA256 required')
    root = Path(root).resolve()
    review_path = _path(root,review_path)
    reader = _Reader(root,max_input_bytes)
    review = _json(reader.read(dict(path=str(review_path),bytes=review_path.stat().st_size,
                                    sha256=expected_review_sha256)))
    _require(review.get('schema') == 'Mora3.MCSARoleListReviewedSourceAdmission/v1'
             and review.get('review_state') == 'accepted-scoped-source'
             and review.get('coordinator_review',{}).get('owner') == 'root-coordinator',
             'Existing scoped source review required')
    from . import mcsa_role_list_admission as admission_module
    _require(review.get('admission_implementation_sha256') == sha256(
        Path(admission_module.__file__).read_bytes()).hexdigest(), 'Reviewed source admission code changed')
    binding = review['approved_model_binding']
    _require(all(_wire(binding.get(k)) == _wire(v) for k,v in _SEMANTICS.items())
             and all(binding.get(k) == v for k,v in _live_code_hashes().items())
             and review['coordinator_review'].get('approved_model_binding_sha256')
                 == sha256(_wire(binding)).hexdigest(), 'Reviewed native model/source semantics differ')
    component_ref = review['component_manifest']
    component = _json(reader.read(component_ref))
    _require((component.get('source_family'),component.get('task_name'),component.get('native_schema'))
                 == (FAMILY,TASK,NATIVE)
             and component.get('scientific_task_admission') is False
             and component.get('model_adapter_supported') is False
             and review['coordinator_review'].get('component_manifest_sha256') == component_ref['sha256'],
             'Exact original nonpromoted source component required')
    count = component['source_splits']['train']
    _require(type(count) is int and 0 < count <= max_records,
             'Complete train cohort exceeds explicit row budget')
    contexts = tuple(_json(line) for line in reader.read(contexts_reference).splitlines())
    _require(len(contexts)==count, 'Complete original train contexts required')
    by_task, last = {}, -1
    for context in contexts:
        _require(type(context) is dict and _CONTEXT_FIELDS <= set(context)
                 and set(context) <= _CONTEXT_FIELDS|{'source_raw_entry_ref'}, 'Exact source-context fields required')
        ordinal = context['source_candidate_record_ordinal']
        _require(type(ordinal) is int and last < ordinal < component['source_entry_views']
                 and type(context['task_id']) is str and _HEX.fullmatch(context['task_id'])
                 and context['task_id'] not in by_task
                 and context['grouping_keys_are_audit_only'] is True,
                 'Original train ordinal/order/task/evidence context differs')
        last=ordinal;by_task[context['task_id']]=context
    vocab_doc = _json(reader.read(component['source_vocabulary']))
    vocabulary = WeakRoleVocabulary.from_document(vocab_doc)
    _require(binding['axis_sha256']==vocabulary.axis_sha256
             and vocab_doc['source_candidate_file']==component['source_candidate_file'],
             'Original source-role axis/candidate identity differs')
    native_manifest = _json(reader.read(component['source_native_manifest']))
    _require(native_manifest['source_candidate_file']==component['source_candidate_file']
             and native_manifest['vocabulary']==component['source_vocabulary'],
             'Original native manifest identity differs')
    groups = {}
    for item in component['group_sidecars']:
        key=item['source_shard']['path']
        _require(key not in groups,'Duplicate source/group sidecar')
        groups[key]=item
    shards = component['shards']
    _require(len({s['path'] for s in shards})==len(shards)
             and all(s.get('split') in ('train','validation','test')
                     and type(s.get('rows')) is int and s['rows']>0 for s in shards)
             and {role:sum(s['rows'] for s in shards if s['split']==role)
                  for role in component['source_splits']}==component['source_splits']
             and sum(component['source_splits'].values())==component['source_entry_views'],
             'Source shard role/count inventory differs')
    import pyarrow.parquet as pq
    payloads=[];fit_coordinates={};seen=set();sites=symbols=characters=0
    for shard in shards:
        if shard['split']!='train':
            continue  # no stat, hash, read, or decoding of heldout payloads
        _require(shard['path'] in groups and groups[shard['path']]['source_shard']==shard,
                 'Exact aligned train sidecar binding required')
        rows=pq.read_table(BytesIO(reader.read(shard)),use_threads=False).to_pylist()
        side=groups[shard['path']]['authoritative_sidecar']
        aligned=pq.read_table(BytesIO(reader.read(side)),use_threads=False).to_pylist()
        _require(len(rows)==len(aligned)==shard['rows'],'Complete train shard/group row counts required')
        for ordinal,(row,group) in enumerate(zip(rows,aligned,strict=True)):
            raw=row['native_payload'];task=row['task_id'];rid=row['record_id']
            _require(type(raw) is bytes and sha256(raw).hexdigest()==row['native_payload_sha256']
                     and task==sha256(b'Mora3.MCSA.source-role-list/v1\0'+raw).hexdigest()
                     and task in by_task and rid not in seen
                     and (row['source_family'],row['task_name'],row['native_schema'],row['split'])
                         ==(FAMILY,TASK,NATIVE,'train')
                     and row['admitted_for_task'] is False and row['model_adapter_supported'] is False
                     and group['task_id']==task and group['effective_groups_json']==row['group_ids_json'],
                     'Original train payload, flags, identity or groups differ')
            keys=_json(row['group_ids_json'])
            _require(type(keys) is list and keys and all(type(k) is str and k for k in keys)
                     and len(set(keys))==len(keys),'Complete nonempty original groups required')
            native=_json(raw);record=parse_weak_site_candidate(native,vocabulary);context=by_task[task]
            observed=sum(len(x.role_ids)+1 for x in record.source_lists)
            _require(record.audit['record_id']==context['source_record_id']==rid
                     and record.audit['proposed_split']=='train'
                     and _wire(record.audit['current_uniprot_provenance'])==_wire(context['current_sequence_provenance'])
                     and context['vocabulary_axis_sha256']==vocabulary.axis_sha256
                     and type(context['observed_source_mapping_views']) is int
                     and context['observed_source_mapping_views']==len(record.source_lists)
                     and type(context['observed_source_role_EOS_symbols']) is int
                     and context['observed_source_role_EOS_symbols']==observed,
                     'Original train context, provenance or complete targets differ')
            if 'source_raw_entry_ref' in context:
                _require(context['source_raw_entry_ref']==native['audit']['original_MCSA_entry_ref'],
                         'Original source entry pointer differs')
            _require(max(len(x.role_ids)+1 for x in record.source_lists)<=binding['max_target_length'],
                     'Complete ordered source-role target exceeds reviewed bound')
            seen.add(rid);sites+=len(record.source_lists);symbols+=observed;characters+=len(record.canonical_sequence)
            payloads.append((context['source_candidate_record_ordinal'],raw))
            fit_coordinates[rid]=(record.condition_inputs(),dict(source_record_id=rid,
                source_task_id=task,source_native_payload_sha256=row['native_payload_sha256'],
                source_shard=shard,source_shard_row_ordinal=ordinal,original_split='train',
                canonical_sequence_sha256=sha256(record.canonical_sequence.encode()).hexdigest(),
                current_sequence_provenance=record.audit['current_uniprot_provenance']))
    _require(len(payloads)==count and sites==component['sparse_mapping_views_by_split']['train']
             and symbols==component['role_EOS_symbols_by_split']['train'],
             'Full training cohort and complete role/EOS denominators differ')
    fit=_json(reader.read(review['payload_fit_receipt']))
    _require(fit.get('schema')=='Mora3.MCSACanonicalConditionPayloadFit/v1'
             and fit.get('status')=='complete-source-fit' and fit.get('target_fields_used') is False
             and fit.get('heldout_rows_used') is False
             and fit.get('fit_fields')==['conditions.canonical_sequence']
             and fit['source_component_manifest']==component_ref
             and fit['payload_vocabulary_artifact']==review['payload_vocabulary_artifact'],
             'Exact source-only fitted condition provenance required')
    fit_seen=set()
    for line in reader.read(fit['fit_input']).splitlines():
        item=_json(line)
        _require(set(item)=={'conditions','audit'} and type(item['audit']) is dict,
                 'Condition-only fit rows required')
        rid=item['audit'].get('source_record_id')
        _require(rid in fit_coordinates and rid not in fit_seen,'Unique complete original train fit rows required')
        conditions,audit=fit_coordinates[rid]
        _require(_wire(item['conditions'])==_wire(conditions) and _wire(item['audit'])==_wire(audit),
                 'Actual train conditions or fit-source coordinates differ')
        fit_seen.add(rid)
    _require(fit_seen==seen and fit['train_views']==count
             and fit['full_canonical_sequence_characters']==characters,'Complete actual train fit population differs')
    result=object.__new__(MCSATrainMaterialization)
    for key,value in dict(_payloads=tuple(sorted(payloads)),
            _pins=tuple((p,identity,digest)for p,(identity,digest)in sorted(reader.pins.items())),
            _review_sha=expected_review_sha256,_component=_wire(component),_component_sha=component_ref['sha256'],_binding=_wire(binding),
            _vocabulary=vocabulary,_contexts_sha=contexts_reference['sha256'],
            _bytes_read=reader.used,_site_count=sites,_symbol_count=symbols).items():
        object.__setattr__(result,key,value)
    result.require_unchanged()
    return result
