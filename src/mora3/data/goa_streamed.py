"""Bounded native GOA batches from immutable selected source rows.

Only exact current canonical sequences qualify. Forms need their own source
coordinate policy. Every raw assertion and explicit NOT is retained; unknown
terms, ND, extension scope and conflicting polarities follow the existing
versioned coarse GOA projector. This supplies data binding, not campaign approval.
"""
from collections import OrderedDict
import copy
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path

from .finite_union_native_rows import FiniteUnionNativeView,SelectedNativeSourceBatch,read_exact,full9
from .annotation_axes import AnnotationAxes
from .goa_packets import VerifiedGOAPacket,verify_goa_packet,bind_goa_applicability
from .uniprot_annotations import _canonical


def _sha(raw):return hashlib.sha256(raw).hexdigest()


def _code_hashes(*,combined_study=False):
    data=Path(__file__).resolve().parent
    paths=(data/'goa_streamed.py',data/'goa_packets.py',data/'finite_union_native_rows.py',
           data/'uniprot_annotations.py',data.parent/'model/uniprot_go.py')
    if combined_study:paths+=(data/'finite_union_annotation_study_filter.py',)
    return {str(p.relative_to(data.parent)):_sha(p.read_bytes()) for p in paths}


_NATIVE_FORM_SEQUENCE_BINDINGS = frozenset((
    ('displayed_isoform', 'sourceDisplayed_alias_exact_canonical_sequence'),
    ('isoform', 'exact_official_same_release_isoform_FASTA'),
    ('processed_product', 'exact_canonical_interval_primary_sequence'),
))


class GOAFormOutsideCanonicalScope(ValueError):
    """An authenticated original form packet requires its own source policy."""


@dataclass(frozen=True)
class StreamedGOABatch:
    source:SelectedNativeSourceBatch
    packets:tuple
    pool:object


class StreamedGOASourcePool:
    def __init__(self,*,root,native_filter_path,native_filter_sha256,registry,
                 canonical_reference_manifest,canonical_reference_sha256,
                 ontology_manifest,ontology_sha256,max_batch_size=128,max_cached_kb_parts=4,
                 native_filter_policy='first-study-overlay-v1'):
        if (not isinstance(registry,AnnotationAxes)
                or any(type(v)is not int or v<1 for v in(max_batch_size,max_cached_kb_parts))):
            raise ValueError('complete native annotation axes and positive bounded batch/cache required')
        if native_filter_policy not in ('first-study-overlay-v1','combined-study-overlay-v2'):
            raise ValueError('explicit known native GOA filter policy required')
        self.root=Path(root).resolve();self.registry=registry
        self.max_batch_size=max_batch_size;self.max_cached_kb_parts=max_cached_kb_parts
        self._cache=OrderedDict()
        def reference(path,digest):
            p=(self.root/path).resolve()
            if not p.is_relative_to(self.root):raise ValueError('source outside project')
            raw=p.read_bytes()
            if _sha(raw)!=digest:raise ValueError('exact source manifest digest required')
            return dict(path=str(p),bytes=len(raw),sha256=digest),json.loads(raw)
        filter_ref,descriptor=reference(native_filter_path,native_filter_sha256)
        combined=native_filter_policy=='combined-study-overlay-v2'
        if combined:
            if descriptor.get('schema')!='Mora3.AdditionalFiniteNativeStudyOrdinalFilter/v1':
                raise ValueError('combined GOA policy requires the exact additional study descriptor')
            from .finite_union_annotation_study_filter import load_study_filtered_native_view
            self.native_view=load_study_filtered_native_view(filter_ref['path'],native_filter_sha256,root=self.root)
            self._combined_view_owner=(self.native_view,)
        else:
            self.native_view=FiniteUnionNativeView.from_effective_filter(filter_ref['path'],native_filter_sha256,root=self.root)
        self._descriptors={d['source_shard']['path']:d for d in self.native_view.descriptors
            if d['source_task_schema_counts'] and all(x['source_family']=='UniProt-GOA-assertion-packets'
                and x['task_name']=='source_GOA_assertion_packet_prediction'
                and x['native_schema']=='mora3.CorpusTask/v1' for x in d['source_task_schema_counts'])}
        if not self._descriptors:raise ValueError('no complete source GOA shards in current native view')
        self.source_paths=tuple(self._descriptors)
        canonical_ref,canonical=reference(canonical_reference_manifest,canonical_reference_sha256)
        if (canonical.get('schema')!='Mora3.NativeUniProtReferenceAssertionBridge/v1'
                or canonical.get('source_release')!='2026_03'
                or canonical.get('exact_current_canonical_sequence_membership_bound')is not True):
            raise ValueError('exact current canonical source index required; no form inference')
        index_ref=canonical['complete_entry_index'];index_raw=read_exact(index_ref,root=self.root)
        self._canonical={}
        for line in index_raw.splitlines():
            row=json.loads(line);accession=row['primary_accession']
            if accession in self._canonical or row['source_release']!='2026_03':
                raise ValueError('duplicate or cross-release current canonical source')
            self._canonical[accession]=row
        if len(self._canonical)!=canonical['complete_current_entry_cohort']:
            raise ValueError('complete frozen current canonical reference index required')
        read_exact(canonical['full_native_manifest'],root=self.root)
        ontology_ref,ontology=reference(ontology_manifest,ontology_sha256)
        if (ontology.get('schema')!='mora3.GOCurrentReferenceAxisBinding/v1'
                or ontology.get('same_snapshot')is not False
                or any(ontology.get(k)is not True for k in('no_automatic_obsolete_or_alt_ID_remapping',
                    'no_GO_ancestry_label_propagation','no_dense_negative_labels'))):
            raise ValueError('explicit current-reference GO ontology policy required')
        axis_refs=[a for a in registry.binding()['axis_files'] if a['axis']=='GO']
        if len(axis_refs)!=1 or any(axis_refs[0][k]!=ontology['source_axis'][k] for k in('path','bytes','sha256')):
            raise ValueError('exact registry/ontology GO source axis required')
        self.axis_ids=tuple(registry.axis('GO'))
        self.ontology_manifest=ontology_ref['path'];self.ontology_sha256=ontology_sha256
        self._binding=dict(schema='Mora3.StreamedCanonicalGOASourcePool/v1',native_filter=filter_ref,
            canonical_reference_manifest=canonical_ref,canonical_entry_index=index_ref,
            ontology_manifest=ontology_ref,axis_sha256=_sha(_canonical(self.axis_ids)),
            source_paths_sha256=_sha(_canonical(self.source_paths)),source_shards=len(self.source_paths),
            source_role='train',canonical_sequence='exact-current-reference-full-string;no-form-transfer',
            supervision='existing-coarse-GOA-explicit-observed-signs;unknown-not-negative;polarity-conflict-quarantine',
            max_batch_size=max_batch_size,source_training_admission=False)
        if combined:
            self._binding.update(native_filter_policy=native_filter_policy,
                effective_filter_payload_sha256=_sha(_canonical(self.native_view.effective_filter)))
        self._binding['source_code']=_code_hashes(combined_study=combined)
        self._binding_sha256=_sha(_canonical(self._binding))

    def head_binding(self):
        combined=self._binding.get('native_filter_policy')=='combined-study-overlay-v2'
        if (_sha(_canonical(self._binding))!=self._binding_sha256
                or self._binding['source_code']!=_code_hashes(combined_study=combined)
                or combined and (self.native_view is not self._combined_view_owner[0]
                    or _sha(_canonical(self.native_view.effective_filter))!=self._binding['effective_filter_payload_sha256'])
                or _sha(_canonical(self.axis_ids))!=self._binding['axis_sha256']
                or _sha(_canonical(self.source_paths))!=self._binding['source_paths_sha256']):
            raise ValueError('streamed source pool/axis identity drift')
        return copy.deepcopy(self._binding)

    def _canonical_sequence(self,packet):
        import pyarrow.parquet as pq
        p=packet.provenance();binding=p['source_audit']['binding'];parent=binding['parent_accession']
        if (binding.get('subject_kind'),binding.get('sequence_binding')) in _NATIVE_FORM_SEQUENCE_BINDINGS:
            raise GOAFormOutsideCanonicalScope('original GOA form packet is outside the canonical-only head')
        if (binding.get('subject_kind')!='canonical'
                or binding.get('sequence_binding')!='exact_current_canonical_source'
                or binding.get('canonical_labels_inherited')is not False
                or binding.get('subject')!='UniProtKB:'+parent or p['subject']!='UniProtKB:'+parent
                or parent not in self._canonical):
            raise ValueError('canonical-only GOA objective; unknown/form source retained outside this task')
        ref=self._canonical[parent];sequence=packet.inputs()['protein_sequence']
        if len(sequence)!=ref['canonical_sequence_length'] or _sha(sequence.encode())!=ref['canonical_sequence_sha256']:
            raise ValueError('GOA sequence differs from complete current canonical reference')
        path=(self.root/ref['source_full_parquet_path']).resolve()
        if not path.is_relative_to(self.root):raise ValueError('canonical source outside project')
        identity=full9(path);key=str(path)
        if key in self._cache:
            expected,table=self._cache.pop(key)
            if expected!=identity:raise ValueError('cached immutable canonical source identity changed')
        else:
            raw=path.read_bytes()
            if _sha(raw)!=ref['source_full_parquet_sha256'] or full9(path)!=identity:
                raise ValueError('canonical native source integrity mismatch')
            table=pq.read_table(io.BytesIO(raw))
        self._cache[key]=(identity,table)
        while len(self._cache)>self.max_cached_kb_parts:self._cache.popitem(last=False)
        ordinal=ref['source_full_parquet_row_ordinal']
        if type(ordinal)is not int or not 0<=ordinal<table.num_rows:raise ValueError('canonical native row outside source')
        row=table.slice(ordinal,1).to_pylist()[0];entry=json.loads(row['full_source_json'])
        if (row['primary_accession']!=parent or entry['primaryAccession']!=parent
                or row['source_structured_sha256']!=ref['source_structured_entry_sha256']
                or _sha(_canonical(entry))!=ref['source_structured_entry_sha256']
                or entry['sequence']['value']!=sequence or entry['sequence']['length']!=len(sequence)
                or full9(path)!=identity):
            raise ValueError('complete original structured canonical source and packet differ')
        p.update(sequence_source_verified=True,historical_GOA_sequence_equivalence_verified=False,
            canonical_sequence_source=dict(native_reference_index=self._binding['canonical_entry_index'],
                source_record=ref))
        return VerifiedGOAPacket(packet.condition_bytes,packet.target_bytes,_canonical(p))

    def bind_native_rows(self,source,*,split):
        import pyarrow.parquet as pq
        self.head_binding()
        if (type(source)is not SelectedNativeSourceBatch or split!='train'
                or len(source.source_row_ordinals)>self.max_batch_size
                or source.source_shard.get('path')not in self._descriptors):
            raise ValueError('explicit bounded canonical native GOA source batch/train role required')
        self.native_view.require_native_rows(source,split=split)
        descriptor=self._descriptors[source.source_shard['path']]
        raw=read_exact(descriptor['original_group_sidecar'],root=self.root)
        groups=pq.read_table(io.BytesIO(raw)).to_pylist()
        if len(groups)!=source.source_shard['rows']:
            raise ValueError('complete original aligned source groups required')
        packets=[]
        for ordinal,row in zip(source.source_row_ordinals,source.native_rows.to_pylist()):
            if row['source_family']!='UniProt-GOA-assertion-packets':raise ValueError('non-GOA native row')
            packet=verify_goa_packet(row,root=self.root);p=packet.provenance();group=groups[ordinal]
            if group['task_id']!=row['task_id']:raise ValueError('exact GOA ordinal group-sidecar identity required')
            p.update(source_groups=json.loads(row['group_ids_json']),groups=json.loads(group['effective_groups_json']),
                authoritative_groups=dict(master_sha256=self.native_view.manifest['source_parent_master']['sha256'],
                    source_shard=source.source_shard,group_sidecar=descriptor['original_group_sidecar'],task_id=row['task_id'],
                    scope='audit-only-source-guards-not-exhaustive-homology'))
            packet=self._canonical_sequence(VerifiedGOAPacket(packet.condition_bytes,packet.target_bytes,_canonical(p)))
            packets.append(bind_goa_applicability(packet,root=self.root,registry=self.registry,
                ontology_manifest=self.ontology_manifest,ontology_sha256=self.ontology_sha256))
        return StreamedGOABatch(source,tuple(packets),self)

    def require_batch(self,batch,*,binding):
        if (type(batch)is not StreamedGOABatch or batch.pool is not self
                or _canonical(binding)!=_canonical(self.head_binding()) or type(batch.packets)is not tuple):
            raise ValueError('exact streamed GOA source batch/pool binding required')
        expected=self.bind_native_rows(batch.source,split='train')
        if batch.packets!=expected.packets:
            raise ValueError('GOA conditions/source/assertions/labels/masks differ from original selected rows')
        return batch.packets
