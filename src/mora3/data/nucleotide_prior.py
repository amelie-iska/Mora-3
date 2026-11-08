"""Lossless source-native nucleotide priors; no translation or paired assay inference."""
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
from .uniprot_annotations import _verified_bytes, _canonical as canonical

ALPHABET='ACGTURYSWKMBDHVNXacgturyswkmbdhvnx'
TASK='source_native_nucleotide_sequence_prior'
POLICY='unpairedreferenceprior_trainonly;all_sourceoccurrencesandparentgroupsretained;homology_and_blindheldoutunresolved'
EVIDENCE='released_reference_sequence_prior;notindependentexperimentalmeasurement'
SOURCES={
    'DNA-native-reference-prior':('source_DNA_sequence','CC-BY-4.0','4e6860536e67a2799c206bb66b0ccce4931e78f1'),
    'Rfam15.1-native-reference-prior':('RNA_family_reference_native_symbols','CC0-1.0','Rfam15.1;January2026;originalCC0FASTA'),
}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class NucleotidePriorView:
    sequence: str
    condition_bytes: bytes
    provenance_bytes: bytes

    def __post_init__(self):
        if (not isinstance(self.sequence,str) or not self.sequence or set(self.sequence)-set(ALPHABET)
                or type(self.condition_bytes) is not bytes or type(self.provenance_bytes) is not bytes):
            raise ValueError('immutable native sequence and serialized context required')
        p=self.provenance()
        if p['sequence_sha256']!=digest(self.sequence.encode()) or p['condition_sha256']!=digest(self.condition_bytes):
            raise ValueError('sequence/condition admission binding mismatch')

    def inputs(self):
        return json.loads(self.condition_bytes)

    def input_bytes(self):
        return self.condition_bytes

    def provenance(self):
        return json.loads(self.provenance_bytes)

    def canonical_bytes(self):
        return canonical(dict(schema='mora3.NucleotidePriorView/v1',sequence=self.sequence,
            conditions=self.inputs(),provenance=self.provenance()))

    def target_tensors(self, *, device='cpu'):
        """Ordered symbol IDs, source-observed mask, base-resolved mask; no padding/crop."""
        import torch
        ids=torch.tensor([ALPHABET.index(c) for c in self.sequence],dtype=torch.long,device=device)
        return ids,torch.ones_like(ids,dtype=torch.bool),torch.tensor(
            [c.upper() in 'ACGTU' for c in self.sequence],dtype=torch.bool,device=device)

    @staticmethod
    def decode(ids):
        import torch
        if ids.ndim!=1 or ids.dtype!=torch.long or bool(((ids<0)|(ids>=len(ALPHABET))).any()):
            raise ValueError('one-dimensional native symbol IDs required')
        return ''.join(ALPHABET[i] for i in ids.tolist())


def nucleotide_prior_view(row, *, effective_groups, corpus_binding):
    raw=row['native_payload']
    if digest(raw)!=row['native_payload_sha256']:raise ValueError('native payload integrity mismatch')
    d=json.loads(raw);receipt=json.loads(row['admission_receipt_json'])
    source=SOURCES.get(row['source_family'])
    if (source is None or row['source_revision']!=source[2] or row['admitted_for_task'] is not True
            or row['split']!='train' or row['split_policy']!=POLICY or row['evidence_class']!=EVIDENCE
            or row['native_schema']!='mora3.CorpusTask/v1' or row['task_name']!=TASK
            or d.get('schema')!='mora3.CorpusTask/v1' or d.get('task')!=TASK):
        raise ValueError('source/train-only evidence admission gate failed')
    c=d['conditions'];seq=d['targets'].get('source_native_sequence')
    if (set(c)!={'prior_domain','source_organism'} or c['prior_domain']!=source[0]
            or (c['source_organism'] is not None and (not isinstance(c['source_organism'],str) or not c['source_organism']))
            or set(d['targets'])!={'source_native_sequence'} or not isinstance(seq,str)
            or not seq or set(seq)-set(ALPHABET)):
        raise ValueError('native symbols or condition allowlist mismatch')
    sha=digest(seq.encode());ref=d['audit']['source_ref']
    if (receipt.get('source_ref')!=ref or ref['native_payload_sha256']!=row['source_sha256']
            or type(ref['source_row_index']) is not int or ref['source_row_index']<0
            or receipt.get('native_sequence_sha256')!=sha or receipt.get('full_axis') is not True
            or receipt.get('native_case_TU_preserved') is not True or receipt.get('source_license')!=source[1]
            or receipt.get('new_uniprot_sequence_admission') is not False
            or row['full_axis_values']!=len(seq)
            or row['task_id']!=digest(canonical([c['prior_domain'],c['source_organism'],sha]))):
        raise ValueError('source identity/rights/full-axis mismatch')
    context=json.loads(row['condition_context_json'])
    if (context.get('source_organism')!=c['source_organism'] or context.get('target_length_audit_only') is not True
            or any(context.get(k) is not False for k in ('raw_image','mask','cell_perturbation_response'))
            or json.loads(row['missing_context_json'])!=d['missing']
            or (c['source_organism'] is None and 'source_organism' not in d['missing'])):
        raise ValueError('missingness/source applicability mismatch')
    groups=tuple(effective_groups)
    if (not groups or any(not isinstance(g,str) or not g for g in groups) or len(set(groups))!=len(groups)
            or 'nucleotide_native:'+sha not in groups):raise ValueError('effective identity groups required')
    if (set(corpus_binding)!={'manifest_sha256','shard_sha256','groups_sha256'}
            or any(not isinstance(v,str) or len(v)!=64 or set(v)-set('0123456789abcdef') for v in corpus_binding.values())):
        raise ValueError('pinned corpus bindings required')
    inputs=canonical(c)
    p=dict(schema='mora3.NucleotidePriorAdmission/v1',sequence_sha256=sha,condition_sha256=digest(inputs),
        envelope={k:v for k,v in row.items() if k!='native_payload'},native_payload=json.loads(raw),
        effective_groups=groups,corpus_binding=corpus_binding,heldout_eligible=False,
        paired_cellular_observation=False,source_evidence=EVIDENCE)
    return NucleotidePriorView(seq,inputs,canonical(p))


def read_nucleotide_priors(manifest, *, root, expected_manifest_sha256, source_family, limit):
    """Read bounded train-only records; verify complete consumed shards and source rows."""
    import pyarrow.parquet as pq
    if source_family not in SOURCES or type(limit) is not int or not 1<=limit<=256:
        raise ValueError('known source and explicit bounded limit 1..256 required')
    root=Path(root).resolve();path=(root/manifest).resolve()
    if not path.is_relative_to(root):raise ValueError('manifest outside project')
    raw=path.read_bytes()
    if digest(raw)!=expected_manifest_sha256:raise ValueError('manifest hash mismatch')
    m=json.loads(raw);out=[];seen=set();sources={}
    for shard in m['shards']:
        if shard.get('source_family') not in (None,source_family) or shard.get('split')!='train':continue
        side=m['effective_group_lookup'][shard['path']]
        rows=pq.read_table(io.BytesIO(_verified_bytes(root,shard))).to_pylist()
        groups=pq.read_table(io.BytesIO(_verified_bytes(root,side))).to_pylist()
        if len(rows)!=shard['rows'] or len(groups)!=side['rows'] or len(rows)!=len(groups):
            raise ValueError('shard/group row counts mismatch')
        for row,group in zip(rows,groups):
            if row['task_id']!=group['task_id']:raise ValueError('group task alignment mismatch')
            if row['source_family']!=source_family:continue
            if row['task_id'] in seen:raise ValueError('duplicate task')
            seen.add(row['task_id'])
            v=nucleotide_prior_view(row,effective_groups=json.loads(group['effective_groups_json']),
                corpus_binding=dict(manifest_sha256=expected_manifest_sha256,shard_sha256=shard['sha256'],groups_sha256=side['sha256']))
            ref=v.provenance()['native_payload']['audit']['source_ref'];key=(ref['native_payload'],ref['native_payload_sha256'])
            if key not in sources:
                p=(root/key[0]).resolve()
                if not p.is_relative_to(root) or digest(p.read_bytes())!=key[1]:raise ValueError('original source integrity mismatch')
                sources[key]=pq.ParquetFile(p)
            pf=sources[key];idx=ref['source_row_index']
            for rg in range(pf.num_row_groups):
                n=pf.metadata.row_group(rg).num_rows
                if idx<n:
                    original=pf.read_row_group(rg,columns=['sequence','organism'] if source_family.startswith('DNA') else ['native_sequence']).slice(idx,1).to_pylist()[0]
                    break
                idx-=n
            else:raise ValueError('source row outside source')
            if original.get('sequence',original.get('native_sequence'))!=v.sequence:
                raise ValueError('original source sequence mismatch')
            if source_family.startswith('DNA') and (original['organism'] or None)!=v.inputs()['source_organism']:
                raise ValueError('original source organism mismatch')
            out.append(v)
            if len(out)==limit:return tuple(out)
    return tuple(out)
