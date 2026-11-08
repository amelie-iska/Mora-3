"""Full fixed-source gene-axis masked native-value cell reconstruction.

This is a low dimensional full-axis numeric encoder sharing the actual TokenGT,
not a per-gene TokenGT or gene-ID MLM. Sparse input sums retain every unmasked
source pair without imputing absent genes. Source ranked order/arrays stay
immutable; the sum does not undo provider median-selected truncation. Targets
are caller-held-out retained provider-transformed values, never raw counts,
absent-gene zeros, cell-type gold, or experimental biological truth. Data-owner
current masks/splits and campaign training admission remain external.
"""
from dataclasses import dataclass,asdict
import copy
import hashlib
import json
import math
import re
import torch
from torch import nn
from torch.nn import functional as F
from mora3.data.sccello_native_rows import (SourceBinding,TokenVocabulary,TOKEN_COUNT,SOURCE_CLS_ID,validate_gene_axes)
from mora3.data.sccello_selected_rows import SelectedNativeGeneBatch
from .annotation_multihead import SharedAnnotationModel
from .physical_trainer import SharedPhysicalModel
from .mixed_payload_projection import exact_wire

GENE_COUNT=TOKEN_COUNT-2
COORDINATE='scCello-pretrain-OOD-provider-log1p-sizefactor10000-median-ranked-retained-v1'
_REVISION='767585b6cc2ef5a3e03234164c665e751a432194'
_SHA=re.compile('[0-9a-f]{64}\\Z')


@dataclass(frozen=True,slots=True)
class NativeCellProtocol:
    evidence_sha256:str
    sources:tuple
    coordinate:str=COORDINATE

    def __post_init__(self):
        if (type(self.evidence_sha256)is not str or not _SHA.fullmatch(self.evidence_sha256)
                or self.coordinate!=COORDINATE or type(self.sources)is not tuple
                or tuple(x[0] for x in self.sources)!=tuple(range(7))
                or any(type(x)is not tuple or len(x)!=3 or type(x[0])is not int
                    or type(x[1])is not str or not x[1] or type(x[2])is not str
                    or re.fullmatch('[0-9a-f]{40}',x[2])is None for x in self.sources)):
            raise ValueError('exact pinned source pretrain/OOD semantic protocol required; GEO not equivalent')

    @classmethod
    def from_source_document(cls,raw,*,expected_sha256):
        if type(raw)is not bytes or hashlib.sha256(raw).hexdigest()!=expected_sha256:
            raise ValueError('exact immutable source protocol document bytes required')
        d=json.loads(raw)
        if (d.get('schema')!='Mora3.scCello.SourceProtocolRolesAndExposure.v1'
                or d.get('pinned_scCello_code_revision')!=_REVISION
                or d.get('source_native_transforms',{}).get('native_full_gene_expression_axis_lossless')is not False
                or d.get('dictionary_lineage',{}).get('median_recomputed_from_scCello_claim')is not False
                or d.get('dictionary_lineage',{}).get('heldout_overlap_or_disjointness')is not None):
            raise ValueError('known source preprocessing/reference exposure scope required')
        source=tuple(sorted((x['release_index'],x['repository'],x['revision']) for x in d['original_export_roles'] if x['release_index']<7))
        return cls(expected_sha256,source)

    def require_source(self,binding):
        self.__post_init__()
        if (type(binding)is not SourceBinding or (binding.release_index,binding.repository,binding.revision)not in self.sources):
            raise ValueError('source not in pinned common-coordinate pretrain/OOD recipe; no GEO fallback')

    def data(self):
        return dict(**asdict(self),reference_fit='public-Geneformer-Genecorpus30M-exact-heldout-study-overlap-unknown',
            source_selection='irreversible-provider-median-dependent-ranking-topk-preserved',
            role_authority='data-owner-current-arm-only;stored-HF-split-or-source-role-not-admission',
            neural_features='no-repository-source-release-row-IDs-or-audit-labels')


@dataclass(frozen=True,slots=True)
class CellMaskedBatch:
    source:SelectedNativeGeneBatch
    input_available:tuple
    target_observed:tuple


class SharedNativeCellModel(nn.Module):
    def __init__(self,shared,vocabulary,protocol,*,max_batch_size=128,max_positions=8_000_000,
                 parameterization_policy='mup-native-cell-v1'):
        super().__init__()
        if (type(shared)not in (SharedAnnotationModel,SharedPhysicalModel) or type(vocabulary)is not TokenVocabulary
                or type(protocol)is not NativeCellProtocol or parameterization_policy!='mup-native-cell-v1'
                or any(type(x)is not int or x<=0 for x in (max_batch_size,max_positions))):
            raise ValueError('exact shared native owner/full passive vocabulary/protocol/budget required')
        shared.validate_parameterization_configuration();protocol.__post_init__()
        keys=vocabulary.gene_keys
        if (type(keys)is not tuple or len(keys)!=TOKEN_COUNT or keys[:2]!=('<pad>','<mask>')
                or any(type(x)is not str or not x for x in keys) or len(set(keys))!=TOKEN_COUNT
                or type(vocabulary.source_sha256)is not str or not _SHA.fullmatch(vocabulary.source_sha256)):
            raise ValueError('complete literal 25424-gene source dictionary and reserved0/1 roles required')
        self.shared=shared;self.vocabulary=vocabulary;self.protocol=protocol
        self._owner_reference=(shared,vocabulary,protocol)
        self.max_batch_size=max_batch_size;self.max_positions=max_positions
        self.parameterization_policy=parameterization_policy
        w=shared.payload.embedding.embedding_dim;spec=shared.get_extra_state()['mup']
        self.value_input=nn.Linear(GENE_COUNT,w)
        self.availability_input=nn.Linear(GENE_COUNT,w,bias=False)
        self.source_cls=nn.Embedding(1,w)
        self.value_prediction=nn.Linear(w,GENE_COUNT)
        with torch.no_grad():
            factor=math.sqrt(w/spec['base_width'])
            self.value_prediction.weight.mul_(factor);self.value_prediction.bias.mul_(factor)
        p=next(shared.parameters())
        for leaf in (self.value_input,self.availability_input,self.source_cls,self.value_prediction):
            leaf.to(device=p.device,dtype=p.dtype)
        self._configuration=self._local_state()

    def _local_state(self):
        cfg=self.shared.get_extra_state();spec=cfg['mup']
        return dict(schema='Mora3.SharedNativeCellModel/v1',policy=self.parameterization_policy,width=cfg['width'],
            base_width=spec['base_width'],output_mult=spec['output_mult'],full_gene_count=GENE_COUNT,
            gene_axis=self.vocabulary.gene_keys[2:],source_vocabulary_sha256=self.vocabulary.source_sha256,
            protocol=self.protocol.data(),source_cls=dict(id=SOURCE_CLS_ID,source_value=0.,gene=False,target=False,learned_role='one-finite-role-embedding'),
            input='sparse-full-gene-axis-values-and-input-availability;heldout-pairs-in-neither-sum',
            source_order='all-native-pairs-immutable;sum-does-not-undo-provider-selection-or-recover-omitted-genes',
            encoder='one-cell-entity-through-identical-shared-TokenGT',
            objective='per-cell-mean-over-explicit-heldout-retained-values-then-mean-active-cells;float64-native-target-MSE',
            model_input_precision='finite-native-float64-values-converted-to-model-dtype;source-arrays-unmodified',
            readout='native-fresh-times-sqrt-width-over-base;input-mult-over-width-ratio',
            missing_gene='no-input-no-target-not-biological-zero',source_admission_issued=False,
            max_batch_size=self.max_batch_size,max_positions=self.max_positions)

    def get_extra_state(self):
        if (self._parameters or self._buffers or any(a is not b for a,b in zip((self.shared,self.vocabulary,self.protocol),self._owner_reference))
                or set(self._modules)!={'shared','value_input','availability_input','source_cls','value_prediction'}
                or not exact_wire(self._local_state(),self._configuration)):
            raise ValueError('native cell source/axis/protocol/shared ownership drift')
        w=self.shared.payload.embedding.embedding_dim
        for name,inside,out,bias in (('value_input',GENE_COUNT,w,True),('availability_input',GENE_COUNT,w,False),('value_prediction',w,GENE_COUNT,True)):
            layer=getattr(self,name)
            if (type(layer)is not nn.Linear or layer._modules or layer._buffers or (layer.in_features,layer.out_features)!=(inside,out)
                    or tuple(layer.weight.shape)!=(out,inside) or (layer.bias is not None)!=bias
                    or bias and tuple(layer.bias.shape)!=(out,)):
                raise ValueError('native cell finite full-axis operator drift')
        if (type(self.source_cls)is not nn.Embedding or self.source_cls._buffers or self.source_cls._modules
                or (self.source_cls.num_embeddings,self.source_cls.embedding_dim,self.source_cls.padding_idx)!=(1,w,None)
                or tuple(self.source_cls.weight.shape)!=(1,w)):
            raise ValueError('separate source CLS role embedding drift')
        return dict(**copy.deepcopy(self._configuration),shared=self.shared.get_extra_state())

    def set_extra_state(self,state):
        if not exact_wire(state,self.get_extra_state()):raise ValueError('native cell checkpoint source/axis/policy mismatch')

    def _load_from_state_dict(self,state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs):
        self.get_extra_state()
        for name,value in self.state_dict().items():
            if not isinstance(value,torch.Tensor) and not exact_wire(state_dict.get(prefix+name),value):
                raise ValueError('native cell checkpoint child/source/policy mismatch')
        super()._load_from_state_dict(state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs)

    def validate_batch(self,batch):
        self.get_extra_state()
        if type(batch)is not CellMaskedBatch or type(batch.source)is not SelectedNativeGeneBatch:
            raise ValueError('typed exact selected native source batch and explicit masks required')
        source=batch.source;self.protocol.require_source(source.binding)
        n=source.num_rows
        if not 1<=n<=self.max_batch_size:raise ValueError('complete source batch resource bound exceeded')
        lengths=validate_gene_axes(source.gene_token_ids,source.gene_expression_nums,max_positions=self.max_positions)
        ordinals=source.source_row_ordinals.to_pylist()
        if (any(type(x)is not int or not 0<=x<source.binding.rows for x in ordinals)
                or tuple(sorted(set(ordinals)))!=tuple(ordinals)):
            raise ValueError('exact original sorted unique source row coordinates required')
        if (type(batch.input_available)is not tuple or type(batch.target_observed)is not tuple
                or len(batch.input_available)!=n or len(batch.target_observed)!=n):
            raise ValueError('one complete explicit pair of mask rows per source cell required')
        for length,inputs,targets in zip(lengths,batch.input_available,batch.target_observed):
            if (type(inputs)is not tuple or type(targets)is not tuple or len(inputs)!=length or len(targets)!=length
                    or any(type(x)is not bool for x in (*inputs,*targets)) or inputs[0]is not True or targets[0]is not False
                    or any(i and t for i,t in zip(inputs,targets))):
                raise ValueError('complete disjoint input/target masks; CLS input role only; no target leakage')
        return source.gene_token_ids.to_pylist(),source.gene_expression_nums.to_pylist()

    def _project_row(self,token_ids,values,available):
        p=self.value_input.weight
        selected=[i for i in range(1,len(token_ids)) if available[i]]
        indices=torch.tensor([token_ids[i]-2 for i in selected],device=p.device,dtype=torch.long)
        x=p.new_tensor([values[i] for i in selected])
        if not torch.isfinite(x).all():raise ValueError('source native values not representable in model input dtype; no rescaling')
        # Empty indexed sums yield only learned bias/CLS, not a fabricated all-
        # gene-zero observation. Absent/heldout gene values are never allocated.
        return F.linear(x,self.value_input.weight[:,indices],self.value_input.bias)+self.availability_input.weight[:,indices].sum(-1)+self.source_cls.weight[0]

    def latents(self,batch):
        ids,values=self.validate_batch(batch);p=self.value_input.weight;graphs=[]
        for row,value,mask in zip(ids,values,batch.input_available):
            features=self._project_row(row,value,mask)
            graphs.append(dict(features=features[None],endpoints=torch.tensor([[0,0]],device=p.device,dtype=torch.long),
                roles=torch.tensor([0],device=p.device,dtype=torch.long),identifiers=self.shared.sequence_identifiers()))
        return self.shared.backbone(graphs)[0][:,0]

    def forward(self,batch):
        latent=self.latents(batch);spec=self._configuration
        return self.value_prediction(latent*(spec['output_mult']/(spec['width']/spec['base_width'])))

    def reference_objective(self,predictions,batch):
        ids,values=self.validate_batch(batch);p=self.value_prediction.weight
        if (not isinstance(predictions,torch.Tensor) or predictions.shape!=(batch.source.num_rows,GENE_COUNT)
                or predictions.dtype!=p.dtype or predictions.device!=p.device or not torch.isfinite(predictions).all()):
            raise ValueError('complete finite native gene-axis predictions in model precision required')
        losses=[]
        for row,value,mask,z in zip(ids,values,batch.target_observed,predictions):
            selected=[i for i in range(1,len(row)) if mask[i]]
            if selected:
                index=torch.tensor([row[i]-2 for i in selected],device=p.device,dtype=torch.long)
                target=torch.tensor([value[i] for i in selected],device=p.device,dtype=torch.float64)
                losses.append((z[index].double()-target).square().mean())
        loss=torch.stack(losses).mean() if losses else predictions.reshape(-1)[:0].sum().double()
        if not torch.isfinite(loss):raise ValueError('nonfinite source coordinate MSE; no automatic clipping/rescaling')
        return loss

    def model_binding(self):
        self.get_extra_state()
        return dict(schema='Mora3.NativeCellReconstructionBinding/v1',protocol=self.protocol.data(),
            source_vocabulary_sha256=self.vocabulary.source_sha256,
            gene_axis_sha256=hashlib.sha256(json.dumps(self.vocabulary.gene_keys[2:],separators=(',',':')).encode()).hexdigest(),
            source_cls_id=SOURCE_CLS_ID,target='caller-masked-provider-transformed-retained-native-expression',
            masks='heldout-values-OUT-of-both-input-projections;target-mask-loss-only',
            normalization='per-cell-mean-target-MSE-then-mean-active-cells;native-float64-targets',
            no_absent_gene_targets=True,source_admission_issued=False,
            model_source_sha256=hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest())
