"""Source-listed EC/Rfam sets and exact-position native residue lists with EOS.

These are source-annotation reproduction objectives, never biological negatives,
site discovery, experimental truth or a new independent source observation. Raw
source completeness, current full-union membership, masks/splits and admission
belong to the data/campaign owner. This module neither reads admission booleans
nor creates an admission certificate. Its reference objectives implement the
loss on explicitly supplied native views after that external authorization.
"""
import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
import torch
from torch import nn
from mora3.data.uniprot_annotations import AnnotationPredictionView
from mora3.data.uniprot_residue import residue_targets
from .uniprot_ec import ec_assertions
from .uniprot_go import annotation_graph_task
from .annotation_multihead import SharedAnnotationModel
from .physical_trainer import SharedPhysicalModel
from .rfam_family import RfamFamilyModel
from .residue_objective import SharedResidueModel
from .source_annotation_list import SourceAnnotationListHead,source_annotation_list_parameter_roles
from .mixed_payload_projection import exact_wire


@dataclass(frozen=True,slots=True)
class SourcePositionList:
    record_index:int
    position_0based:int
    ordered_label_ids:tuple


class SourceAnnotationObjectives(nn.Module):
    def __init__(self,annotation,rfam,residue,*,residue_max_target_length):
        super().__init__()
        if (type(annotation)not in (SharedAnnotationModel,SharedPhysicalModel) or type(rfam)is not RfamFamilyModel
                or type(residue)is not SharedResidueModel or rfam.shared is not annotation or residue.shared is not annotation
                or type(residue_max_target_length)is not int or residue_max_target_length<1):
            raise ValueError('exact shared native annotation/Rfam/residue owners and complete source-list budget required')
        annotation.validate_parameterization_configuration();rfam._validate_recurrent_parameterization();residue.get_extra_state()
        spec=annotation.get_extra_state()['mup'];w=annotation.payload.embedding.embedding_dim
        self.annotation=annotation;self.rfam=rfam;self.residue=residue
        self._owners=(annotation,rfam,residue)
        options=dict(base_width=spec['base_width'],output_mult=spec['output_mult'])
        self.ec_list=SourceAnnotationListHead(w,annotation.ec.axis_ids,
            source_namespace='UniProt-unscoped-source-listed-EC-canonical-set-v1',max_target_length=len(annotation.ec.axis_ids)+1,**options)
        self.rfam_list=SourceAnnotationListHead(w,rfam.family_axis,
            source_namespace='Rfam15.1-source-listed-family-canonical-set-v1',max_target_length=len(rfam.family_axis)+1,**options)
        self.residue_list=SourceAnnotationListHead(w,residue.head.feature_axis,
            source_namespace='UniProt-native-source-feature-order-at-exact-canonical-position-v1',max_target_length=residue_max_target_length,**options)
        p=next(annotation.parameters())
        for head in (self.ec_list,self.rfam_list,self.residue_list):head.to(device=p.device,dtype=p.dtype)
        self._local_configuration=self._local_state()

    def _local_state(self):
        return dict(schema='Mora3.SourceAnnotationObjectives/v1',
            ec_semantics='canonical-sorted-unscoped-source-listed-set-plus-EOS;partial-EC-independent-no-hierarchy',
            rfam_semantics='canonical-sorted-source-listed-family-set-plus-EOS;unknown-biology-not-negatives',
            residue_semantics='native-feature-order-including-repetition-at-exact-source-listed-canonical-positions-plus-EOS',
            unlisted_residues='unlabeled-no-loss',fuzzy_or_otherform_residues='retained-source-audit-not-spatial-loss',
            empty_list='observed-empty-source-list-EOS-only;not-absence-in-biology',
            condition='complete-native-sequence-only;no-source-IDs-target-positions-labels-or-audit-predictors',
            target_history='separate-causal-BOS-previous-observed-source-labels',
            source_readout_prior='existing-EC-Rfam-residue-readout-label-logits-added-to-categorical-next-source-symbol;EOS-prior-zero;not-BCE-or-biological-negatives',
            source_admission='external-current-union-reader-and-campaign-owned;none-issued-here',
            ec_list=self.ec_list.get_extra_state(),rfam_list=self.rfam_list.get_extra_state(),
            residue_list=self.residue_list.get_extra_state())

    def get_extra_state(self):
        if (any(a is not b for a,b in zip((self.annotation,self.rfam,self.residue),self._owners))
                or self.rfam.shared is not self.annotation or self.residue.shared is not self.annotation
                or set(self._modules)!={'annotation','rfam','residue','ec_list','rfam_list','residue_list'}
                or self.ec_list.axis!=self.annotation.ec.axis_ids or self.rfam_list.axis!=self.rfam.family_axis
                or self.residue_list.axis!=self.residue.head.feature_axis
                or not exact_wire(self._local_state(),self._local_configuration)):
            raise ValueError('source annotation list shared owner/axis/objective drift')
        return dict(**copy.deepcopy(self._local_configuration),annotation=self.annotation.get_extra_state(),
            rfam=self.rfam.get_extra_state(),residue=self.residue.get_extra_state())

    def set_extra_state(self,state):
        if not exact_wire(state,self.get_extra_state()):raise ValueError('source annotation list checkpoint mismatch')

    def _load_from_state_dict(self,state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs):
        self.get_extra_state()
        for name,value in self.state_dict().items():
            if not isinstance(value,torch.Tensor) and not exact_wire(state_dict.get(prefix+name),value):
                raise ValueError('source annotation list child/source checkpoint mismatch')
        super()._load_from_state_dict(state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs)

    @staticmethod
    def _protein_views(views):
        views=tuple(views)
        if not views or any(type(v)is not AnnotationPredictionView for v in views):
            raise ValueError('explicit full-native annotation prediction views required')
        for v in views:
            target=v.targets()
            if type(target)is not dict or set(target)!={'annotation_packet','provider_computed_sequence_properties'} or type(target['annotation_packet'])is not dict:
                raise ValueError('full-native annotation packet required; missing is not observed-empty')
        return views

    def ec_observed_lists(self,views):
        views=self._protein_views(views);index={x:i for i,x in enumerate(self.ec_list.axis)};result=[]
        for v in views:
            terms=ec_assertions(v)
            if not set(terms)<=set(index):raise ValueError('complete source EC list outside frozen axis; never drop to empty EOS')
            result.append(tuple(index[t] for t in terms))
        return tuple(result)

    def ec_context(self,views):
        views=self._protein_views(views);graphs=[]
        for v in views:
            record,task=annotation_graph_task(v)
            graphs.append(self.annotation.payload.forward_task(record,task=task,identifiers=self.annotation.sequence_identifiers()))
        return self.annotation.backbone(graphs)[0][:,0]

    def ec_forward(self,views,prefixes):
        self.get_extra_state();context=self.ec_context(views)
        labels=self.annotation.ec(self.annotation._readout_features(context))
        prior=torch.cat((labels,labels.new_zeros(len(labels),1)),-1)
        return self.ec_list(context,prefixes,prior_logits=prior)

    def ec_reference_objective(self,views):
        views=self._protein_views(views);rows=self.ec_observed_lists(views)
        return self.ec_list.objective(self.ec_forward(views,self.ec_list.teacher_prefixes(rows)),rows)

    def rfam_observed_lists(self,views):
        views=self.rfam._views(views,'prediction');index={x:i for i,x in enumerate(self.rfam_list.axis)}
        # Native projection already verifies sorted unique lists and all axis IDs.
        return tuple(tuple(index[x] for x in v.targets()['source_Rfam_family_ids']) for v in views)

    def rfam_context(self,views):
        return self.rfam._latent(tuple(v.inputs() for v in views),'prediction')

    def rfam_forward(self,views,prefixes):
        self.get_extra_state();context=self.rfam_context(views)
        labels=self.rfam.family_output(self.annotation._readout_features(context))
        prior=torch.cat((labels,labels.new_zeros(len(labels),1)),-1)
        return self.rfam_list(context,prefixes,prior_logits=prior)

    def rfam_reference_objective(self,views):
        views=tuple(views);rows=self.rfam_observed_lists(views)
        return self.rfam_list.objective(self.rfam_forward(views,self.rfam_list.teacher_prefixes(rows)),rows)

    def residue_observed_lists(self,views):
        views=self._protein_views(views);index={x:i for i,x in enumerate(self.residue_list.axis)};result=[]
        for record,v in enumerate(views):
            positions={}
            for row in residue_targets(v):
                if row.feature_type not in index:raise ValueError('complete source feature type outside frozen axis')
                if row.coordinate_status!='exact_canonical':continue
                coords=range(*row.interval0) if row.interval0 is not None else tuple(dict.fromkeys(row.endpoint_indices0))
                for i in coords:positions.setdefault(i,[]).append(index[row.feature_type])
            for i,labels in sorted(positions.items()):result.append(SourcePositionList(record,i,tuple(labels)))
        if not result:raise ValueError('no exact source-listed residue positions; no invented empty-site/negative loss')
        self.residue_list._observed_lists(tuple(r.ordered_label_ids for r in result))
        return tuple(result)

    def residue_forward(self,views,positions,prefixes):
        self.get_extra_state();views=self._protein_views(views);positions=tuple(positions)
        if (not positions or len(positions)!=len(prefixes) or any(type(r)is not SourcePositionList
                or type(r.record_index)is not int or not 0<=r.record_index<len(views)
                or type(r.position_0based)is not int or not 0<=r.position_0based<len(views[r.record_index].sequence) for r in positions)):
            raise ValueError('exact sparse output positions required outside encoder conditions')
        encoded=self.residue.encode_sequences(tuple(v.sequence for v in views))
        context=torch.stack([encoded[r.record_index][r.position_0based] for r in positions])
        head=self.residue.head
        labels=head.output(context*(head.output_mult/(head.width/head.base_width)))
        prior=torch.cat((labels,labels.new_zeros(len(labels),1)),-1)
        return self.residue_list(context,prefixes,prior_logits=prior)

    def residue_reference_objective(self,views):
        views=self._protein_views(views);positions=self.residue_observed_lists(views);rows=tuple(r.ordered_label_ids for r in positions)
        return self.residue_list.objective(self.residue_forward(views,positions,self.residue_list.teacher_prefixes(rows)),rows)

    def model_binding(self):
        """Width-independent exact semantics/code identity, never an admission."""
        from . import source_annotation_list,uniprot_ec
        import mora3.data.uniprot_residue as residue_reader
        import mora3.data.rfam_family as rfam_reader
        self.get_extra_state()
        return dict(schema='Mora3.SourceAnnotationObjectivesBinding/v1',
            ec_axis_sha256=self.ec_list.axis_sha256,rfam_axis_sha256=self.rfam_list.axis_sha256,
            residue_axis_sha256=self.residue_list.axis_sha256,
            residue_max_target_length=self.residue_list.max_target_length,
            protein_payload_vocabulary_sha256=self.annotation.get_extra_state()['mup']['vocabulary_sha256'],
            payload_max_tokens=self.annotation.payload.max_tokens,
            source_admission_issued=False,conditions='complete-native-protein-or-nucleotide-sequence-only',
            label_semantics='observed-source-listed-sets-or-position-lists-with-observed-EOS;no-biological-negatives',
            code_sha256={name:hashlib.sha256(Path(path).read_bytes()).hexdigest() for name,path in
                [('model',__file__),('list',source_annotation_list.__file__),('ec',uniprot_ec.__file__),
                 ('residue_reader',residue_reader.__file__),('rfam_reader',rfam_reader.__file__)]})


def source_annotation_objectives_parameter_roles(model,*,identifier_axis):
    from .annotation_mup import annotation_mup_parameter_roles
    from .sequence_mup import sequence_mup_parameter_roles
    from .residue_mup import residue_mup_parameter_roles
    from .parameterization import parameter_ledger
    from .mu_optimizer import _storage_signature
    if type(model)is not SourceAnnotationObjectives:raise ValueError('exact shared source-list objectives owner required')
    model.get_extra_state();roles={}
    for name,builder in [('annotation',annotation_mup_parameter_roles),('rfam',sequence_mup_parameter_roles),('residue',residue_mup_parameter_roles)]:
        roles.update({name+'.'+n:r for n,r in builder(getattr(model,name),identifier_axis=identifier_axis).items()})
    for name in ('ec_list','rfam_list','residue_list'):
        roles.update({name+'.'+n:r for n,r in source_annotation_list_parameter_roles(getattr(model,name)).items()})
    _storage_signature(model);parameter_ledger(model,roles)
    return roles


def source_annotation_objectives_width_family(base,delta,target,*,identifier_axis):
    from .parameterization import parameter_ledger,width_family
    from .full_owner_mup import _without_member_dimensions
    models=(base,delta,target)
    ledgers=[parameter_ledger(m,source_annotation_objectives_parameter_roles(m,identifier_axis=identifier_axis)) for m in models]
    expected=_without_member_dimensions(base.get_extra_state())
    for m in models:
        if (not exact_wire(m.model_binding(),base.model_binding())
                or not exact_wire(_without_member_dimensions(m.get_extra_state()),expected)):
            raise ValueError('source annotation finite axis/scope/budget/base/operator configuration drift')
    if base.annotation.payload.embedding.embedding_dim!=base.annotation.get_extra_state()['mup']['base_width']:
        raise ValueError('actual source annotation base width must match declaration')
    return width_family(*ledgers)
