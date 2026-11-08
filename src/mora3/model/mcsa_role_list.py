"""Sparse source annotation LIST/EOS reconstruction, never catalytic truth.

Canonical sequence alone conditions the shared native encoder. A source position
selects an output feature, never an encoder marker. Prefixes are decoder target
history. Repetitions/order and observed empty lists are retained. No loss exists
at unlisted sites and no biological negative is inferred. Candidate reader flags
cannot authorize training: objective currently exposes an explicit implementation
diagnostic only unless a reviewed, source-bound data admission authenticates the
complete batch and its requested split against this exact task implementation.
"""
import copy
import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F

from mora3.data.mcsa_weak_sites import WeakRoleVocabulary,WeakSiteBatch,SourceListSupervision
from .mixed_payload_projection import exact_wire
from .residue_objective import SharedResidueModel
from .residue_mup import residue_mup_parameter_roles,residue_mup_width_family
from .parameterization import Axis,ParameterRole,parameter_ledger,width_family
from .mu_optimizer import _storage_signature

SOURCE_FAMILY='MCSA-source-role-list-current-reference'
TASK_NAME='source_role_list_reconstruction_on_current_reference'
NATIVE_SCHEMA='Mora3.MCSAWeakReferenceSiteCandidate/v1'


class SourceRoleListDecoder(nn.Module):
    """Explicit dense GRU-gate width adaptation with a fixed source-role axis.

    PyTorch GRUCell uses separate input/hidden packed reset/update/new gates.
    Every gate is a W-by-W affine map, kernel multiplicity is fixed at three.
    Native uniform weights already scale as 1/sqrt(W); fresh biases are restored
    to the declared base distribution. μAdam uses base/W for both matrices.
    This named finite-width adaptation is not a formal recurrent TP theorem.
    """
    def __init__(self,width,vocabulary,*,base_width,output_mult=1.0):
        super().__init__()
        if (type(width) is not int or type(base_width) is not int or base_width<2
                or width<base_width or type(vocabulary) is not WeakRoleVocabulary
                or type(output_mult) not in (int,float) or not math.isfinite(output_mult) or output_mult<=0):
            raise ValueError('exact source vocabulary and positive declared GRU width/readout required')
        self.width,self.base_width,self.output_mult=width,base_width,float(output_mult)
        self.vocabulary=vocabulary
        self.embedding=nn.Embedding(vocabulary.decoder_input_class_count,width)
        self.cell=nn.GRUCell(width,width)
        self.output=nn.Linear(width,vocabulary.prediction_class_count)
        with torch.no_grad():
            ratio=math.sqrt(width/base_width)
            self.cell.bias_ih.mul_(ratio);self.cell.bias_hh.mul_(ratio)
            self.output.weight.mul_(ratio);self.output.bias.mul_(ratio)
        self._configuration=self._current_configuration()

    def _current_configuration(self):
        return dict(schema='mcsa-source-role-list-decoder-v1',width=self.width,
            base_width=self.base_width,output_mult=self.output_mult,
            axis_sha256=self.vocabulary.axis_sha256,roles=self.vocabulary.roles,
            prediction_classes=self.vocabulary.prediction_class_count,
            input_classes=self.vocabulary.decoder_input_class_count,
            eos=self.vocabulary.eos_id,bos=self.vocabulary.bos_id,
            gates='pytorch-GRUCell-reset-update-new-three-equal-slices',
            gate_initialization='native-uniform-bias-times-sqrt-width-over-base',
            gate_adam_lr='base-over-width',embedding_initialization='native-normal-width-invariant',
            readout='native-times-sqrt-width-over-base-input-divided-by-width-over-base')

    def get_extra_state(self):
        modules=dict(self.named_modules(remove_duplicate=False))
        expected={'':SourceRoleListDecoder,'embedding':nn.Embedding,'cell':nn.GRUCell,'output':nn.Linear}
        if (set(modules)!=set(expected) or any(type(modules[n]) is not t for n,t in expected.items())
                or (self.cell.input_size,self.cell.hidden_size,self.cell.bias)!=(self.width,self.width,True)
                or (self.embedding.num_embeddings,self.embedding.embedding_dim,self.embedding.padding_idx,
                    self.embedding.max_norm,self.embedding.norm_type,self.embedding.scale_grad_by_freq,self.embedding.sparse)
                   !=(self.vocabulary.decoder_input_class_count,self.width,None,None,2.0,False,False)
                or (self.output.in_features,self.output.out_features)!=(self.width,self.vocabulary.prediction_class_count)):
            raise ValueError('source role decoder operator configuration drift')
        shapes={'embedding.weight':(self.vocabulary.decoder_input_class_count,self.width),
            'cell.weight_ih':(3*self.width,self.width),'cell.weight_hh':(3*self.width,self.width),
            'cell.bias_ih':(3*self.width,),'cell.bias_hh':(3*self.width,),
            'output.weight':(self.vocabulary.prediction_class_count,self.width),
            'output.bias':(self.vocabulary.prediction_class_count,)}
        params=dict(self.named_parameters(remove_duplicate=False))
        if set(params)!=set(shapes) or any(tuple(params[n].shape)!=shape for n,shape in shapes.items()):
            raise ValueError('source role decoder packed parameter shape drift')
        if not exact_wire(self._configuration,self._current_configuration()):
            raise ValueError('source role-list decoder configuration drift')
        return copy.deepcopy(self._configuration)

    def set_extra_state(self,state):
        if not exact_wire(state,self.get_extra_state()):raise ValueError('source role decoder configuration mismatch')

    def validate_checkpoint(self,state_dict,prefix):
        if not exact_wire(state_dict.get(prefix+'_extra_state'),self.get_extra_state()):
            raise ValueError('source role decoder checkpoint configuration/axis mismatch')

    def _load_from_state_dict(self,state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs):
        self.validate_checkpoint(state_dict,prefix)
        super()._load_from_state_dict(state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs)

    def step(self,input_ids,hidden):
        self.get_extra_state()
        if (input_ids.dtype!=torch.long or input_ids.ndim!=1 or hidden.shape!=(len(input_ids),self.width)
                or input_ids.device!=hidden.device or hidden.device!=self.embedding.weight.device
                or hidden.dtype!=self.embedding.weight.dtype or not len(input_ids)
                or (input_ids<0).any() or (input_ids>=self.vocabulary.decoder_input_class_count).any()):
            raise ValueError('source-list decoder integer prefix IDs and aligned hidden state required')
        state=self.cell(self.embedding(input_ids),hidden)
        return self.output(state*(self.output_mult/(self.width/self.base_width))),state

    @torch.no_grad()
    def generate(self,feature,*,max_target_length):
        if type(max_target_length) is not int or max_target_length<1:
            raise ValueError('finite positive generation budget required')
        if feature.shape!=(self.width,):raise ValueError('one selected residue feature required')
        history=[];state=feature[None]
        token=self.vocabulary.bos_id
        for _ in range(max_target_length):
            logits,state=self.step(torch.tensor([token],device=feature.device),state)
            token=int(logits.argmax(-1).item())
            if token==self.vocabulary.eos_id:
                return dict(role_ids=tuple(history),termination='EOS',symbols=len(history)+1)
            history.append(token)
        return dict(role_ids=tuple(history),termination='budget-exhausted',symbols=len(history))


class SourceRoleListModel(nn.Module):
    """Source-specific, source-list decoder attached to one native residue owner."""
    def __init__(self,residue,vocabulary,*,max_target_length=75):
        super().__init__()
        if (type(residue) is not SharedResidueModel or residue.parameterization_policy!='mup-residue-v1'
                or type(vocabulary) is not WeakRoleVocabulary
                or type(max_target_length) is not int or max_target_length<1):
            raise ValueError('explicit native residue owner and frozen source-list vocabulary required')
        cfg=residue.get_extra_state();spec=cfg['head']['mup']
        self.residue=residue;self._residue_reference=(residue,)
        self.max_target_length=max_target_length
        self.decoder=SourceRoleListDecoder(residue.head.width,vocabulary,
            base_width=spec['base_width'],output_mult=spec['output_mult'])
        self.decoder.to(device=residue.shared.payload.embedding.weight.device,
                        dtype=residue.shared.payload.embedding.weight.dtype)
        self._configuration=dict(schema='mcsa-source-role-list-model-v1',source_family=SOURCE_FAMILY,
            task_name=TASK_NAME,native_schema=NATIVE_SCHEMA,max_target_length=max_target_length,
            condition_inputs=('canonical_sequences',),position_role='sparse-output-selector-only',
            objective='ordered-source-list-cross-entropy-including-observed-EOS',
            unknown_positions='no-target-no-loss',biological_negatives=False,
            admission='external-reviewed-source-role-list-packet-and-complete-batch-binding')

    def get_extra_state(self):
        if (self.residue is not self._residue_reference[0]
                or self.max_target_length!=self._configuration['max_target_length']):
            raise ValueError('source role-list model configuration drift')
        return dict(**copy.deepcopy(self._configuration),residue=self.residue.get_extra_state(),
                    decoder=self.decoder.get_extra_state())

    def set_extra_state(self,state):
        if not exact_wire(state,self.get_extra_state()):raise ValueError('source role-list model configuration mismatch')

    def _load_from_state_dict(self,state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs):
        if not exact_wire(state_dict.get(prefix+'_extra_state'),self.get_extra_state()):
            raise ValueError('source role-list checkpoint configuration mismatch')
        self.decoder.validate_checkpoint(state_dict,prefix+'decoder.')
        self.residue._validate_parameterization_checkpoint(state_dict,prefix+'residue.')
        super()._load_from_state_dict(state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs)

    def validate_batch(self,batch):
        v=self.decoder.vocabulary
        if (type(batch) is not WeakSiteBatch or batch.task_admitted is not False
                or batch.vocabulary_axis_sha256!=v.axis_sha256
                or type(batch.prediction_class_count) is not int or batch.prediction_class_count!=v.prediction_class_count
                or type(batch.decoder_input_class_count) is not int or batch.decoder_input_class_count!=v.decoder_input_class_count
                or type(batch.canonical_sequences) is not tuple or not batch.canonical_sequences
                or any(type(s) is not str or not s or any(c not in 'ACDEFGHIKLMNPQRSTVWYBXZJUO' for c in s)
                       for s in batch.canonical_sequences)
                or type(batch.source_list_supervision) is not tuple or not batch.source_list_supervision):
            raise ValueError('exact candidate source-list batch/axis and false admission state required')
        for row in batch.source_list_supervision:
            if (type(row) is not SourceListSupervision or type(row.record_index) is not int
                    or not 0<=row.record_index<len(batch.canonical_sequences)
                    or type(row.position_0based) is not int
                    or not 0<=row.position_0based<len(batch.canonical_sequences[row.record_index])
                    or type(row.target_ids) is not tuple or not 1<=len(row.target_ids)<=self.max_target_length
                    or any(type(i) is not int or not 0<=i<v.eos_id for i in row.target_ids[:-1])
                    or type(row.target_ids[-1]) is not int or row.target_ids[-1]!=v.eos_id
                    or row.decoder_input_ids!=(v.bos_id,)+row.target_ids[:-1]):
                raise ValueError('exact sparse position, complete role-prefix and terminal EOS required')
        self.get_extra_state()

    def admission_binding(self):
        """Pure task semantics and exact implementation, independent of model width.

        This is an identity, not an approval. The separately reviewed source data
        packet must accept this exact binding and all supplied native records.
        Family construction and launch qualification remain separate contracts.
        """
        from pathlib import Path
        import mora3.data.mcsa_weak_sites as reader
        from . import residue_objective
        self.get_extra_state()
        return dict(schema='Mora3.MCSASourceRoleListModelBinding/v1',
            source_family=SOURCE_FAMILY,task_name=TASK_NAME,native_schema=NATIVE_SCHEMA,
            axis_sha256=self.decoder.vocabulary.axis_sha256,max_target_length=self.max_target_length,
            condition_inputs=('canonical_sequences',),
            encoder_sequence_contract='complete-native-sequence-no-crop-or-truncation',
            payload_max_tokens=self.residue.shared.payload.max_tokens,
            target_history='teacher-forced-ordered-role-prefix-BOS-input-only',
            targets='observed-ordered-source-role-list-including-EOS',
            source_position='output-selector-only-never-encoder-marker',
            unknown_positions='no-target-no-loss',biological_negatives=False,
            payload_vocabulary_sha256=self.residue.shared.get_extra_state()['mup']['vocabulary_sha256'],
            model_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            residue_source_sha256=hashlib.sha256(Path(residue_objective.__file__).read_bytes()).hexdigest(),
            reader_source_sha256=hashlib.sha256(Path(reader.__file__).read_bytes()).hexdigest())

    def _authorize(self,batch,*,purpose,admission,split):
        self.validate_batch(batch)
        if purpose=='implementation-diagnostic' and admission is None and split is None:return
        if (purpose not in ('training','evaluation') or admission is None
                or (purpose=='training' and split!='train')
                or (purpose=='evaluation' and split not in ('validation','test'))):
            raise ValueError('explicit reviewed admission and matching train/evaluation split required')
        try:
            from mora3.data.mcsa_role_list_admission import MCSARoleListAdmission
        except ImportError as exc:
            raise ValueError('reviewed source-list admission implementation unavailable') from exc
        if type(admission) is not MCSARoleListAdmission:
            raise ValueError('exact reviewed source-list admission required')
        admission.require_batch(batch,model_binding=self.admission_binding(),split=split)

    def encode_conditions(self,batch):
        self.validate_batch(batch)
        return self.residue.encode_sequences(batch.canonical_sequences)

    def forward(self,batch,*,purpose='implementation-diagnostic',admission=None,split=None):
        self._authorize(batch,purpose=purpose,admission=admission,split=split)
        features=self.encode_conditions(batch)
        result=[]
        for row in batch.source_list_supervision:
            state=features[row.record_index][row.position_0based][None]
            logits=[]
            for token in row.decoder_input_ids:
                z,state=self.decoder.step(torch.tensor([token],device=state.device),state)
                logits.append(z[0])
            result.append(torch.stack(logits))
        return tuple(result)

    def objective(self,logits,batch,*,purpose,admission=None,split=None):
        self._authorize(batch,purpose=purpose,admission=admission,split=split)
        if type(logits) is not tuple or len(logits)!=len(batch.source_list_supervision):
            raise ValueError('one complete source-list logit matrix per sparse annotation view required')
        values=[];targets=[]
        for z,row in zip(logits,batch.source_list_supervision):
            if (not isinstance(z,torch.Tensor) or z.shape!=(len(row.target_ids),batch.prediction_class_count)
                    or not z.is_floating_point() or z.device!=self.decoder.output.weight.device
                    or z.dtype!=self.decoder.output.weight.dtype or not torch.isfinite(z).all()):
                raise ValueError('finite complete source-list logits on model dtype/device required')
            values.append(z);targets.extend(row.target_ids)
        loss=F.cross_entropy(torch.cat(values),torch.tensor(targets,device=values[0].device))
        return dict(source_role_list=loss,total=loss)


def source_role_list_parameter_roles(model,*,identifier_axis):
    if type(model) is not SourceRoleListModel or set(model._modules)!={'residue','decoder'}:
        raise ValueError('exact source-role-list owner required')
    model.get_extra_state();d=model.decoder
    if (type(d) is not SourceRoleListDecoder or set(d._modules)!={'embedding','cell','output'}
            or type(d.embedding) is not nn.Embedding or type(d.cell) is not nn.GRUCell or type(d.output) is not nn.Linear
            or (d.cell.input_size,d.cell.hidden_size,d.cell.bias)!=(d.width,d.width,True)
            or (d.embedding.num_embeddings,d.embedding.embedding_dim)!=(d.vocabulary.decoder_input_class_count,d.width)
            or (d.output.in_features,d.output.out_features)!=(d.width,d.vocabulary.prediction_class_count)):
        raise ValueError('source GRU gate/embedding/readout operator configuration drift')
    shapes={'embedding.weight':(d.vocabulary.decoder_input_class_count,d.width),
        'cell.weight_ih':(3*d.width,d.width),'cell.weight_hh':(3*d.width,d.width),
        'cell.bias_ih':(3*d.width,),'cell.bias_hh':(3*d.width,),
        'output.weight':(d.vocabulary.prediction_class_count,d.width),
        'output.bias':(d.vocabulary.prediction_class_count,)}
    params=dict(d.named_parameters(remove_duplicate=False))
    if set(params)!=set(shapes) or any(tuple(params[n].shape)!=shape for n,shape in shapes.items()):
        raise ValueError('source role-list packed gate parameter shape/ownership drift')
    roles={'residue.'+n:r for n,r in residue_mup_parameter_roles(model.residue,identifier_axis=identifier_axis).items()}
    h=Axis('width','hidden')
    # Gate packed rows are three equal width slices with fixed named ordering.
    gates=Axis('width','gru-three-gates')
    inputs=Axis('finite',hashlib.sha256(('mcsa-role-prefix:'+d.vocabulary.axis_sha256).encode()).hexdigest())
    outputs=Axis('finite',hashlib.sha256(('mcsa-role-target:'+d.vocabulary.axis_sha256).encode()).hexdigest())
    roles.update({'decoder.embedding.weight':ParameterRole('embedding',(inputs,h)),
        'decoder.cell.weight_ih':ParameterRole('hidden',(gates,h)),
        'decoder.cell.weight_hh':ParameterRole('hidden',(gates,h)),
        'decoder.cell.bias_ih':ParameterRole('vector',(gates,)),
        'decoder.cell.bias_hh':ParameterRole('vector',(gates,)),
        'decoder.output.weight':ParameterRole('output',(outputs,h)),
        'decoder.output.bias':ParameterRole('vector',(outputs,))})
    _storage_signature(model);parameter_ledger(model,roles)
    return roles


def source_role_list_width_family(base,delta,target,*,identifier_axis):
    models=(base,delta,target)
    ledgers=[parameter_ledger(m,source_role_list_parameter_roles(m,identifier_axis=identifier_axis)) for m in models]
    residue_mup_width_family(*(m.residue for m in models),identifier_axis=identifier_axis)
    expected=None
    for m in models:
        cfg=copy.deepcopy(m.get_extra_state());cfg.pop('residue');cfg['decoder'].pop('width')
        if expected is None:expected=cfg
        if not exact_wire(cfg,expected):raise ValueError('source-list axis/base/gate/objective policy drift')
    return width_family(*ledgers)
