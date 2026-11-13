"""Inactive sparse raw-count head on the existing shared TokenGT encoder."""
from fractions import Fraction
from hashlib import sha256
import math
import torch
from torch import nn
from torch.nn import functional as F

from mora3.model.annotation_multihead import SharedAnnotationModel
from mora3.model.annotation_mup import annotation_mup_parameter_roles
from mora3.model.mixed_payload_projection import exact_wire
from mora3.model.parameterization import Axis,ParameterRole,parameter_ledger,width_family
from mora3.model.physical_trainer import SharedPhysicalModel
from mora3.data.sccello_raw_cell_packets import RawFitAxis,RawMaskedPacket,canonical,require


class SharedRawCountCellModel(nn.Module):
    def __init__(self,shared,axis,*,max_batch_size=128,max_positions=8_000_000):
        super().__init__()
        require(type(shared) in (SharedAnnotationModel,SharedPhysicalModel) and type(axis) is RawFitAxis
                and all(type(n) is int and n>0 for n in (max_batch_size,max_positions)),
                "exact shared owner, fitted raw axis and positive budgets required")
        shared.validate_parameterization_configuration()
        cfg=shared.get_extra_state()
        self.shared=shared
        self.axis=axis
        self._anchors=(shared,axis)
        self.genes=len(axis.gene_keys)
        self.max_batch_size=max_batch_size
        self.max_positions=max_positions
        require(self.genes>0,"finite fitted gene axis required")
        width=shared.payload.embedding.embedding_dim
        base=cfg["mup"]["base_width"]
        multiplier=cfg["mup"]["output_mult"]
        self._config=dict(schema="Mora3.InactiveSharedRawCountCell/v1",width=width,base_width=base,
            output_mult=multiplier,fit_sha256=axis.fit_sha256,fit_evidence_sha256=axis.evidence_sha256,
            finite_gene_axis_sha256=sha256(canonical(axis.gene_keys)).hexdigest(),genes=self.genes,
            max_batch_size=max_batch_size,max_positions=max_positions,source_cls_id=2,
            coordinate="exact-rational-raw-counts-scaled-by-visible-stored-library",
            roles_assigned=False,training_authorized=False)
        reference=next(shared.parameters())
        options=dict(device=reference.device,dtype=reference.dtype)
        self.value_input=nn.Linear(self.genes,width,**options)
        self.availability_input=nn.Linear(self.genes,width,bias=False,**options)
        self.source_cls=nn.Embedding(1,width,**options)
        self.value_prediction=nn.Linear(width,self.genes,**options)
        with torch.no_grad():
            factor=math.sqrt(width/base)
            self.value_prediction.weight.mul_(factor)
            self.value_prediction.bias.mul_(factor)

    def get_extra_state(self):
        require(self.shared is self._anchors[0] and self.axis is self._anchors[1]
                and self.axis.fit_sha256==self._config["fit_sha256"] and self.axis.evidence_sha256==self._config["fit_evidence_sha256"]
                and self.genes==self._config["genes"] and self.max_batch_size==self._config["max_batch_size"]
                and self.max_positions==self._config["max_positions"]
                and set(self._modules)=={"shared","value_input","availability_input","source_cls","value_prediction"}
                and not self._parameters and not self._buffers,"raw source or finite-axis ownership drift")
        width=self._config["width"]
        specs=(("value_input",self.genes,width,True),("availability_input",self.genes,width,False),
               ("value_prediction",width,self.genes,True))
        for name,inputs,outputs,bias in specs:
            operator=getattr(self,name)
            require(type(operator) is nn.Linear and not operator._buffers and not operator._modules
                    and (operator.in_features,operator.out_features)==(inputs,outputs)
                    and tuple(operator.weight.shape)==(outputs,inputs) and (operator.bias is not None)==bias
                    and (not bias or tuple(operator.bias.shape)==(outputs,)),"raw finite-axis parameter role changed")
        require(type(self.source_cls) is nn.Embedding and tuple(self.source_cls.weight.shape)==(1,width),
                "separate raw source CLS role changed")
        mup=self.shared.get_extra_state()["mup"]
        require(self.shared.payload.embedding.embedding_dim==width and mup["base_width"]==self._config["base_width"]
                and mup["output_mult"]==self._config["output_mult"],"shared raw-cell width policy changed")
        return dict(self._config)

    def set_extra_state(self,state):
        require(exact_wire(state,self.get_extra_state()),"raw-cell source, axes or width checkpoint mismatch")

    def _load_from_state_dict(self,state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs):
        self.set_extra_state(state_dict.get(prefix+"_extra_state"))
        super()._load_from_state_dict(state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs)

    def _batch(self,packets):
        self.get_extra_state()
        require(type(packets) is tuple and 0<len(packets)<=self.max_batch_size,
                "nonempty raw packet batch within fixed budget required")
        total=0
        for packet in packets:
            require(type(packet) is RawMaskedPacket and packet.fitted_axis_evidence_sha256==self.axis.evidence_sha256
                    and type(packet.visible_library_count) is int and packet.visible_library_count>0
                    and type(packet.visible) is tuple and packet.visible
                    and type(packet.targets) is tuple and packet.targets,"exact fit-bound raw packet required")
            for values in (packet.visible,packet.targets):
                require(all(type(pair) is tuple and len(pair)==2 and type(pair[0]) is int
                            and 3<=pair[0]<self.genes+3 for pair in values),"exact fitted gene tokens required")
            seen={token for token,_ in packet.visible+packet.targets}
            require(len(seen)==len(packet.visible)+len(packet.targets),
                    "visible and target coordinates must be unique and disjoint")
            require(all(type(value) is Fraction and value>=0 for _,value in packet.visible)
                    and sum(value for _,value in packet.visible)==10000,"invalid visible-only native normalization")
            total+=len(seen)
        require(total<=self.max_positions,"raw batch exceeds exact position budget")

    def forward(self,packets):
        self._batch(packets)
        p=self.value_input.weight
        graphs=[]
        for packet in packets:
            indices=torch.tensor([token-3 for token,_ in packet.visible],device=p.device,dtype=torch.long)
            values=p.new_tensor([float(value) for _,value in packet.visible])
            features=F.linear(values,self.value_input.weight[:,indices],self.value_input.bias)
            features=features+self.availability_input.weight[:,indices].sum(-1)+self.source_cls.weight[0]
            graphs.append(dict(features=features[None],endpoints=torch.tensor([[0,0]],device=p.device,dtype=torch.long),
                               roles=torch.tensor([0],device=p.device,dtype=torch.long),identifiers=self.shared.sequence_identifiers()))
        context=self.shared.backbone(graphs)[0][:,0]
        return self.value_prediction(context*(self._config["output_mult"]/(self._config["width"]/self._config["base_width"])))

    def objective(self,packets):
        predictions=self(packets)
        losses=[]
        for prediction,packet in zip(predictions,packets,strict=True):
            require(all(type(value) is Fraction and value>=0 for _,value in packet.targets),"invalid native target ratio")
            ids=torch.tensor([token-3 for token,_ in packet.targets],device=prediction.device,dtype=torch.long)
            values=torch.tensor([float(value) for _,value in packet.targets],device=prediction.device,dtype=torch.float64)
            losses.append((prediction[ids].double()-values).square().mean())
        loss=torch.stack(losses).mean()
        require(torch.isfinite(loss).item(),"nonfinite native raw-count objective")
        return loss


def raw_cell_parameter_roles(model,*,identifier_axis):
    require(type(model) is SharedRawCountCellModel,"exact inactive raw-cell model required")
    meta=model.get_extra_state()
    roles={"shared."+name:value for name,value in annotation_mup_parameter_roles(model.shared,identifier_axis=identifier_axis).items()}
    width=Axis("width","hidden")
    genes=Axis("finite",meta["finite_gene_axis_sha256"])
    cls=Axis("finite",sha256(b"raw-cell-CLS-role-token-2").hexdigest())
    roles.update({"value_input.weight":ParameterRole("input",(width,genes)),
                  "value_input.bias":ParameterRole("vector",(width,)),
                  "availability_input.weight":ParameterRole("input",(width,genes)),
                  "source_cls.weight":ParameterRole("embedding",(cls,width)),
                  "value_prediction.weight":ParameterRole("output",(genes,width)),
                  "value_prediction.bias":ParameterRole("vector",(genes,))})
    parameter_ledger(model,roles)
    return roles


def raw_cell_width_family(base,delta,target,*,identifier_axis):
    models=(base,delta,target)
    meta=[model.get_extra_state() for model in models]
    require(meta[0]["width"]==meta[0]["base_width"] and all(
        exact_wire({k:v for k,v in item.items() if k!="width"},{k:v for k,v in meta[0].items() if k!="width"})
        for item in meta),"same fitted axis, source proof and non-width configuration required")
    return width_family(*(parameter_ledger(model,raw_cell_parameter_roles(model,identifier_axis=identifier_axis))
                          for model in models))
