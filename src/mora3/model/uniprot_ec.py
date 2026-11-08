"""Exact source-observed EC prediction; missing is unknown, never negative.

Partial EC identifiers are independent explicit source labels. No descendant or
ancestor propagation, reaction matching, evidence upgrade or form transfer.
"""
import re
import torch
from torch import nn
from mora3.data.uniprot_annotations import AnnotationPredictionView
from mora3.model.uniprot_go import observed_go_loss


# Immutable syntax only; every value still receives a fresh full match.
_EC_SYNTAX = re.compile(r'[1-7]\.(?:\d+|-)\.(?:\d+|-)\.(?:\d+|n\d+|-)')


def _valid(value):
    if not isinstance(value,str) or _EC_SYNTAX.fullmatch(value) is None:
        return False
    parts=value.split('.')
    return all(x=='-' for x in parts[parts.index('-'):]) if '-' in parts else True


def _axis(values):
    axis=tuple(values)
    if not axis or any(not _valid(x) for x in axis) or len(set(axis))!=len(axis):
        raise ValueError('nonempty unique ordered EC axis required')
    return axis


def ec_assertions(view):
    """Unscoped entry assertions only; full evidence remains on the source view.

    Read only entry recommended/alternative names and catalytic activity reaction.
    Exclude molecule/form parent scopes and description contains/includes chains.
    Do not recursively collect EC-looking strings from arbitrary annotation text.
    """
    if not isinstance(view,AnnotationPredictionView):
        raise ValueError('prediction view required')
    packet=view.targets()['annotation_packet'];found=set()
    def add(value):
        if not _valid(value):raise ValueError('invalid source EC identifier')
        found.add(value)
    def unscoped(obj):
        return not any(obj.get(k) for k in ('molecule','isoformId','sequence'))
    desc=packet.get('proteinDescription',{})
    if unscoped(desc):
        for name in [desc.get('recommendedName',{}),*desc.get('alternativeNames',[])]:
            if unscoped(name):
                for item in name.get('ecNumbers',[]):
                    if unscoped(item):add(item['value'])
    for comment in packet.get('comments',[]):
        if comment.get('commentType')=='CATALYTIC ACTIVITY' and unscoped(comment):
            reaction=comment.get('reaction',{})
            if unscoped(reaction) and 'ecNumber' in reaction:add(reaction['ecNumber'])
    return tuple(sorted(found))


def ec_targets(views,axis_ids,*,device='cpu',dtype=torch.float32):
    axis=_axis(axis_ids)
    if not views:raise ValueError('nonempty prediction view batch required')
    index={x:i for i,x in enumerate(axis)}
    y=torch.zeros(len(views),len(axis),device=device,dtype=dtype)
    mask=torch.zeros_like(y,dtype=torch.bool)
    for row,view in enumerate(views):
        for term in ec_assertions(view):
            if term in index:y[row,index[term]]=1;mask[row,index[term]]=True
    return y,mask


def observed_ec_loss(logits,targets,observed):
    """Observed-only BCE, using the tested shared Boolean-mask reference."""
    return observed_go_loss(logits,targets,observed)


class ECAnnotationHead(nn.Module):
    def __init__(self,width,axis_ids):
        super().__init__();self.axis_ids=_axis(axis_ids)
        self.output=nn.Linear(width,len(self.axis_ids))

    def forward(self,latent):return self.output(latent)

    def get_extra_state(self):
        return dict(version=1,axis_ids=self.axis_ids,missing_is_negative=False,
                    objective='source-unscoped-entry-EC-observed-BCE-v1',hierarchy_propagation=False)

    def set_extra_state(self,state):
        if state!=self.get_extra_state():raise ValueError('EC axis/objective checkpoint mismatch')
