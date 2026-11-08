"""Train-only RNA annotation diagnostic; not a complete Mora-3 training system."""
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .payload import PayloadEncoder
from .tokengt import TokenGT, orthogonal_identifiers


class RNAClassifier(nn.Module):
    def __init__(self, tokenizer, classes, *, pooling='positional', feature_width=16,
                 width=32, heads=4, layers=2, id_width=4, attention='vanilla', tropical_block_size=32):
        super().__init__()
        if len(classes)<2 or len(set(classes))!=len(classes):
            raise ValueError('Require at least two distinct classes')
        self.classes=tuple(classes)
        self.id_width=id_width
        self.payload=PayloadEncoder(tokenizer,feature_width,pooling=pooling)
        self.backbone=TokenGT(feature_width,id_width,width=width,heads=heads,layers=layers,ffn_width=width,attention=attention,tropical_block_size=tropical_block_size)
        self.head=nn.Linear(width,len(classes))

    def targets(self, records):
        labels=[]
        for record in records:
            matches=[m for n in record.nodes for m in n.measurements if m.name=='rna_type']
            if (len(matches)!=1 or not matches[0].available or matches[0].status!='observed'
                    or matches[0].value not in self.classes):
                raise ValueError('Require one known observed rna_type target')
            labels.append(self.classes.index(matches[0].value))
        return torch.tensor(labels,dtype=torch.long,device=self.head.weight.device)

    def forward(self, records):
        graphs=[]
        for record in records:
            selected={( 'node',i,m.name) for i,n in enumerate(record.nodes)
                      for m in n.measurements if m.name=='sequence' and m.status=='observed'
                      and m.available and isinstance(m.value,str)}
            if len(selected)!=1:
                raise ValueError('Require exactly one observed sequence input')
            ids=orthogonal_identifiers(len(record.nodes),self.id_width,seed=19,
                    dtype=self.head.weight.dtype,device=self.head.weight.device)
            graphs.append(self.payload(record,input_fields=selected,identifiers=ids))
        out,_=self.backbone(graphs)
        return self.head(out[:,0])


def classification_loss(logits, targets):
    if (logits.ndim!=2 or logits.shape[0]==0 or targets.shape!=(logits.shape[0],)
            or targets.dtype!=torch.long or targets.device!=logits.device
            or not torch.isfinite(logits).all() or (targets<0).any()
            or (targets>=logits.shape[1]).any()):
        raise ValueError('Invalid classification logits or targets')
    return F.cross_entropy(logits,targets,reduction='mean')


def save_checkpoint(path, model, optimizer, *, step, binding):
    """Exclusive create protects prior checkpoints. Only load locally produced files."""
    state=dict(version=1,model=model.state_dict(),optimizer=optimizer.state_dict(),
               step=step,binding=binding,cpu_rng=torch.get_rng_state(),
               cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])
    with Path(path).open('xb') as stream:
        torch.save(state,stream)


def load_checkpoint(path, model, optimizer, *, binding):
    state=torch.load(path,map_location='cpu',weights_only=True)
    if state['version']!=1 or state['binding']!=binding:
        raise ValueError('Checkpoint source/configuration binding mismatch')
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    torch.set_rng_state(state['cpu_rng'])
    if state['cuda_rng']:
        torch.cuda.set_rng_state_all(state['cuda_rng'])
    return state['step']
