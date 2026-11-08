"""Complete native byte+EOS likelihood from an authenticated hidden context.

This shared neural equation accepts no raw source conditions or admission claims.
The caller validates its source-specific condition before producing context.
"""
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .pubchem_decoder import PubChemDecoder


def validate_byte_targets(targets,*,max_batch_size,max_total_target_bytes,chunk_size,recompute):
    targets=tuple(targets)
    if (type(max_batch_size)is not int or max_batch_size<1
            or type(max_total_target_bytes)is not int or max_total_target_bytes<1
            or not targets or len(targets)>max_batch_size or any(type(raw)is not bytes for raw in targets)
            or sum(map(len,targets))>max_total_target_bytes
            or type(chunk_size)is not int or chunk_size<1 or type(recompute)is not bool):
        raise ValueError('complete native target bytes and explicit compute/resource bounds required')
    for raw in targets:raw.decode('utf8')
    return targets


def source_byte_objective(decoder,context,targets,*,chunk_size=1024,recompute=False,
        max_batch_size=64,max_total_target_bytes=64*1024*1024):
    targets=validate_byte_targets(targets,max_batch_size=max_batch_size,
        max_total_target_bytes=max_total_target_bytes,chunk_size=chunk_size,recompute=recompute)
    if type(decoder)is not PubChemDecoder or decoder.get_extra_state()['version']!=2:
        raise ValueError('exact native shared primary byte decoder required')
    weight=decoder.embedding.weight
    if (not isinstance(context,torch.Tensor)or context.shape!=(len(targets),decoder.rnn.hidden_size)
            or context.dtype!=weight.dtype or context.device!=weight.device or not torch.isfinite(context).all()):
        raise ValueError('complete finite source context with exact decoder shape/dtype/device required')
    token_sequences=tuple([decoder.codec.bos,*decoder.codec.tokenizer.encode(raw),decoder.codec.eos]for raw in targets)
    return _source_token_objective(decoder,context,token_sequences,chunk_size=chunk_size,recompute=recompute)


def _source_token_objective(decoder,context,token_sequences,*,chunk_size,recompute):
    """Internal exact recurrence shared by validated byte and native-symbol owners."""
    codec=decoder.codec;total=context.sum()*0;count=0
    def segment(tokens,truth,hidden):
        values,next_hidden=decoder.rnn(decoder.embedding(tokens),hidden)
        logits=decoder._readout(values)
        return F.cross_entropy(logits.reshape(-1,codec.size),truth.reshape(-1),reduction='sum'),next_hidden
    for row,sequence in enumerate(token_sequences):
        ids=torch.tensor(sequence,device=context.device,dtype=torch.long)
        hidden=context[row:row+1].unsqueeze(0).contiguous();count+=len(ids)-1
        for start in range(0,len(ids)-1,chunk_size):
            stop=min(start+chunk_size,len(ids)-1)
            args=(ids[start:stop][None],ids[start+1:stop+1][None],hidden)
            loss,hidden=(checkpoint(segment,*args,use_reentrant=False,preserve_rng_state=True)
                if recompute and torch.is_grad_enabled()else segment(*args))
            total=total+loss
    return total/count
