"""First-order exact chunk adjoint for complete native recurrent byte targets."""
import math

import torch
from torch.autograd.function import once_differentiable
from torch.nn import functional as F

from .pubchem_decoder import PubChemDecoder
from .source_byte_objective import validate_byte_targets


def _autocast(device):
    try:
        return torch.is_autocast_enabled(device.type)
    except TypeError:
        return torch.is_autocast_enabled() if device.type=='cuda' else torch.is_autocast_cpu_enabled()


def validate_bounded_adjoint_decoder(decoder):
    if type(decoder)is not PubChemDecoder or decoder.get_extra_state()['version']!=2:
        raise ValueError('exact native shared byte decoder required')
    w=decoder.embedding.weight
    rnn=decoder.rnn
    if (w.dtype not in (torch.float32,torch.float64) or w.device.type not in ('cpu','cuda')
            or _autocast(w.device) or rnn.num_layers!=1 or rnn.dropout!=0 or rnn.bidirectional):
        raise ValueError('deterministic one-layer FP32/FP64 decoder without autocast required')
    if torch.is_grad_enabled() and w.device.type=='cuda' and torch.backends.cudnn.enabled and not rnn.training:
        raise ValueError('CUDA cuDNN recurrent gradients require training mode with dropout disabled')
    return w,rnn


def source_byte_bounded_adjoint(decoder,context,targets,*,chunk_size,max_batch_size,max_total_target_bytes):
    """Score full UTF8 plus EOS; save boundary states and recompute one block at a time.

    Only deterministic one-layer unidirectional decoder recurrence is allowed.
    The requested gradients are first order, in the original forward precision.
    Source supervision, token reductions and chunk boundaries are unchanged.
    """
    targets=validate_byte_targets(targets,max_batch_size=max_batch_size,
        max_total_target_bytes=max_total_target_bytes,chunk_size=chunk_size,recompute=True)
    w,rnn=validate_bounded_adjoint_decoder(decoder)
    if (not isinstance(context,torch.Tensor) or context.shape!=(len(targets),rnn.hidden_size)
            or context.device!=w.device or context.dtype!=w.dtype or not torch.isfinite(context).all()):
        raise ValueError('complete finite source context with exact decoder shape, dtype and device required')
    def execution_config():
        return (decoder.rnn.training,torch.backends.cudnn.enabled,torch.backends.cudnn.allow_tf32,
                torch.backends.cudnn.deterministic,torch.backends.cudnn.benchmark,
                torch.are_deterministic_algorithms_enabled(),torch.get_float32_matmul_precision())
    configuration=execution_config()
    codec=decoder.codec
    rows=tuple(torch.tensor([codec.bos,*codec.tokenizer.encode(raw),codec.eos],
                             dtype=torch.long,device=w.device)for raw in targets)
    count=sum(len(ids)-1 for ids in rows)
    segments=tuple((row,i,min(i+chunk_size,len(ids)-1))
       for row,ids in enumerate(rows)for i in range(0,len(ids)-1,chunk_size))
    if not segments or len(segments)!=sum(math.ceil((len(ids)-1)/chunk_size)for ids in rows):
        raise ValueError('exact complete ordered byte chunks required')
    def params_now():
        return tuple(p for layer in (decoder.embedding,decoder.rnn,decoder.output)for p in layer.parameters())
    params=params_now()
    if len({id(p)for p in params})!=len(params) or any(p.device!=w.device or p.dtype!=w.dtype for p in params):
        raise ValueError('exact unique recurrent parameter storage and dtype required')
    trainable=tuple(i for i,p in enumerate(params)if p.requires_grad)

    def segment(ids,i,j,hidden):
        values,end=decoder.rnn(decoder.embedding(ids[i:j][None]),hidden)
        logits=decoder._readout(values)
        loss=F.cross_entropy(logits.reshape(-1,codec.size),ids[i+1:j+1],reduction='sum')
        return loss,end

    class BoundedAdjoint(torch.autograd.Function):
        @staticmethod
        def forward(ctx,initial,*owned):
            states=[]
            total=initial.new_zeros(())
            last=-1
            for row,i,j in segments:
                if row!=last:
                    hidden=initial[row:row+1].unsqueeze(0).contiguous();last=row
                states.append(hidden)
                term,hidden=segment(rows[row],i,j,hidden)
                total=total+term
            ctx.save_for_backward(*owned,*states)
            return total/count

        @staticmethod
        @once_differentiable
        def backward(ctx,output_gradient):
            owned=ctx.saved_tensors[:len(params)]
            states=ctx.saved_tensors[len(params):]
            if any(a is not b for a,b in zip(params,params_now(),strict=True)) or _autocast(w.device) or execution_config()!=configuration:
                raise ValueError('recurrent owner or backward precision changed after forward')
            active=tuple(owned[i]for i in trainable)
            total_grad=[None]*len(owned)
            context_grad=[None]*len(rows)
            last=-1
            for (row,i,j),state in zip(reversed(segments),reversed(states),strict=True):
                if row!=last:
                    carry=torch.zeros_like(state);last=row
                with torch.enable_grad():
                    hidden=state.detach().requires_grad_(True)
                    term,end=segment(rows[row],i,j,hidden)
                    values=torch.autograd.grad((term,end),(hidden,*active),
                        grad_outputs=(output_gradient/count,carry),allow_unused=True)
                carry=values[0]
                if carry is None:raise ValueError('exact recurrent boundary gradient required')
                if i==0:context_grad[row]=carry.squeeze(0)
                for idx,grad in zip(trainable,values[1:],strict=True):
                    if grad is not None:
                        total_grad[idx]=grad if total_grad[idx]is None else total_grad[idx]+grad
            if any(x is None for x in context_grad):
                raise ValueError('all recurrent source contexts require exact adjoints')
            return torch.cat(context_grad,dim=0),*total_grad

    return BoundedAdjoint.apply(context,*params)
