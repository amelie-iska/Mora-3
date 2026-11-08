"""Current source-specific prior neural path; synthetic targets, no admission."""
import json
import pytest
import torch
from torch.nn import functional as F
from test_pubchem_admet_mup import pubchem
from mora3.model.nucleotide_decoder import NucleotideDecoder


def make():
    from mora3.model.native_sequence_priors import NativeSequencePriorModel
    d=pubchem(16).eval();n=NucleotideDecoder(d.shared,parameterization_policy='mup-sequence-v1').eval()
    return NativeSequencePriorModel(n,d).eval()


def records():
    from mora3.data.source_native_priors import NativePriorView
    specs=[('dna','DNA','sequence',b'AcGTnN',{'prior_domain':'source_DNA_sequence','source_organism':None}),
           ('protein','protein','sequence',b'MGUEK*',{'prior_domain':'protein_reference_native_amino_acids'}),
           ('native_smiles','native_SMILES','native_smiles',b'C[C@H](O)N',{'prior_domain':'molecular_reference_native_SMILES'})]
    return tuple(NativePriorView(route=route,datatype=datatype,target_field=field,target_bytes=target,
        condition_bytes=json.dumps(condition).encode(),audit_bytes=b'{}')for route,datatype,field,target,condition in specs)


@pytest.mark.parametrize('chunk,recompute',[(1,False),(3,True),(1000,False)])
def test_native_full_symbols_bytes_eos_all_gradients_and_recurrent_history(chunk,recompute):
    m=make();rs=records();actual=m._reference_objective(rs,chunk_size=chunk,recompute=recompute)
    total=next(m.parameters()).sum()*0;count=0
    for r in rs:
        d=m.nucleotide if r.route=='dna'else m.decoder
        ids=d.codec.encode(r.target_bytes.decode())if r.route=='dna'else[d.codec.bos,*d.codec.tokenizer.encode(r.target_bytes),d.codec.eos]
        context=m._condition(r);tokens=torch.tensor(ids,dtype=torch.long)
        h,_=d.rnn(d.embedding(tokens[:-1][None]),context[None].contiguous())
        total+=F.cross_entropy(d._readout(h).reshape(-1,d.codec.size),tokens[1:],reduction='sum');count+=len(ids)-1
    expected=total/count;torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
    ps=tuple(m.parameters());a=torch.autograd.grad(actual,ps,allow_unused=True);b=torch.autograd.grad(expected,ps,allow_unused=True)
    for x,y in zip(a,b):
        assert(x is None)==(y is None)
        if x is not None:torch.testing.assert_close(x,y,rtol=1e-10,atol=1e-11)
    g={id(p):x for p,x in zip(ps,a)}
    for p in(m.nucleotide.rnn.weight_hh_l0,m.decoder.rnn.weight_hh_l0):assert g[id(p)]is not None and g[id(p)].abs().sum()>0
    assert m.nucleotide.shared is m.decoder.shared


def test_prior_target_refusal_and_audit_exclusion_before_encoder():
    from dataclasses import replace
    m=make();r=records()[0];a=m._condition(r);b=m._condition(replace(r,target_bytes=b'UUU',audit_bytes=b'{"record":"secret-audit"}'))
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    def fail(*args,**kw):raise AssertionError('encoder entered before targetvalidation')
    m._condition=fail
    for bad in(replace(r,target_bytes=b'not-DNA!'),replace(r,condition_bytes=b'{"prior_domain":"source_DNA_sequence","source_organism":null,"target_length":8}')):
        with pytest.raises(ValueError):m._reference_objective((bad,))


def test_current_prior_public_objective_requires_authenticated_reader_first():
    m=make()
    with pytest.raises((ValueError,TypeError,AttributeError)):m.objective(None,None)


def test_current_native_prior_aliases_width_roles_checkpoint_and_no_new_parameters():
    import copy
    from mora3.model.native_sequence_priors import NativeSequencePriorModel,native_sequence_prior_parameter_roles
    from mora3.model.parameterization import parameter_ledger,width_family
    from mora3.model.mu_optimizer import MuAdamW
    models=[];roles=[]
    for width in(8,12,16):
        d=pubchem(width).eval();n=NucleotideDecoder(d.shared,parameterization_policy='mup-sequence-v1').eval()
        m=NativeSequencePriorModel(n,d);models.append(m);roles.append(native_sequence_prior_parameter_roles(m,identifier_axis=('protein-coordinate',)))
    family=width_family(*(parameter_ledger(m,r)for m,r in zip(models,roles)));m=models[-1];r=roles[-1]
    assert family.total_numel[-1]==sum(p.numel()for p in m.parameters())
    assert r['nucleotide.shared.backbone.layers.0.attention.qkv.weight']==r['decoder.shared.backbone.layers.0.attention.qkv.weight']
    opt=MuAdamW(m,family,r,base_lr=.001,weight_decay=0.,decay_aliases=());assert not opt.state
    clone=copy.deepcopy(m);clone.load_state_dict(m.state_dict());assert clone.nucleotide.shared is clone.decoder.shared
    assert clone.get_extra_state()==m.get_extra_state()
