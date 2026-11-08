"""Exercise typed query conditioning and exact observed/missing byte loss behavior."""
from dataclasses import dataclass
import torch
import pytest

from mora3.model.uniprot_native_objectives import SharedUniProtNativeAnnotationModel,_known_graph,uniprot_native_annotation_parameter_roles
from test_pubchem_admet_mup import pubchem


@dataclass(frozen=True)
class FixtureRow:
    target: bytes|None

    def model_condition_fields(self):
        return dict(amino_acid_sequence='ACDEFGHIKLMNPQRSTVWY',organism_taxon=9606)

    def target_bytes(self,field):
        assert field=='source_curated_function'
        return self.target


def test_observed_native_field_full_utf8_eos_and_missing_mask():
    model=SharedUniProtNativeAnnotationModel(pubchem(8)).double().eval()
    literal='Complete source text with γ and β.'.encode('utf-8')
    logits=[]
    hook=model.decoder.output.register_forward_hook(lambda m,a,o:logits.append(o.detach().clone()[0]))
    try:
        with torch.inference_mode():
            result=model._score((FixtureRow(None),FixtureRow(literal),FixtureRow(None)),'source_curated_function',
                chunk_size=7,recompute=False)
    finally:hook.remove()
    code=model.decoder.codec;encoded=code.tokenizer.encode(literal)+(code.eos,)
    assert result['observed']==1 and result['missing']==2 and result['exact_target_UTF8_bytes']==len(literal)
    log=torch.cat(logits)
    assert log.shape==(len(encoded),code.size)
    target=torch.tensor(encoded,dtype=torch.long)
    expected=(torch.logsumexp(log,1)-log.gather(1,target[:,None])[:,0]).mean()
    torch.testing.assert_close(result['loss'],expected,rtol=1e-12,atol=1e-12)
    with pytest.raises(ValueError,match='unobserved'):
        model._score((FixtureRow(None),),'source_curated_function',chunk_size=7,recompute=False)
    roles=uniprot_native_annotation_parameter_roles(model,identifier_axis=('context',))
    assert set(roles)=={name for name,_ in model.named_parameters(remove_duplicate=False)}
    assert all(a is b for a,b in zip(model.parameters(),model.decoder.parameters(),strict=True))


def test_known_query_rejects_sequence_or_audit_leak_in_inverse():
    known=dict(source_experimentally_supported_function_texts=('reported function',),
        source_experimental_associated_GO_ids=('GO:0000001',),source_explicit_experimental_NOT_GO_ids=(),
        source_experimentally_supported_EC_ids=('1.1.1.-',),source_kinetic_conditions=(),organism_taxon_id=9606)
    r,t=_known_graph(known,'native_canonical_amino_acid_sequence')
    assert all(x[2] not in ('requested_native_target','sequence','accession') for x in t.input_fields(r))
    for extra in ({'sequence':'ACDE'},{'accession':'P00000'},{'native_shard':'private.parquet'}):
        with pytest.raises(ValueError,match='whitelist'):_known_graph(dict(known,**extra),'native_canonical_amino_acid_sequence')
