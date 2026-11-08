"""Original train-only native-symbol and byte-prior campaign contracts."""

import pytest

from mora3.data.source_native_priors import NativePriorView
from mora3.model.annotation_vocabulary import AnnotationVocabulary
from mora3.model.pubchem_decoder import PubChemByteCodec
from mora3.research.native_prior_campaign import _prior_target_count


def test_full_native_symbol_and_fitted_byte_denominators_keep_eos():
    codec = PubChemByteCodec(AnnotationVocabulary(
        '{"learned_hex":["4343"],"controls":["pad","node","edge"]}').tokenizer())
    dna = NativePriorView('dna','DNA','source_native_sequence',b'ACCGT',b'{}',b'{}')
    smiles = NativePriorView('native_smiles','native_SMILES','native_smiles',b'CCO',b'{}',b'{}')
    assert _prior_target_count((dna,),'dna',codec)==6
    assert _prior_target_count((smiles,),'native_smiles',codec)==3
    with pytest.raises(ValueError,match='native prior route'):
        _prior_target_count((dna,smiles),'dna',codec)
    unsupported = NativePriorView('dna','DNA','source_native_sequence',b'AC!',b'{}',b'{}')
    with pytest.raises(ValueError,match='native nucleotide'):
        _prior_target_count((unsupported,),'dna',codec)
