"""Exact campaign observation counts across native assay partitions."""

from dataclasses import replace
from decimal import Decimal

import pytest
import torch

from mora3.data.nci_current_assays import CONCLUSION, ENDPOINT, PAIRED, NciAssayPartition
from mora3.model.pubchem_decoder import PubChemByteCodec
from mora3.model.annotation_vocabulary import AnnotationVocabulary
from mora3.research.assay_campaign_routes import (
    _nci_target_count, _nci_partition_layout, _partition_weighted_mean, _validation_reduction,
)
from test_nci_assay_model import make, tasks


def test_nci_prediction_uses_native_coordinate_denominators_and_exact_partition_membership():
    originals = tasks('prediction')
    layout = _nci_partition_layout(originals)
    assert tuple(count for _,_,count in layout)==(1,1,2)
    branch = make(16)
    results = branch._prediction_objectives(originals)
    terms = tuple((result.partition,result.indices,result.loss) for result in results)
    actual = _partition_weighted_mean(layout,terms)
    expected = (results[0].loss+results[1].loss+2*results[2].loss)/4
    torch.testing.assert_close(actual,expected,rtol=1e-13,atol=1e-13)
    value = _validation_reduction(actual.detach(),4)
    assert value.observed_targets==4 and value.loss_sum==Decimal(str(float(actual.detach())))*4
    wrong = (terms[0],(terms[1][0],(0,),terms[1][2]),terms[2])
    with pytest.raises(ValueError,match='partition'):
        _partition_weighted_mean(layout,wrong)
    with pytest.raises(ValueError,match='partition'):
        _partition_weighted_mean(layout,terms[:-1])


def test_native_generation_enrollment_counts_fitted_byte_merges_and_eos():
    vocabulary = AnnotationVocabulary('{"learned_hex":["4343"],"controls":["pad","node","edge"]}')
    codec = PubChemByteCodec(vocabulary.tokenizer())
    generation = tasks('generation')
    changed = tuple(replace(t,target=body) for t,body in zip(generation,(b'CCO',b'O',b'CCCC'),strict=True))
    # CCO->CC,O,EOS. O->O,EOS. CCCC->CC,CC,EOS.
    assert _nci_target_count(changed,'generation',codec)==8
    assert _nci_target_count(tasks('prediction'),'prediction',codec)==4
    with pytest.raises(ValueError,match='direction'):
        _nci_target_count(changed,'prediction',codec)


def test_numeric_partition_counts_never_derive_ratios_or_merge_units():
    base = tasks('prediction')
    both_units = (replace(base[2],partition=NciAssayPartition(PAIRED,None,'M')),base[2],base[1])
    layout = _nci_partition_layout(both_units)
    assert len(layout)==3 and [count for _,_,count in layout]==[2,2,1]
    assert {partition.unit for partition,_,_ in layout if partition.family==PAIRED}=={'M','u'}
    assert _nci_partition_layout((base[0],base[0]))==((NciAssayPartition(CONCLUSION,None,None),(0,1),2),)


def test_scalar_owner_accepts_exact_current_rebinding_and_rejects_proof_or_partition_drift():
    from types import SimpleNamespace
    from mora3.data.admet import canonical, digest
    from mora3.model.source_scalar_objectives import SourceScalarAxis
    from mora3.research.assay_campaign_routes import _scalar_owner_view

    original = SimpleNamespace(effective_filter={'selected':'original'})
    current = SimpleNamespace(effective_filter={'selected':'current'})
    key = ('FreeSolv','experimental_hydration_free_energy','kcal/mol',None,None,None,'point')
    old = SourceScalarAxis('a'*64,digest(canonical(original.effective_filter)),((key,42),))
    rebound = replace(old,filter_payload_sha256=digest(canonical(current.effective_filter)))
    adapter = SimpleNamespace(axis=old,original_view=original,current_view=current)
    assert _scalar_owner_view(adapter,old) is original
    assert _scalar_owner_view(adapter,rebound) is current
    for drift in (replace(rebound,proof_sha256='b'*64),
                  replace(rebound,partitions=((key,41),)),
                  replace(rebound,filter_payload_sha256='c'*64)):
        with pytest.raises(ValueError,match='scalar source axis'):
            _scalar_owner_view(adapter,drift)
