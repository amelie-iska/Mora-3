"""Source-listed categorical objectives, synthetic native views; no admission."""
import copy
import json
from dataclasses import replace
import pytest
import torch
from torch.nn import functional as F
from mora3.data.uniprot_annotations import AnnotationPredictionView
from mora3.model.source_annotation_objectives import SourceAnnotationObjectives,source_annotation_objectives_parameter_roles,source_annotation_objectives_width_family
from mora3.model.residue_objective import SharedResidueModel
from mora3.model.nucleotide_decoder import NucleotideDecoder
from mora3.model.rfam_family import RfamFamilyModel
from mora3.model.parameterization import parameter_ledger,width_family
from test_annotation_mup import make as annotation
from test_rfam_family_objective import AXIS,view as rfam_view


def make(w=8):
    shared=annotation(w,max_tokens=8192)
    nucleotide=NucleotideDecoder(shared,parameterization_policy='mup-sequence-v1')
    rfam=RfamFamilyModel(nucleotide,AXIS,manifest_sha256='a'*64,parameterization_policy='mup-rfam-v1')
    residue=SharedResidueModel(shared,('Active site','Disulfide bond'),parameterization_policy='mup-residue-v1')
    return SourceAnnotationObjectives(shared,rfam,residue,residue_max_target_length=8).double()


def protein(features=None,ec=('1.1.1.1','2.2.2.2')):
    if features is None:
        features=[dict(type='Active site',location=dict(start=dict(value=2,modifier='EXACT'),end=dict(value=2,modifier='EXACT'))),
                  dict(type='Disulfide bond',location=dict(start=dict(value=1,modifier='EXACT'),end=dict(value=4,modifier='EXACT'))),
                  dict(type='Active site',location=dict(start=dict(value=2,modifier='EXACT'),end=dict(value=2,modifier='EXACT')))]
    packet=dict(proteinDescription=dict(recommendedName=dict(ecNumbers=[dict(value=e) for e in reversed(ec)])),comments=[],features=features)
    provenance=dict(split='train',license='CC-BY-4.0',source_ref=dict(source_release='synthetic-unit-only',raw_file_sha256='0'*64))
    return AnnotationPredictionView('unit-only','ACDE',json.dumps(dict(annotation_packet=packet,provider_computed_sequence_properties=[])).encode(),json.dumps(provenance).encode())


def test_ec_canonical_source_set_eos_and_all_primary_readout_kernel_gradients():
    m=make(16);views=(protein(),protein(ec=()))
    rows=m.ec_observed_lists(views);assert rows==((0,1),())
    context=m.ec_context(views);prefix=m.ec_list.teacher_prefixes(rows)
    prior=m.annotation.ec(m.annotation._readout_features(context));prior=torch.cat((prior,prior.new_zeros(2,1)),-1)
    independent=m.ec_list(context,prefix,prior_logits=prior)
    loss=F.cross_entropy(torch.cat(independent),torch.tensor([0,1,2,2]))
    actual=m.ec_reference_objective(views);torch.testing.assert_close(actual,loss,rtol=0,atol=0)
    actual.backward()
    for p in (m.annotation.ec.output.weight,m.ec_list.cell.weight_hh,m.ec_list.output.weight,m.annotation.backbone.feature.weight):
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
    changed=replace(views[0],accession='untrusted-audit',provenance_bytes=json.dumps(dict(views[0].provenance(),audit_extra='changed')).encode(),target_bytes=protein(ec=()).target_bytes)
    # Same causal history and sequence means no target/audit effect on predictors.
    a=m.ec_forward(views,prefix);b=m.ec_forward((changed,views[1]),prefix)
    for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=0,atol=0)
    bad=protein(ec=('7.9.9.9',))
    with pytest.raises(ValueError,match='frozen axis'):m.ec_observed_lists((bad,))
    missing=replace(views[0],target_bytes=b'{}')
    with pytest.raises(ValueError,match='missing'):m.ec_observed_lists((missing,))


def test_ec_scope_and_canonical_set_not_recursive_text_or_fabricated_empty():
    m=make();v=protein(ec=('1.1.1.1',));t=v.targets();p=t['annotation_packet']
    p['proteinDescription']['alternativeNames']=[dict(molecule='chain',ecNumbers=[dict(value='2.2.2.2')])]
    p['proteinDescription']['contains']=[dict(recommendedName=dict(ecNumbers=[dict(value='2.2.2.2')]))]
    p['comments']=[dict(commentType='CATALYTIC ACTIVITY',reaction=dict(ecNumber='1.1.1.1')),
                   dict(commentType='FUNCTION',texts=[dict(value='2.2.2.2')])]
    v=replace(v,target_bytes=json.dumps(t).encode());assert m.ec_observed_lists((v,))==((0,),)


def test_rfam_source_set_uses_existing_readout_prior_and_preserves_target_exclusion():
    m=make(16);views=(rfam_view(),rfam_view(family='RF00001'))
    rows=m.rfam_observed_lists(views);assert rows==((1,),(0,))
    prefixes=m.rfam_list.teacher_prefixes(rows);loss=m.rfam_reference_objective(views);loss.backward()
    for p in (m.rfam.family_output.weight,m.rfam_list.cell.weight_hh,m.rfam_list.output.weight):
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
    a=m.rfam_forward(views,prefixes);b=m.rfam_forward((rfam_view(family='RF00003'),views[1]),prefixes)
    for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=0,atol=0)


def test_residue_native_order_duplicates_bond_endpoints_and_only_source_positions_loss():
    m=make(16);v=protein();positions=m.residue_observed_lists((v,))
    assert [(r.position_0based,r.ordered_label_ids) for r in positions]==[(0,(1,)),(1,(0,0)),(3,(1,))]
    rows=tuple(r.ordered_label_ids for r in positions);assert sum(len(x)+1 for x in rows)==7
    loss=m.residue_reference_objective((v,));loss.backward()
    for p in (m.residue.head.output.weight,m.residue.head.local.weight,m.residue_list.cell.weight_hh,m.residue_list.output.weight):
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
    # Source positions select loss/outputs; they never mark sequence features.
    prefixes=m.residue_list.teacher_prefixes(rows);a=m.residue_forward((v,),positions,prefixes)
    changed=replace(v,target_bytes=protein(features=[]).target_bytes,provenance_bytes=b'{}')
    b=m.residue_forward((changed,),positions,prefixes)
    for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=0,atol=0)
    with pytest.raises(ValueError,match='no exact'):m.residue_observed_lists((changed,))
    fuzzy=v.targets();fuzzy['annotation_packet']['features'][0]['location']['start']['modifier']='UNKNOWN'
    f=replace(v,target_bytes=json.dumps(fuzzy).encode());rows=m.residue_observed_lists((f,))
    assert next(r.ordered_label_ids for r in rows if r.position_0based==1)==(0,)


def test_all_three_source_lists_complete_shared_family_and_checkpoint_refusal():
    ms=tuple(make(w) for w in (8,12,16))
    ledgers=[parameter_ledger(m,source_annotation_objectives_parameter_roles(m,identifier_axis=('protein-coordinate',))) for m in ms]
    family=source_annotation_objectives_width_family(*ms,identifier_axis=('protein-coordinate',));assert family.total_numel==tuple(sum(p.numel() for p in m.parameters()) for m in ms)
    assert ms[0].model_binding()==ms[-1].model_binding()
    m=ms[-1];before=[p.detach().clone() for p in m.parameters()];state=copy.deepcopy(m.state_dict())
    first=next(n for n,v in state.items() if isinstance(v,torch.Tensor));state[first]+=8
    state['residue_list._extra_state']['source_namespace']='wrong'
    with pytest.raises(ValueError):m.load_state_dict(state)
    assert all(torch.equal(a,b) for a,b in zip(before,m.parameters()))
