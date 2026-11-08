from dataclasses import replace
from hashlib import sha256

import pytest

from mora3.data.bioelectric_evidence import (BioelectricJATSEvidenceSpan, NativeBioelectricBaseline,
    NativeBioelectricEndpoint, NativeBioelectricEvent, NativeBioelectricObservation)


def test_authenticates_original_unicode_slot_and_refuses_mutated_bound(tmp_path):
    original=b'<article><p>alpha \xce\xb2 gamma</p><!--x-->\n</article>'
    p=tmp_path/'source.xml';p.write_bytes(original)
    span=BioelectricJATSEvidenceSpan(p.name,len(original),sha256(original).hexdigest(),(0,),'text',6,7,
        sha256('alpha β gamma'.encode()).hexdigest(),sha256('β'.encode()).hexdigest())
    assert span.read(tmp_path)=='β'
    with pytest.raises(ValueError,match='fingerprint'):replace(span,end_codepoint=8).read(tmp_path)
    with pytest.raises(ValueError,match='immutable'):replace(span,original_sha256='a'*64).read(tmp_path)


def observation():
    evidence=BioelectricJATSEvidenceSpan('fixture.xml',1,'a'*64,(0,),'text',0,1,'b'*64,'c'*64)
    return NativeBioelectricObservation(NativeBioelectricBaseline('fixture organism','culture',None,None,None),
        (NativeBioelectricEvent('alter_junction_coupling','native assay perturbation','gap_junction',
            ('reported junction',),None,None,None,None),),
        NativeBioelectricEndpoint('transfer detection','paired-cell assay',None,'native_functional_coupling_class',None,
            ('observed transfer','observed no transfer')),
        'primary_observation','paired','fixture-work','fixture-experiment','fixture-v1',(evidence,),
        category_target='observed transfer')


def test_outcome_rights_and_control_audit_never_enter_predictor():
    first=observation()
    changed=replace(first,category_target='observed no transfer',exact_work_group='unseen work',
        exact_experiment_parent='unseen experiment',original_version='other version')
    assert changed.condition_inputs()==first.condition_inputs()
    assert changed.condition_sha256()==first.condition_sha256() and first.target()!=changed.target()
    assert replace(first,category_target=None).target() is None
    with pytest.raises(ValueError,match='literal observed'):replace(first,category_target='unobserved').target()


@pytest.mark.parametrize('kind',['simulation','review','hypothesis'])
def test_nonobservational_roles_never_produce_observed_labels(kind):
    with pytest.raises(ValueError,match='not observed'):replace(observation(),evidence_class=kind).target()


def test_requires_reported_control_and_distinguishes_proximity_proposal():
    row=observation()
    with pytest.raises(ValueError,match='comparator'):
        replace(row,evidence_class='primary_reported_contrast',control_evidence='unresolved').target()
    event=replace(row.interventions[0],action='propose_linkage',intended_coupling_partners=('first hemichannel','second hemichannel'))
    planned=replace(row,interventions=(event,))
    assert planned.condition_inputs()['interventions'][0]['intended_coupling_partners']==('first hemichannel','second hemichannel')
    with pytest.raises(ValueError,match='not a measured'):planned.target()
