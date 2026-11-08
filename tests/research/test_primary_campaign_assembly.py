from dataclasses import asdict,replace
from decimal import Decimal
from hashlib import sha256
import json

import pytest
import torch

from mora3.data.admet import canonical
from mora3.model.full_owner import PrimaryConfiguration
from mora3.research.matched_validation import MatchedValidationCohort,MatchedValidationTask,ValidationSourceRow
from mora3.research.mup_campaign import FrozenPrimaryProxyCampaign,NativeTrainRoutePlan
from mora3.research.mup_sweep_selection import planned_proxy_trials
from mora3.research import primary_campaign_assembly as assembly


def fixture(root):
    def write(name,doc,raw=False):
        data=doc if raw else canonical(doc)
        (root/name).write_bytes(data)
        return dict(path=name,sha256=sha256(data).hexdigest())
    source=write('source.json',dict(schema='test-source-only'))
    snapshot=source['sha256']
    family=dict(schema='mora3.declared-width-family.v1',widths=[['hidden',4,6,8]],total_numel=[4,6,8])
    for key,width in [('base',4),('delta',6),('target',8)]:
        family[key]=dict(schema='mora3.parameter-alias-inventory.v1',total_numel=width,trainable_numel=width,
            parameters=[dict(aliases=['vector'],shape=[width],axes=[['width','hidden']],role='vector',numel=width,trainable=True,dtype='torch.float32')])
    family_ref=write('family.json',family)
    cohort=MatchedValidationCohort(snapshot,'b'*64,'c'*64,'d'*64,1,1,1024,
        (MatchedValidationTask('task','family','native','schema',
        (ValidationSourceRow('source.arrow','e'*64,0,'row','f'*64,'revision'),),1,Decimal('1'),Decimal('.01'),'loss','target'),),())
    validation=write('validation.json',cohort.wire(),True)
    evidence={}
    for key in ['source_role_schedule','complete_owner_evidence','runtime_profile','existing_worker']:
        evidence[key]=write(key+'.json',dict(source_snapshot_sha256=snapshot,unreviewed_fixture=True))
    campaign=FrozenPrimaryProxyCampaign(snapshot,family_ref['sha256'],4,4,
        *(evidence[key]['sha256'] for key in ['source_role_schedule','complete_owner_evidence','runtime_profile','existing_worker']),
        cohort,(('task','a'*64),),(NativeTrainRoutePlan('task','b'*64,'c'*64,1),),('task',),1.,(),Decimal('1'),Decimal('1'))
    campaign_ref=write('campaign.json',campaign.wire(),True)
    configurations={key:write(key+'.json',asdict(PrimaryConfiguration(width,4,snapshot,readout_mult=.5,attn_mult=.5))) for key,width in [('base',4),('delta',6),('target',8)]}
    request=dict(schema='Mora3.PrimaryCampaignConstructionRequest/v1',source_snapshot=source,
        trial_id=planned_proxy_trials()[0].trial_id,campaign=campaign_ref,validation=validation,
        width_family=family_ref,configurations=configurations,evidence=evidence,
        source_view_policy='effective-study-v2')
    ref=write('request.json',request)
    return request,ref,write


def test_load_cross_binds_without_tensor_allocation_or_admission(tmp_path,monkeypatch):
    request,ref,_=fixture(tmp_path)
    def forbidden(*args,**kwargs): pytest.fail('tensor allocation forbidden')
    monkeypatch.setattr(torch,'empty',forbidden)
    loaded=assembly.load_primary_campaign_request(tmp_path,ref['path'],ref['sha256'])
    assert loaded.campaign.source_snapshot_sha256==request['source_snapshot']['sha256']
    assert loaded.family.total_numel==(4,6,8)
    assert loaded.allocation_authorized is False
    assert 'semantic_evidence_admission' in loaded.unresolved


@pytest.mark.parametrize('case',['config-snapshot','config-width','family-count','evidence-snapshot','missing-evidence','extra-field','trial','tamper','traversal'])
def test_invalid_request_refuses_before_allocation(tmp_path,case):
    request,ref,write=fixture(tmp_path)
    if case=='config-snapshot':
        request['configurations']['base']=write('bad-config.json',asdict(PrimaryConfiguration(4,4,'a'*64,readout_mult=.5,attn_mult=.5)))
    elif case=='config-width':
        request['configurations']['base']=write('bad-config.json',asdict(PrimaryConfiguration(6,4,request['source_snapshot']['sha256'],readout_mult=.5,attn_mult=.5)))
    elif case=='family-count':
        doc=json.loads((tmp_path/'family.json').read_bytes());doc['base']['total_numel']=999
        request['width_family']=write('bad-family.json',doc)
    elif case=='evidence-snapshot':
        request['evidence']['runtime_profile']=write('bad-runtime.json',dict(source_snapshot_sha256='f'*64))
    elif case=='missing-evidence': del request['evidence']['existing_worker']
    elif case=='extra-field': request['construct_anyway']=True
    elif case=='trial': request['trial_id']='unplanned'
    elif case=='traversal': request['source_snapshot']['path']='../outside.json'
    elif case=='tamper':
        with (tmp_path/'base.json').open('ab') as stream: stream.write(b'\n')
    ref=write('request.json',request)
    with pytest.raises(ValueError): assembly.load_primary_campaign_request(tmp_path,ref['path'],ref['sha256'])


def test_nonwidth_configuration_drift_refuses(tmp_path):
    request,ref,write=fixture(tmp_path)
    document=json.loads((tmp_path/'target.json').read_bytes())
    document['encoder_var']=2.
    request['configurations']['target']=write('changed-target.json',document)
    ref=write('request.json',request)
    with pytest.raises(ValueError,match='Non-width'):
        assembly.load_primary_campaign_request(tmp_path,ref['path'],ref['sha256'])


def test_evidence_json_never_accepts_duplicate_snapshot_fields():
    with pytest.raises(ValueError):
        assembly._document(b'{"source_snapshot_sha256":"a","source_snapshot_sha256":"b"}',canonical_required=False)
