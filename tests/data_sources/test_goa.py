import copy
import importlib.util
import pathlib
import pytest

ROOT=pathlib.Path.cwd()
def module():
    path=ROOT/'scripts/data_sources_goa.py'
    assert path.is_file(), 'GOA bounded source pager is not implemented yet'
    spec=importlib.util.spec_from_file_location('data_sources_goa',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def row(id='a',subject='UniProtKB:P05067'):
    return dict(id=id,geneProductId=subject,goId='GO:0003674',qualifier='NOT|enables',evidenceCode='ECO:0000314',goEvidence='IDA',assignedBy='UniProt',date='20260701',reference='PMID:1',withFrom=[{'connectedXrefs':[{'db':'UniProtKB','id':'Q12345'},{'db':'UniProtKB','id':'P12345'}]}],extensions=[{'connectedXrefs':[{'db':'CL','id':'0000000','relation':'occurs_in'},{'db':'GO','id':'0005737','relation':'part_of'}]},{'connectedXrefs':[{'db':'CL','id':'0000001','relation':'occurs_in'}]}],unexpected_future_field={'keep':'exact'})

def page(rows,hits,current=1,total=1,limit=200):
    return dict(results=rows,numberOfHits=hits,pageInfo=dict(current=current,total=total,resultsPerPage=limit))

def test_filter_rejects_unbounded_invalid_and_noncanonical_requests():
    m=module()
    for value in [[],[''],['P05067-2'],['UniProtKB:P05067'],['P05067,Q12345']]:
        with pytest.raises(ValueError):m.query_params(value,1)
    assert m.query_params(['P05067','P05067','Q12345'],1)['geneProductId']=='UniProtKB:P05067,UniProtKB:Q12345'

def test_canonical_namespace_keeps_subject_forms_and_excludes_foreign_roots():
    m=module();raw=row(subject='UniProtKB:P05067-PRO_0000000093');original=copy.deepcopy(raw)
    out=m.summarize_pages([page([raw],1)],['P05067'])
    assert raw==original
    assert out['subject_counts']=={'UniProtKB:P05067-PRO_0000000093':1}
    assert out['negated_rows']==1 and out['with_from_rows']==1 and out['extension_rows']==1
    with pytest.raises(ValueError):m.summarize_pages([page([row(subject='UniProtKB:Q12345')],1)],['P05067'])

def test_page_count_unique_ids_and_declared_total_are_verified():
    m=module();p1=page([row('a'),row('b')],3,1,2,2);p2=page([row('c')],3,2,2,2)
    assert m.summarize_pages([p1,p2],['P05067'],limit=2)['annotation_rows']==3
    with pytest.raises(ValueError):m.summarize_pages([p1],['P05067'],limit=2)
    with pytest.raises(ValueError):m.summarize_pages([p1,page([row('a')],3,2,2,2)],['P05067'],limit=2)
    with pytest.raises(ValueError):m.summarize_pages([p1,page([row('c')],4,2,2,2)],['P05067'],limit=2)

def test_zero_annotations_is_complete_and_not_a_negative_label():
    m=module();out=m.summarize_pages([page([],0,0,0)],['P05067'])
    assert out['annotation_rows']==0 and out['missing_annotations_create_negative_labels'] is False

def test_snapshot_drift_stops_mixed_release_admission():
    m=module();m.check_snapshot({'annotation':{'timestamp':'2026-07-28 10:08'}},'2026-07-28 10:08')
    with pytest.raises(ValueError):m.check_snapshot({'annotation':{'timestamp':'2026-08-01 00:00'}},'2026-07-28 10:08')

def test_gaf_form_column_and_not_are_retained_without_transfer():
    m=module();fields=['UniProtKB','P05067','APP','NOT|located_in','GO:0005737','PMID:1','IDA','','C','APP','','protein','taxon:9606','20260701','UniProt','occurs_in(CL:0000000),part_of(GO:0005737)|occurs_in(CL:0000001)','UniProtKB:P05067-PRO_0000000093']
    s='!gaf-version: 2.2\n'+'\t'.join(fields)+'\n';out=m.gaf_info(s,['P05067'])
    assert out['annotation_rows']==1 and out['form_rows']==1 and out['negated_rows']==1
    assert out['subject_counts']=={'UniProtKB:P05067-PRO_0000000093':1}
    with pytest.raises(ValueError):m.gaf_info(s.replace('gaf-version: 2.2','gaf-version: 2.1'),['P05067'])
    with pytest.raises(ValueError):m.gaf_info(s.replace('\t'.join(fields),'\t'.join(fields[:-1])),['P05067'])

def test_retry_policy_is_finite_and_respects_service_delay():
    m=module();assert m.retry_delay(429,0,'120')==120
    assert m.retry_delay(503,1,None)==15
    for args in [(429,0,'900'),(503,2,None),(401,0,None)]:
        with pytest.raises(ValueError):m.retry_delay(*args)


def test_finite_batch_verifies_post_snapshot_and_full_gaf_before_quality_pass():
    m=module();calls=[];r=row();fields=['UniProtKB','P05067','APP','NOT|enables','GO:0003674','PMID:1','IDA','UniProtKB:Q12345,UniProtKB:P12345','F','APP','','protein','taxon:9606','20260701','UniProt','occurs_in(CL:0000000),part_of(GO:0005737)|occurs_in(CL:0000001)','']
    def request(path,params,name,accept='application/json'):
        calls.append((path,params,name,accept))
        if path=='annotation/about':return {'annotation':{'timestamp':'fixed'}}
        if path=='annotation/search':return page([r],1,limit=params['limit'])
        return '!gaf-version: 2.2\n'+'\t'.join(fields)+'\n'
    result=m.acquire_batch(['P05067'],request,'fixed')
    assert result['quality_pass_annotations']==1 and result['task_admitted_annotations']==0
    assert calls[0][0]=='annotation/about' and calls[-1][0]=='annotation/about'
    assert sum(x[0]=='annotation/search' for x in calls)==2
    assert any(x[3]=='text/gaf' for x in calls)

def test_batch_fails_closed_when_snapshot_changes_after_native_payload():
    m=module();i=0
    def request(path,params,name,accept='application/json'):
        nonlocal i
        if path=='annotation/about':
            i+=1;return {'annotation':{'timestamp':'fixed' if i==1 else 'new'}}
        if path=='annotation/search':return page([],0,0,0,1 if name=='count-probe' else 200)
        return '!gaf-version: 2.2\n'
    with pytest.raises(ValueError,match='snapshot'):m.acquire_batch(['P05067'],request,'fixed')


def test_production_driver_refuses_unleased_or_foreign_scope_before_api(tmp_path):
    import sys,json
    sys.path.insert(0,str(ROOT/'scripts'))
    import data_sources_goa_acquire as driver
    (tmp_path/'audits').mkdir()
    (tmp_path/'audits/remaining-canonical-scope-v1.json').write_text(json.dumps({'accessions':['P05067']}))
    plan={'job_id':'fixture','workers':1,'requests_per_second_max':1,'batches':[['Q12345']],'accessions':['Q12345'],'canonical_scope_count':1}
    pp=tmp_path/'plan.json';pp.write_text(json.dumps(plan));lp=tmp_path/'lease.json';lp.write_text(json.dumps({'job_id':'fixture','one_source_connection_reserved':False}))
    with pytest.raises(ValueError,match='lease'):driver.run(pp,lp)
    lp.write_text(json.dumps({'job_id':'fixture','one_source_connection_reserved':True}))
    with pytest.raises(ValueError,match='qualified baseline'):driver.run(pp,lp)
    assert not (tmp_path/'lease-fixture.json').exists()
