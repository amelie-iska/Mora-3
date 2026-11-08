import importlib.util,pathlib,sys,json
import pytest
ROOT=pathlib.Path.cwd();sys.path.insert(0,str(ROOT/'scripts'))
def module():
    p=ROOT/'scripts/data_sources_goa_v2.py'
    assert p.is_file(), 'source-faithful GPAD fallback for null GO shortcodes not implemented'
    s=importlib.util.spec_from_file_location('data_sources_goa_v2',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def records():
    common=dict(geneProductId='UniProtKB:Q96EB6',qualifier='is_active_in',goId='GO:0005634',reference='PMID:35732914',withFrom=None,extensions=None,interactingTaxonId=None,date='20220623',assignedBy='UniProt')
    return [dict(common,id='one',goEvidence='IDA',evidenceCode='ECO:0000314'),dict(common,id='two',goEvidence=None,evidenceCode='ECO:0000302')]

def exports():
    gaf='!gaf-version: 2.2\n'+'\t'.join(['UniProtKB','Q96EB6','SIRT1','is_active_in','GO:0005634','PMID:35732914','IDA','','C','','','protein','taxon:9606','20220623','UniProt','',''])+'\n'
    gpad='!gpa-version: 1.1\n'
    for eco,short in [('ECO:0000314','IDA'),('ECO:0000302','')]:
        gpad+='\t'.join(['UniProtKB','Q96EB6','is_active_in','GO:0005634','PMID:35732914',eco,'','','20220623','UniProt','','goEvidence='+short])+'\n'
    return gaf,gpad

def test_null_short_code_preserved_and_all_native_rows_must_match_full_gpad():
    m=module();rows=records();gaf,gpad=exports();original=json.dumps(rows,sort_keys=True)
    q=m.crosscheck_exports(rows,['Q96EB6'],gaf,gpad)
    assert q['native_annotation_rows']==2 and q['GPAD_full_native_semantic_multiset_equal'] and q['GAF_unrepresentable_native_rows']==1
    assert q['GAF_representable_native_rows']==1 and json.dumps(rows,sort_keys=True)==original and rows[1]['goEvidence'] is None

def test_null_short_code_cannot_skip_independent_full_count_or_eco_gate():
    m=module();rows=records();gaf,gpad=exports()
    for bad in [None,gpad.rsplit('\n',2)[0]+'\n',gpad.replace('ECO:0000302','ECO:0000315')]:
        with pytest.raises(ValueError):m.crosscheck_exports(rows,['Q96EB6'],gaf,bad)

def test_representable_gaf_loss_is_still_rejected_even_with_complete_gpad():
    m=module();rows=records();gaf,gpad=exports()
    with pytest.raises(ValueError):m.crosscheck_exports(rows,['Q96EB6'],'!gaf-version: 2.2\n',gpad)

def test_not_form_and_extension_changes_are_still_rejected():
    m=module();rows=records();gaf,gpad=exports()
    for bad in [gpad.replace('is_active_in','NOT|is_active_in',1),gpad.replace('Q96EB6','Q96EB6-2',1),gpad.replace('goEvidence=IDA','goEvidence=IMP')]:
        with pytest.raises(ValueError):m.crosscheck_exports(rows,['Q96EB6'],gaf,bad)
