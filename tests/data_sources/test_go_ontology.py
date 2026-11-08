import importlib.util,pathlib,json
import pytest
ROOT=pathlib.Path.cwd()
def module():
    p=ROOT/'scripts/data_sources_go_ontology.py'
    assert p.is_file(), 'bounded versioned GO ontology parser not implemented'
    s=importlib.util.spec_from_file_location('data_sources_go_ontology',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_fixed25_term_pages_require_complete_stable_count_and_order_metadata():
    m=module();page={'numberOfHits':26,'pageInfo':{'current':1,'total':2,'resultsPerPage':25},'results':[{'id':'GO:%07d'%i} for i in range(25)]}
    assert m.validate_page(page,1,26)==25
    bad=dict(page,results=page['results'][:-1])
    with pytest.raises(ValueError):m.validate_page(bad,1,26)
    with pytest.raises(ValueError):m.validate_page(page,2,26)

def test_term_name_aspect_obsolete_unknown_fields_and_snapshot_are_preserved():
    m=module();term={'id':'GO:0042803','name':'protein homodimerization activity','aspect':'molecular_function','isObsolete':False,'future_field':{'native':[1,2]}}
    node,edges=m.project_term(term,'raw.json',0,'ontology2026-09-27')
    assert node['go_id']=='GO:0042803' and node['aspect']=='molecular_function' and not node['is_obsolete']
    assert json.loads(node['native_term_json'])==term and node['ontology_version']=='ontology2026-09-27' and edges==[]

def test_native_children_edge_direction_is_child_to_parent_and_history_is_not_current_graph():
    m=module();term={'id':'GO:0000001','name':'source','isObsolete':True,'children':[{'id':'GO:0000002','relation':'part_of'}],'history':[{'action':'Deleted','category':'RELATION','text':'is_a GO:0000003'}]}
    node,edges=m.project_term(term,'raw.json',2,'v')
    assert node['aspect'] is None and node['is_obsolete']
    assert edges==[{'child_id':'GO:0000002','relation':'part_of','parent_id':'GO:0000001','ontology_version':'v','source_raw_path':'raw.json','source_result_index':2}]

def test_ontology_snapshot_cannot_be_silently_replaced_with_annotation_date():
    m=module();uri='http://purl.obolibrary.org/obo/go/releases/2026-09-27/extensions/go-plus.ofn'
    d={'go':{'timestamp':'2026-09-27','version':uri}};m.check_version(d,'2026-09-27',uri)
    with pytest.raises(ValueError):m.check_version(d,'2026-07-28',uri)
