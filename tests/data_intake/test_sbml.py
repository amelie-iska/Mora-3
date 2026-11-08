import importlib.util
from pathlib import Path
import xml.etree.ElementTree as ET
import pytest
p=next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').exists())/'scripts/data_intake_sbml.py'
s=importlib.util.spec_from_file_location('sbml',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_qualifiers_belong_to_enclosing_element_not_nested_entities():
 x=ET.fromstring('''<model xmlns="http://www.sbml.org/sbml/level3/version1/core" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:bq="http://biomodels.net/biology-qualifiers/"><annotation><rdf:RDF><rdf:Description><bq:is><rdf:Bag><rdf:li rdf:resource="pathway"/></rdf:Bag></bq:is></rdf:Description></rdf:RDF></annotation><listOfSpecies><species><annotation><rdf:RDF><rdf:Description><bq:is><rdf:Bag><rdf:li rdf:resource="complex"/></rdf:Bag></bq:is><bq:hasPart><rdf:Bag><rdf:li rdf:resource="protein"/></rdf:Bag></bq:hasPart><bq:isHomologTo><rdf:Bag><rdf:li rdf:resource="ortholog"/></rdf:Bag></bq:isHomologTo></rdf:Description></rdf:RDF></annotation></species></listOfSpecies></model>''')
 assert m.biological_qualifiers(x)=={'is':['pathway']}
 q=m.biological_qualifiers(next(e for e in x.iter() if e.tag.endswith('species')))
 assert q=={'is':['complex'],'hasPart':['protein'],'isHomologTo':['ortholog']}
 assert m.exact_source_resources(q)==['complex']
def test_SBML3_missing_stoichiometry_unknown_and_modifiers_not_reactants():
 assert m.participant('reactant',{'species':'A'})['stoichiometry_numeric'] is None
 assert m.participant('modifier',{'species':'A'})['role']=='modifier'
 assert m.participant('product',{'species':'B','stoichiometry':'0.5'})['stoichiometry_numeric']==.5
 for value in ['0','-1','nan','inf']:
  with pytest.raises(ValueError):m.participant('reactant',{'species':'A','stoichiometry':value})

def test_nested_qualifier_evidence_is_not_enclosing_entity_identity():
 x=ET.fromstring('''<species xmlns="http://www.sbml.org/sbml/level3/version1/core" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:bq="http://biomodels.net/biology-qualifiers/"><annotation><rdf:RDF><rdf:Description><bq:is><rdf:Bag><rdf:li rdf:resource="entity"/></rdf:Bag><bq:isDescribedBy><rdf:Bag><rdf:li rdf:resource="paper"/></rdf:Bag></bq:isDescribedBy></bq:is></rdf:Description></rdf:RDF></annotation></species>''')
 assert m.biological_qualifiers(x)=={'is':['entity']}

import copy,json

def graph():return dict(compartments={'c':{'attributes':{'name':'cytosol'},'qualifiers':{}}},species={'A':{'attributes':{'name':'substrate','compartment':'c','sboTerm':'SBO:0000247'},'qualifiers':{'is':['https://identifiers.org/chebi:CHEBI:1']}},'B':{'attributes':{'name':'product','compartment':'c'},'qualifiers':{}}},reactions=[{'attributes':{'id':'reaction-secret','name':'A converts to B','reversible':'false'},'notes':['targetdescription'],'qualifiers':{'isDescribedBy':['paper']},'participants':[{'species':'A','role':'reactant','stoichiometry_source':'1','source_attributes':{'id':'input_A','sboTerm':'SBO:0000010'}},{'species':'B','role':'product','stoichiometry_source':None,'source_attributes':{'id':'output_B','sboTerm':'SBO:0000011'}}]}])
def test_masked_role_stoichiometry_sbo_and_text_cannot_reach_inputs():
 g=graph();a=m.masked_pathway_task(g,[0]);d=copy.deepcopy(g);d['reactions'][0]['attributes']['name']='mutatedtargettext';d['reactions'][0]['notes']=['changed'];d['reactions'][0]['participants'][0].update(role='product',stoichiometry_source='42');d['reactions'][0]['participants'][0]['source_attributes']['sboTerm']='revealingrole';b=m.masked_pathway_task(d,[0]);assert a.input_bytes()==b.input_bytes();assert a.canonical_bytes()!=b.canonical_bytes();assert 'reaction-secret' not in a.input_bytes().decode() and 'input_A' not in a.input_bytes().decode()
def test_unknown_stoichiometry_mask_and_qualifier_role_preserved():
 t=json.loads(m.masked_pathway_task(graph(),[0]).canonical_bytes());targets=t['targets']['masked_reaction_participants'][0];assert any(x['stoichiometry_source'] is None for x in targets);assert all(x['role'] is None and x['stoichiometry_source'] is None for x in t['conditions']['reactions'][0]['participants']);assert len(t['conditions']['species'])==2

def test_legacy_Chebi_and_UniProt_context_retained_with_qualifier_scope():
 c='https://www.ebi.ac.uk/chebi/searchId.do?chebiId=CHEBI:15377';u='http://purl.uniprot.org/uniprot/P00001'
 q={'is':[c,'https://identifiers.org/reactome:R-HSA-1'],'hasPart':[u],'isHomologTo':['https://identifiers.org/uniprot:MOUSE']}
 assert m.qualified_biological_context(q)=={'is':[c],'hasPart':[u]}
