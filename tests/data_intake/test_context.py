import importlib.util,pathlib,pytest,sys
sys.path.insert(0,str(pathlib.Path(__file__).parents[2]/'scripts'))
p=pathlib.Path(__file__).parents[2]/'scripts/data_intake_context.py';spec=importlib.util.spec_from_file_location('context',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_kinetics_keeps_exact_units_and_does_not_infer_assay_conditions():
 e={'primaryAccession':'P1','entryType':'UniProtKB reviewed (Swiss-Prot)','annotationScore':4.,'sequence':{'value':'ACDE'},'comments':[{'commentType':'BIOPHYSICOCHEMICAL PROPERTIES','kineticParameters':{'michaelisConstants':[{'constant':2.6,'unit':'uM','substrate':'X','evidences':[{'source':'PubMed','id':'123'}]}]},'phDependence':{'texts':[{'value':'Optimum pH7'}]}}]}
 row=m.kinetic_measurements(e)[0]
 assert row['value']==2.6 and row['unit']=='uM' and row['measurement_type']=='Km'
 assert row['assay_ph'] is None and row['substrate_smiles'] is None
 e['annotationScore']=3.
 with pytest.raises(ValueError):m.kinetic_measurements(e)

XML=b'''<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core"><model id="p" name="P"><listOfSpecies><species id="a"/><species id="b"/><species id="c"/></listOfSpecies><listOfReactions><reaction id="r" reversible="false"><listOfReactants><speciesReference species="a" stoichiometry="2"/></listOfReactants><listOfProducts><speciesReference species="b" stoichiometry="1"/></listOfProducts><listOfModifiers><modifierSpeciesReference species="c" sboTerm="SBO:0000013"/></listOfModifiers></reaction></listOfReactions></model></sbml>'''

def test_sbml_preserves_direction_stoichiometry_and_modifier_role():
 s,r=m.parse_sbml(XML,'p.sbml');row=r[0]
 assert row['reactants'][0]['species']=='a' and row['reactants'][0]['stoichiometry']=='2'
 assert row['products'][0]['species']=='b' and row['modifiers'][0]['species']=='c'
 assert row['reversible_source']=='false'
 assert row['kinetic_law_xml'] is None

def test_sbml_undeclared_participant_fails():
 with pytest.raises(ValueError):m.parse_sbml(XML.replace(b'species="b"',b'species="missing"'),'p.sbml')

def test_null_species_annotations_have_stable_nullable_string_schema():
 a=m.species_table([{'species_id':'a','sbo_term':None}]);b=m.species_table([{'species_id':'b','sbo_term':'SBO:0000252'}])
 assert a.schema==b.schema
 assert str(a.schema.field('sbo_term').type)=='string'
 assert a['sbo_term'].to_pylist()==[None]

def test_annotation_admission_is_task_specific_and_sequence_exact():
 current=dict(Annotation='4.0',Reviewed='reviewed',Sequence='ACDE')
 a=m.admission_decision(current,'ACDE',False)
 assert a['sequence_annotation_quality_pass'] and not a['molecular_pair_quality_pass']
 b=m.admission_decision(current,'ACDF',True)
 assert not b['sequence_annotation_quality_pass'] and not b['molecular_pair_quality_pass']
 c=m.admission_decision(current,'ACDE',True)
 assert c['sequence_annotation_quality_pass'] and c['molecular_pair_quality_pass']
