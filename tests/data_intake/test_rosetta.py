import importlib.util,sys,pathlib
sys.path.insert(0,str(pathlib.Path(__file__).parents[2]/'scripts'))
spec=importlib.util.spec_from_file_location('rosetta',pathlib.Path(__file__).parents[2]/'scripts/data_intake_rosetta.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def test_chemical_link_uses_exact_sample_and_source_coordinate_without_units():
 a=m.link_key('CDRPBIO-BBBC036-Bray','CellPainting',{'Metadata_Sample_Dose':'BRD-K123-001_50.0','pert_type':'trt'})
 b=m.link_key('CDRPBIO-BBBC036-Bray','L1000',{'pert_sample_dose':'BRD-K123-001_50','pert_type':'trt'})
 assert a==b
 assert a!=m.link_key('CDRPBIO-BBBC036-Bray','L1000',{'pert_sample_dose':'BRD-K123-002_50','pert_type':'trt'})
 assert m.link_key('CDRPBIO-BBBC036-Bray','L1000',{'pert_sample_dose':'BRD-K123-001_50','pert_type':'control'}) is None

def test_genetic_link_cannot_collapse_clone_or_cell_context():
 a=m.link_key('LUAD-BBBC041-Caicedo','CellPainting',{'Metadata_pert_mfc_id':'BRDN00001','Metadata_cell_id':'A549','pert_type':'trt'})
 b=m.link_key('LUAD-BBBC041-Caicedo','L1000',{'pert_mfc_id':'BRDN00001','cell_id':'U2OS','pert_type':'trt'})
 assert a!=b
 assert m.link_key('LUAD-BBBC041-Caicedo','L1000',{'pert_mfc_id':'','cell_id':'A549','pert_type':'trt'}) is None
