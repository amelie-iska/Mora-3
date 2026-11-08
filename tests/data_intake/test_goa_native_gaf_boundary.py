import importlib.util,copy
from pathlib import Path
import pytest
root=next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').exists())
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
m=load('curategoaboundary',root/'scripts/data_intake_curate_goa_snapshot.py');source=load('realgoasource',root/'scripts/data_sources_goa.py')
def fixture():
 a=dict(geneProductId='UniProtKB:P00001',qualifier='NOT|enables',goId='GO:0000001',reference='PMID:1',goEvidence='IDA',date='20200101',assignedBy='UniProt',withFrom=None,extensions=None)
 gaf='\t'.join(['UniProtKB','P00001','TEST','NOT|enables','GO:0000001','PMID:1','IDA','','F','','','protein','taxon:9606','20200101','UniProt','',''])+'\n'
 return a,gaf

def test_actual_gaf_generator_is_compared_as_multiset_and_not_mutation_rejected():
 assert hasattr(m,'validate_native_gaf_annotations'),'native/GAF boundary validation missing'
 a,gaf=fixture();m.validate_native_gaf_annotations([a],gaf,source)
 b=copy.deepcopy(a);b['qualifier']='enables'
 with pytest.raises(ValueError):m.validate_native_gaf_annotations([b],gaf,source)
