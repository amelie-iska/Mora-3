import importlib.util,json
from pathlib import Path
import pytest
root=next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').exists())
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
base=None
if (root/'scripts/data_intake_curate_goa_snapshot_v2.py').exists():base=load('curategoagpadv2',root/'scripts/data_intake_curate_goa_snapshot_v2.py')
projection=load('projectnullgo',root/'scripts/data_intake_goa_projection.py');source=load('sourcegoav2gpad',root/'scripts/data_sources_goa_v2.py')

def test_null_short_code_is_retained_with_ECO_and_no_inferred_short_code():
 a=dict(id='a',geneProductId='UniProtKB:P00001',qualifier='is_active_in',goId='GO:0005634',goName=None,goAspect='cellular_component',goEvidence=None,evidenceCode='ECO:0000302',reference='PMID:1',date='20200101',assignedBy='UniProt',withFrom=None,extensions=None)
 r=projection.annotation_projection(a);assert r['GO_evidence'] is None and r['ECO_evidence']=='ECO:0000302' and r['negated'] is False and r['missing_is_negative'] is False

def test_full_GPAD_allows_null_shortcode_only_when_full_ECO_semantics_match():
 # GAF has no row for this assertion; GPAD retains ECO and the blank goEvidence property.
 a=dict(id='a',geneProductId='UniProtKB:P00001',qualifier='is_active_in',goId='GO:0005634',goEvidence=None,evidenceCode='ECO:0000302',reference='PMID:1',date='20200101',assignedBy='UniProt',withFrom=None,extensions=None)
 gaf='!gaf-version: 2.2\n';gpad='!gpa-version: 1.1\n'+'\t'.join(['UniProtKB','P00001','is_active_in','GO:0005634','PMID:1','ECO:0000302','','','20200101','UniProt','','goEvidence='])+'\n'
 assert base is not None,'GPAD-aware corpus projection path missing'
 proof=base.validate_native_exports([a],gaf,gpad,['P00001'],source)
 assert proof['GPAD_full_native_semantic_multiset_equal'] is True and proof['GAF_unrepresentable_native_rows']==1
 with pytest.raises(ValueError):base.validate_native_exports([a],gaf,None,['P00001'],source)
 with pytest.raises(ValueError):base.validate_native_exports([a],gaf,gpad.replace('ECO:0000302','ECO:0000269'),['P00001'],source)
