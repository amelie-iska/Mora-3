"""Bounded CPU-only reader handoff, no optimizer/model-code changes."""
from pathlib import Path
import sys,json,hashlib,datetime,time
R=Path('/home/iska/Documents/amelie/bio/Mora-3');sys.path[:0]=[str(R/'src'),str(R/'scripts')]
from data_intake_corpus import read_corpus_tasks
from mora3.data.nucleotide_prior import read_nucleotide_priors
from mora3.data.rfam_family import read_rfam_families
from mora3.data.molecule_profile import read_molecule_profile_views
from mora3.data.goa_packets import verify_goa_packet
M=R/'outputs/data-intake/20261101-RNAcentral-pilot-union-v1/master-corpus-manifest-v18-RNAcentral-native-pilot-v1.json';O=M.parent/'model-handoff-v1';O.mkdir(exist_ok=True);start=time.perf_counter()
def ref(p):return dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
modules={name:ref(R/'src/mora3/data'/name) for name in ['nucleotide_prior.py','rfam_family.py','molecule_profile.py','goa_packets.py']};mh=ref(M);results=[]
for source in ['DNA-native-reference-prior','Rfam15.1-native-reference-prior']:
 rows=list(read_nucleotide_priors(M,root=R,expected_manifest_sha256=mh['sha256'],source_family=source,limit=2));assert len(rows)==2
 results.append(dict(source=source,module='nucleotide_prior.py',views=2,data_projection='boundednativeview_and_fulltarget_roundtrip',model_optimizer_steps=0,canonical_view_bytes=[len(r.canonical_bytes()) for r in rows],inputs_keys=[sorted(r.inputs()) for r in rows]))
for direction in ['prediction','generation']:
 family_manifest=R/'outputs/data-intake/20261101-rfam-family-annotation-v1/curated-v1/manifest.json';fm=json.loads(family_manifest.read_text());master=json.loads(M.read_text());assert all(master['effective_group_lookup'][key]==value for key,value in fm['effective_group_lookup'].items());rows=list(read_rfam_families(family_manifest,root=R,expected_manifest_sha256=ref(family_manifest)['sha256'],direction=direction,limit=2));assert len(rows)==2
 results.append(dict(source='Rfam15.1-source-family-prior',module='rfam_family.py',reader_manifest=ref(family_manifest),authoritative_groups_equal_current_master=True,direction=direction,views=2,inputs_keys=[sorted(r.inputs()) for r in rows],target_keys=[sorted(r.targets()) for r in rows]))
for modality in ['CellPainting','L1000']:
 for direction in ['molecule_to_profile','profile_to_molecule']:
  rows=list(read_molecule_profile_views(M,root=R,expected_manifest_sha256=mh['sha256'],modality=modality,direction=direction,limit=2));assert len(rows)==2
  results.append(dict(source='Rosetta-CDRP-molecule-profile-'+modality,module='molecule_profile.py',direction=direction,views=2,inputs_keys=[sorted(r.inputs()) for r in rows],target_keys=[sorted(r.targets()) for r in rows]))
try:
 list(read_nucleotide_priors(M,root=R,expected_manifest_sha256=mh['sha256'],source_family='RNAcentral-reference-priors',limit=2))
except ValueError as exc:
 results.append(dict(source='RNAcentral-reference-priors',module='nucleotide_prior.py',views=0,status='explicit_source_or_domain_gate_not_yet_registered',reason=str(exc)))
else:raise AssertionError('RNAcentral support changed; inspect before reporting unsupported')
raw=list(read_corpus_tasks(M,root=R,source='UniProt-GOA-assertion-packets',supported_only=False,limit=2));verified=[verify_goa_packet(row,root=R) for row in raw];assert len(verified)==2;results.append(dict(source='UniProt-GOA-assertion-packets',module='goa_packets.py',views=2,scope='raw_source_packet_verification_only;thisverifierdoesnotgrantsequence/group/tasktrainingeligibility',inputs_keys=[sorted(v.inputs()) for v in verified],target_keys=[sorted(v.targets()) for v in verified]))
assert ref(M)==mh
for name,expected in modules.items():assert ref(R/'src/mora3/data'/name)==expected
result=dict(schema='mora3.CurrentBoundedNativeReaderProbe/v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),master=mh,reader_modules_unchanged_during_probe=modules,results=results,view_count=sum(x['views'] for x in results),full_current_pool_consumption=False,neural_objectives_or_training_updates_not_tested=True,source_quality_or_admission_not_promoted=True,historical_generic_flags_unchanged=True,source_homology_blindheldout_and_mixture_gates_unchanged=True,model_code_changed=False,optimizer_steps=0,seconds=time.perf_counter()-start)
(O/'runtime-reader-probe-v1.json').open('x').write(json.dumps(result,indent=2)+'\n');print('COMPLETE',result['view_count'],result['seconds'])
