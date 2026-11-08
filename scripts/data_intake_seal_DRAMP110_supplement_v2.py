from pathlib import Path
import ast,copy,datetime,hashlib,json,os,sys,time
R=Path('/home/iska/Documents/amelie/bio/Mora-3');os.chdir(R);sys.path.insert(0,str(R/'scripts'))
from data_intake_DRAMP_symbolic_component_v1 import validate_approval,SCOPE
P=R/'planning/research/data_intake/20261101-DRAMP-offline-preparation-v1';O=P/'writer-execution-review-v2';O.mkdir(exist_ok=False)
def ref(p):
 p=Path(p);h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=h.hexdigest())
def checked(v):
 p=R/v['path'];assert ref(p)==v,v['path'];return p
def load(p):return json.loads(p.read_text())
def write(p,d):p.open('x').write(json.dumps(d,indent=2)+'\n')
prepfile=P/'writer-preparation-v1/preparation-manifest.json';prep=load(prepfile);assert ref(prepfile)['sha256']=='34d291c16e319d5d94da10d9f12cb781948ff933937eadfb767fbd448c6459e4'
draftfile=checked(prep['draft_supplement']);draft=load(draftfile);assert ref(draftfile)['sha256']=='34998c97adf10261297cf371c9c88c9edadebe069f0ea3c378c657ffe79b1c16'
att=R/'outputs/monitoring/attestation-20261101T195813Z-constructor-quiescence-for-DRAMP-supplement-v1/attestation.json';assert ref(att)['sha256']=='dd673d6f834d4d1c3850df73d065cbd08d1c5f10c50e010f6e955e5553a1ef16';a=load(att)
assert a['verified_no_live_constructor_phase'] and a['old_owned_PIDs_absent'] and all(all(not v for v in row.values()) for row in a['wholeSIDs_twiceempty'])
policy=checked(a['active_source_version_policy']);assert ref(policy)['sha256']=='0f77c5310a55161c5d1723c0688b04addc56b5ede7241d6afcb522fd8414133b'
static=R/'outputs/monitoring/review-20261101T192703Z-DRAMP-candidate-writer-static-only-v1/review.json';assert ref(static)['sha256']=='059bd184e16134bf24f8b35b169cce4dbc730f5f9789c8ca33771e09238885f7'
master=checked(a['current_master']);handoff=checked(a['current_handoff'])
progress=R/'planning/data_intake_progress.json';queue=R/'configs/dataset_acquisition_queue.json';expkg=R/'planning/research/data_intake/20261101-expanded-scope-v4/ExpansionRx-post-RNA-exact-union-preparation-v1/frozen-review-package-v1/execution-review-package.json';assert ref(expkg)['sha256']=='c151e6dacc99dc668714f52b1b02370a7d6d2bcc19f41d6a3a60aa890ee097da'
expplanfile=R/load(expkg)['plan']['path'];expplan=load(expplanfile)
for pin in expplan['input_pins']:
 if pin['path'] in ['planning/data_intake_progress.json','configs/dataset_acquisition_queue.json']:checked(pin)
assert load(progress)['current_master_corpus_manifest']==a['current_master']['path'] and load(progress)['running_jobs']==[]
assert load(queue)['current_corpus_manifest']==a['current_master']['path'] and len(load(queue)['entries'])==114
for p in a['closed_receipt_refs']:checked(p)
for p in draft['code_refs']+list(draft['inputs'].values())+[draft['parent_plan'],*prep['tests'],prep['synthetic_log']]:checked(p)
# Source/logic bytes remain identical; this is a JSON-supplement operation only.
s=copy.deepcopy(draft);s.update(status='SEALED_EXACT_ROOT_REVIEW_REQUIRED_NOT_APPROVED_NOT_LAUNCHED',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),supersedes_draft_without_overwrite=ref(draftfile),frozen_writer_preparation=ref(prepfile),root_static_review=ref(static))
s['inputs'].update(current_published_master=a['current_master'],current_immutable_handoff=a['current_handoff'],active_inherited_RGD1_source_policy=ref(policy),current_intake_progress=ref(progress),current_acquisition_queue=ref(queue))
s['actual_current_master_snapshot']=dict(master=a['current_master'],handoff=a['current_handoff'],current_v23_task_views=81054784,current_v23_shards=82141,full_dataset_ready=False,actual_model_use=None,actual_model_tokens=None,weights=None)
s['output_namespace']='outputs/data-intake/20261101-ADMET-corpus-v1/DRAMP4-stability-source-symbolic-candidate-component-v1';s['exclusive_log']=s['output_namespace']+'.log';s['launch_receipt']=str((O/'DRAMP110-candidate-component-actual-launch-v2.json').relative_to(R))
reserved_exp_receipt='planning/research/data_intake/20261101-expanded-scope-v4/ExpansionRx-post-RNA-exact-union-preparation-v1/ExpansionRx-v24-exact-union-launch-v1.json'
absent=[expplan['output_namespace'],expplan['new_log'],reserved_exp_receipt]
s['quiescence']=dict(root_attestation=ref(att),old_owned_PIDs=a['old_owned_PIDs'],closed_receipt_refs=a['closed_receipt_refs'],must_remain_absent_paths=absent,state='ROOT_ATTESTED_C1_GO_C2_PUBLISHER_NORMAL_CLOSED;future_launch_requires_fresh_all-owner_preflight',mode='sequential_no_other_live_constructor_phase;ExpansionRxmustremainunlauncheduntilDRAMPnormalclosure',ExpansionRx_unapproved_review_package=ref(expkg),ExpansionRx_exact_plan=ref(expplanfile),reserved_future_ExpansionRx_launch_receipt=reserved_exp_receipt,absence_scope='Exact futureExpansionRx output/log and canonicalreserved launchreceipt; log must be openedexclusively BEFOREfuturePopen; no guess of arbitrary alternative receipts. Fresh /proc scan is additionally required at actual launch.')
s['active_source_policy_precedence']=dict(policy=ref(policy),scope='RGD1v6 energy roles remain hardclosed, regardless historicaladmittedflags; this peptide report component does not useRGD1labels or liftits gate',runtime_enforcement_verified=False)
s['runtime_phase_guards_explained']=dict(writer_existing_guards='Eachsource row checks alloldPIDsabsent, closedreceiptwholeSHAs, progresslivejobs and exactfutureExpansionRx output/log/receiptabsence. Firstguard is beforeproject_row.',fresh_launch_preflight='Separate root exactapproval then repeatfullsource/code/supplementpins, currentv23/intakebytes, alloldPIDs/SIDsabsent, noother live data_intake script/process except currentpreflight/launcher ancestors, freshoutput/log/receipt and allfuturephaseabsencepaths, CPU/RAM/free policy; preserve failures and donotspawn onfailure.',race_boundary='Root sequential lease/coordination is required. No filesystemlock or crossagent processsignalling is introduced. Any futurephase detected by writer stops only this candidate writer; no repair/retry/phasechain.')
s['candidate_and_split_roles']=dict(complete_source_rows=110,source_local_modification_and_descriptor_groups='NONAPPLYING; no assumed samephysicalmaterial/ordinaryprotein sequence/homology/study/assay equivalence',split=None,original_official_holdouts_and_global_exposure='not_evaluated; not an allocated train/evalcorpus',source_evidence='database_curated_native_report;notoriginalprimaryassayindependentvalidation',report_vs_scalar='source_report plus only strictwhole-string nativehalf-life scalar; nonadditive views, exactcensor/unit spelling retained',missing_reports='missing/nonstring/blank report cannotcreate supervisedlabel',physical_ADMET=False,global_task_admission=False,model_use=False,full_dataset_ready=False,training=False)
sp=O/'execution-supplement-sealed-v2.json';write(sp,s)
notapproved=dict(approved=False,scope=SCOPE,supplement_sha256=ref(sp)['sha256'],script_sha256=s['script_ref']['sha256'],launcher_sha256=s['launcher_ref']['sha256'],parent_plan_sha256=s['parent_plan']['sha256'],physical_ADMET=False,task_admission=False,master_publication=False,model_use=False,model_training=False,full_dataset_ready=False,reason='NOT_APPROVED template; separateexactrootapprovalandfreshquiescencepreflight required; no actual110projection')
write(O/'NOT_APPROVED-template.json',notapproved)
# Validate exact current supplement bindings and AST, not any source candidate.
allpins=[s['parent_plan'],*s['inputs'].values(),*s['code_refs'],ref(att),*a['closed_receipt_refs'],ref(static),ref(prepfile),ref(draftfile),ref(expkg),ref(expplanfile),*prep['tests'],prep['synthetic_log']]
allpins=list({p['path']:p for p in allpins}.values())
for pin in allpins:checked(pin)
codepaths=[checked(p) for p in s['code_refs']+prep['tests']]
for p in codepaths:ast.parse(p.read_bytes(),filename=str(p))
for path in [s['output_namespace'],s['exclusive_log'],s['launch_receipt'],*absent]:assert not (R/path).exists(),path
for pid in a['old_owned_PIDs']:assert not (Path('/proc')/str(pid)).exists(),pid
ancestors={os.getpid()};pid=os.getpid()
while pid>1:
 try:pid=int((Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[1]);ancestors.add(pid)
 except (FileNotFoundError,ProcessLookupError):break
live=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name) in ancestors:continue
 try:argv=[os.fsdecode(x) for x in (p/'cmdline').read_bytes().split(b'\0') if x]
 except (FileNotFoundError,PermissionError,ProcessLookupError):continue
 if any(Path(x).name.startswith('data_intake_') and Path(x).name.endswith('.py') for x in argv):live.append(dict(pid=int(p.name),argv=argv))
assert not live,live
sids=[]
for _ in range(2):
 observed={str(x):[] for x in [3756182,4039642,4086786,4148373]}
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:sid=os.getsid(int(p.name))
  except ProcessLookupError:continue
  if str(sid) in observed:observed[str(sid)].append(int(p.name))
 assert all(not v for v in observed.values());sids.append(observed);time.sleep(.05)
free=os.statvfs(R);mem=next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'));assert free.f_bavail*free.f_frsize>=64*1024**3 and mem>=8*1024**3
pre=dict(schema='mora3.DRAMP110FinalSupplementStaticPreflight/v2',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),supplement=ref(sp),root_attestation=ref(att),all_source_code_helper_test_policy_and_intake_pins_current=len(allpins),AST_files=len(codepaths),old_PIDs_absent=True,SIDs_twiceempty=sids,other_live_constructor_scripts=[],fresh_output_log_launch_and_ExpansionRx_absence_paths=[s['output_namespace'],s['exclusive_log'],s['launch_receipt'],*absent],free_bytes=free.f_bavail*free.f_frsize,host_MemAvailable_bytes=mem,historical_synthetic_checks=16,synthetic_tests_rerun=False,reason_no_rerun='writer/launcher/logic/helpers/tests byte-for-byte unchanged; rootstaticreview059bdalready reran16checks',source_rows_projected=0,candidate_builder_imported_or_called=False,writer_imported_only_for_pure_approval_contract=True,actual_component_or_ExpansionRx_launch=False,full_dataset_ready=False,training=False)
write(O/'preflight-current-pins-AST-quiescence.json',pre)
argv=['/usr/bin/env','PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','NUMEXPR_NUM_THREADS=1','CUDA_VISIBLE_DEVICES=','/usr/bin/taskset','-c','27','/usr/bin/nice','-n','10','/home/iska/miniconda3/envs/tokengt/bin/python','-B',str(R/s['launcher_ref']['path']),'--supplement',str(sp),'--approval','ROOT_EXACT_APPROVAL_PATH_REQUIRED']
write(O/'future-exact-argv.json',dict(argv=argv,launcher_creates_inner_timeout=dict(signal='TERM',kill_after='15s',duration='150s'),writer_internal_seconds=120,launch_isolated_session=True,stdin='DEVNULL',log_exclusive_create=s['exclusive_log'],actual_execution=False))
S=O/'frozen-small-inputs';S.mkdir();snap=[]
for pin in allpins:
 if pin['bytes']>1024*1024:continue
 dest=S/pin['path'];dest.parent.mkdir(parents=True,exist_ok=True);dest.open('xb').write(checked(pin).read_bytes());snap.append(dict(source=pin,snapshot=ref(dest)))
README='''# Sealed DRAMP110 candidate execution supplement v2\n\nPREPARATION ONLY: separate exact rootapproval and a fresh actual launchpreflight are still required. No110-row candidate projection, component, shard, split allocation, master admission/publisher, model use or training ran. The previousDRAFT34998…, offlineparent2729…, writerpreparation34d291… and allsource/logic bytes remain immutable. The unchanged writer775ffc…, launcherc1d78… and logice1d42… already passed rootstaticreview059bd… and16syntheticchecks; those checks were not repeated here. Currentallpins+7ASTs+realquiescence/absence/resourcechecks are recorded separately, not new native-sourcequality evidence.\n\nThe NEW sealed supplement binds rootattestationdd673d…, normalC1/GOv4/C2/publisherclosures, actual publishedv23f8b09… and handoff23cb85…, the two intakeJSONbytes also pinned by the ExpansionRxpackagec151…, current source/context/identity/rights manifests, unchanged code/helpers/tests, and ACTIVE RGD1policy0f77…. The peptide candidate writer does not lift anyRGD1gate; model runtimeenforcement remains separately pending. No giantindexhash, nativeRNAquery or sourceETL replay occurred.\n\nSequentialownership is explicit. Both all8oldPIDsabsence and4wholeSIDs twiceempty are pinned historic evidence, not a future guarantee. At actual launch root must freshly recheck allother liveconstructor script owners, exactcurrentv23/intakepins, outputs/log/launchreceiptabsence and resources. The existing writer checks oldPIDs, allclosedreceiptSHAs, currentprogresslivejobs and futureExpansionRx absence before projection and on every source row. The supplement pins the exact futureExpansionRx output+exclusive log and a reservedcanonical launchreceipt. FutureExpansionRx must open that planned log exclusively beforePopen and stay unlaunched until the DRAMPcandidate writer closes; these guards do not claim to detect arbitrary renamed owner paths. No lockservice or signalling is introduced. Any newphase detected stops this writer only; allpartials are retained with no retry/reset/cleanup/rollback.\n\nBounds are unchanged:CPU27/one thread/nice10,256MiBownpeakRSS,hostavailable8GiB,hardwarefree64GiB,120sinternal/150souter+15skill,noGPU/network/noartificialdiskcap. Existinglauncher creates the isolated child session/DEVNULLstdin/newexclusive log and exactreceipt. The outer futureargv additionally freezes singlethread/GPUhidden/no-bytecode environment without any environmentinstallation or production dependency.\n\nAll110source dispositions/unknown fields/literalmodifiedO/X/stereo/termini descriptors remain source-symbolic. Missing/nonstring/blank reports create no label; Type/Assay roles come from actual110-row reviewed literals. Only complete native report and strictsource-literal scalar candidates are possible, nonadditive and unallocated. Nativeuncertainty/censor/unit text is retained; no physicalADMET, clinicalPK, safety/protein-annotation/homology/material/same-assay or independentprimary-validation claim. NONAPPLYING source-local group proposals remain separate from originalheldouts/globalexposure, both not evaluated. All admission/model/fullready/training gates are false; tokens/weights/exposure are unmeasured.\n\n`future-exact-argv.json` and `NOT_APPROVED-template.json` are concrete review artifacts, not authority. An exact fresh rootapproval must bind supplement+unchangedwriter+launcher+parenthashes and keep all scientific gatesfalse. Preserve every oldfile/source/index/task/master/checkpoint and the pending ExpansionRxpackage; the two intakeJSONbytes were not modified.\n'''
(O/'README.md').open('x').write(README)
for pin in allpins:checked(pin)
write(O/'execution-review-package.json',dict(schema='mora3.DRAMP110SealedSupplementReview/v2',status='SEALED_ROOT_REVIEW_REQUIRED_NO_EXECUTION',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),supplement=ref(sp),writer=s['script_ref'],launcher=s['launcher_ref'],parent_plan=s['parent_plan'],root_attestation=ref(att),static_root_review=ref(static),frozen_writer_preparation=ref(prepfile),current_pins=allpins,preflight=ref(O/'preflight-current-pins-AST-quiescence.json'),future_argv=ref(O/'future-exact-argv.json'),approval_template=ref(O/'NOT_APPROVED-template.json'),README=ref(O/'README.md'),binary_snapshots=snap,source_rows_projected=0,tests_rerun=False,historical_static_tests=16,unchanged_pinned_intakeJSONs=True,ExpansionRx_package_unchanged=ref(expkg),physical_ADMET=False,task_admission=False,master_publication=False,model_use=False,full_dataset_ready=False,training=False))
print(json.dumps(ref(O/'execution-review-package.json')));print('SUPPLEMENT',ref(sp));print('PRECHECK',ref(O/'preflight-current-pins-AST-quiescence.json'))
