"""Exact inactive v6 + BETSE owner authority. No numerical source payloads."""

from hashlib import sha256
import json
from pathlib import Path


REL = Path('outputs/betse-v6-owner-composition-v1')
V6 = 'outputs/native-common-chemical-quarantine-v6-20261108/'
CTOR = 'outputs/v6-owner-construction-staged-20261108/'
V5 = 'outputs/betse-joint-homology-v5-v3/'
FACTORY = 'outputs/betse-production-owner-patch-v1/receipt.json'
PHYS = 'outputs/betse-physiology-native-capture-run-v4/'
PREPARED = 'outputs/betse-membrane-prepared-targets-v2/receipt.json'
MODEL = 'outputs/registered-head-composition-20261108/counts/receipt.json'
CURRENT = 'outputs/v6-fresh-owner-20261108/receipt.json'
PINS = {
    V6+'receipt.json': '74e687b0f2461a3c204faee7cd9c3b59b00c1e539aea399e6e1138f42485b4a3',
    V6+'scoped-source-registry-snapshot.json': 'c47f7a2b55da718c64f9910167830e5667cf4806249889d57eae336747e34ec2',
    V6+'common-scope-grid.json': '216755f233448f6037f198ac0aa5b9bf75242efed99903723ae51122f7a19a12',
    V6+'route-binding-plan.json': '36326812f7529a565ca26dff5a58bb16cefd2a56662f50c914914f22c51e6e7b',
    CTOR+'construct_v6.py': 'a70b1d1a9a2975b49e90198e4768113b48849b35ec4cb90136287baf870402f0',
    CTOR+'v6_entry.py': '3e87d9453c0945870bf5b0603621c10bfc8bb98ce15c132fea21e25a88ce3230',
    CTOR+'runtime-pins.json': 'e8a135486c9b42ce92f6af8957932a37ba65893985767be732442afe1eec231d',
    V5+'receipt.json': 'caa5090e0f540689695dcc3117e63f2992eb62c426693cf21635ccbfcc636a7f',
    V5+'betse_membrane.py': '16b61dacda2254b0ec3631366f96b6922e561ec8bdddec14d0531a3037176801',
    V5+'joint-homology-v5-snapshot.json': '5af1a09f547dc81db4ea8e722814973ca170ce4d9d4f1e31efe00685ce3d520c',
    FACTORY: 'c84c01fe0ed3b4ae05e74a0c05f8a5a025ca0e0a04cdf906bfa2fc5feaa3bdd2',
    PREPARED: '9656464856da4b3e3724ab899edfb1cc04a862fec2d6c45a5ae0261d2299ba1d',
    MODEL: '46af756d5ec163a5fc46e8f0e236a2dbb9acdc5226f79c5d890084b1e5904103',
    CURRENT: 'a758ceeec21b7fbca9fc265f6e06b0d759337b4c73f7a5cb8458dd5c415c9efd',
    PHYS+'capture.json': '69cadaab5ba077fe7dee103e00a5fceb6077e82876cade288eb65addb49159ee',
    PHYS+'readback.json': '378de4ee2957ae5564b2e536e1df3ec587f74bc43bfd25c3fe5b6a6f3fdf0372',
}
# Stage-local copies of the exact already-sealed data. Paths are scope-bound, not model features.
STAGE_PINS = {'condition-proof-final.json':'79a008fb5c135d1d0c610286fbdc3c325b5a0478150bc55c84d2a228df6e5eff',
              'owner-review-proposal.json':'448d1b689e63129643388ad24fc57e02aec5a223a3aaccfca202befa8630b8a9',
              'physiology_source.py':'0d58463f6efb9f9fe5533919cd929edf3ca874ebbe088ac0525b7e7f686ed02e',
              'review_gate.py':'49d3bae45f355f4a8cd555116fec971c47485bbc73b2236bd14700ce9ed9ef78',
              'physiology_head.py':'52a611836ef8d24e372c66165a0962ff388ec8041cefc89a0eba0466b45d311f'}
WIDTHS = (36,52,156)
TRIAL = 'hp-01-2026110201'
REPLAY_PIN = PINS[V6+'receipt.json']
ROUTE = 'inactive-betse-train-candidate'


def need(ok, msg):
    if not ok:
        raise ValueError(msg)


def sha(raw):
    return sha256(raw).hexdigest()


def wire(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def ref(path, pin=None):
    return {'path':str(path),'sha256':PINS[str(path)] if pin is None else pin}


def pinned(root, spec, cap=512*1024):
    p=root/spec['path']
    need(p.is_absolute() and p.resolve()==p and p.is_relative_to(root) and p.is_file()
         and p.stat().st_size<=cap,'Exact bounded immutable composition file required')
    raw=p.read_bytes()
    need(sha(raw)==spec['sha256'],'Exact pin differs: '+str(spec['path']))
    return raw


def doc(root,spec,cap=512*1024):
    return json.loads(pinned(root,spec,cap))


def build_joint_document(root, replay_pin):
    need(replay_pin==REPLAY_PIN,'Reviewed exact completed v6 replay pin required')
    replay=doc(root,ref(V6+'receipt.json'))
    snap=doc(root,ref(V6+'scoped-source-registry-snapshot.json'))
    plan=doc(root,ref(V6+'route-binding-plan.json'))
    grid=doc(root,ref(V6+'common-scope-grid.json'))
    prior=doc(root,ref(V5+'receipt.json'))
    factory=doc(root,ref(FACTORY))
    prepared=doc(root,ref(PREPARED))
    model=doc(root,ref(MODEL))
    current=doc(root,ref(CURRENT))
    condition=doc(root,ref(REL/'condition-proof-final.json',STAGE_PINS['condition-proof-final.json']))
    proposed=doc(root,ref(REL/'owner-review-proposal.json',STAGE_PINS['owner-review-proposal.json']))
    cap=doc(root,ref(PHYS+'capture.json'))
    rb=doc(root,ref(PHYS+'readback.json'))
    snapshot=ref(V6+'scoped-source-registry-snapshot.json')
    need(replay['scoped_registry_source_snapshot']==plan['source_snapshot']==grid['source_snapshot']==current['source_snapshot']==snapshot
         and replay['source_routes']==62 and replay['replayed_source_updates']==5000 and replay['source_windows_consumed']==5940
         and replay['matched_validation_scopes']==5 and replay['training_authorized'] is False
         and replay['validation_reference_qualified'] is replay['role_changes'] is snap['global_split_accepted'] is False
         and current['training_authorized'] is False and current['registered_heads']==27
         and current['route_replay']==ref(V6+'receipt.json')
         and all('betse' not in route['task_key'].casefold() for route in plan['train_routes']),
         'Exact current inactive v6 routes and owner snapshot required')
    route=prepared['staged_binding']
    need(prior['actual_28_head_factory_rebuilt'] and prior['active_roles']==prior['betse_scheduled_routes']==0
         and prior['joint_snapshot']==ref(V5+'joint-homology-v5-snapshot.json')
         and factory['active_roles']==factory['data_admissions']==0
         and route['role']==ROUTE and route['training_authorized'] is False and route['active_roles']==0
         and route['coordinates']==[['sim_cell_mean_membrane_voltage_V','V','cell',.1,11571],
                                   ['sampled_potassium_membrane_diffusion_m2_s','m^2/s','membrane',1e-18,63726],
                                   ['sampled_gap_junction_open_fraction','fraction','membrane',1.0,63726]]
         and len(route['case_order'])==9 and factory['BETSE_binding']['route']==route
         and model['widths']==list(WIDTHS), 'Exact verified BETSE incumbent source, axes and factory required')
    expected=dict(zip(WIDTHS,model['scoped_counts'],strict=True))
    members=factory['width_members']
    need(all(m['baseline']==expected[m['width']] and m['BETSE_increment']==24*m['width']+75
             and m['composed']==m['baseline']+m['BETSE_increment'] and m['incremental_tensors']==6
             and m['shared_annotation_tensors_once']==45 for m in members)
         and tuple(m['width'] for m in members)==WIDTHS, 'Exact prior width roles and count contract differs')
    trial=next(t for t in grid['trial_grid'] if t['trial_id']==TRIAL)
    need(tuple(m['width'] for m in trial['models'])==WIDTHS
         and all(m['configuration']['source_snapshot_sha256']==snapshot['sha256']
                 and sha(wire(m['configuration']))==m['configuration_sha256'] for m in trial['models']),
         'Exact common-grid v6 width trial required')
    b=condition['physiology_binding']
    need(b['capture_sha256']==PINS[PHYS+'capture.json'] and b['training_admission'] is False
         and cap['active_role'] is None and rb['active_roles']==0
         and rb['sim_observed_count']==condition['readback_pins']['target_voltage_positions']==1330
         and condition['readback_pins']['readback_sha256']==PINS[PHYS+'readback.json']
         and condition['readback_pins']['target_fields_accessed_here']==0
         and proposed['owner_review_status']=='pending_separate_owner_attestation'
         and proposed['condition_runtime_binding']==b and not proposed['evaluator_dispatch_authorized'],
         'Exact separate inactive physiology condition proof and pending review required')
    return {'schema':'Mora3.StagedV6BETSEJointOwnerScope/v1','v6_source_snapshot':snapshot,
            'v6_replay':ref(V6+'receipt.json'),'v6_route_plan':ref(V6+'route-binding-plan.json'),
            'v6_width_grid':ref(V6+'common-scope-grid.json'),'v6_baseline_owner':ref(CURRENT),
            'inherited_v6_routes':62,'inherited_v6_source_windows':5940,'inherited_v6_updates':5000,
            'betse_scheduled_routes':0,'betse_selection_routes':0,'active_roles':0,'training_authorized':False,
            'global_split_accepted':False,'matched_validation_reference_qualified':False,
            'factory':ref(FACTORY),'v5_constructor_proof':ref(V5+'receipt.json'),'incumbent_prepared_route':ref(PREPARED),
            'incumbent_betse_role':ROUTE,'incumbent_native_axes':route['coordinates'],
            'incumbent_finite_axis_sha256':route['finite_axis_sha256'],
            'width_trial':TRIAL,'v6_trial_members':trial['models'],'expected_width_contract':members,
            'physiology_source_binding':b,'physiology_owner_attestation_present':False,
            'physiology_condition_proof':ref(REL/'condition-proof-final.json',STAGE_PINS['condition-proof-final.json']),
            'physiology_owner_review_proposal':ref(REL/'owner-review-proposal.json',STAGE_PINS['owner-review-proposal.json']),
            'physiology_registration':'separate inactive diagnostic; not a registered parameter branch'}
