"""Metadata-only preflight for the witnessed native 29-head source factory."""
from dataclasses import asdict, dataclass
from hashlib import sha256
import argparse
import json
from pathlib import Path

JOINT = 'outputs/betse-v6-owner-composition-v1/'
ORIGINAL = 'outputs/native-common-chemical-quarantine-v6-20261108/'
SIZING = 'outputs/sccello-raw-full-family-sizing-20261108/'
ENDPOINTS = 'outputs/sccello-raw-full-family-cuda-betse-constructor-rehearsal-20261108-v3/'
COUNT = 'outputs/registered-head-composition-20261108/counts/'
REGISTRATION = 'outputs/raw-sccello-native-owner-registration-20261108-frozen-axes/'
CANDIDATE = 'outputs/v6-freesolv-train-only-successor-20261108-v9/'
PINNED = {
    JOINT+'joint-owner-snapshot.json':'79d328f542d1158581875ef325e56f213b407992f95db6ee0c7352a3459669d4',
    ORIGINAL+'scoped-source-registry-snapshot.json':'c47f7a2b55da718c64f9910167830e5667cf4806249889d57eae336747e34ec2',
    ORIGINAL+'receipt.json':'74e687b0f2461a3c204faee7cd9c3b59b00c1e539aea399e6e1138f42485b4a3',
    ORIGINAL+'route-binding-plan.json':'36326812f7529a565ca26dff5a58bb16cefd2a56662f50c914914f22c51e6e7b',
    SIZING+'measurement.json':'2d129945f4f2114ae5951828b068da22c94e0915c8428aaf53b8df87f31b35ec',
    ENDPOINTS+'plan.json':'21bdd32098bf21c9e381c01bc8d923fba70798ef7e12b51b816fed36c192c9a2',
    ENDPOINTS+'seal.json':'6c607588189a6833c734605d7deb2bbd584ce008822eb602ffea97527d480dc6',
    REGISTRATION+'run/proof.json':'145273a4ef7f4f21041201469f96441c9b7933e7748a78c2f2865e8d8a53d2d5',
    COUNT+'receipt.json':'46af756d5ec163a5fc46e8f0e236a2dbb9acdc5226f79c5d890084b1e5904103',
    COUNT+'extended-source-snapshot.json':'8c1803ed4aac1967004fd539d23e24ce36f351537803e15678747825de2d23b4',
    CANDIDATE+'receipt.json':'f08ae0d452000103fbe04651081347f0d066b9833c310323ed763562cb0b5661',
}
OWNER_FAMILY = '759600b7de0eebd17bfaad184236bd360ee905d6efc9b1c8f07387ad1df4a710'
WIDTHS = (24, 36, 130)
TOTALS = (10145408, 15944576, 80283064)
RAW_FIT = {
    'fit.json':'3bc95f7a30f00e8158f6d9c11d95806aa53caf80ec06f020de6bfb70bdb9f0e2',
    'fit-selection.json':'baddd3c1981f661fe7360065d5ef9b07b95ffa012b0dd0d46ff7a92697c8c9f3',
    'fit-fullsort.json':'11c73b866603ffecb1fb28b8443e00b927dd87ce948991d6b0c9fbce2a4d0254',
}
ASSAYS = ('CEMA Figure 8g','Sempou stage 9','Sempou stage 10')
MAX_METADATA_BYTES = 4*2**20
CANDIDATE_SNAPSHOT = '1091e3f27d57cb70203ab9fce1136836d0cd96eef955b6682a3e6264abadc733'


@dataclass(frozen=True)
class Ref:
    path: str
    sha256: str


@dataclass(frozen=True)
class Native29SourcePlan:
    owner_snapshot: Ref
    original_route_snapshot: Ref
    candidate_route_snapshot: Ref
    native_family_sha256: str
    constructor_proofs: tuple[Ref,...]
    source_assets: Ref
    bioelectric: Ref
    profile_group_manifest: Ref
    raw_gene_extent: int
    owner_widths: tuple[int,...]
    owner_parameter_counts: tuple[int,...]
    constructor_coordinate_shards: int
    constructor_train_readers: int


def require(ok, message):
    if not ok:
        raise ValueError(message)


def _unique_pairs(items):
    result={}
    for key,value in items:
        require(key not in result,'Duplicate source metadata key')
        result[key]=value
    return result


def document(root, ref):
    rel=Path(ref.path)
    path=root/rel
    require(rel.is_relative_to('outputs') and path.suffix=='.json' and path.name!='private-packet.json'
            and str(rel)==ref.path and '..' not in rel.parts and path.resolve()==path and path.is_file(),
            'Exact source metadata path required')
    with path.open('rb') as stream:
        raw=stream.read(MAX_METADATA_BYTES+1)
    require(len(raw)<=MAX_METADATA_BYTES and sha256(raw).hexdigest()==ref.sha256,
            'Source metadata digest differs: '+path.name)
    doc=json.loads(raw,object_pairs_hook=_unique_pairs)
    require(type(doc) is dict,'Exact source metadata object required')
    return doc


def _ref(path):
    return Ref(path,PINNED[path])


def _field(value):
    require(type(value) is dict and {'path','sha256'}<=set(value),'Exact native source reference required')
    return Ref(value['path'],value['sha256'])


def preflight_native29_source_inputs(root):
    """Authenticate the actual fixed constructor evidence without importing numerical readers."""
    root=Path(root).resolve()
    owner_ref=_ref(JOINT+'joint-owner-snapshot.json')
    source_ref=_ref(ORIGINAL+'scoped-source-registry-snapshot.json')
    joint=document(root,owner_ref)
    replay=_ref(ORIGINAL+'receipt.json')
    route_plan=_ref(ORIGINAL+'route-binding-plan.json')
    for ref in (source_ref,replay,route_plan):
        document(root,ref)
    require(_field(joint['v6_source_snapshot'])==source_ref and _field(joint['v6_replay'])==replay
            and _field(joint['v6_route_plan'])==route_plan and joint['inherited_v6_routes']==62
            and joint['inherited_v6_updates']==5000 and joint['inherited_v6_source_windows']==5940
            and joint['training_authorized'] is joint['global_split_accepted'] is False
            and joint['active_roles']==joint['betse_scheduled_routes']==0,
            'Exact inactive joint constructor and inherited v6 lineage required')

    plan_ref=_ref(ENDPOINTS+'plan.json')
    plan=document(root,plan_ref)
    seal=document(root,_ref(ENDPOINTS+'seal.json'))
    sizing=document(root,_ref(SIZING+'measurement.json'))
    native=document(root,_ref(REGISTRATION+'run/proof.json'))
    require(seal['status']==native['status']=='complete' and seal['plan_sha256']==plan_ref.sha256
            and plan['source_snapshot_sha256']==sizing['source_snapshot_sha256']==owner_ref.sha256
            and plan['native_family_sha256']==native['family_sha256']==OWNER_FAMILY
            and tuple(plan['widths'])==(24,130) and tuple(sizing['width_family_total_numel'])==TOTALS
            and tuple(row['width'] for row in sizing['configuration'])==WIDTHS
            and all(row['source_snapshot_sha256']==owner_ref.sha256 and row['base_width']==24
                    for row in sizing['configuration'])
            and native['registered_heads']==29 and native['protected_task_shard_opens']==0
            and seal['training_authorized'] is plan['training_authorized'] is False
            and seal['GPU_queries']==seal['optimizer_updates']==seal['source_witness_outcome_reads']==0
            and (plan['coordinate_shards'],plan['constructor_coordinate_reads'],plan['constructor_original_train_readers'])==(148,888,8),
            'Exact 29-head source, mixed width family and constructor-only boundary required')
    proofs=[]
    require(tuple(row['width'] for row in seal['results'])==(24,130),'Exact two-width witness required')
    for row in seal['results']:
        width=row['width']
        base=ENDPOINTS+f'runs/width-{width}/'
        proof=Ref(base+'proof.json',row['proof_sha256'])
        obs=document(root,proof)
        terminal=document(root,Ref(base+'terminal.json',row['terminal_sha256']))
        require(terminal['status']==obs['status']=='complete' and terminal['proof_sha256']==proof.sha256
                and obs['source_snapshot_sha256']==owner_ref.sha256 and obs['native_family_sha256']==OWNER_FAMILY
                and obs['width']==width and obs['plan_sha256']==plan_ref.sha256
                and obs['registered_heads']==29 and obs['unique_tensors']==413
                and obs['unique_parameters']==TOTALS[WIDTHS.index(width)]
                and obs['source_readback']['protected_task_shard_opens']==0 and obs['native_shared_identity_exact']
                and obs['BETSE_archive_reads']==0 and obs['source_witness_outcome_reads']==0,
                'Exact completed constructor endpoint proof required')
        proofs.append(proof)

    count=document(root,_ref(COUNT+'receipt.json'))
    extended=_ref(COUNT+'extended-source-snapshot.json')
    require(_field(count['exact_scope'])==extended,'Exact source scope behind actual constructor required')
    assets=document(root,extended)
    for name in ('effective_assets','cartesian_construction','rgb_source_envelope','rgb_independent_axes',
                 'rgb_reversible_alpha','profile_group_source','profile_independent_groups','bioelectric_owner'):
        document(root,_field(assets[name]))
    bio=_field(assets['bioelectric_owner'])
    manifest=document(root,bio)
    require(set(manifest['assay_axes'])==set(ASSAYS)
            and tuple(manifest['assay_axes'][label]['count'] for label in ASSAYS)==(7,62,20)
            and manifest['training_authorized'] is manifest['split_roles_assigned'] is False
            and assets['source_role_or_training_admission'] is False
            and (manifest['calcium_trace_columns'],manifest['calcium_trace_timepoints'],manifest['laterality_groups'])==(20,892,9)
            and sizing['raw_fit_gene_extent']==51282 and sizing['fit_sha256']==RAW_FIT['fit.json']
            and sizing['selection_sha256']==RAW_FIT['fit-selection.json']
            and sizing['fullsort_sha256']==RAW_FIT['fit-fullsort.json']
            and native['source_fit_sha256']==sizing['fit_sha256'],
            'Exact source-only scope, literal assay axes or raw fit provenance differs')
    candidate=document(root,_ref(CANDIDATE+'receipt.json'))
    candidate_ref=_field(candidate['candidate_source_snapshot'])
    proposed=document(root,candidate_ref)
    require(candidate_ref.sha256==CANDIDATE_SNAPSHOT and proposed['training_authorized'] is False
            and candidate['candidate_train_replay_completed'] is candidate['training_authorized'] is False
            and len({candidate_ref.sha256,owner_ref.sha256,source_ref.sha256})==3,
            'Exact inactive successor route candidate must remain distinct from the owner')
    return Native29SourcePlan(owner_ref,source_ref,candidate_ref,OWNER_FAMILY,tuple(proofs),extended,bio,
                              _field(assets['profile_group_source']),51282,WIDTHS,TOTALS,148,8)


def require_common_snapshot(plan, route_snapshot, width_family_sha256):
    """Do not turn an inherited source reference into a same-snapshot campaign."""
    require(type(plan) is Native29SourcePlan and type(route_snapshot) is Ref
            and route_snapshot==plan.owner_snapshot and width_family_sha256==plan.native_family_sha256,
            'The executable campaign must bind the exact 29-head owner snapshot and native width family')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    plan=preflight_native29_source_inputs(parser.parse_args(argv).root)
    print(json.dumps(asdict(plan)|{'owner_and_routes_rebound':False,'training_authorized':False},sort_keys=True))


if __name__=='__main__':
    main()
