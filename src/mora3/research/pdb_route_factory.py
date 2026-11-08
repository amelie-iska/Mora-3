"""Original two-packet deposited PDB train route; no runtime/owner admission."""
from dataclasses import dataclass,asdict
from pathlib import Path
import json
from mora3.data.admet import canonical,digest
from mora3.data.finite_union_annotation_study_filter import load_study_filtered_native_view
from mora3.data.finite_union_native_rows import full9
from mora3.data.pdb_masked_coordinates import complete_pdb_coordinate_source,FAMILY,TASK
from mora3.model.source_assembly_inputs import PINS,CURRENT_VIEW_PINS
from .primary_campaign_assembly import _reference,_read
from .pdb_campaign_routes import PDBCampaignAdapter
from .mup_campaign import NativeTrainRoutePlan,executable_adapter_sha256

@dataclass(frozen=True)
class PDBRouteBudget:
    batch_size:int
    max_windows:int
    native_heads:int
    native_layers:int
    max_attention_elements:int
    def __post_init__(self):
        if any(type(v) is not int or v<1 for v in asdict(self).values()):
            raise ValueError('Explicit positive PDB route and declared native resource budgets required')

@dataclass(frozen=True)
class PartialPDBRegistry:
    binding:object
    plan:NativeTrainRoutePlan
    budget:PDBRouteBudget
    resource_binding_sha256:str
    source_availability:dict
    asset_pins:tuple
    @property
    def complete_campaign(self):return False
    @property
    def training_authorized(self):return False
    @property
    def allocation_authorized(self):return False
    @property
    def unresolved(self):return ('No native validation/test packets; explicit campaign train-only disposition required',
        'Only two source packets and one finite pass; repeat/campaign capacity policy not supplied',
        'Declared heads/layers and attention cap must be bound to the eventual exact owner/runtime evidence; no timing or memory qualification here')

def load_original_pdb_assets(root):
    root=Path(root).resolve()
    pins=dict(view=CURRENT_VIEW_PINS['effective_source_view'],source=CURRENT_VIEW_PINS['pdb_source_axes'],nonintersection=CURRENT_VIEW_PINS['unchanged_route_scope'])
    refs={k:_reference(dict(path=p,sha256=h)) for k,(p,h) in pins.items()}
    raws={k:_read(root,r) for k,r in refs.items()}
    proof=json.loads(raws['source']);bridge=json.loads(raws['nonintersection'])
    if (proof.get('schema')!='Mora3.CompletePDBNativeCoordinateSourceProof/v1'
            or proof.get('complete_original_labels_observations_and_targets_preserved') is not True
            or proof.get('train_packets')!=2 or proof.get('validation_or_test_packets')!=0
            or bridge.get('schema')!='Mora3.ExactAdditionalStudyFilterRouteNonintersection/v1'
            or bridge['prior_effective_source_filter']['sha256']!=PINS['effective_source_view'][1]
            or bridge['combined_source_filter']['sha256']!=refs['view'].sha256
            or bridge['scopes']['PDB-nucleic-masked-native-coordinates']['source_paths_intersecting_additive_exclusions']!=0):
        raise ValueError('Exact original PDB roles and v1-to-v2 nonintersection proof required')
    view=load_study_filtered_native_view(root/refs['view'].path,refs['view'].sha256,root=root)
    axes,packets=complete_pdb_coordinate_source(view)
    old=dict(proof['source_axes']);current=dict(axes.binding())
    del old['native_view_policy_sha256'];del current['native_view_policy_sha256']
    if canonical(old)!=canonical(current):raise ValueError('Original complete PDB axes/coordinates changed')
    observed=sum(sum(p is not None for p in packet.condition.positions_angstrom) for packet in packets)
    masked=sum(sum(p is not None for p in packet.masked_targets_angstrom) for packet in packets)
    if len(packets)!=2 or observed!=proof['observed_coordinates'] or masked!=proof['withheld_coordinates']:
        raise ValueError('Exact two native packets and complete coordinate counts required')
    availability=dict(train_packets=2,validation_packets=0,test_packets=0,max_atoms=axes.max_atoms,
        observed_atoms=observed,masked_atoms=masked,observed_target_cartesian_components=3*masked,
        source_paths=len({p for p,_,_ in axes.source_coordinates}),single_finite_pass=True)
    return view,axes,availability,refs,raws

def _resource_binding(axes,availability,budget):
    if type(budget) is not PDBRouteBudget:raise ValueError('Exact typed PDB route budget required')
    budget.__post_init__()
    required=budget.native_heads*budget.native_layers*(axes.max_atoms+1)**2*min(budget.batch_size,availability['train_packets'])
    if budget.max_attention_elements<required:
        raise ValueError('Declared attention cap cannot preserve complete native PDB batch')
    return digest(canonical(dict(schema='Mora3.PendingPDBRouteResourceBinding/v1',
        source_axes=axes.binding(),budgets=asdict(budget),conservative_attention_elements=required,
        measured_runtime=False)))

def _construct(root,view,axes,snapshot,budget):
    adapter=PDBCampaignAdapter(view,view,snapshot,axes)
    binding=adapter.train_binding(batch_size=budget.batch_size)
    plan=NativeTrainRoutePlan(TASK,digest(binding.route.initial_cursor),executable_adapter_sha256(binding.prepare,root=root),budget.max_windows)
    return binding,plan

def build_original_pdb_registry(root,*,source_snapshot,expected_view_policy_sha256,budget):
    if type(budget) is not PDBRouteBudget:raise ValueError('Exact typed PDB route budget required')
    budget.__post_init__()
    root=Path(root).resolve();snapshot=_reference(source_snapshot);before=_read(root,snapshot)
    view,axes,availability,refs,raws=load_original_pdb_assets(root)
    if axes.native_view_policy_sha256!=expected_view_policy_sha256:raise ValueError('Exact source-bound PDB view policy required')
    resource=_resource_binding(axes,availability,budget)
    binding,plan=_construct(root,view,axes,snapshot.sha256,budget)
    by_path={d['source_shard']['path']:d for d in view.descriptors}
    if any(full9(root/p)!=tuple(by_path[p]['source_full9']) for p,_,_ in axes.source_coordinates):
        raise ValueError('Original deposited-coordinate source changed during assembly')
    if _read(root,snapshot)!=before or any(_read(root,r)!=raws[k] for k,r in refs.items()):raise ValueError('Pinned PDB source assembly assets changed')
    refs['source_snapshot']=snapshot
    return PartialPDBRegistry(binding,plan,budget,resource,availability,tuple(sorted(refs.items())))
