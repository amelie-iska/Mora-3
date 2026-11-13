"""Native SourceAssemblyInputs for the exact witnessed 29-head owner.

The launcher checks its reviewed release before entering this module. Historical
source metadata and construction scope are rechecked before loading train-side
native schemas. No owner, optimizer, device context or output is created here.
"""
from dataclasses import replace
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

from mora3.research.primary_source_factory_preflight import (
    ASSAYS, ORIGINAL, PINNED, RAW_FIT, SIZING, Ref, document,
    preflight_native29_source_inputs, require, require_common_snapshot,
)


def _ref(doc):
    return Ref(doc['path'],doc['sha256'])


def _rgb(root, evidence):
    import pyarrow.parquet as pq
    from .raster_source_envelope import RasterSourceEnvelope

    ref=_ref(evidence['rgb_source_envelope'])
    native=document(root,ref)
    independent=document(root,_ref(evidence['rgb_independent_axes']))
    alpha=document(root,_ref(evidence['rgb_reversible_alpha']))
    require(native['complete_planes_without_mask_cap']
            and (native['matrix_rows'],native['native_plane_positions'],native['empty_native_plane_positions'])==(665,30704,1)
            and _ref(independent['producer_manifest'])==ref
            and independent['every_original_plane_scalar_bit_and_native_order_checked']
            and independent['one_empty_source_layer_preserved'] and alpha['exact_bytewise_reconstruction']
            and alpha['alpha_all255']
            and alpha['native_RGBA_C_order_SHA256']==alpha['constant_alpha_reconstruction_RGBA_SHA256'],
            'Exact RGB geometry, native empty layer and reversible alpha evidence required')
    matrices=native['matrices']
    path=root/matrices['path']
    size=matrices['bytes']
    require(path.is_relative_to(root/'outputs') and path.resolve()==path and type(size) is int and 0<size<=2**20,
            'Exact bounded NuInsSeg matrix-coordinate metadata required')
    with path.open('rb') as stream:
        raw=stream.read(size+1)
    require(len(raw)==size and sha256(raw).hexdigest()==matrices['sha256'],
            'NuInsSeg matrix coordinate digest differs')
    table=pq.read_table(BytesIO(raw),columns=['native_shape','complete_native_plane_axis']).to_pylist()
    counts=[]
    for row in table:
        shape, layers=row['native_shape'],row['complete_native_plane_axis']
        require(shape[:2]==[512,512] and (len(shape)==2 and layers==1 or len(shape)==3 and shape[2]==layers),
                'NuInsSeg frame or complete layer extent differs')
        counts.append(layers)
    require(len(counts)==665 and sum(counts)==30704 and max(counts)==297,
            'NuInsSeg observed complete layer denominator differs')
    return RasterSourceEnvelope(ref.sha256,512,512,max(counts))


def _bioelectric(root, ref):
    from .bioelectric_outcomes import BioelectricAssayAxis

    manifest=document(root,ref)
    axes=[]
    for label in ASSAYS:
        record=manifest['assay_axes'][label]
        native=record['axis']
        axis=BioelectricAssayAxis(native['measurement'],native['native_assay'],native['literal_unit'],
                                  native['coordinate'],tuple(native['category_axis']))
        require(json.loads(json.dumps(axis.binding()))==native and len(axis.outputs)==record['finite_outputs'],
                'Sealed native assay axis differs from its production representation')
        axes.append(axis)
    require(len({axis.key for axis in axes})==3,'Duplicate native bioelectric assay axis')
    return dict(bioelectric_assay_axes=tuple(axes),bioelectric_source_contract_sha256=ref.sha256,
                cema_trajectory_source_sha256=manifest['sources']['cema_trajectory']['sha256'],
                hcn4_laterality_source_sha256=manifest['sources']['hcn4_group_counts']['sha256'])


def load_reviewed_native29_source_inputs(root, *, release_path, release_sha256, launch):
    """Resolve fixed native typed constructors after independent release and snapshot checks."""
    from mora3.research.primary_campaign_entrypoint import preflight_primary_worker_launch

    root=Path(root).resolve()
    authenticated=preflight_primary_worker_launch(root,release_path=release_path,release_sha256=release_sha256)
    require(launch==authenticated,'Exact pinned launch required for native source assembly')
    plan=preflight_native29_source_inputs(root)
    require_common_snapshot(plan, _ref(vars(launch.inputs.source_snapshot)), launch.inputs.campaign.width_family_sha256)
    require(_ref(vars(launch.profile_group_manifest))==plan.profile_group_manifest
            and launch.inputs.source_view_policy=='effective-study-v2',
            'Exact Rosetta group scope and source view required')
    inputs=launch.inputs
    sizing=document(root,Ref(SIZING+'measurement.json',PINNED[SIZING+'measurement.json']))
    require({k:vars(v) for k,v in inputs.configurations}==dict(zip(('base','delta','target'),sizing['configuration'],strict=True)),
            'Exact witnessed source-bound width and multiplier configurations required')

    from mora3.data.betse_membrane import BETSESource
    from mora3.data.sccello_raw_cell_packets import RawFitAxis
    from .cartesian_gfn_policy import CartesianBudget
    from .full_owner import SourceAssemblyInputs
    from .native29_assay_bindings import HeadBudgets, bind_axes, load_axes, minimum_budgets
    from .native29_extra_bindings import AdditionalBudgets, bind_additional, load_additional, minima
    from .source_assembly_inputs import assemble_source_inputs, load_actual_source_assets

    source=BETSESource(root,plan.owner_snapshot.path,plan.owner_snapshot.sha256,PINNED[ORIGINAL+'receipt.json'])
    for name,digest in RAW_FIT.items():
        document(root,Ref(SIZING+name,digest))
    raw=RawFitAxis.load_native_dual(root,*(root/SIZING/name for name in RAW_FIT),
                                    selection_sha256=RAW_FIT['fit-selection.json'],
                                    fullsort_sha256=RAW_FIT['fit-fullsort.json'])
    require(raw.fit_sha256==sizing['fit_sha256'] and raw.evidence_sha256==sizing['raw_fit_evidence_sha256']
            and len(raw.gene_keys)==plan.raw_gene_extent,'Exact train-fitted raw-cell axis required')

    evidence=document(root,plan.source_assets)
    assets=load_actual_source_assets(root,source_view_policy=inputs.source_view_policy)
    facts=document(root,_ref(evidence['effective_assets']))
    require(all(facts[key]==json.loads(json.dumps(value)) for key,value in assets.facts.items()),
            'Actual source asset inventory differs from the two-width constructor')
    rgb=_rgb(root,evidence)
    cart_ref=_ref(evidence['cartesian_construction'])
    cartesian=document(root,cart_ref)
    groups=document(root,plan.profile_group_manifest)
    require(cartesian['model_construction_only'] is True and cartesian['properness_admission'] is False
            and groups['train_condition_groups']==18293 and not groups['encoding_failures'],
            'Exact native Cartesian construction and observed Rosetta group proof required')
    geometry=assets.facts['grayscale_required_native_geometry_rank']
    color=1+rgb.max_instances
    require(launch.geometry_identifier_axis==tuple(f'grayscale-observed-geometry-node-ordinal:{i}' for i in range(geometry))
            and launch.color_identifier_axis==tuple(f'NuInsSeg-frame-or-source-mask-plane-ordinal:{i}' for i in range(color)),
            'Exact source-derived geometry and color axes required')
    runtime=dict(
        reconstruction_dimension=32,reconstruction_horizon=8,reconstruction_hidden=64,
        reconstruction_source=dict(policy='source-guided-frozen-profile-likelihood-v1',stage_refresh_updates=250,
            observed_group_manifest=evidence['profile_group_source'],
            independent_group_proof=evidence['profile_independent_groups']),
        geometry_id_width=geometry,color_id_width=color,instance_slots=rgb.max_instances,
        instance_max_output_bytes=rgb.max_instances*512*512*4,
        payload_max_tokens=assets.facts['known_full_protein_entity_token_bound'],
        color_input_policy='source-envelope-v2',instance_slot_policy='source-layer-list-v4',raster_source_envelope=rgb,
        native_prior_max_target_bytes=64*2**20,source_text_max_target_bytes=64*2**20,
        source_text_root_list_policy='observed-native-root-list-v1',lincs_inverse_max_target_bytes=64*2**20,
        cartesian_budget=CartesianBudget(**cartesian['budget']),cartesian_id_width=cartesian['atomic_rank'],
        cartesian_horizon=cartesian['horizon'],cartesian_teacher_sha256=cartesian['teacher_sha256'],
        cartesian_input_contract_sha256=cart_ref.sha256,**_bioelectric(root,plan.bioelectric),
    )
    axes,refs,_=load_axes(root)
    required=minimum_budgets(axes)
    required['payload_max_tokens']=max(required['payload_max_tokens'],runtime['payload_max_tokens'])
    require(dict(batch_size=1,**required)==evidence['explicit_head_budgets'] and refs==evidence['additional_source_refs']
            and json.loads(json.dumps({key:axis.binding() for key,axis in axes.items()}))==evidence['additional_source_axes'],
            'Exact source relational/assay axes and budgets required')
    sources=replace(assemble_source_inputs(assets,runtime=runtime,betse_source=source),
                    **bind_axes(axes,HeadBudgets(**evidence['explicit_head_budgets'])))
    axes,proof=load_additional(root,assets)
    required=minima(axes)
    required['payload_max_tokens']=max(required['payload_max_tokens'],sources.payload_max_tokens)
    require(dict(batch_size=1,**required)==evidence['remaining_five_budget']
            and json.loads(json.dumps(proof))==evidence['remaining_five_proof'],
            'Exact remaining native source axes and budget proof required')
    result=replace(sources,raw_cell_axis=raw,**bind_additional(axes,AdditionalBudgets(**evidence['remaining_five_budget'])))
    require(type(result) is SourceAssemblyInputs and result.betse_source is source and result.raw_cell_axis is raw
            and not result.admet_contracts,'Exact native 29-head input types and inactive source binding required')
    return result
