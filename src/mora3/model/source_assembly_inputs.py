"""Concrete verified native source assets for the primary owner factory.

Reads fixed immutable artifacts and actual retained source rows. It never fits a
vocabulary from a diagnostic slice, replaces source axes, authorizes training,
or chooses a physical resource/rank/reward contract from fixture defaults.
Runtime settings and complete source codec/streamed GOA bindings are explicit
inputs. The returned facts expose unresolved production fields separately.
"""
from dataclasses import dataclass,fields
from pathlib import Path
import hashlib
import json
import pyarrow.parquet as pq
from mora3.data.annotation_axes import read_annotation_axes
from mora3.data.goa_streamed import StreamedGOASourcePool
from mora3.data.mcsa_weak_sites import WeakRoleVocabulary
from mora3.data.sccello_numeric_dictionary import decode_numeric_mapping
from mora3.data.sccello_native_rows import TokenVocabulary
from mora3.data.finite_union_native_rows import FiniteUnionNativeView,read_exact
from mora3.data.sciplex_admission import ObservationalSciPlexTask
from mora3.data.molecule_profile import molecule_profile_view
from mora3.data.lincs_native import _axis as lincs_axis
from mora3.data.admet import canonical
from mora3.data.finite_union_annotation_study_filter import load_study_filtered_native_view
from .sealed_axis_replay import frozen_pdb_axes, verify_frozen_grayscale
from mora3.data.rosetta_unconditional_profiles import load_rosetta_profile_axes, rosetta_view_policy_sha256
from .raster_source_envelope import RasterSourceEnvelope
from .annotation_vocabulary import AnnotationVocabulary
from .cell_native import NativeCellProtocol
from .full_owner import SourceAssemblyInputs
from .profile_decoder import SelfiesCodec

# Exact reviewed source snapshots, not a discovery of a mutable 'latest' path.
PINS={
 'grayscale_envelope':('outputs/implementation/readiness-20261106-model-mup/current-native-image-envelopes-v1/manifest.json','30ae777a61ebcb4a043abac0d6d1e357d3438449682f27a2de45271ca2747bdf'),
 'protein_payload_bound':('outputs/implementation/readiness-20261106-model-mup/full-source-protein-payload-bound-v1/manifest.json','8528d55388e63b91e3426c989a2435bfe4ea0cf7d3fd7e334d2531cf2b54ff39'),
 'goa_canonical':('outputs/data-intake/readiness-20261106-corpus-continuation/uniprot-native-reference-bridge-v1/manifest.json','c06d024a1b1bf0ed91ccd10d5ade26ed8d0220afa6d159ba8b52c66540e18477'),
 'goa_ontology':('outputs/data-intake/20261101-GO-ontology-reference-binding-full-GOA-v3/manifest.json','9c4a7009a2807bc4a9b7bfdf0944dc1f26577bbadd41025a0c52a70e221cfd8f'),
 'rosetta_groups':('outputs/implementation/readiness-20261106-model-mup/rosetta-complete-source-groups-v1/manifest.json','9d652e709cf26937db3ca037a18f51a907c5731f94354d72938e13cd09d91313'),
 'rosetta_codec':('outputs/implementation/readiness-20261106-model-mup/rosetta-complete-source-groups-v1/source-train-selfies-codec.json','7070001d504ed4d0b89be577f38bafa81fe4e117fabe10c6d45ad5814e1ff295'),
 'payload_vocabulary':('outputs/implementation/readiness-20261106-mcsa-train-tokenizer-ipm-v3-20261106T184643Z/annotation-vocabulary.json','bdca187d520cf639e2f51c08b71578d6a40223458667af982835e41f317c67d2'),
 'payload_fit':('outputs/data-intake/readiness-20261106-corpus-continuation/mcsa-canonical-condition-payload-fit-v1/fit-receipt.json','252ae14451b4b3886606462e96fe36139955f2353198b6a5b53f716fb094bee8'),
 'annotation_axes':('outputs/data-intake/20261101-annotation-target-axes-full-GOA-v2/manifest.json','bfb9024ddfacbb0b8a80b7f3645aac691b4794474d0cebd4a568cb48afc9459e'),
 'physical_axes':('outputs/implementation/readiness-20261106-model-mup/full-source-physical-axes-v1/manifest.json','72d11f4ee6642d8db597e48dfb63e225c15753b9f05198775cff856d750c3377'),
 'mcsa_roles':('outputs/data-intake/readiness-20261106-corpus-continuation/mcsa-source-role-list-axis-v1/frozen-role-list-vocabulary.json','fe57f88bf23b558f66a62e2db7840128a9a12fe365634d1378d12562822807bf'),
 'residue_census':('outputs/data-intake/readiness-20261106-corpus-continuation/uniprot-residue-list-full-source-census-v1/manifest.json','92916d6ea56ee0bd66c4b4ed6418ce83a5e21223d46d2aa37352f1a8f95a7a02'),
 'cell_dictionary':('outputs/data-intake/readiness-20261106-corpus/numeric-dictionaries-v1/manifest.json','e3da2e4cbda1599e6192c77c89c27db4a902d6463ef0c61ceb1e6b3f7660bb58'),
 'cell_protocol':('outputs/data-sources/readiness-20261106-sccello-native/source-protocol-roles-and-exposure.json','a0b700ae63f22f485576cc1d2911bc6796a6bf3e66de8aeb690205465db9a74c'),
 'rfam_manifest':('outputs/data-intake/20261101-rfam-family-annotation-v1/curated-v1/manifest.json','d1d684839333aa7d43a1620bf90d950b743c75746ed85dc0905dbbcf4c569190'),
 'lincs_manifest':('outputs/data-intake/20261101-LINCS-corpus-v1/source-dose-coordinate-qualified-v2/manifest.json','52f5055af4e71297af793ac9ad55c6ae975654e5982341dd1f2cd9600bcfb29d'),
 'pubchem_manifest':('outputs/data-intake/20261101-LINCS-corpus-v1/independent-master-union-v22/master-corpus-manifest-v22-LINCS-qualified-source-coordinate-v1.json','82ef478d53ccbd2bb8a87607b1ce5dd35a6a0f58f6afa7d912aa214129c1e6f2'),
 'effective_source_view':('outputs/data-intake/readiness-20261106-corpus-continuation/current32-effective-annotation-study-filter-v1/effective-filter-descriptor.json','80eb0975b80db33f32afb221dbbeb8759593ae92ce41989a0db2d661efddd860'),
}


CURRENT_VIEW_PINS = {
    'effective_source_view': (
        'outputs/data-intake/readiness-20261106-corpus-continuation/current32-effective-annotation-study-filter-v2/effective-filter-descriptor.json',
        '5d60c917c5592959ea7ca909e8ddfb9cbd06221aa13af8ddd2c754c103fbae3e',
    ),
    'unchanged_route_scope': (
        'outputs/data-intake/readiness-20261106-corpus-continuation/additional-study-route-compatibility-v1/route-scope-nonintersection-proof.json',
        'f8e9971db2335e3a4378aa36a6456a752cdfb86402c7ec8a53e5763d4c091b93',
    ),
    'effective_axis_census': (
        'outputs/data-intake/readiness-20261106-corpus-continuation/current-readiness-overlay-v7b/effective-source-axis-census.json',
        '10f671df30eb221df3e22a744178897c712707145da0588fe98b3f3a41289634',
    ),
    'scalar_source_projection': (
        'outputs/data-intake/readiness-20261106-corpus-continuation/source-scalar-reader-stage-v4/all-native-scalar-projection-verification.json',
        'f1a818227281b03202672a92a3f1fcfd4de28e94e68a2c90f2fee6502b83fbd7',
    ),
    'grayscale_envelope': (
        'outputs/implementation/readiness-20261106-model-mup/current-native-image-envelopes-v2/manifest.json',
        'd573b3fe01c07b440d9475b87907170b08babced54bef7b703490084906ff63b',
    ),
    'scalar_original_guards': (
        'outputs/data-intake/readiness-20261106-corpus-continuation/scalar-preexisting-guard-dispositions-v1/exact-4-original-kinetic-guard-exclusions.json',
        '9fee1f9d81e7d24329468c795870e39ff4cff050473542598dfce36dd74cba42',
    ),
    'pdb_source_axes': (
        'outputs/implementation/readiness-20261106-model-mup/pdb-coordinate-source-v1/proof.json',
        '5ed07df1fd73ae9cad330685291f25981ce7361f0a1d1838f520333400c136ee',
    ),
    'rosetta_unconditional_array_census': (
        'outputs/implementation/readiness-20261106-model-mup/rosetta-unconditional-array-census-v1/manifest.json',
        'd1b905c63885fd537fad23b549a3d55cebe78e3f777a8e7658c8c6ad56c17b15',
    ),
}


@dataclass(frozen=True)
class CurrentNativeRouteAxes:
    grayscale_source_envelope: RasterSourceEnvelope
    scalar_source_axis: object
    pdb_coordinate_axes: object
    rosetta_flow_axes: tuple
    rosetta_flow_source_census_sha256: str
    rosetta_flow_native_view_policy_sha256: str


def _pin(root,reference):
    name,expected=reference;path=Path(name);path=path if path.is_absolute() else Path(root)/path
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('native source artifact SHA mismatch: '+name)
    return raw


def _ref_pin(root,reference):
    raw=_pin(root,(reference['path'],reference['sha256']))
    if len(raw)!=reference['bytes']:raise ValueError('native source artifact byte count mismatch')
    return raw


def _descriptor_paths(view,family,task):
    return tuple(d['source_shard']['path'] for d in view.descriptors if any(
        x['source_family']==family and x['task_name']==task and x['split']=='train'
        and x['eligible_view_disposition']==0 for x in d['source_task_schema_counts']))


def _first_native_row(view,family,task):
    paths=_descriptor_paths(view,family,task)
    if not paths:raise ValueError('no retained native source route: '+family+'/'+task)
    batches=view.iter_native_task_batches(split='train',batch_size=64,source_paths=paths)
    try:
        for batch in batches:
            view.require_native_rows(batch,split='train')
            for ordinal,row in zip(batch.source_row_ordinals,batch.native_rows.to_pylist()):
                if row['source_family']==family and row['task_name']==task:
                    raw=row['native_payload']
                    if hashlib.sha256(raw).hexdigest()!=row['native_payload_sha256']:
                        raise ValueError('native source payload SHA mismatch')
                    return row,dict(source_shard=batch.source_shard,absolute_original_row_ordinal=ordinal,
                        native_payload_sha256=row['native_payload_sha256'],task_id=row['task_id'],
                        scope='actual retained row used for source schema, not full-pool consumption')
    finally:batches.close()
    raise ValueError('retained source route has no actual native row')


def _reference(path_and_sha):
    return dict(zip(('path', 'sha256'), path_and_sha))


def _current_route_axes(root, view, image):
    from .source_scalar_objectives import SourceScalarAxis

    proof = json.loads(_pin(root, CURRENT_VIEW_PINS['unchanged_route_scope']))
    old_sha = PINS['effective_source_view'][1]
    new_sha = CURRENT_VIEW_PINS['effective_source_view'][1]
    scopes = proof['scopes']
    if (
        proof['schema'] != 'Mora3.ExactAdditionalStudyFilterRouteNonintersection/v1'
        or proof['prior_effective_source_filter']['sha256'] != old_sha
        or proof['combined_source_filter']['sha256'] != new_sha
        or view.effective_filter['combined_excluded_taskviews'] != 227
        or any(scopes[name]['source_paths_intersecting_additive_exclusions'] != 0 for name in (
            'scalar-native-3', 'PDB-nucleic-masked-native-coordinates', 'Rosetta-CP-and-L1000-all-native-routes',
        ))
    ):
        raise ValueError('Exact current source-view and original route nonintersection proof required')
    scalar_ref = CURRENT_VIEW_PINS['scalar_source_projection']
    original_scalar = json.loads(_pin(root, scalar_ref))
    current_census = json.loads(_pin(root, CURRENT_VIEW_PINS['effective_axis_census']))
    scalar_families = {row['source_family'] for row in scopes['scalar-native-3']['original_route_and_role_counts']}
    original_roles = {(row['source_family'], row['original_split']): row['rows']
                      for row in original_scalar['original_role_counts']}
    current_roles = {(row['source_family'], row['split']): row['retained_rows']
                     for row in current_census['task_schema_role_routes'] if row['source_family'] in scalar_families}
    if (
        original_scalar['source_filter']['sha256'] != old_sha
        or current_census['combined_filter']['sha256'] != new_sha
        or current_census['retained_taskviews'] != view.effective_filter['effective_retained_taskviews']
        or current_roles != original_roles
    ):
        raise ValueError('Native scalar partition scope changed under the new filter')
    verified_scalar = SourceScalarAxis.from_verified_projection(*scalar_ref, root=root)
    scalar_axis = SourceScalarAxis(
        verified_scalar.proof_sha256, hashlib.sha256(canonical(view.effective_filter)).hexdigest(),
        verified_scalar.partitions,
    )
    pdb_proof = json.loads(_pin(root, CURRENT_VIEW_PINS['pdb_source_axes']))
    pdb = frozen_pdb_axes(view, pdb_proof, root)
    original_pdb = dict(pdb_proof['source_axes'])
    rebound_pdb = json.loads(json.dumps(pdb.binding()))
    del original_pdb['native_view_policy_sha256'], rebound_pdb['native_view_policy_sha256']
    if rebound_pdb != original_pdb:
        raise ValueError('Native PDB axes or original rows changed under the new filter')
    _pin(root, CURRENT_VIEW_PINS['scalar_original_guards'])
    if (
        image['schema'] != 'Mora3.CompleteCurrentNativeImageEnvelope/v2'
        or image['source_effective_view']['sha256'] != new_sha
        or image['original_test_development_exposed'] is not True
    ):
        raise ValueError('Complete native grayscale train and development-exposed test envelope required')
    verify_frozen_grayscale(view, image, root)
    grayscale = RasterSourceEnvelope(
        CURRENT_VIEW_PINS['grayscale_envelope'][1], image['max_height'], image['max_width'], None)
    rosetta_ref = CURRENT_VIEW_PINS['rosetta_unconditional_array_census']
    rosetta = load_rosetta_profile_axes(root, *rosetta_ref, view)
    return CurrentNativeRouteAxes(
        grayscale, scalar_axis, pdb, rosetta, rosetta_ref[1], rosetta_view_policy_sha256(view),
    )


@dataclass(frozen=True)
class NativeSourceAssets:
    root:Path
    vocabulary:object
    annotation_registry:object
    codec:object
    max_decoder_tokens:int
    response_axis:tuple
    physical_parameter_axis:tuple
    physical_method_axis:tuple
    profile_axes:dict
    lincs_axes:dict
    rfam_family_axis:tuple
    rfam_manifest_sha256:str
    mcsa_role_vocabulary:object
    cell_vocabulary:object
    cell_protocol:object
    native_view:object
    goa_pool:object
    facts:dict
    current_route_axes: CurrentNativeRouteAxes | None = None


def load_actual_source_assets(root,*,rfam_manifest=PINS['rfam_manifest'][0],rfam_sha256=PINS['rfam_manifest'][1],lincs_manifest=PINS['lincs_manifest'][0],lincs_sha256=PINS['lincs_manifest'][1],
                              source_view_policy='effective-study-v1'):
    """Resolve actual full finite axes and source-built vocabulary; no NN calls.

    Rfam and LINCS versions are caller-pinned until their exact current component
    references are recorded. The native profile row is checked against the
    selected source view. The new policy includes the 227-exclusion view and
    the additional source-bound scalar, PDB, grayscale and unconditional axes.
    """
    root=Path(root).resolve()
    if Path.cwd().resolve()!=root:raise ValueError('native registry requires explicit repository working directory')
    if source_view_policy not in ('effective-study-v1', 'effective-study-v2'):
        raise ValueError('Explicit known native source view policy required')
    raws={key:_pin(root,ref) for key,ref in PINS.items()}
    registry=read_annotation_axes(root/PINS['annotation_axes'][0],expected_sha256=PINS['annotation_axes'][1])
    vocabulary=AnnotationVocabulary(raws['payload_vocabulary'].decode('utf8'));fit=json.loads(raws['payload_fit'])
    if (fit['status']!='complete-source-fit' or fit['train_views']!=602 or fit['target_fields_used']is not False
            or fit['heldout_rows_used']is not False or fit['payload_vocabulary_artifact']['sha256']!=PINS['payload_vocabulary'][1]):
        raise ValueError('actual complete602 condition-only payload fit required')
    physical=json.loads(raws['physical_axes']);residue=json.loads(raws['residue_census'])
    if (physical['source_entries']!=107778 or physical['numeric_objective_admission_issued']is not False
            or residue['required_max_target_length_including_one_EOS']!=47):
        raise ValueError('complete source physical axes/residue multiplicity required')
    roles=WeakRoleVocabulary.from_document(json.loads(raws['mcsa_roles']))
    dictionary=json.loads(raws['cell_dictionary']);token_ref=next(x['source'] for x in dictionary['sources'] if x['ordinal']==1411)
    cells=TokenVocabulary.from_passive_mapping(decode_numeric_mapping(_ref_pin(root,token_ref)))
    protocol=NativeCellProtocol.from_source_document(raws['cell_protocol'],expected_sha256=PINS['cell_protocol'][1])
    current = source_view_policy == 'effective-study-v2'
    filter_ref = (CURRENT_VIEW_PINS if current else PINS)['effective_source_view']
    loader = load_study_filtered_native_view if current else FiniteUnionNativeView.from_effective_filter
    view = loader(root / filter_ref[0], filter_ref[1], root=root)
    goa_filter_options = dict(native_filter_policy='combined-study-overlay-v2') if current else {}
    goa_pool=StreamedGOASourcePool(root=root,native_filter_path=filter_ref[0],
        native_filter_sha256=filter_ref[1],registry=registry,
        canonical_reference_manifest=PINS['goa_canonical'][0],canonical_reference_sha256=PINS['goa_canonical'][1],
        ontology_manifest=PINS['goa_ontology'][0],ontology_sha256=PINS['goa_ontology'][1],**goa_filter_options)
    row,response_ref=_first_native_row(view,'GPerturb-sciPlex2','response')
    task=ObservationalSciPlexTask.from_bytes(row['native_payload']);sidecar=json.loads(task.task.sidecar_bytes)
    response_axis=tuple(sidecar['targets']['expression']['axis_ids'])
    if len(response_axis)!=5000 or sidecar['targets']['expression']['axis_identity']!='source_column_position':
        raise ValueError('complete5000-position source SciPlex schema required')
    profile_axes={};profile_refs={}
    for modality,length in (('CellPainting',1783),('L1000',977)):
        row,ref=_first_native_row(view,'Rosetta-CDRP-molecule-profile-'+modality,'profile_to_molecule')
        # Native groups remain audit inputs to source schema validation only.
        v=molecule_profile_view(row,effective_groups=json.loads(row['group_ids_json']))
        if len(v.axis)!=length:raise ValueError('complete native Rosetta axis required')
        profile_axes[modality]=v.axis;profile_refs[modality]=ref
    lincs=json.loads(_pin(root,(lincs_manifest,lincs_sha256)));lincs_axes={}
    for modality in ('CellPainting','L1000'):
        ref=lincs['source_full_axes'][modality]['axis_file'];axis=json.loads(_ref_pin(root,ref))
        lincs_axis(axis,modality);lincs_axes[modality]=axis
    rfam=json.loads(_pin(root,(rfam_manifest,rfam_sha256)));rfam_axis=tuple(rfam['family_ids'])
    if len(rfam_axis)!=4091 or rfam_axis!=tuple(sorted(set(rfam_axis))):raise ValueError('complete4091-family source axis required')
    groups=json.loads(raws['rosetta_groups']);codec_document=json.loads(raws['rosetta_codec'])
    if (groups['encoding_failures']or groups['source_effective_view']!=dict(path=PINS['effective_source_view'][0],sha256=PINS['effective_source_view'][1])
            or codec_document['heldout_symbols_added']is not False or any(codec_document['heldout_unknown_symbols'].values())):
        raise ValueError('complete same-source train codec without encoding failures or silent heldout expansion required')
    codec=SelfiesCodec.from_ordered_source_symbols(codec_document['symbols']);maximum=codec_document['max_tokens_including_BOS_EOS']
    if type(maximum)is not int or maximum<3:raise ValueError('complete native source target length required')
    image = json.loads(_pin(root, CURRENT_VIEW_PINS['grayscale_envelope'])) if current else json.loads(raws['grayscale_envelope'])
    protein = json.loads(raws['protein_payload_bound'])
    facts=dict(schema='Mora3.ActualNativeSourceAssemblyAssets/v1',source_pins={k:dict(path=v[0],sha256=v[1])for k,v in PINS.items()},
        rfam_manifest=dict(path=str(rfam_manifest),sha256=rfam_sha256),lincs_manifest=dict(path=str(lincs_manifest),sha256=lincs_sha256),
        axis_counts=dict(response=len(response_axis),GO=len(registry.axis('GO')),EC=len(registry.axis('EC')),
            residue=len(registry.axis('feature_type')),physical_parameters=len(physical['parameter_axis']),physical_methods=len(physical['method_axis']),
            Rosetta={k:len(v) for k,v in profile_axes.items()},LINCS={k:len(v['ordered_feature_names'])for k,v in lincs_axes.items()},
            Rfam=len(rfam_axis),cell_genes=len(cells.gene_keys)-2,mcsa_roles=roles.prediction_class_count-1),
        response_axis_identity='source_column_position',response_distinct_labels=len(set(response_axis)),
        schema_rows=dict(response=response_ref,profiles=profile_refs),source_vocab_fit='complete602MCSAcanonicaltrainconditions-not-allcorpusfit',
        residue_list_max_target_length=47,source_axes_are_schema_not_training_admission=True,
        source_inverse_training_rows=sum(x['rows']for x in groups['complete_retained_inverse_rows']if x['split']=='train'),
        source_inverse_condition_groups=groups['train_condition_groups'],source_inverse_unknown_symbols=codec_document['heldout_unknown_symbols'],
        source_decoder_max_tokens=maximum,source_SELFIES_vocabulary=codec.size,
        grayscale_complete_source_image_count=image['distinct_source_images'],
        grayscale_required_native_geometry_rank=image['max_source_input_graph_nodes'],
        grayscale_max_attention_entities=image['max_source_input_attention_entities'],
        rgb_source_view_bound=image['rgb_source_view_bound'],rgb_instance_slot_bound=None,
        known_full_protein_entity_token_bound=protein['proven_complete_single_protein_entity_stream_bound'],
        neural_forward_calls=0,optimizer_updates=0,
        unresolved=['source-derived full geometry/color ranks and full native instance slots/pixel byte envelope',
            'full Cartesian input contract/fullmaxN/budget and target properness',
            'frozen complete per-profile observed Ygroups + staged source-guided reward implementation',
            'complete task-route native model adapters/global model-data signatures/campaign admission'])
    extra = _current_route_axes(root, view, image) if current else None
    if extra is not None:
        facts.update(
            schema='Mora3.ActualNativeSourceAssemblyAssets/v2',
            source_view_policy=source_view_policy,
            effective_source_view=_reference(filter_ref),
            current_scope_pins={key: _reference(ref) for key, ref in CURRENT_VIEW_PINS.items()},
            retained_taskviews=view.effective_filter['effective_retained_taskviews'],
            retained_original_roles=view.effective_filter['effective_retained_split_counts'],
            scalar_native_partitions=len(extra.scalar_source_axis.partitions),
            pdb_source_atoms_max=extra.pdb_coordinate_axes.max_atoms,
            rosetta_unconditional_axis_dimensions={axis.sha256: len(axis.axis_ids) for axis in extra.rosetta_flow_axes},
            gray_native_shape=(extra.grayscale_source_envelope.max_height, extra.grayscale_source_envelope.max_width),
            unresolved=[
                'Complete native image route ingress/objectives and same-source eligible RGB geometry/slots/resources',
                'Full Cartesian input contract, maxN, budget and target properness',
                'Staged source-guided profile reward campaign checkpoint/update controller',
                'Complete native objective route bindings and global model/data/campaign signatures',
            ],
        )
    return NativeSourceAssets(root,vocabulary,registry,codec,maximum,response_axis,tuple(physical['parameter_axis']),tuple(physical['method_axis']),
        profile_axes,lincs_axes,rfam_axis,rfam_sha256,roles,cells,protocol,view,goa_pool,facts,extra)


def assemble_source_inputs(assets,*,codec=None,max_decoder_tokens=None,goa_packets=None,runtime,admet_contracts=(),betse_source=None):
    """Concrete factory inputs from actual assets plus explicit reviewed choices.

    Caller supplies the real source-built codec/complete streamed-GOA route and
    runtime/source budgets. No fixture values or missing physical hashes are
    synthesized here. This function itself never issues a training approval.
    """
    if type(assets)is not NativeSourceAssets or type(runtime)is not dict:
        raise ValueError('actual native source assets and explicit reviewed runtime required')
    if betse_source is not None:
        from mora3.data.betse_membrane import BETSESource
        if type(betse_source)is not BETSESource or betse_source.root!=assets.root.resolve():
            raise ValueError('Verified BETSE source must share the exact native asset root')
    wanted={f.name for f in fields(SourceAssemblyInputs)}
    values=dict(vocabulary=assets.vocabulary,annotation_registry=assets.annotation_registry,
        response_axis=assets.response_axis,residue_feature_axis=assets.annotation_registry.axis('feature_type'),
        physical_parameter_axis=assets.physical_parameter_axis,physical_method_axis=assets.physical_method_axis,
        profile_axes=assets.profile_axes,lincs_axes=assets.lincs_axes,codec=assets.codec if codec is None else codec,
        max_decoder_tokens=assets.max_decoder_tokens if max_decoder_tokens is None else max_decoder_tokens,
        rfam_family_axis=assets.rfam_family_axis,rfam_manifest_sha256=assets.rfam_manifest_sha256,
        goa_packets=assets.goa_pool if goa_packets is None else goa_packets,mcsa_role_vocabulary=assets.mcsa_role_vocabulary,
        residue_list_max_target_length=47,cell_vocabulary=assets.cell_vocabulary,cell_protocol=assets.cell_protocol,
        admet_contracts=tuple(admet_contracts),betse_source=betse_source,pubchem_source_root=assets.root,
        pubchem_manifest=PINS['pubchem_manifest'][0],pubchem_manifest_sha256=PINS['pubchem_manifest'][1])
    if assets.current_route_axes is not None:
        extra = assets.current_route_axes
        values.update(
            grayscale_source_envelope=extra.grayscale_source_envelope,
            scalar_source_axis=extra.scalar_source_axis,
            pdb_coordinate_axes=extra.pdb_coordinate_axes,
            rosetta_flow_axes=extra.rosetta_flow_axes,
            rosetta_flow_source_census_sha256=extra.rosetta_flow_source_census_sha256,
            rosetta_flow_native_view_policy_sha256=extra.rosetta_flow_native_view_policy_sha256,
        )
        if not set(values) <= wanted:
            raise ValueError('Current verified source route axes require the matching native owner factory schema')
    if set(runtime)-wanted or set(runtime)&set(values):raise ValueError('unknown/overriding source runtime setting')
    values.update(runtime)
    required={f.name for f in fields(SourceAssemblyInputs)if f.default is __import__('dataclasses').MISSING and f.default_factory is __import__('dataclasses').MISSING}
    missing=required-set(values)
    if missing:raise ValueError('unresolved actual source/runtime inputs: '+','.join(sorted(missing)))
    if (values['geometry_id_width']<assets.facts['grayscale_required_native_geometry_rank']
            or values['payload_max_tokens']<assets.facts['known_full_protein_entity_token_bound']):
        raise ValueError('actual complete source geometry/protein payload envelope required; no crop')
    if (values['codec'].symbols!=assets.codec.symbols or values['max_decoder_tokens']<assets.max_decoder_tokens):
        raise ValueError('actual complete source SELFIES codec/target bound required')
    if (values.get('reconstruction_dimension'),values.get('reconstruction_horizon'),values.get('reconstruction_hidden'))!=(32,8,64):
        raise ValueError('root primary fixed latent32/horizon8/hidden64 required across widths')
    return SourceAssemblyInputs(**values)
