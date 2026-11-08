"""Exact 18-route nonintersection witness, not generic filter equivalence."""
from pathlib import Path
import json
from mora3.data.admet import canonical,digest
from mora3.data.finite_union_annotation_study_filter import load_study_filtered_native_view
from mora3.data.finite_union_inverse_literal_filter import load_inverse_literal_filtered_native_view,FAMILY,TASK
from mora3.data.finite_union_native_rows import decode_disposition, full9
from mora3.data.tox21 import INVERSE,SOURCE
from mora3.model.source_assembly_inputs import CURRENT_VIEW_PINS
from .primary_campaign_assembly import _reference,_read
from .primary_native_routes import finite_view_policy_sha256
from .native_prior_campaign import _SOURCE_ROUTES
from .finite_assay_registry import ROUTES as ASSAY_ROUTES
from .matched_validation import ValidationSourceRow, MatchedValidationTask

BASE_POLICY='c3812ecfcfea8a01f81769bf5ef6fb3880e04a0ac054a6349e9272db14765393'
INVERSE_POLICY='11b7bb1121567ec3098c044c67a9254b954242f37e9f43a4002655c87c29c046'
FILTER=('outputs/data-intake/readiness-20261106-corpus-continuation/current32-effective-inverse-target-fragment-filter-v2/effective-filter-descriptor.json','2ac36e62d203f85a5db8f8a17d28bfabe0db2eba94f069f095262d3fa5ad5aba')
ENROLLMENT=('outputs/implementation/readiness-20261106-model-mup/primary-campaign-driver-v1/actual-tox21-inverse-campaign-validation-enrollment-v1.json','d5a8e6c2f6e011f88348668effe31157df4a889cad102b32b364b429025b7f42')
SUPPORTED=tuple(sorted(tuple('native-prior:'+k for k in _SOURCE_ROUTES)+('enzyme_annotated_reaction_product',)+tuple(ASSAY_ROUTES)))

def require_policy_scope(train_policy,validation_policy,train_axes,validation_axes):
    if (train_policy!=BASE_POLICY or validation_policy!=INVERSE_POLICY
            or type(train_axes) is not tuple or train_axes!=SUPPORTED
            or validation_axes!=(INVERSE,)):
        raise ValueError('Only exact base18 train plus enrolled inverse-filter Tox21 validation scope is compatible')

def resolve_view_compatibility(root,*,train_policy,validation_policy,train_axes,validation_axes,validation_tasks):
    if INVERSE in validation_axes or train_policy!=validation_policy:
        return verify_exact_view_compatibility(root,train_policy=train_policy,validation_policy=validation_policy,
            train_axes=train_axes,validation_axes=validation_axes,validation_tasks=validation_tasks)
    return None

def _require_enrolled_task(enrollment,validation_tasks):
    """Bind the caller's actual task, not just the separately pinned source proof."""
    if (type(validation_tasks) is not tuple or len(validation_tasks)!=1
            or type(validation_tasks[0]) is not MatchedValidationTask):
        raise ValueError('Exact single typed frozen Tox21 validation task required')
    task=validation_tasks[0];task.__post_init__()
    expected=tuple(ValidationSourceRow(**r['coordinate']) for r in enrollment['inverse_validation_source_coordinates'])
    keys=tuple((c.source_path,c.original_row_ordinal) for c in expected)
    if len(expected)!=114 or keys!=tuple(sorted(set(keys))):
        raise ValueError('Pinned original enrollment coordinates must be unique and ordered')
    witness,=tuple(w for w in enrollment['native_route_binding_witnesses'] if w['direction']==INVERSE)
    if (task.task_key!=INVERSE or task.native_task_name!=INVERSE or task.source_family!=SOURCE
            or task.native_schema!='mora3.CorpusTask/v1'
            or task.coordinates!=expected or type(task.observed_targets) is not int
            or task.observed_targets!=5225 or enrollment['exact_payload_tokens_and_EOS']!=5225
            or task.loss_unit!=witness['loss_unit'] or task.target_unit!=witness['target_unit']):
        raise ValueError('Caller frozen validation coordinates/denominator/route units differ from exact enrollment')
    return expected

def verify_exact_view_compatibility(root,*,train_policy,validation_policy,train_axes,validation_axes,validation_tasks):
    require_policy_scope(train_policy,validation_policy,train_axes,validation_axes)
    root=Path(root).resolve()
    pins=dict(base=CURRENT_VIEW_PINS['effective_source_view'],inverse=FILTER,enrollment=ENROLLMENT)
    refs={k:_reference(dict(path=p,sha256=h)) for k,(p,h) in pins.items()}
    raws={k:_read(root,r) for k,r in refs.items()}
    filtered=json.loads(raws['inverse']);enrollment=json.loads(raws['enrollment'])
    coords=_require_enrolled_task(enrollment,validation_tasks)
    candidate_ref=_reference({k:filtered['source_inverse_literal_guard_proposal'][k] for k in ('path','sha256')})
    candidate_raw=_read(root,candidate_ref);proposal=json.loads(candidate_raw)
    refs['candidate']=candidate_ref;raws['candidate']=candidate_raw
    base=load_study_filtered_native_view(root/refs['base'].path,refs['base'].sha256,root=root)
    inverse=load_inverse_literal_filtered_native_view(FILTER[0],FILTER[1],root=root)
    if (finite_view_policy_sha256(base)!=train_policy or finite_view_policy_sha256(inverse)!=validation_policy
            or canonical(base.manifest)!=canonical(inverse.manifest)
            or base._sealed_descriptors!=inverse._sealed_descriptors
            or not base._ordinal_exclusions<inverse._ordinal_exclusions):
        raise ValueError('Exact same originals and stricter inverse guard required')
    delta=inverse._ordinal_exclusions-base._ordinal_exclusions
    candidate,=proposal['candidates']
    coordinate=(candidate['original_shard']['path'],candidate['absolute_original_row_ordinal'])
    if (delta!={coordinate} or candidate['source_family']!=FAMILY or candidate['task_name']!=TASK
            or candidate['original_split']!='train' or candidate['role_reassigned'] is not False):
        raise ValueError('Only the proven original inverse UniProt train omission is allowed')
    # Bind each supported family/task identity to its unchanged original descriptors.
    identities={(f,t) for f,t,_ in _SOURCE_ROUTES.values()}
    identities.add(('ReactZyme','enzyme_annotated_reaction_product'))
    for task,(kind,family,_) in ASSAY_ROUTES.items():
        identities.add((family if kind=='nci' else SOURCE if kind=='tox21' else 'ExpansionRx-source-native-ADMET-readouts',task))
    touched={d['source_shard']['path'] for d in base.descriptors if any(
        (r['source_family'],r['task_name']) in identities and r['split']=='train' and r['eligible_view_disposition']==0
        for r in d['source_task_schema_counts'])}
    if coordinate[0] in touched:raise ValueError('Guarded original source shard intersects scoped train routes')
    before=base.effective_filter['effective_retained_split_counts'];after=inverse.effective_filter['effective_retained_split_counts']
    if after!={**before,'train':before['train']-1}:raise ValueError('Other source roles changed')
    if (enrollment['source_scope']['current_native_view_policy_sha256']!=validation_policy
            or enrollment['source_scope']['complete_model_source_snapshot'] is not False
            or enrollment['initial_reference_loss'] is not None):raise ValueError('Exact pending validation proof required')
    descriptors={d['source_shard']['path']:d for d in inverse.descriptors};masks={}
    for c in coords:
        d=descriptors[c.source_path]
        if c.source_path not in masks:
            if full9(root/c.source_path)!=tuple(d['source_full9']):raise ValueError('Original enrolled validation shard changed')
            masks[c.source_path]=decode_disposition(d,root=root)
        if (d['source_shard']['sha256']!=c.shard_sha256 or masks[c.source_path][c.original_row_ordinal]!=1
                or (c.source_path,c.original_row_ordinal) in inverse._ordinal_exclusions):
            raise ValueError('Exact enrolled coordinate is not original eligible validation')
    if any(_read(root,r)!=raws[k] for k,r in refs.items()):raise ValueError('View compatibility evidence changed')
    return dict(schema='Mora3.ExactBase18TrainInverseTox21ValidationCompatibility/v1',
        train_view_policy_sha256=train_policy,validation_view_policy_sha256=validation_policy,
        train_axes=train_axes,validation_axes=validation_axes,unchanged_train_source_shards=len(touched),
        additional_excluded_rows=1,excluded_source_family=FAMILY,excluded_native_task=TASK,
        original_validation_coordinates_verified=114,original_validation_and_test_roles_unchanged=True,
        exact_caller_task_source_sha256=validation_tasks[0].source_sha256,exact_caller_observed_targets=validation_tasks[0].observed_targets,
        enrollment_sha256=ENROLLMENT[1],training_authorized=False,
        limitation='Not generic view equivalence. Guarded UniProt inverse route and all other validation axes remain outside this witness.')
