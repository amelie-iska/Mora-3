from pathlib import Path
from dataclasses import replace
from decimal import Decimal
import json
import pytest
from mora3.research.finite_view_compatibility import BASE_POLICY,INVERSE_POLICY,SUPPORTED,INVERSE,SOURCE,ENROLLMENT,require_policy_scope,verify_exact_view_compatibility,_require_enrolled_task
from mora3.research.matched_validation import MatchedValidationTask,ValidationSourceRow

@pytest.fixture(scope='module')
def enrolled():
    proof=json.loads(Path(ENROLLMENT[0]).read_text())
    coordinates=tuple(ValidationSourceRow(**r['coordinate']) for r in proof['inverse_validation_source_coordinates'])
    witness,=tuple(w for w in proof['native_route_binding_witnesses'] if w['direction']==INVERSE)
    # Synthetic loss/floor only for validator fixtures; no production authority.
    task=MatchedValidationTask(INVERSE,SOURCE,INVERSE,'mora3.CorpusTask/v1',coordinates,5225,
        Decimal('1'),Decimal('1'),witness['loss_unit'],witness['target_unit'])
    return proof,task

@pytest.mark.parametrize('kind',['base_substitution','reverse','other_validation','guarded_inverse_axis','missing_axis','unknown_filter'])
def test_no_generalized_filter_equivalence(kind):
    train,validation,axes,validation_axes=BASE_POLICY,INVERSE_POLICY,SUPPORTED,(INVERSE,)
    if kind=='base_substitution':validation=BASE_POLICY
    if kind=='reverse':train,validation=validation,train
    if kind=='other_validation':validation_axes=('unknown',)
    if kind=='guarded_inverse_axis':axes=tuple(sorted(SUPPORTED+('source_UniProt_annotation_conditioned_canonical_sequence',)))
    if kind=='missing_axis':axes=axes[:-1]
    if kind=='unknown_filter':validation='a'*64
    with pytest.raises(ValueError):require_policy_scope(train,validation,axes,validation_axes)

def test_real_original_one_row_guard_is_disjoint_and_validation_unchanged(enrolled):
    witness=verify_exact_view_compatibility(Path.cwd(),train_policy=BASE_POLICY,validation_policy=INVERSE_POLICY,
        train_axes=SUPPORTED,validation_axes=(INVERSE,),validation_tasks=(enrolled[1],))
    assert witness['additional_excluded_rows']==1
    assert witness['original_validation_coordinates_verified']==114
    assert witness['original_validation_and_test_roles_unchanged']
    assert not witness['training_authorized']
    assert witness['unchanged_train_source_shards']>0
    assert witness['exact_caller_observed_targets']==5225
    assert witness['exact_caller_task_source_sha256']==enrolled[1].source_sha256

def test_v2_does_not_bypass_base_substitution_check():
    from mora3.research.finite_view_compatibility import resolve_view_compatibility
    with pytest.raises(ValueError):
        resolve_view_compatibility('/missing',train_policy=BASE_POLICY,validation_policy=BASE_POLICY,
            train_axes=SUPPORTED,validation_axes=(INVERSE,),validation_tasks=())

def test_exact_ordered_coordinate_equality_positive(enrolled):
    proof,task=enrolled
    assert _require_enrolled_task(proof,(task,))==task.coordinates

@pytest.mark.parametrize('kind',['subset','reordered','payload','shard','task_id','revision','denominator','units','source_family'])
def test_different_caller_cohort_same_task_and_view_refused(enrolled,kind):
    proof,task=enrolled
    if kind=='subset':task=replace(task,coordinates=task.coordinates[:-1])
    elif kind=='reordered':
        task=replace(task);object.__setattr__(task,'coordinates',tuple(reversed(task.coordinates)))
    elif kind in ('payload','shard','task_id','revision'):
        field={'payload':'payload_sha256','shard':'shard_sha256','task_id':'task_id','revision':'source_revision'}[kind]
        value='a'*64 if kind in ('payload','shard') else 'different'
        task=replace(task,coordinates=(replace(task.coordinates[0],**{field:value}),)+task.coordinates[1:])
    elif kind=='denominator':task=replace(task,observed_targets=5226)
    elif kind=='units':task=replace(task,target_unit='different')
    else:task=replace(task,source_family='different')
    with pytest.raises(ValueError):_require_enrolled_task(proof,(task,))
