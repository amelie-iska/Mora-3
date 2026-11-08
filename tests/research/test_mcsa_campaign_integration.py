"""Original M-CSA reviewed route and mixed matched-validation scope."""

from dataclasses import replace
from decimal import Decimal
import inspect
from pathlib import Path

import pytest

from mora3.data.mcsa_role_list_admission import FAMILY, NATIVE, TASK
from mora3.research.matched_validation import (
    CampaignValidationSources, MatchedValidationTask, matched_validation_source_policy_sha256,
)
from mora3.research.mup_campaign import executable_adapter_sha256
from mora3.research.primary_native_routes import select_native_route_targets
from test_matched_validation import cohort
from test_mcsa_campaign_route import review_fixture, route
from test_primary_native_routes import fixture as original_view


def test_mixed_validation_keeps_exact_original_roles_and_pass_binding(tmp_path,review_fixture):
    mcsa = route(review_fixture)
    original = original_view(tmp_path)
    mixed = CampaignValidationSources(original,mcsa)
    coordinates,count = mcsa.validation_enrollment(max_selected_bytes=50_000)
    assert len(coordinates)==1 and count==3
    scope = replace(cohort(original),view_policy_sha256=matched_validation_source_policy_sha256(mixed))
    # The source view must share the exact campaign snapshot.
    validation = MatchedValidationTask(TASK,FAMILY,TASK,NATIVE,coordinates,count,Decimal('1'),Decimal('.01'),
                                       mcsa.loss_unit,mcsa.target_unit)
    scope = replace(scope,source_snapshot_sha256=mcsa.source_snapshot_sha256,
                    tasks=tuple(sorted((*scope.tasks,validation),key=lambda task:task.task_key)))
    assert scope.batches(mixed,'fit-objective')
    windows = scope.batches(mixed,TASK)
    assert sum(batch.original_source_rows for batch in windows)==1
    assert windows[0].coordinates==coordinates and windows[0].observed_targets==count
    assert windows[0].batch.canonical_sequences==('ACE',)
    assert windows[0].batch.audit[0]['proposed_split']=='validation'
    with pytest.raises(ValueError,match='snapshot'):
        scope.batches(original,TASK)
    with pytest.raises(ValueError,match='snapshot'):
        replace(scope,source_snapshot_sha256='b'*64).batches(mixed,TASK)
    altered = replace(validation,coordinates=(replace(coordinates[0],original_row_ordinal=1),))
    with pytest.raises(ValueError,match='complete.*validation'):
        replace(scope,tasks=tuple(sorted((scope.tasks[0],altered),key=lambda t:t.task_key))).batches(mixed,TASK)
    with pytest.raises(ValueError,match='budget'):
        replace(scope,max_selected_arrow_bytes=1).batches(mixed,TASK)


def test_mcsa_preparation_is_source_only_and_executable_pin_covers_review_policy(review_fixture):
    mcsa = route(review_fixture)
    binding = mcsa.train_binding()
    selected = select_native_route_targets(binding,mcsa.initial_cursor,max_windows=1)
    assert selected.status=='observed' and selected.targets.observed_positions==3
    assert selected.window.original_source_rows==1 and selected.window.batch.canonical_sequences==('ACD',)
    with pytest.raises(ValueError,match='owner'):
        selected.targets.evaluate(None)
    root = Path(inspect.getfile(type(mcsa))).resolve().parents[3]
    adapter = executable_adapter_sha256(binding.prepare,root=root)
    assert adapter!=executable_adapter_sha256(mcsa.evaluate_validation,root=root)
    other = route(review_fixture,seed=30)
    assert adapter!=executable_adapter_sha256(other.train_binding().prepare,root=root)


def test_explicit_pass_capacity_and_wrong_train_objective_refuse(review_fixture):
    mcsa = route(review_fixture,passes=3)
    cycle = (TASK,'other',TASK)
    mcsa.require_task_cycle_capacity(cycle,4)
    with pytest.raises(ValueError,match='cannot cover'):
        mcsa.require_task_cycle_capacity(cycle,6)
    binding = mcsa.train_binding()
    with pytest.raises(ValueError,match='reviewed M-CSA train objective'):
        replace(binding,prepare=lambda _:None)
    with pytest.raises(ValueError,match='reviewed M-CSA train objective'):
        replace(binding,target_unit='raw biological negatives')
