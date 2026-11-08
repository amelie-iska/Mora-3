from dataclasses import replace
from decimal import Decimal

import pytest

from mora3.data.admet import digest
from mora3.research.matched_validation import (
    MatchedValidationCohort, MatchedValidationTask, ValidationSourceRow,
)
from mora3.research.primary_native_routes import finite_view_policy_sha256
from test_primary_native_routes import fixture


def task(view, ordinal=3, task_id='a3'):
    reference=view.descriptors[0]['source_shard']
    coordinate=ValidationSourceRow(reference['path'],reference['sha256'],ordinal,
                                   task_id,digest(task_id.encode()),'revision-1')
    return MatchedValidationTask('fit-objective','native','fit','native-schema/v1',
        (coordinate,),3,Decimal('0.10'),Decimal('0.01'),'native likelihood','observed scalar positions')


def cohort(view, matched=None):
    return MatchedValidationCohort('a'*64,finite_view_policy_sha256(view),'b'*64,'c'*64,2500,16,2**20,
                                  (matched or task(view),),(('train-only-objective','no original validation role'),))


def test_same_original_validation_cohort_exact_selection_and_wire(tmp_path):
    view=fixture(tmp_path);m=cohort(view)
    path=tmp_path/'validation.json'
    path.write_bytes(m.wire())
    reopened=MatchedValidationCohort.load(path,m.sha256)
    assert reopened==m and reopened.wire()==m.wire()
    batches=reopened.batches(view,'fit-objective')
    assert len(batches)==1 and batches[0].original_split=='validation'
    assert batches[0].source_row_ordinals==(3,)
    assert batches[0].native_rows['task_id'].to_pylist()==['a3']
    contract=reopened.selection_contract('f'*64)
    assert contract.validation_exposure_sha256==m.sha256
    assert contract.tasks[0].eligible_observed_targets==3
    assert contract.tasks[0].initial_reference_loss==Decimal('0.10')
    assert contract.tasks[0].validation_source_sha256==m.tasks[0].source_sha256
    assert contract.comparison_update==2500
    with pytest.raises(ValueError):
        MatchedValidationCohort.load(path,'0'*64)


@pytest.mark.parametrize('drift',('test','train','wrong-hash','wrong-revision','budget','view'))
def test_wrong_original_role_identity_or_resource_bound_refuses_before_validation(drift,tmp_path):
    view=fixture(tmp_path);m=cohort(view);a=m.tasks[0];coordinate=a.coordinates[0]
    if drift=='test':
        m=cohort(view,task(view,ordinal=5,task_id='a5'))
    elif drift=='train':
        m=cohort(view,task(view,ordinal=4,task_id='a4'))
    elif drift=='wrong-hash':
        m=cohort(view,replace(a,coordinates=(replace(coordinate,payload_sha256='0'*64),)))
    elif drift=='wrong-revision':
        m=cohort(view,replace(a,coordinates=(replace(coordinate,source_revision='wrong'),)))
    elif drift=='budget':
        m=replace(m,max_selected_arrow_bytes=1)
    else:
        view.effective_filter={'changed':True}
    with pytest.raises(ValueError):
        m.batches(view,'fit-objective')


def test_no_selection_disposition_cannot_overlap_eligible_validation(tmp_path):
    view=fixture(tmp_path);m=cohort(view)
    with pytest.raises(ValueError):
        replace(m,train_only_dispositions=(('fit-objective','fabricated absence'),))
    with pytest.raises(ValueError):
        replace(m,comparison_update=True)
