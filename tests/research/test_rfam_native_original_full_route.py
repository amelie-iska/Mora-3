"""Actual original-source Rfam finite native route integration; never trains."""
from pathlib import Path
from mora3.research.rfam_native_complete_replay import replay_complete_rfam_native_routes


def test_actual_rfam_native_predictions_and_generation_entire_original_population():
    result = replay_complete_rfam_native_routes(Path.cwd())
    assert result['schema'] == 'Mora3.OriginalRfamNativeFullSourceReplay/v1'
    assert result['source_snapshot_sha256'] == '79d328f542d1158581875ef325e56f213b407992f95db6ee0c7352a3459669d4'
    assert result['rows']['prediction'] == result['rows']['generation'] == 18423
    assert result['original_parent_observations'] == 18423
    assert result['native_preparation_success'] is True
    assert result['candidate_common_result_match'] is True
    assert result['restart_checks'] >= 4
    assert result['full_model_training_admission'] is False
    assert result['optimizer_updates'] == result['actual_proxy_trials'] == result['actual_target_runs'] == 0
