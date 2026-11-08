import ast
import json
from pathlib import Path
import tempfile
import unittest

from mora3.extensions.scri.readiness import build_sweep_manifest,inspect_primary_admission,preflight,write_new_json


def plan():
    return dict(schema='mora3.mup_hyperparameter_transfer.USER_plan.v1',
        proposed_primary_sweep=dict(master_learning_rates=[.0001,.0003,.001],readout_multipliers=[.5,1],
            ordinary_attention_key_multipliers=[.5,1],seeds=[2026110201,2026110202],runs=24,
            max_updates_each=5000,max_seconds_each=1800,max_gpu_hours=12),
        proposed_baseline=dict(proxy_runs=8,max_gpu_hours=4),
        transfer=dict(seeds_per_width=3,max_updates_each=1000,max_seconds_each=1800,max_gpu_hours=4.5,
            naive_SP_target_runs=3,naive_SP_max_gpu_hours=1.5))


class ReadinessTests(unittest.TestCase):
    def test_grid_has_exact_same_seeds_and_twenty_four_unique_trials(self):
        d=build_sweep_manifest(plan(),catalog_sha256='a'*64)
        trials=d['primary_trials']
        self.assertEqual(len(trials),24)
        self.assertEqual(len({x['trial_id'] for x in trials}),24)
        for hp in {x['hp_id'] for x in trials}:
            self.assertEqual({x['seed'] for x in trials if x['hp_id']==hp},{2026110201,2026110202})
        self.assertEqual(d['maximum_gpu_hours'],22.)
        self.assertEqual(d['total_planned_runs'],44)
        self.assertFalse(d['ready_to_run'])
        self.assertIsNone(d['widths'])
        self.assertIsNone(d['selected_transfer_hyperparameters'])

    def test_budget_or_task_omission_cannot_silently_change_campaign(self):
        p=plan();p['proposed_primary_sweep']['max_updates_each']=9999
        with self.assertRaisesRegex(ValueError,'budget'):
            build_sweep_manifest(p,catalog_sha256='a'*64)
        with self.assertRaisesRegex(ValueError,'complete SCRI'):
            build_sweep_manifest(plan(),catalog_sha256='a'*64,task_names=('identity_state',))

    def test_preflight_reads_source_without_importing_or_executing(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'owner.py'
            path.write_text("raise RuntimeError('would execute')\nclass SourceBoundOwner:\n    def require_training_admission(self):\n        raise ValueError('closed')\n")
            got=inspect_primary_admission(path)
            self.assertEqual(got['status'],'unconditional_refusal')
            self.assertFalse(got['ready'])

    def test_ready_json_labels_do_not_override_actual_training_refusal(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'planning').mkdir();(root/'src/mora3/model').mkdir(parents=True)
            (root/'planning/data_worker_coordination.json').write_text('{"full_dataset_ready":true}')
            (root/'planning/data_intake_progress.json').write_text('{"full_dataset_training_gate":{"ready":true}}')
            (root/'src/mora3/model/full_owner.py').write_text('class SourceBoundOwner:\n def require_training_admission(self):\n  raise ValueError("closed")\n')
            catalog=root/'catalog.json';catalog.write_text(json.dumps({'schema':'Mora3.SCRISourceCatalog/v1','sources':[]}))
            got=preflight(root,catalog,tasmorpho_revision='b'*40)
            self.assertFalse(got['ready_to_run'])
            self.assertTrue(any('unconditionally' in x for x in got['blockers']))

    def test_reports_cannot_overwrite_prior_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'receipt.json';write_new_json(path,{'x':1})
            with self.assertRaises(FileExistsError):write_new_json(path,{'x':2})
            self.assertEqual(json.loads(path.read_text()),{'x':1})


if __name__=='__main__': unittest.main()
