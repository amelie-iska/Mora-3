from dataclasses import replace
import unittest
from test_scri_contracts import record,evidence
from mora3.extensions.scri.contracts import NativeValue
from mora3.extensions.scri.localization import Region,ActionProposal,constrain_actions,localization_metrics


def spatial(key,x,value,time=0.):
    return record(key,source=evidence(source_locator=key,lineage_id=key),time_value=time,
                  coordinates=(x,0.),coordinate_unit='micrometre',coordinate_frame='image-1',
                  measurements=(NativeValue('voltage','membrane',value,'mV',value is not None),))


class LocalizationTests(unittest.TestCase):
    def setUp(self):
        self.region=Region('image-1','micrometre',(-1.,-1.),(1.,1.))

    def test_one_off_region_action_rejects_entire_candidate(self):
        rows=(spatial('inside',0.,-60.),spatial('outside',2.,-60.))
        with self.assertRaisesRegex(ValueError,'off-region'):
            constrain_actions(rows,self.region,(ActionProposal('inside','A'),ActionProposal('outside','A')),
                              admissible_actions=('A',),max_actions=2)
        got=constrain_actions(rows,self.region,(ActionProposal('inside','A'),),admissible_actions=('A',),max_actions=2)
        self.assertFalse(got['physical_localization_validated'])

    def test_coordinates_cannot_be_assumed_or_converted(self):
        for r in (record(),replace(spatial('a',0.,-60.),coordinate_unit='millimetre')):
            with self.assertRaises(ValueError): self.region.contains(r)

    def test_absent_outside_measurement_does_not_claim_zero_off_target_effect(self):
        a=(spatial('in',0.,-60.),spatial('out',2.,None))
        b=(spatial('in',0.,-30.,24.),spatial('out',2.,None,24.))
        got=localization_metrics(a,b,self.region,modality='voltage',feature_id='membrane')
        self.assertIsNone(got['outside_absolute_change'])
        self.assertIsNone(got['outside_fraction_of_measured_absolute_change'])

    def test_measured_ratio_and_denominators(self):
        a=(spatial('in',0.,-60.),spatial('out',2.,-60.))
        b=(spatial('in',0.,-30.,24.),spatial('out',2.,-50.,24.))
        got=localization_metrics(a,b,self.region,modality='voltage',feature_id='membrane')
        self.assertEqual(got['outside_fraction_of_measured_absolute_change'],.25)
        self.assertEqual((got['inside_observed'],got['outside_observed']),(1,1))

    def test_no_change_ratio_is_undefined(self):
        a=(spatial('in',0.,-60.),spatial('out',2.,-60.))
        b=tuple(replace(x,time_value=24.) for x in a)
        self.assertIsNone(localization_metrics(a,b,self.region,modality='voltage',feature_id='membrane')
                          ['outside_fraction_of_measured_absolute_change'])

    def test_duplicate_cells_and_mixed_evidence_cannot_change_metric(self):
        a=(spatial('in',0.,-60.),spatial('out',2.,-60.))
        b=tuple(replace(x,time_value=24.) for x in a)
        with self.assertRaisesRegex(ValueError,'duplicate'):
            localization_metrics(a+(a[0],),b,self.region,modality='voltage',feature_id='membrane')
        b=(replace(b[0],source=replace(b[0].source,evidence_kind='simulation')),b[1])
        with self.assertRaisesRegex(ValueError,'pooled'):
            localization_metrics(a,b,self.region,modality='voltage',feature_id='membrane')

    def test_unrelated_cohorts_and_intervals_cannot_share_a_localization_fraction(self):
        a=(spatial('in',0.,-60.),spatial('out',2.,-60.))
        b=(spatial('in',0.,-30.,24.),spatial('out',2.,-50.,24.))
        variants=[
            (a,(b[0],replace(b[1],time_value=48.))),
            (a,(b[0],replace(b[1],intervention_ids=('different-treatment',)))),
            ((a[0],replace(a[1],organism='mouse')),(b[0],replace(b[1],organism='mouse'))),
            ((a[0],replace(a[1],source=replace(a[1].source,study_id='other-study'))),
             (b[0],replace(b[1],source=replace(b[1].source,study_id='other-study')))),
        ]
        for baseline,outcome in variants:
            with self.assertRaisesRegex(ValueError,'cohort context'):
                localization_metrics(baseline,outcome,self.region,modality='voltage',feature_id='membrane')


if __name__=='__main__': unittest.main()
