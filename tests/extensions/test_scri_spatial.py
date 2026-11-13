from dataclasses import replace
import unittest

from mora3.data.schema import GraphRecord, Measurement
from mora3.data.tasks import ConditionalTask
from mora3.extensions.scri.contracts import NativeValue, SourceEvidence, StateRecord
from mora3.extensions.scri.spatial import build_neighborhood_query, validate_spatial_query


AXIS = (('rna', 'GENE_A', 'count'), ('bioelectric', 'vm', 'mV'))


def cell(identifier, xy, *, missing=False):
    source = SourceEvidence('dataset-secret', 'revision-secret', 'a' * 64,
                            'file:///source-secret', 'CC0-1.0', 'experimental',
                            'study-secret', 'specimen-secret', 'donor-secret', 'experiment-secret', None)
    return StateRecord(identifier, source, 'human', None, 0, 'h', 'pre_intervention',
        (NativeValue('rna', 'GENE_A', None if missing else 3, 'count', not missing),),
        xy, 'um', 'specimen-frame-secret', (), 'train')


def build(rows=None, **kwargs):
    rows = (cell('query-secret', (0., 0.)), cell('near-secret', (3., 4.), missing=True),
            cell('far-secret', (10., 0.))) if rows is None else rows
    args = dict(query_record_id='query-secret', target_axis=AXIS, radius=5.)
    args.update(kwargs)
    return build_neighborhood_query(rows, **args)


class SpatialQueryTests(unittest.TestCase):
    def test_native_graph_closed_radius_units_targets_and_reciprocity(self):
        query = build()
        self.assertIs(type(query.record), GraphRecord)
        self.assertIs(type(query.task), ConditionalTask)
        self.assertEqual(query.query_node_index, 0)
        self.assertEqual(query.target_axis, AXIS)
        self.assertEqual(len(query.record.nodes), 2)
        self.assertEqual([(e.source, e.target, e.directed) for e in query.record.edges],
                         [(0, 1, True), (1, 0, True)])
        self.assertEqual(query.record.edges[0].measurements[0].value, 5.)
        self.assertEqual(query.record.edges[0].measurements[0].units, 'um')
        self.assertEqual(query.task.available_targets(query.record), ())
        self.assertFalse(query.audit['training_admitted'])
        self.assertIs(validate_spatial_query(query), query)

    def test_missing_values_remain_absent_and_do_not_break_condition_encoding(self):
        query = build()
        fields = query.task._fields(query.record)
        ref = ('node', 1, 'native:["rna","GENE_A"]')
        self.assertIsNone(fields[ref].value)
        self.assertEqual(fields[ref].units, 'count')
        self.assertEqual(dict(query.task.roles)[ref], 'unused')
        self.assertNotIn(ref, query.task.input_fields(query.record))
        self.assertFalse(fields[('node', 1, 'observed:["rna","GENE_A"]')].value)

    def test_audit_identifiers_do_not_enter_predictive_field_values(self):
        query = build()
        fields = query.task._fields(query.record)
        predictive = [fields[ref].value for ref in query.task.input_fields(query.record)]
        for identifier in ('query-secret', 'near-secret', 'study-secret', 'donor-secret',
                           'dataset-secret', 'revision-secret', 'specimen-frame-secret',
                           'file:///source-secret', 'specimen-secret', 'experiment-secret'):
            self.assertNotIn(identifier, str(predictive))
        self.assertEqual(query.audit['node_record_ids'], ('query-secret', 'near-secret'))

    def test_relabeling_ids_and_reordering_candidates_preserves_predictive_graph(self):
        a, b, c = cell('query-secret', (0., 0.)), cell('b', (1., 0.)), cell('c', (0., 1.))
        left = build((a, b, c)).record
        right = build((replace(c, record_id='new-c'), replace(a, record_id='new-query'),
                       replace(b, record_id='new-b')), query_record_id='new-query').record
        self.assertEqual(left.nodes, right.nodes)
        self.assertEqual(left.edges, right.edges)

    def test_no_nearest_neighbor_fallback_and_no_truncation(self):
        rows = (cell('query-secret', (0., 0.)), cell('other', (10., 0.)))
        query = build(rows, radius=1.)
        self.assertEqual(len(query.record.nodes), 1)
        self.assertEqual(query.record.edges, ())
        with self.assertRaisesRegex(ValueError, 'truncation'):
            build(max_cells=1)

    def test_mixed_original_source_rights_cannot_form_a_native_spatial_cohort(self):
        # Original source licensing cannot be reconciled by geometric proximity.
        # The supplied specimen population must be audited as one rights stratum,
        # including excluded cells; do not mutate or silently drop source records.
        query = cell('query-secret', (0., 0.))
        for position in ((1., 0.), (100., 0.)):
            other = cell('other', position)
            incompatible = replace(other, source=replace(other.source, license_id='restricted-source'))
            with self.subTest(position=position), self.assertRaisesRegex(ValueError, 'license'):
                build((query, incompatible))
        # Independent file locators with the SAME declared rights remain legal.
        same_rights = replace(other, source=replace(other.source, source_locator='file:///other-file'))
        legitimate = build((query, same_rights))
        self.assertIs(validate_spatial_query(legitimate), legitimate)

    def test_unknown_or_mixed_cohorts_fail_even_outside_radius(self):
        a, b = cell('query-secret', (0., 0.)), cell('other', (100., 0.))
        variants = [replace(b, coordinate_unit='mm'), replace(b, coordinate_frame='different'),
                    replace(b, coordinates=(100., 0., 0.)), replace(b, time_value=1),
                    replace(b, organism='mouse'), replace(b, intervention_ids=('drug',)),
                    replace(b, official_split='test'), replace(b, time_value=None, time_unit=None, time_reference=None),
                    replace(b, coordinates=None, coordinate_unit=None, coordinate_frame=None),
                    replace(b, source=replace(b.source, donor_id=None))]
        for name in ('study_id', 'sample_id', 'donor_id', 'experiment_id', 'evidence_kind'):
            variants.append(replace(b, source=replace(b.source, **{name: 'simulation' if name == 'evidence_kind' else 'other'})))
        for variant in variants:
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                build((a, variant))

    def test_invalid_radius_axis_budget_and_query_fail(self):
        for kwargs in ({'radius': 0}, {'radius': -1}, {'radius': float('nan')}, {'radius': True},
                       {'max_cells': 129}, {'max_cells': True}, {'query_record_id': 'absent'},
                       {'target_axis': ()}, {'target_axis': (AXIS[0], AXIS[0])},
                       {'target_axis': (('rna', 'GENE_A', 'normalized'),)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                build(**kwargs)

    def test_duplicate_records_and_conflicting_native_units_fail(self):
        a = cell('query-secret', (0., 0.))
        with self.assertRaises(ValueError):
            build((a, a))
        b = replace(cell('b', (1., 0.)), measurements=(NativeValue('rna', 'GENE_A', 3, 'arbitrary', True),))
        with self.assertRaisesRegex(ValueError, 'native unit'):
            build((a, b))

    def test_spatial_query_withholds_observed_query_target_and_rejects_reinjection(self):
        query_cell = cell('query-secret', (0., 0.))
        neighbor = cell('neighbor', (1., 0.))
        query = build((query_cell, neighbor))
        native_name = 'native:["rna","GENE_A"]'
        observed_name = 'observed:["rna","GENE_A"]'
        fields = {m.name: m for m in query.record.nodes[0].measurements}
        self.assertIsNone(fields[native_name].value)
        self.assertEqual(fields[native_name].status, 'missing')
        self.assertFalse(fields[native_name].available)
        self.assertFalse(fields[observed_name].value)
        self.assertEqual({m.name: m for m in query.record.nodes[1].measurements}[native_name].value, 3)
        changed_cell = replace(query_cell, measurements=(NativeValue('rna', 'GENE_A', 999, 'count', True),))
        self.assertEqual(query.record, build((changed_cell, neighbor)).record)
        # A caller cannot bypass the builder by supplying a forged query graph.
        nodes = list(query.record.nodes)
        injected = tuple(replace(m, value=999, available=True, status='observed')
                         if m.name == native_name else
                         replace(m, value=True) if m.name == observed_name else m
                         for m in nodes[0].measurements)
        nodes[0] = replace(nodes[0], measurements=injected)
        roles = tuple((ref, 'condition') if ref == ('node', 0, native_name) else (ref, role)
                      for ref, role in query.task.roles)
        with self.assertRaisesRegex(ValueError, 'query target'):
            validate_spatial_query(replace(query, record=replace(query.record, nodes=tuple(nodes)),
                                           task=replace(query.task, roles=roles)))

    def test_target_values_cannot_be_inserted_into_query(self):
        query = build()
        nodes = list(query.record.nodes)
        values = tuple(replace(m, value=9, available=True, status='observed')
                       if m.name == 'target:0' else m for m in nodes[0].measurements)
        nodes[0] = replace(nodes[0], measurements=values)
        with self.assertRaisesRegex(ValueError, 'remain missing'):
            validate_spatial_query(replace(query, record=replace(query.record, nodes=tuple(nodes))))

    def test_injected_conditions_wrong_roles_and_target_derived_edges_fail(self):
        query = build()
        injected = Measurement('future_target', 999, 'count', 'observed', True)
        nodes = (replace(query.record.nodes[0], measurements=(*query.record.nodes[0].measurements, injected)),
                 *query.record.nodes[1:])
        task = replace(query.task, roles=(*query.task.roles, (('node', 0, 'future_target'), 'condition')))
        with self.assertRaisesRegex(ValueError, 'undeclared predictor'):
            validate_spatial_query(replace(query, record=replace(query.record, nodes=nodes), task=task))
        roles = tuple((ref, 'unused' if ref == ('node', 0, 'is_query') else role)
                      for ref, role in query.task.roles)
        with self.assertRaisesRegex(ValueError, 'roles'):
            validate_spatial_query(replace(query, task=replace(query.task, roles=roles)))
        bad_edges = tuple(replace(e, measurements=(replace(e.measurements[0], value=1.),))
                          for e in query.record.edges)
        with self.assertRaisesRegex(ValueError, 'baseline coordinates'):
            validate_spatial_query(replace(query, record=replace(query.record, edges=bad_edges)))

    def test_spatial_query_validator_rejects_native_unit_spoofing(self):
        # Source-unit binding must survive graph serialization and validation;
        # the query's redacted target value is not permission to relabel units.
        query = build((cell('query-secret', (0., 0.)), cell('neighbor', (1., 0.))))
        native_name = 'native:["rna","GENE_A"]'
        for node_index, replacement_unit in ((0, 'arbitrary'), (1, 'arbitrary')):
            nodes = list(query.record.nodes)
            changed = tuple(replace(m, units=replacement_unit) if m.name == native_name else m
                            for m in nodes[node_index].measurements)
            nodes[node_index] = replace(nodes[node_index], measurements=changed)
            forged = replace(query, record=replace(query.record, nodes=tuple(nodes)))
            with self.subTest(node_index=node_index), self.assertRaisesRegex(ValueError, 'native.*unit'):
                validate_spatial_query(forged)
        # Even a condition outside the supervised axis cannot combine values
        # in incompatible units in the same native neighborhood.
        rows = tuple(replace(row, measurements=(*row.measurements,
                     NativeValue('metabolite', 'ATP', 5, 'micromolar', True)))
                     for row in (cell('query-secret', (0., 0.)), cell('neighbor', (1., 0.))))
        query = build(rows, target_axis=(AXIS[0],))
        nodes = list(query.record.nodes)
        changed = tuple(replace(m, units='millimolar') if m.name == 'native:["metabolite","ATP"]' else m
                        for m in nodes[1].measurements)
        nodes[1] = replace(nodes[1], measurements=changed)
        with self.assertRaisesRegex(ValueError, 'native feature unit'):
            validate_spatial_query(replace(query, record=replace(query.record, nodes=tuple(nodes))))

    def test_audit_payload_is_excluded_from_query_validation(self):
        query = build()
        changed = replace(query, audit={**query.audit, 'held_out_truth': 999, 'private_id': 'secret'})
        self.assertIs(validate_spatial_query(changed), changed)
        self.assertEqual(query.record, changed.record)


if __name__ == '__main__':
    unittest.main()
