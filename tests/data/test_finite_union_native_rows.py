import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from mora3.data.finite_union_native_rows import (FiniteUnionNativeView, SelectedNativeSourceBatch,
    decode_disposition, selected_ordinals, full9)


class NativeViewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def put(self, name, raw):
        path = self.root / name
        path.write_bytes(raw)
        return dict(path=name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())

    def descriptor(self, codes):
        return dict(source_shard=dict(path='original.parquet', rows=len(codes)),
            primary_retained_rows=sum(value <= 2 for value in codes), disposition=dict(
                kind='original-row-ordinal-uint8', rows=len(codes), reference=self.put('mask.u8', bytes(codes))))

    def test_noncontiguous_selection_retains_absolute_original_coordinates(self):
        codes = bytes([0, 6, 0, 1, 7, 0, 2, 8])
        self.assertEqual(selected_ordinals(codes, split='train', row_start=2, row_stop=7), (2, 5))
        self.assertEqual(selected_ordinals(codes, split='validation', row_start=2, row_stop=7), (3,))
        self.assertEqual(selected_ordinals(codes, split='test', row_start=2, row_stop=7), (6,))

    def test_all_exclusion_codes_stay_unselected_in_every_role(self):
        for split in ('train', 'validation', 'test'):
            self.assertEqual(selected_ordinals(bytes(range(3, 11)), split=split, row_start=0, row_stop=8), ())

    def test_original_row_interval_and_role_are_exact(self):
        for args in (dict(split='train', row_start=True, row_stop=2),
                dict(split='train', row_start=-1, row_stop=2),
                dict(split='train', row_start=0, row_stop=3),
                dict(split='heldout', row_start=0, row_stop=2)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                selected_ordinals(b'\0\1', **args)

    def test_mask_byte_mutation_is_not_a_new_selection(self):
        descriptor = self.descriptor([0, 6, 2])
        (self.root / 'mask.u8').write_bytes(b'\0\0\2')
        with self.assertRaises(ValueError):
            decode_disposition(descriptor, root=self.root)

    def test_complete_codes_conserve_original_ordinals(self):
        descriptor = self.descriptor(list(range(11)))
        self.assertEqual(decode_disposition(descriptor, root=self.root), bytes(range(11)))

    def test_permission_quarantine_code10_valid_in_homogeneous_and_u8_views(self):
        descriptor = self.descriptor([10, 10])
        self.assertEqual(decode_disposition(descriptor, root=self.root), bytes([10, 10]))
        descriptor['disposition'] = dict(kind='homogeneous-original-row-disposition', code=10, rows=2)
        self.assertEqual(decode_disposition(descriptor, root=self.root), bytes([10, 10]))

    def test_unknown_code_and_truncated_mask_refuse(self):
        for values in ([0, 11], [0, 2]):
            descriptor = self.descriptor(values)
            if values == [0, 2]:
                descriptor['source_shard']['rows'] = descriptor['disposition']['rows'] = 3
            with self.subTest(values=values), self.assertRaises(ValueError):
                decode_disposition(descriptor, root=self.root)

    def test_homogeneous_hold_preserves_full_source_accounting(self):
        descriptor = dict(source_shard=dict(path='held.parquet', rows=6), primary_retained_rows=0,
            disposition=dict(kind='homogeneous-original-row-disposition', code=3, rows=6))
        self.assertEqual(decode_disposition(descriptor, root=self.root), bytes([3]) * 6)
        descriptor['disposition']['code'] = True
        with self.assertRaises(ValueError):
            decode_disposition(descriptor, root=self.root)

    def test_false_retained_count_cannot_expand_consumer(self):
        descriptor = self.descriptor([0, 6, 2])
        descriptor['primary_retained_rows'] = 3
        with self.assertRaises(ValueError):
            decode_disposition(descriptor, root=self.root)

    def manifest(self, descriptors):
        reference = self.put('views.jsonl', b''.join(json.dumps(d).encode() + b'\n' for d in descriptors))
        document = dict(schema='Mora3.ImmutableFiniteEligibleNativeUnion/v1',
            full_project_training_ready=False, canonical_master_pointer_unchanged=True,
            original_immutable_source_views=reference,
            source_taskviews=sum(row['source_shard']['rows'] for row in descriptors),
            primary_retained_taskviews=sum(row['primary_retained_rows'] for row in descriptors),
            primary_retained_split_counts=dict(train=sum(row['primary_retained_rows'] for row in descriptors), validation=0, test=0))
        ref = self.put('manifest.json', json.dumps(document).encode())
        return self.root / ref['path'], ref['sha256']

    def test_view_loads_without_arrow_or_source_payload_execution(self):
        path, sha = self.manifest([self.descriptor([0, 6, 2])])
        view = FiniteUnionNativeView(path, sha, root=self.root)
        self.assertEqual(len(view.descriptors), 1)
        self.assertFalse(view.manifest['full_project_training_ready'])

    def test_descriptor_hash_and_duplicate_sources_are_required(self):
        descriptor = self.descriptor([0, 6, 2])
        path, sha = self.manifest([descriptor, descriptor])
        with self.assertRaises(ValueError):
            FiniteUnionNativeView(path, sha, root=self.root)
        path, sha = self.manifest([descriptor])
        (self.root / 'views.jsonl').write_bytes(b'{}\n')
        with self.assertRaises(ValueError):
            FiniteUnionNativeView(path, sha, root=self.root)

    def test_untrusted_manifest_cannot_be_loaded(self):
        path, sha = self.manifest([self.descriptor([0])])
        with self.assertRaises(ValueError):
            FiniteUnionNativeView(path, '0' * 64, root=self.root)


class ArrowNativeBindingTests(unittest.TestCase):
    setUp = NativeViewTests.setUp
    put = NativeViewTests.put
    descriptor = NativeViewTests.descriptor
    manifest = NativeViewTests.manifest
    def native_view(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        table = pa.table({'task_id': ['a', 'held', 'b', 'c'],
            'source_family': ['native'] * 4, 'task_name': ['native_task'] * 4, 'split': ['train'] * 4,
            'record_id': ['r0', 'r1', 'r2', 'r3'],
            'native_payload_sha256': ['a'*64, 'b'*64, 'c'*64, 'd'*64],
            'payload': ['{"conditions":{"x":"A"},"targets":{"y":1}}',
                None, '{"conditions":{"x":"B"},"targets":{"y":2}}',
                '{"conditions":{"x":"C"},"targets":{"y":3}}'],
            'nullable_native': pa.array([1, None, 3, 4], type=pa.int16())})
        table = table.replace_schema_metadata({b'native': b'unchanged'})
        pq.write_table(table, self.root / 'original.parquet', row_group_size=2)
        d = self.descriptor([0, 6, 0, 0])
        raw = (self.root / 'original.parquet').read_bytes()
        d['source_shard'].update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        d['source_full9'] = list(full9(self.root / 'original.parquet'))
        d['original_group_sidecar'] = self.put('groups.parquet', b'exact source guard sidecar')
        path, sha = self.manifest([d])
        return FiniteUnionNativeView(path, sha, root=self.root), table

    def one_batch(self):
        view, table = self.native_view()
        batch = next(view.iter_native_task_batches(split='train', batch_size=4))
        return view, batch, table

    def test_arrow_noncontiguous_full_payload_roundtrip(self):
        import pyarrow as pa
        view, batch, original = self.one_batch()
        self.assertEqual(batch.source_row_ordinals, (0, 2, 3))
        self.assertTrue(pa.Table.from_batches([batch.native_rows]).equals(
            original.take(pa.array([0, 2, 3])), check_metadata=True))
        self.assertIsNone(view.require_native_rows(batch, split='train'))
        self.assertFalse(view.manifest['full_project_training_ready'])

    def test_arrow_chunk_boundaries_keep_absolute_ordinals(self):
        view, _ = self.native_view()
        batches = list(view.iter_native_task_batches(split='train', batch_size=2))
        self.assertEqual([b.source_row_ordinals for b in batches], [(0,), (2, 3)])
        for batch in batches:
            view.require_native_rows(batch, split='train')

    def test_forged_target_payload_refuses(self):
        import pyarrow as pa
        from dataclasses import replace
        view, batch, _ = self.one_batch()
        i = batch.native_rows.schema.get_field_index('payload')
        altered = batch.native_rows.set_column(i, batch.native_rows.schema.field(i),
            pa.array(['forged'] * 3))
        with self.assertRaises(ValueError):
            view.require_native_rows(replace(batch, native_rows=altered), split='train')

    def test_ordinal_alias_does_not_authorize_different_native_payload(self):
        from dataclasses import replace
        view, batch, _ = self.one_batch()
        for ordinals in ((0, 0, 3), (0, 1, 3), (0, 2, 4), (True, 2, 3), (3, 2, 0)):
            with self.subTest(ordinals=ordinals), self.assertRaises(ValueError):
                view.require_native_rows(replace(batch, source_row_ordinals=ordinals), split='train')

    def test_original_role_and_source_reference_cannot_be_relabelled(self):
        from dataclasses import replace
        view, batch, _ = self.one_batch()
        for changed in (replace(batch, original_split='validation'),
                replace(batch, source_shard={**batch.source_shard, 'rows': 100}),
                replace(batch, source_shard={**batch.source_shard, 'path': 'other.parquet'})):
            with self.subTest(changed=changed.source_shard), self.assertRaises(ValueError):
                view.require_native_rows(changed, split='train')
        with self.assertRaises(ValueError):
            view.require_native_rows(batch, split='test')

    def test_projected_payload_cannot_claim_complete_native_task_binding(self):
        view, _ = self.native_view()
        batch = next(view.iter_native_batches(split='train', columns=['payload'], batch_size=4))
        with self.assertRaises(ValueError):
            view.require_native_rows(batch, split='train')

    def test_same_values_different_native_type_refuse(self):
        import pyarrow as pa
        from dataclasses import replace
        view, batch, _ = self.one_batch()
        i = batch.native_rows.schema.get_field_index('nullable_native')
        altered = batch.native_rows.set_column(i, 'nullable_native',
            pa.array([1, 3, 4], type=pa.int64()))
        with self.assertRaises(ValueError):
            view.require_native_rows(replace(batch, native_rows=altered), split='train')

    def test_native_schema_metadata_cannot_be_rewritten(self):
        from dataclasses import replace
        view, batch, _ = self.one_batch()
        altered = batch.native_rows.replace_schema_metadata({b'native': b'altered'})
        with self.assertRaises(ValueError):
            view.require_native_rows(replace(batch, native_rows=altered), split='train')

    def test_mutable_public_descriptor_audit_cannot_expand_selection(self):
        view, _ = self.native_view()
        view.descriptors[0]['disposition'] = dict(
            kind='homogeneous-original-row-disposition', code=0, rows=4)
        view.descriptors[0]['primary_retained_rows'] = 4
        batch = next(view.iter_native_task_batches(split='train', batch_size=4))
        self.assertEqual(batch.source_row_ordinals, (0, 2, 3))
        view.require_native_rows(batch, split='train')

    def test_source_and_guard_mutation_refuse_before_consumption(self):
        view, _ = self.native_view()
        (self.root / 'groups.parquet').write_bytes(b'mutated guard')
        with self.assertRaises(ValueError):
            next(view.iter_native_task_batches(split='train'))

    def test_source_mutation_while_consumer_holds_batch_refuses_resume(self):
        view, _ = self.native_view()
        stream = view.iter_native_task_batches(split='train', batch_size=2)
        next(stream)
        with (self.root / 'original.parquet').open('ab') as out:
            out.write(b'changed')
        with self.assertRaises(ValueError):
            next(stream)

    def test_boolean_batchsize_and_implicit_columns_refuse(self):
        view, _ = self.native_view()
        for kwargs in (dict(split='train', columns=['payload'], batch_size=True),
                dict(split='train', columns=[]), dict(split='other', columns=['payload'])):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                next(view.iter_native_batches(**kwargs))


class EffectiveFilterTests(unittest.TestCase):
    setUp=NativeViewTests.setUp
    put=NativeViewTests.put
    descriptor=NativeViewTests.descriptor
    manifest=NativeViewTests.manifest
    native_view=ArrowNativeBindingTests.native_view
    # Do not inherit test methods twice; helpers are used by the dedicated
    # source-only overlay cases below.
    def filter_descriptor(self):
        view, table = self.native_view()
        parent = dict(path='manifest.json', **{k:v for k,v in self.put('parent-copy.json',
            (self.root/'manifest.json').read_bytes()).items() if k != 'path'})
        native = table.slice(2,1).to_pylist()[0]
        row = dict(source_shard=view.descriptors[0]['source_shard'],
            absolute_original_row_ordinal=2, original_split='train', parent_view_disposition_code=0,
            direct_source_annotation_paper_assertions_consumed=1,
            same_experiment_or_physical_identity_proven=False,
            **{k:native[k] for k in ('task_id','native_payload_sha256','task_name','source_family')})
        overlay = self.put('overlay.jsonl',json.dumps(row).encode()+b'\n')
        evidence = self.put('consumption.json',json.dumps(dict(parent_finite_native_view=parent,
            proposed_ordinal_overlay=overlay)).encode())
        proof = self.put('independent.json',json.dumps(dict(producer_manifest=evidence,
            exact_source_task_coordinate_dispositions_both_directions=True,
            false_positives=0,false_negatives=0,counts=dict(exact_exclusion_taskviews=1))).encode())
        filt = dict(schema='Mora3.AdditiveFiniteNativeOrdinalFilter/v1',
            reason='heldout-MCSA-direct-source-annotation-evidence',source_role='train',
            source_and_parent_masks_unchanged=True,full_project_training_ready=False,
            parent_native_view=parent,source_task_consumption_manifest=evidence,
            independent_source_ordinal_verification=proof,excluded_taskviews=1,
            effective_retained_split_counts=dict(train=2,validation=0,test=0),effective_retained_taskviews=2)
        return view,table,row,filt

    def filtered(self,filt):
        ref=self.put('filter.json',json.dumps(filt).encode())
        return FiniteUnionNativeView.from_effective_filter(self.root/ref['path'],ref['sha256'],root=self.root)

    def test_additive_filter_removes_only_exact_ordinal_and_preserves_base_codes(self):
        base,table,row,filt=self.filter_descriptor()
        effective=self.filtered(filt)
        self.assertEqual(next(base.iter_native_task_batches(split='train',batch_size=4)).source_row_ordinals,(0,2,3))
        batch=next(effective.iter_native_task_batches(split='train',batch_size=4))
        self.assertEqual(batch.source_row_ordinals,(0,3))
        effective.require_native_rows(batch,split='train')
        self.assertEqual(decode_disposition(base.descriptors[0],root=self.root),bytes([0,6,0,0]))
        self.assertEqual(effective.manifest['primary_retained_taskviews'],3)
        self.assertEqual(effective.effective_filter['effective_retained_taskviews'],2)

    def test_excluded_row_cannot_be_reintroduced_as_otherwise_valid_source_batch(self):
        base,table,row,filt=self.filter_descriptor()
        original=next(base.iter_native_task_batches(split='train',batch_size=4))
        effective=self.filtered(filt)
        with self.assertRaisesRegex(ValueError,'additive task filter'):
            effective.require_native_rows(original,split='train')

    def test_filter_does_not_relabel_unknown_reason_or_heldout_scope(self):
        base,table,row,filt=self.filter_descriptor()
        for key,value in [('reason','generic-paper-mask'),('source_role','test'),('excluded_taskviews',True),
                ('effective_retained_taskviews',3),('full_project_training_ready',True)]:
            changed=dict(filt,**{key:value})
            with self.subTest(key=key),self.assertRaises(ValueError):self.filtered(changed)

    def test_corrupt_or_wrong_snapshot_exclusion_evidence_refuses(self):
        base,table,row,filt=self.filter_descriptor()
        path=self.root/'overlay.jsonl';path.write_bytes(path.read_bytes()+b'{}\n')
        with self.assertRaises(ValueError):self.filtered(filt)

    def test_mutable_display_filter_does_not_reintroduce_exclusions(self):
        base,table,row,filt=self.filter_descriptor()
        effective=self.filtered(filt)
        effective.effective_filter['excluded_taskviews']=0
        effective.effective_filter['reason']='changed display'
        batch=next(effective.iter_native_task_batches(split='train',batch_size=4))
        self.assertEqual(batch.source_row_ordinals,(0,3))


class SourcePathSelectionTests(unittest.TestCase):
    setUp=NativeViewTests.setUp
    put=NativeViewTests.put
    descriptor=NativeViewTests.descriptor
    manifest=NativeViewTests.manifest
    native_view=ArrowNativeBindingTests.native_view
    filter_descriptor=EffectiveFilterTests.filter_descriptor
    filtered=EffectiveFilterTests.filtered

    def test_explicit_bound_source_selection_keeps_exact_native_ordinals(self):
        view,table=self.native_view()
        batch=next(view.iter_native_task_batches(split='train',source_paths=('original.parquet',),batch_size=4))
        self.assertEqual(batch.source_row_ordinals,(0,2,3))
        view.require_native_rows(batch,split='train')

    def test_unknown_duplicate_empty_or_implicit_paths_refuse_without_fallback(self):
        view,table=self.native_view()
        for paths in ((),[],True,'original.parquet',('other.parquet',),
                ('original.parquet','original.parquet'),('original.parquet','other.parquet'),(1,)):
            with self.subTest(paths=paths),self.assertRaises(ValueError):
                next(view.iter_native_task_batches(split='train',source_paths=paths))

    def test_source_selection_preserves_additive_study_filter(self):
        base,table,row,filt=self.filter_descriptor()
        view=self.filtered(filt)
        batch=next(view.iter_native_task_batches(split='train',source_paths=['original.parquet'],batch_size=4))
        self.assertEqual(batch.source_row_ordinals,(0,3))
        view.require_native_rows(batch,split='train')


if __name__=='__main__':unittest.main()
