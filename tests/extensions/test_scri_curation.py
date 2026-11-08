from dataclasses import replace
import csv
import json
from pathlib import Path
import tempfile
import unittest

from test_scri_contracts import record,evidence
from mora3.extensions.scri.contracts import canonical
from mora3.extensions.scri.curation import (
    read_native_jsonl,read_long_cell_csv,curate_records,inspect_catalog,strict_json,
)


FIELDS=['record_id','study_id','sample_id','donor_id','experiment_id','lineage_id',
        'organism','cell_type','time_value','time_unit','time_reference',
        'coordinates','coordinate_unit','coordinate_frame','intervention_ids',
        'official_split','modality','feature_id','value','unit','observed']


class CurationTests(unittest.TestCase):
    def test_duplicate_keys_and_nonfinite_json_are_refused(self):
        for payload in ('{"x":1,"x":2}','{"x":NaN}','{"x":Infinity}','{"x":1e999}'):
            with self.assertRaises(ValueError): strict_json(payload)

    def test_json_arrays_are_not_coerced_from_strings(self):
        from mora3.extensions.scri.contracts import StateRecord
        source=json.loads(canonical(record().wire()))
        source['intervention_ids']='drug'
        with self.assertRaisesRegex(ValueError,'no implicit coercion'):
            StateRecord.from_mapping(source)

    def test_numeric_units_reject_string_values(self):
        from mora3.extensions.scri.contracts import NativeValue
        for unit in ('count','mV','FPKM'):
            with self.assertRaisesRegex(ValueError,'numeric units'):
                NativeValue('RNA','G','not-a-number',unit,True)
        self.assertEqual(NativeValue('cell','type','fibroblast','category',True).value,'fibroblast')

    def test_long_csv_retains_zero_missingness_order_and_native_lineage(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'native.csv'
            rows=[]
            for feature,value,observed in [('Z','0','true'),('A','','false')]:
                rows.append(['cell','study','sample','donor','exp','','human','fibroblast',
                    '0','hour','baseline','[1,2]','micrometre','section','[]','',
                    'RNA',feature,value,'count',observed])
            with path.open('w',newline='') as f:
                w=csv.writer(f);w.writerow(FIELDS);w.writerows(rows)
            got=read_long_cell_csv(path,dataset_id='fixture',revision='1',license_id='CC0',evidence_kind='synthetic_fixture')
            self.assertEqual([m.feature_id for m in got[0].measurements],['Z','A'])
            self.assertEqual([m.value for m in got[0].measurements],[0,None])
            self.assertIn('csv-lines:2,3',got[0].source.source_locator)
            self.assertEqual(got[0].coordinates,(1,2))
            from hashlib import sha256
            self.assertEqual(got.source_sha256,sha256(path.read_bytes()).hexdigest())
            self.assertEqual(got[0].source.artifact_sha256,got.source_sha256)
            self.assertEqual(got.parser_output_sha256,
                sha256(b''.join(canonical(r.wire())+b'\n' for r in got)).hexdigest())
            receipt=curate_records(got,source_path=path,output_directory=Path(d)/'curation')
            self.assertEqual(receipt['records_sha256'],got.parser_output_sha256)

    def test_prefix_limit_never_returns_partial_training_coverage(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'x.jsonl'
            path.write_bytes(canonical(record('a').wire())+b'\n'+canonical(record('b').wire())+b'\n')
            with self.assertRaisesRegex(ValueError,'no silent prefix'):
                read_native_jsonl(path,max_records=1)

    def test_immutable_curation_does_not_admit_synthetic_data(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'x.jsonl';out=Path(d)/'curation'
            path.write_bytes(canonical(record().wire())+b'\n')
            rows=read_native_jsonl(path)
            receipt=curate_records(rows,source_path=path,output_directory=out)
            self.assertFalse(receipt['training_admitted'])
            self.assertFalse(receipt['complete_benchmark_splits'])
            self.assertEqual(receipt['missing_measurements'],1)
            self.assertEqual(receipt['evidence_counts'],{'synthetic_fixture':1})
            with self.assertRaises(FileExistsError):
                curate_records(rows,source_path=path,output_directory=out)

    def test_curation_refuses_changed_input_before_writing_and_unbound_records(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'x.jsonl';out=Path(d)/'curation'
            path.write_bytes(canonical(record('first').wire())+b'\n')
            rows=read_native_jsonl(path)
            path.write_bytes(canonical(record('second').wire())+b'\n')
            with self.assertRaisesRegex(ValueError,'exact bytes consumed'):
                curate_records(rows,source_path=path,output_directory=out)
            self.assertFalse(out.exists())
            with self.assertRaisesRegex(ValueError,'parser-bound'):
                curate_records(tuple(rows),source_path=path,output_directory=out)
            self.assertFalse(out.exists())

    def test_curation_refuses_replaced_measurements_with_retained_parser_digest(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'x.jsonl';out=Path(d)/'curation'
            path.write_bytes(canonical(record().wire())+b'\n')
            parsed=read_native_jsonl(path)
            changed_measurement=replace(parsed[0].measurements[0],value=999)
            changed_record=replace(parsed[0],
                measurements=(changed_measurement,*parsed[0].measurements[1:]))
            changed=replace(parsed,records=(changed_record,*parsed.records[1:]))
            self.assertEqual(changed.source_path,parsed.source_path)
            self.assertEqual(changed.source_sha256,parsed.source_sha256)
            self.assertEqual(changed.parser_output_sha256,parsed.parser_output_sha256)
            self.assertNotEqual(changed[0].measurements[0].value,parsed[0].measurements[0].value)
            with self.assertRaisesRegex(ValueError,'canonical parser output'):
                curate_records(changed,source_path=path,output_directory=out)
            self.assertFalse(out.exists())
            receipt=curate_records(parsed,source_path=path,output_directory=out)
            self.assertTrue(receipt['input_to_parser_output_bound'])
            self.assertEqual(receipt['parser_output_sha256'],parsed.parser_output_sha256)
            self.assertEqual(receipt['records_sha256'],parsed.parser_output_sha256)

    def test_catalog_is_not_data_readiness(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'catalog.json'
            path.write_text(json.dumps({'schema':'Mora3.SCRISourceCatalog/v1','sources':[
                {'dataset_id':'example','data_kind':'experimental','downloaded':False,'license_status':'unresolved',
                 'revision':None,'training_status':'candidate_only'}]}))
            got=inspect_catalog(path)
            self.assertFalse(got['ready']);self.assertEqual(len(got['blockers']),4)

    def test_method_reference_is_not_reported_as_missing_experimental_data(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'catalog.json'
            path.write_text(json.dumps({'schema':'Mora3.SCRISourceCatalog/v1','sources':[
                {'dataset_id':'method','data_kind':'reference','artifact_admission_allowed':False,
                 'downloaded':False,'license_status':'unresolved','revision':None,
                 'training_status':'candidate_only'}]}))
            got=inspect_catalog(path)
            self.assertEqual(got['training_data_candidates'],0)
            self.assertEqual(got['blockers'],[])
            self.assertFalse(got['reference_only_entries'][0]['artifact_use_authorized'])
            self.assertFalse(got['ready'])


if __name__=='__main__': unittest.main()
