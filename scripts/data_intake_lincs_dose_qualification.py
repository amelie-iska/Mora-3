"""Preserve source dose coordinate while masking unresolved physical-unit claims."""
import json,math
try:
    from data_intake_corpus import CorpusTask
except ModuleNotFoundError:
    from scripts.data_intake_corpus import CorpusTask


def qualify_source_dose(task,primary_provenance):
    raw=task.canonical_bytes() if isinstance(task,CorpusTask) else task
    d=json.loads(raw)
    if 'source_nominal_dose' not in d['conditions'] or 'source_export_dose_coordinate' in d['conditions']:
        raise ValueError('one original source dose condition required; no double projection')
    original=d['conditions'].pop('source_nominal_dose');value=original.get('value')
    if type(value) not in (float,int) or not math.isfinite(value) or value<=0:
        raise ValueError('positive original source dose coordinate required; no repair')
    d['conditions']['source_export_dose_coordinate']=dict(value=value,unit=None,physical_unit_confirmed=False,cross_assay_conversion_allowed=False,scope='assay-specific source export coordinate; not calibrated concentration')
    d['audit']['source_unit_semantic_qualification']=dict(original_source_dose_condition=original,primary_provenance=primary_provenance,source_curated_unit_label='micromolar',upstream_CP_unit_labels=['profiling README mmoles/L','IDR author library uM'],source_unit_status='unresolved provenance; no automatic physical-unit or material equivalence',GE_primary_unit_confirmation='not asserted by this projection; source-specific coordinate retained',source_numeric_value_unchanged=True,physical_dose_condition_valid=False,cross_assay_unit_conversion_allowed=False)
    if 'confirmed_physical_dose_unit' not in d['missing']:d['missing'].append('confirmed_physical_dose_unit')
    return CorpusTask(d['task'],d['conditions'],d['targets'],d['audit'],d['missing'])
