"""Original sciPlex2 candidate context; no physical units or RNA joins inferred.

Source CTG measurements and sample-hash labels are separate assays. These helpers
preserve both original coordinates and source strings. Public access is not an
admission grant; candidate outputs remain gated on source rights and context.
"""
import json, math, hashlib, re

def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()

def finite(value):
    result=float(value)
    if not math.isfinite(result):raise ValueError('finite source value required')
    return result

def condition_key(treatment,dose,cell_type='A549'):
    return 'sciPlex2_source_condition:'+hashlib.sha256(canonical([cell_type,treatment,finite(dose)])).hexdigest()

def hash_sample(line,source_line):
    fields=line.rstrip('\r\n').split('\t')
    if len(fields)!=4 or fields[3]!='':raise ValueError('source four-column tab schema required')
    label,oligo,column3,_=fields
    parts=label.split('_')
    if len(parts)!=3:raise ValueError('source sample label schema required')
    valid_well=bool(re.fullmatch('[A-Z]{2}[0-9]{2}',parts[2]))
    if not re.fullmatch('[ACGT]{10}',oligo):raise ValueError('source ten-base hash oligo required')
    dose=finite(parts[1])
    if dose<0:raise ValueError('nonnegative source dose coordinate required')
    return dict(source_line=source_line,source_sample_label=label,source_treatment_label=parts[0],source_dose_label=parts[1],source_dose_coordinate=dose,source_well_oligo=parts[2],source_culture_plate=parts[2][0] if valid_well else None,source_culture_well=parts[2][1:4] if valid_well else None,source_well_context_status='source_label_parsed_as_notebook' if valid_well else 'incomplete_or_invalid_source_well_label',source_column3=column3,hash_oligo=oligo,hash_oligo_role='synthetic_sample_label_not_genomic_sequence',dose_unit=None,rna_cell_barcode=None,condition_group=condition_key(parts[0],dose),join_granularity='source_condition_coordinate_separate_assays',same_cell_link=False,admitted_for_task=False,source_line_text=line.rstrip('\r\n'))

def ctg_record(raw,source_line):
    if set(raw)!={'CTG','replicate','cell_type','treatment','dose','real_dose'}:raise ValueError('exact CTG source columns required')
    ctg,dose,real_dose=(finite(raw[k]) for k in ['CTG','dose','real_dose'])
    if not raw['replicate'].isdigit() or int(raw['replicate'])<1:raise ValueError('positive source assay replicate index required')
    if dose<0 or real_dose<0 or not raw['cell_type'] or not raw['treatment']:raise ValueError('source condition required')
    return dict(source_line=source_line,source_CTG=ctg,source_replicate_index=int(raw['replicate']),replicate_role='source_CTG_assay_index_not_RNA_cell_replicate',cell_type=raw['cell_type'],source_treatment_label=raw['treatment'],source_dose_coordinate=dose,source_real_dose_coordinate=real_dose,dose_unit=None,readout_unit=None,physical_dose_mapping_verified=False,condition_group=condition_key(raw['treatment'],dose,raw['cell_type']),join_granularity='source_condition_coordinate_separate_assays',same_cell_link=False,admitted_for_task=False,source_fields_json=canonical(raw).decode())
