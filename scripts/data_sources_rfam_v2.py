"""Rfam15.1 native nominal truncation codes, independent of stale Boolean README."""
import decimal
import json
import math
import re
from data_sources_rfam import open_seed_text,seed_families,clan_bindings

OBSERVED_RELEASE_CODES=frozenset(['0','3','5','53'])

def parse_region(line):
    f=line.rstrip('\r\n').split()
    if len(f)!=10 or not re.fullmatch(r'RF[0-9]{5}',f[0]):raise ValueError('unsupported Rfam15.1 region schema')
    try:
        start,end,cmstart,cmend=map(int,[f[2],f[3],f[6],f[7]]);score=decimal.Decimal(f[4]);evalue=decimal.Decimal(f[5])
    except (ValueError,decimal.InvalidOperation):raise ValueError('invalid native numericfields')
    if not score.is_finite() or not evalue.is_finite() or evalue<0 or f[8] not in OBSERVED_RELEASE_CODES or f[9] not in ('seed','full'):raise ValueError('unsupported sourceevidence or releasecode')
    sf=float(score)
    return dict(family_accession=f[0],sequence_accession_version=f[1],native_start=start,native_end=end,native_fasta_identifier_key=f[1]+'/'+f[2]+'-'+f[3],native_coordinate_order='decreasing' if end<start else 'increasing' if end>start else 'same',native_bitscore_text=f[4],bitscore_float64=sf if math.isfinite(sf) else None,native_evalue_text=f[5],native_CM_start=cmstart,native_CM_end=cmend,native_truncation_code_text=f[8],native_truncation_code_integer=int(f[8]),truncation_boolean_interpretation_inferred=False,source_membership_type=f[9],native_fields_json=json.dumps(f,separators=(',',':')),source_quality_fields_are_inference_conditions=False)
