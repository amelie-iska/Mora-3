"""Native Rfam region-confidence and seed/clan context; no biological label inference."""
import collections
import gzip
import decimal
import json
import math
import re

def parse_region(line):
    f=line.rstrip('\r\n').split()
    if len(f)!=10 or not re.fullmatch(r'RF[0-9]{5}',f[0]):raise ValueError('unsupported Rfamfull_region schema')
    try:
        start,end,cmstart,cmend,trunc=map(int,[f[2],f[3],f[6],f[7],f[8]]);score=decimal.Decimal(f[4]);evalue=decimal.Decimal(f[5])
    except (ValueError,decimal.InvalidOperation):raise ValueError('invalidnative regionnumeric fields')
    if not score.is_finite() or not evalue.is_finite() or evalue<0 or trunc not in (0,1) or f[9] not in ('seed','full'):raise ValueError('invalid source evidence value')
    sf=float(score)
    return dict(family_accession=f[0],sequence_accession_version=f[1],native_start=start,native_end=end,native_fasta_identifier_key=f[1]+'/'+f[2]+'-'+f[3],native_coordinate_order='decreasing' if end<start else 'increasing' if end>start else 'same',native_bitscore_text=f[4],bitscore_float64=sf if math.isfinite(sf) else None,native_evalue_text=f[5],native_CM_start=cmstart,native_CM_end=cmend,native_truncation_flag=trunc,source_membership_type=f[9],native_fields_json=json.dumps(f,separators=(',',':')),source_quality_fields_are_inference_conditions=False)

def seed_families(stream):
    gf=collections.defaultdict(list);rawgf=[];aligned=collections.defaultdict(list);ss=[];rf=[]
    for line in stream:
        text=line.rstrip('\r\n')
        if text=='//':
            ac=gf.get('AC',[]);name=gf.get('ID',[])
            if len(ac)!=1 or len(name)!=1 or not re.fullmatch(r'RF[0-9]{5}',ac[0]):raise ValueError('missing/ambiguous Rfamfamily identity')
            lengths=[sum(map(len,v)) for v in aligned.values()];columns=lengths[0] if lengths else 0;cons=''.join(ss);reference=''.join(rf)
            yield dict(family_accession=ac[0],family_name=name[0],source_GF_fields=dict(gf),source_GF_native_lines=list(rawgf),SS_cons_native=cons,RF_reference_native=reference,seed_sequences=len(aligned),alignment_columns=columns,alignment_axis_quality_pass=bool(lengths) and len(set(lengths))==1 and (not cons or len(cons)==columns) and (not reference or len(reference)==columns),structure_provenance_native=gf.get('SS',[]),sequence_provenance_native=gf.get('SE',[]),experimental_structure_truth_inferred=False)
            gf=collections.defaultdict(list);rawgf=[];aligned=collections.defaultdict(list);ss=[];rf=[];continue
        if text.startswith('#=GF'):
            f=text.split(maxsplit=2)
            if len(f)<3:raise ValueError('incomplete source GFfield')
            gf[f[1]].append(f[2]);rawgf.append(text)
        elif text.startswith('#=GC'):
            f=text.split(maxsplit=2)
            if len(f)==3 and f[1]=='SS_cons':ss.append(f[2])
            if len(f)==3 and f[1]=='RF':rf.append(f[2])
        elif text and not text.startswith('#'):
            f=text.split()
            if len(f)!=2:raise ValueError('unsupported native seedalignment row')
            aligned[f[0]].append(f[1])
    if gf or aligned:raise ValueError('incomplete Stockholmfamily terminator')

def clan_bindings(lines,family_name_to_accessions):
    rows=[]
    for line in lines:
        fields=line.split()
        if not fields:continue
        if not re.fullmatch(r'CL[0-9]{5}',fields[0]):raise ValueError('unsupported clanID')
        for name in fields[1:]:
            candidates=family_name_to_accessions.get(name,[]);exact=len(candidates)==1
            rows.append(dict(clan_id=fields[0],source_family_name=name,family_accession=candidates[0] if exact else None,exact_seed_family_name_bound=exact,candidate_family_accessions=list(candidates)))
    return rows


def open_seed_text(path):
    """Explicit one-to-one byte mapping; native seed citations are not UTF-8."""
    return gzip.open(path,"rt",encoding="latin-1",errors="strict",newline="")
