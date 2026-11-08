"""Exact CEMA Supplementary Fig. 7 native assay trajectories and masks."""
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import struct

ARMS_SHA='751a49dde1c2cba97ecf1a12f9591d6082a381502b5ab54a9ef9a00e09972691'
ORACLE_SHA='4a7cdfd63d1283ee67dcced5b1b945e609c3a180e831d7230136c22053a9a646'
WORKBOOK_SHA='a849a1167be0db404704b4fb2b412e5e13f3001e2405dc32546600768afc0ee7'
ARTICLE_SHA='da1476a2aa22cfc14e490c77ad81d8dcc67b073e7bdeb583f9f987c27eb2d572'
WORK='doi:10.1038/s41467-023-44522-2'
MEMBER='xl/worksheets/sheet21.xml'
ROLES=tuple((exposure,subject) for exposure in ('untreated','suramin-treated') for subject in ('injured','receiver'))
MISSING=('original-cell-absent','original-explicit-no-value')
N_TIME=892


def _json(raw):
    def pairs(items):
        d={}
        for k,v in items:
            if k in d:raise ValueError('duplicate exact source field')
            d[k]=v
        return d
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda s:(_ for _ in ()).throw(ValueError(s)))


def _stamp(p):
    s=p.stat()
    return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)


def _decode(raw):return struct.unpack('>d',raw)[0]


def _number(row):
    n=row['source_value_float64'];lex=row['original_value_lexeme'];bits=row['source_Float64_bits']
    if (type(n) is not float or not math.isfinite(n) or type(lex) is not str or type(bits) is not str
            or re.fullmatch('[0-9a-f]{16}',bits) is None):
        raise ValueError('exact finite native Float64/number lexeme required')
    try:b=struct.pack('>d',float(lex))
    except (ValueError,OverflowError) as exc:raise ValueError('native numeric lexeme invalid')from exc
    if b.hex()!=bits or struct.pack('>d',n)!=b:raise ValueError('native scalar bit/lexeme mismatch')
    return b


@dataclass(frozen=True)
class CemaNativeTrajectory:
    """Audit identity is deliberately separate from the four literal query arms."""
    native_column: str
    source_local_trace_label: str
    exposure: str
    subject_context: str
    value_bits: tuple
    missing_dispositions: tuple

    def condition_inputs(self):
        return dict(species='Xenopus laevis',source_exposure=self.exposure,source_subject_context=self.subject_context,
            assay='source-reported normalized calcium fluorescence time course',time_unit='min')


class CemaNativeTrajectorySource:
    """Authenticated original-derived source view. It issues no optimization roles."""
    def __init__(self,root,arms_manifest,independent_proof):
        self._root=Path(root).resolve()
        self._refs={}
        def read(path,digest,expected_size=None):
            p=(self._root/path).resolve();p.relative_to(self._root)
            before=_stamp(p)
            with p.open('rb') as f:raw=f.read(16*2**20+1)
            if (before!=_stamp(p) or sha256(raw).hexdigest()!=digest or len(raw)>16*2**20
                    or expected_size is not None and len(raw)!=expected_size):
                raise ValueError('exact bounded immutable CEMA source artifact required')
            self._refs[p]=before
            return raw
        def ref(x):
            return read(x['path'],x['sha256'],x['bytes'])
        arms=_json(read(arms_manifest,ARMS_SHA))
        oracle=_json(read(independent_proof,ORACLE_SHA))
        if (oracle['original_calcium_arms_manifest']['sha256']!=ARMS_SHA
                or not all(oracle[k] is True for k in (
                    'all17840_native_trace_positions_verified','all892_exact_time_axis_positions_verified',
                    'independent_original_Open_XML_DOM_and_typed_Float64_checks'))
                or oracle['exact_observed_native_trace_values']!=17650 or oracle['original_empty_or_absent_trace_values']!=190
                or arms['same_exact_study_group']!=WORK or arms['training_authorized'] is not False
                or arms['original_or_prospective_training_role_assigned'] is not False
                or arms['original_publisher_workbook']['sha256']!=WORKBOOK_SHA):
            raise ValueError('exact native source, original independent verification and source-only role required')
        ref(arms['original_publisher_workbook'])
        times=_json(ref(arms['shared_original_Supplementary_Fig7_time_axis']))
        traces=_json(ref(arms['all20_native_normalized_intensity_trajectory_rows']))
        article=traces[0]['source_article']
        if article['sha256']!=ARTICLE_SHA:raise ValueError('exact qualified publisher article required')
        article_body=_json(ref(article))
        qual=article_body['qualification']
        if qual['source_candidate'] is not True or qual['expected_PMCID']!='PMC10794468' or 'CC-BY-4.0' not in qual['selected_original_grant_identifiers']:
            raise ValueError('original version and grant mismatch')
        ref(article_body['original_source_ref'])
        if len(times)!=N_TIME or len(traces)!=20:raise ValueError('all source-native time and trace axes required')
        time=tuple(_number(row) for row in times)
        tf=tuple(map(_decode,time))
        if tf[0]!=0 or any(b<=a for a,b in zip(tf,tf[1:])):
            raise ValueError('exact strictly increasing native minute axis required')
        def location(row,column,index):
            if (row['source_cell']['cell_coordinate']!=f'{column}{index+3}'
                    or row['source_cell']['archive_member_path']!=MEMBER
                    or row['source_cell']['sheet_ordinal']!=20
                    or row['source_cell']['native_row_ordinal']!=index+2):
                raise ValueError('native time/trace cell coordinate differs')
        for i,row in enumerate(times):location(row,'A',i)
        out=[];counts=Counter()
        for i,row in enumerate(traces):
            col=chr(ord('B')+i);exposure,subject=ROLES[i//5]
            label=row['exact_source_local_embryo_trace_label']
            values=row['ordered_source_values']
            if (row['source_column']!=col or type(label) is not str or not label or len(values)!=N_TIME
                    or row['source_article']!=article or row['source_workbook']!=arms['original_publisher_workbook']
                    or row['study_group']!=WORK or row['training_authorized'] is not False
                    or row['original_role'] is not None or row['prospective_role'] is not None
                    or row['native_time_unit']!='min' or row['native_value_coordinate']!='source-reported normalized intensity'
                    or row['known_workbook_source_condition']!=dict(source_exposure=exposure,source_subject_context=subject)):
                raise ValueError('exact source trace/arm/units/roles differ')
            bits=[];missing=[]
            for j,v in enumerate(values):
                kind=v['native_value_disposition'];counts[kind]+=1
                if kind=='original-observed-finite':
                    location(v,col,j)
                    bits.append(_number(v));missing.append(None)
                elif kind in MISSING:
                    if kind=='original-explicit-no-value':location(v,col,j)
                    if v.get('source_value_float64') is not None or v.get('source_Float64_bits') is not None:
                        raise ValueError('missing positions must have no native number')
                    bits.append(None);missing.append(kind)
                else:raise ValueError('unknown native trajectory value disposition')
            out.append(CemaNativeTrajectory(col,label,exposure,subject,tuple(bits),tuple(missing)))
        if counts['original-observed-finite']!=17650 or sum(counts[k] for k in MISSING)!=190:
            raise ValueError('complete original observation and missingness census differs')
        self._rows=tuple(out);self._time=time

    @property
    def rows(self):return self._rows

    @property
    def time(self):return tuple(map(_decode,self._time))

    def require_rows(self,rows):
        if any(_stamp(p)!=stamp for p,stamp in self._refs.items()):
            raise ValueError('immutable original native evidence changed')
        rows=tuple(rows)
        if (not rows or any(type(r) is not CemaNativeTrajectory or r not in self._rows for r in rows)
                or len({r.native_column for r in rows})!=len(rows)):
            raise ValueError('exact unique native trajectory specimens required')
        return tuple(tuple((i,_decode(x)) for i,x in enumerate(r.value_bits) if x is not None) for r in rows)
