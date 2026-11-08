"""Read complete original Figure 8c/e/f native values in three separate assay contexts."""
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import struct
import zipfile

from defusedxml.ElementTree import fromstring
from .bioelectric_evidence import (BioelectricJATSEvidenceSpan,BioelectricWorkbookNumericCell,
    NativeBioelectricBaseline,NativeBioelectricEndpoint,NativeBioelectricEvent,NativeBioelectricObservation)
from .cema_figure8g_source import NS,CAPTION,WORK
from .cema_native_trajectories import ARMS_SHA,ORACLE_SHA,WORKBOOK_SHA,ARTICLE_SHA
from .source_scalar_native_rows import ScalarTarget

P='outputs/data-intake/readiness-20261106-corpus-continuation/'
ARMS=P+'CEMA-ordered-native-calcium-source-arms-v1/manifest.json'
ORACLE=P+'CEMA-native-source-cells-and-arms-independent-v1/independent-native-CEMA-source-verification.json'
SHEETS=('Figure 8c','Figure 8e','Figure 8f')
EXPECTED=((7,8,9),(10,5),(10,5))


def _sha(raw):return sha256(raw).hexdigest()


def load_cema_native_wave_points(root):
    root=Path(root).resolve()
    def read(path,expected,n=None):
        p=(root/path).resolve();p.relative_to(root)
        with p.open('rb')as f:raw=f.read(16*2**20+1)
        if len(raw)>16*2**20 or _sha(raw)!=expected or n is not None and len(raw)!=n:
            raise ValueError('exact immutable CEMA source artifact required')
        return raw
    def ref(x):return read(x['path'],x['sha256'],x['bytes'])
    manifest=json.loads(read(ARMS,ARMS_SHA));proof=json.loads(read(ORACLE,ORACLE_SHA))
    if (proof['original_calcium_arms_manifest']['sha256']!=ARMS_SHA or proof['all61_explicit_source_calcium_scalar_positions_verified']is not True
            or manifest['training_authorized']is not False or manifest['original_or_prospective_training_role_assigned']is not False):
        raise ValueError('exact source coordinate audit and source-only scope required')
    rows=json.loads(ref(manifest['exact_source_Figure8c_e_f_points']))
    workbook=manifest['original_publisher_workbook'];projection=rows[0]['source_authoritative_article_projection']
    if len(rows)!=54 or workbook['sha256']!=WORKBOOK_SHA or projection['sha256']!=ARTICLE_SHA:
        raise ValueError('complete 54 original numeric positions and original source versions required')
    article=json.loads(ref(projection));q=article['qualification'];original=article['original_source_ref']
    if q['source_candidate']is not True or q['expected_PMCID']!='PMC10794468' or q['selected_original_grant_identifiers']!=['CC-BY-4.0']:
        raise ValueError('exact original CEMA work and grant required')
    cap=[z for z in article['native_character_data_segments'] if tuple(z['source_child_ordinal_path'][:len(CAPTION)])==CAPTION]
    caption=''.join(z['text_exact_native']for z in cap)
    if not all(x in caption.casefold() for x in ('suramin','gcamp6s','speed')):
        raise ValueError('original published assay caption missing')
    segment=next(z for z in cap if 'suramin' in z['text_exact_native'].casefold())
    txt=segment['text_exact_native'];txt_sha=_sha(txt.encode('utf8'))
    span=BioelectricJATSEvidenceSpan(original['path'],original['bytes'],original['sha256'],
        tuple(segment['source_child_ordinal_path']),segment['slot'],0,len(txt),txt_sha,txt_sha)
    span.read(root)
    raw=ref(workbook);sheets={}
    with zipfile.ZipFile(BytesIO(raw))as z:
        strings=fromstring(z.read('xl/sharedStrings.xml'),forbid_dtd=True)
        for ordinal in range(16,19):
            name=f'xl/worksheets/sheet{ordinal}.xml';b=z.read(name);xml=fromstring(b,forbid_dtd=True)
            cells=xml.findall(f'{NS}sheetData/{NS}row/{NS}c');indexed={c.attrib['r']:c for c in cells}
            if len(indexed)!=len(cells):raise ValueError('duplicate original source cell')
            sheets[name]=(ordinal,_sha(b),indexed)

    def header(c):
        if c.get('t')!='s' or len(c)!=1 or c[0].tag!=NS+'v':raise ValueError('original shared-string source header required')
        return ''.join(t.text or ''for t in strings[int(c[0].text)].iter(NS+'t'))

    arms=Counter();coords=set();out=[]
    for r in rows:
        panel=r['native_source_sheet'];i=SHEETS.index(panel);value=r['target'];target=value['source_cell']
        named=r['exact_literal_condition_header_cell'];name=f'xl/worksheets/sheet{16+i}.xml'
        ordinal,digest,cells=sheets[name];col=target['cell_coordinate'][0]
        if ((name,target['cell_coordinate'])in coords or target['archive_member_path']!=name or named['archive_member_path']!=name
                or any(x['archive_member_SHA256']!=digest or x['sheet_ordinal']!=ordinal-1 for x in (target,named))
                or named['cell_coordinate']!=col+'2' or target['cell_coordinate']==named['cell_coordinate']
                or tuple(r['exact_original_published_caption_path'])!=CAPTION or r['source_study']!=WORK
                or r['source_authoritative_article_projection']!=projection or r['original_role']is not None
                or r['prospective_role']is not None or r['training_authorized']is not False):
            raise ValueError('exact native original position, source scope and roles required')
        coords.add((name,target['cell_coordinate']));literal=header(cells[col+'2'])
        if i==0:
            known=dict(source_measurement_location=literal);unit='um/s';control='none'
            coordinate='source-reported derived wave speed'
            baseline=NativeBioelectricBaseline('Xenopus laevis','calcium excitation wave',None,None,None,literal)
            event=NativeBioelectricEvent('other_reported_perturbation','source injury assay','unknown',(),None,None,None,None)
            endpoint=NativeBioelectricEndpoint('wave speed','CEMA native source wave speed assay',unit,'native_scalar',None)
        else:
            context=('injured','receiver')[i-1];unit=None;control='reported_comparator'
            known=dict(source_exposure=literal,source_subject_context=context)
            coordinate='source-normalized peak GCaMP6S intensity'
            baseline=NativeBioelectricBaseline('Xenopus laevis',context+' embryo',None,None,None)
            event=NativeBioelectricEvent('untreated_control' if literal=='control' else 'other_reported_perturbation',
                'source-reported exposure arm','not_applicable' if literal=='control' else 'unknown',
                (),literal,None,None,None)
            endpoint=NativeBioelectricEndpoint('peak GCaMP6S fluorescence','CEMA '+context+' embryo normalized source peak assay',
                None,'native_scalar',None)
        literal_axis=('between','injured','receiver') if i==0 else ('control','suramin')
        if (literal not in literal_axis or r['known_source_assay_context']!=known or r['original_measurement_unit']!=unit
                or r['literal_source_reporting_coordinate']!=coordinate):
            raise ValueError('exact original source condition, coordinate and unit required')
        arms[i,literal_axis.index(literal)]+=1
        c=cells[target['cell_coordinate']];lex=value['original_value_lexeme']
        observation=value['source_value_float64']
        if type(observation)is not float or struct.pack('>d',observation).hex()!=value['source_Float64_bits']:
            raise ValueError('exact native original Float64 representation required')
        numeric=BioelectricWorkbookNumericCell(workbook['path'],workbook['bytes'],workbook['sha256'],name,digest,
            target['cell_coordinate'],c.get('t'),lex)
        scalar=ScalarTarget('reported_point',observation,None,None,None,None,None,lex,
            'exact-original-openxml-v-number-lexeme',b'{}',None,None,b'{}')
        row=NativeBioelectricObservation(baseline,(event,),endpoint,'primary_observation',control,WORK,WORK+'#'+panel,
            'PMC10794468.1',(span,numeric),numeric_target=scalar)
        row.verify_evidence(root);out.append(row)
    if arms!=Counter({(i,j):n for i,ns in enumerate(EXPECTED)for j,n in enumerate(ns)}):
        raise ValueError('complete original point counts by source arm required')
    return tuple(out)
