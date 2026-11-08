"""Exact original CEMA Figure 8g source readout, with no assigned train/test role."""
from collections import Counter
from io import BytesIO
from pathlib import Path
import hashlib
import json
import struct
import zipfile

from defusedxml.ElementTree import fromstring
from .bioelectric_evidence import (BioelectricJATSEvidenceSpan,BioelectricWorkbookNumericCell,
    NativeBioelectricBaseline,NativeBioelectricEndpoint,NativeBioelectricEvent,NativeBioelectricObservation)
from .source_scalar_native_rows import ScalarTarget


NS='{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
MEMBER='xl/worksheets/sheet19.xml'
CAPTION=(2,1,8,1,4,1)
WORK='doi:10.1038/s41467-023-44522-2'
CELLS=('A3','A4','A5','B3','B4','B5','B6')


def _sha(x):return hashlib.sha256(x).hexdigest()


def load_cema_figure8g_source(root,manifest_path,expected_sha256):
    root=Path(root).resolve()
    def read(path):
        p=(root/path).resolve();p.relative_to(root)
        with p.open('rb')as f:b=f.read(32*2**20+1)
        if len(b)>32*2**20:raise ValueError('bounded exact CEMA source metadata required')
        return b
    def exact(ref):
        b=read(ref['path'])
        if len(b)!=ref['bytes'] or _sha(b)!=ref['sha256']:raise ValueError('CEMA original source reference differs')
        return b
    b=read(manifest_path)
    if _sha(b)!=expected_sha256:raise ValueError('exact CEMA source-cell evidence manifest SHA required')
    m=json.loads(b)
    if (m.get('schema')!='Mora3.CEMACompleteNativeWorkbookCellAndFigureEvidence/v1'
            or m.get('literal_work_lineage_group')!=WORK
            or m.get('original_or_prospective_split_assigned') is not False
            or m.get('retained_Figure8g_native_experimental_source_candidates')!=7):
        raise ValueError('exact source-only seven-observation CEMA scope required')
    rows=json.loads(exact(m['full_Figure8g_native_source_scalar_candidates']))
    if tuple(r['target']['source_cell']['original_cell']for r in rows)!=CELLS:
        raise ValueError('all original control and ATP source cells required')
    workbook=rows[0]['original_workbook'];raw=exact(workbook)
    with zipfile.ZipFile(BytesIO(raw))as archive:
        if any(archive.namelist().count(n)!=1 for n in (MEMBER,'xl/sharedStrings.xml')):
            raise ValueError('exact original workbook members required')
        sheet=archive.read(MEMBER);strings=fromstring(archive.read('xl/sharedStrings.xml'),forbid_dtd=True)
    xml=fromstring(sheet,forbid_dtd=True);cell_index={}
    for cell in xml.findall(f'{NS}sheetData/{NS}row/{NS}c'):
        coordinate=cell.get('r')
        if coordinate in cell_index:raise ValueError('duplicate native source cell')
        cell_index[coordinate]=cell
    def shared_text(ref):
        cell=cell_index[ref]
        if cell.get('t')!='s' or len(cell)!=1 or cell[0].tag!=NS+'v':raise ValueError('original shared string cell required')
        return ''.join(t.text or '' for t in strings[int(cell[0].text)].iter(NS+'t'))
    if (shared_text('A1'),shared_text('A2'),shared_text('B2'))!=('peak intensity of receiver embryo','control','ATP'):
        raise ValueError('original source endpoint or comparator labels differ')
    projection=json.loads(exact(m['exact_original_article_JATS_projection']));qual=projection['qualification']
    if (qual.get('source_candidate') is not True or qual.get('expected_PMCID')!='PMC10794468'
            or 'CC-BY-4.0' not in qual.get('selected_original_grant_identifiers',[])):
        raise ValueError('exact original source article and grant required')
    ref=projection['original_source_ref'];caption_slots=[]
    for x in projection['native_character_data_segments']:
        p=tuple(x['source_child_ordinal_path'])
        if p[:len(CAPTION)]==CAPTION and (p!=CAPTION or x['slot']=='text'):
            caption_slots.append(x)
    cap=''.join(x['text_exact_native']for x in caption_slots).casefold()
    if not all(x in cap for x in ('gcamp6s','atp','mmr')):raise ValueError('native Figure 8g caption evidence differs')
    match=next(x for x in caption_slots if 'gcamp6s' in x['text_exact_native'].casefold())
    text=match['text_exact_native'];digest=_sha(text.encode('utf8'))
    span=BioelectricJATSEvidenceSpan(ref['path'],ref['bytes'],ref['sha256'],tuple(match['source_child_ordinal_path']),
        match['slot'],0,len(text),digest,digest)
    span.read(root)
    target_rows=[];exposure_counts=Counter()
    for row in rows:
        target=row['target'];coord=target['source_cell'];cell=coord['original_cell'];column=cell[0]
        header=row['predictor_condition_evidence']['native_group_header_cell']
        exposure='MMR medium (control)' if column=='A' else 'ATP'
        known=dict(exposure=exposure,organism='Xenopus laevis',readout='peak GCaMP6S fluorescence')
        if (row['schema']!='Mora3.CEMANativeSourceScalarCandidate/v1' or row['source_group']!=WORK
                or row['original_workbook']!=workbook or row['exact_original_JATS_projection']!=m['exact_original_article_JATS_projection']
                or tuple(row['exact_original_caption_path'])!=CAPTION or row['assay_input_condition']!=known
                or row['original_role'] is not None or row['prospective_role'] is not None or row['original_unit'] is not None
                or row['training_authorized'] is not False or row['outcome_dependent_groups_used_as_conditions'] is not False
                or header['original_cell']!=column+'2' or coord['original_archive_member_path']!=MEMBER
                or coord['original_archive_member_SHA256']!=_sha(sheet) or coord['source_sheet_ordinal']!=18):
            raise ValueError('exact source target-free condition, provenance and unassigned role required')
        cell_proof=BioelectricWorkbookNumericCell(workbook['path'],workbook['bytes'],workbook['sha256'],MEMBER,
            coord['original_archive_member_SHA256'],cell,cell_index[cell].get('t'),target['original_number_lexeme'])
        value=target['reported_native_Float64']
        if type(value) is not float or struct.pack('>d',value).hex()!=target['original_float64_bits']:
            raise ValueError('exact original CEMA Float64 coordinate required')
        scalar=ScalarTarget('reported_point',value,None,None,None,None,None,target['original_number_lexeme'],
            'exact-original-openxml-v-number-lexeme',b'{}',None,None,b'{"source_coordinate":"normalized reported intensity; original unit unspecified"}')
        observation=NativeBioelectricObservation(
            NativeBioelectricBaseline('Xenopus laevis','receiver embryo',None,None,None),
            (NativeBioelectricEvent('vehicle_control' if column=='A' else 'other_reported_perturbation',
                'source-reported bolus assay','not_applicable' if column=='A' else 'unknown',(),exposure,
                None,None,None),),
            NativeBioelectricEndpoint('peak intensity of receiver embryo','peak GCaMP6S fluorescence; published normalized source coordinate',
                None,'native_scalar',None),'primary_observation','reported_comparator',WORK,WORK+'#source-Figure8g',
            'PMC10794468.1',(span,cell_proof),numeric_target=scalar)
        observation.verify_evidence(root);target_rows.append(observation);exposure_counts[exposure]+=1
    if exposure_counts!={'MMR medium (control)':3,'ATP':4}:raise ValueError('complete exact observed Figure 8g cohorts required')
    return tuple(target_rows)
