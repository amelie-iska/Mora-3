"""Bounded GOA source validation; raw evidence stays immutable and form scoped."""
import collections
import math
import re

ACCESSION=re.compile(r'(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})')

def query_params(accessions,page,limit=200):
    values=list(dict.fromkeys(accessions))
    if not 1<=len(values)<=100 or any(not isinstance(x,str) or not ACCESSION.fullmatch(x) for x in values):
        raise ValueError('query must contain1..100exact canonical UniProt accessions')
    if not 1<=page<=10000 or not 1<=limit<=200:raise ValueError('page/limit exceed documented caps')
    return dict(geneProductId=','.join('UniProtKB:'+x for x in values),geneProductType='protein',limit=limit,page=page)

def parent_accession(subject,allowed):
    if not isinstance(subject,str) or not subject.startswith('UniProtKB:'):raise ValueError('foreign source namespace')
    text=subject[len('UniProtKB:'):];root=text.split('-',1)[0]
    if not ACCESSION.fullmatch(root) or root not in allowed:raise ValueError('subject outside exact accession scope')
    if '-' in text and not re.fullmatch(r'(?:[0-9]+|PRO_[0-9]+)',text.split('-',1)[1]):
        raise ValueError('unrecognized UniProt form syntax; retain/quarantine rather than collapse')
    return root

def check_snapshot(response,expected):
    if response.get('annotation',{}).get('timestamp')!=expected:raise ValueError('GOA snapshot drift or unavailable version')

def summarize_pages(pages,accessions,limit=200):
    allowed=set(accessions);query_params(accessions,1,limit);seen=set();subjects=collections.Counter();counts=collections.Counter();hits=None;expected_pages=None
    for i,page in enumerate(pages,1):
        n=page.get('numberOfHits');info=page.get('pageInfo',{});rows=page.get('results')
        if not isinstance(n,int) or n<0 or not isinstance(rows,list):raise ValueError('invalid annotation response schema')
        if hits is None:hits=n;expected_pages=max(1,math.ceil(n/limit))
        if n!=hits or expected_pages>10000:raise ValueError('changing count or query exceeds pagination; split batch')
        if info.get('resultsPerPage')!=limit or info.get('total')!=(math.ceil(n/limit) if n else 0):raise ValueError('inconsistent server page metadata')
        if info.get('current')!=(i if n else 0) or i>expected_pages:raise ValueError('missing/extra page')
        if len(rows)!=(min(limit,max(0,n-(i-1)*limit))):raise ValueError('truncated or inconsistent page')
        for row in rows:
            for key in ['id','geneProductId','goId','qualifier','evidenceCode','goEvidence','reference','assignedBy','date','withFrom','extensions']:
                if key not in row:raise ValueError('missing source field '+key)
            if row['id'] in seen:raise ValueError('duplicate annotation ID across pages')
            seen.add(row['id']);parent_accession(row['geneProductId'],allowed);subjects[row['geneProductId']]+=1
            counts['negated_rows']+='NOT' in str(row['qualifier']).split('|')
            counts['with_from_rows']+=bool(row['withFrom']);counts['extension_rows']+=bool(row['extensions'])
    if hits is None or i!=expected_pages or len(seen)!=hits:raise ValueError('incomplete annotation pages')
    return dict(annotation_rows=len(seen),source_annotation_ids_unique=True,source_declared_rows=hits,subject_counts=dict(subjects),negated_rows=counts['negated_rows'],with_from_rows=counts['with_from_rows'],extension_rows=counts['extension_rows'],missing_annotations_create_negative_labels=False,task_admitted_rows=0)

def gaf_info(text,accessions):
    allowed=set(accessions);query_params(accessions,1);lines=text.splitlines();subjects=collections.Counter();count=forms=negated=0
    if '!gaf-version: 2.2' not in lines:raise ValueError('observed source format must be GAF2.2')
    for line in lines:
        if not line or line.startswith('!'):continue
        fields=line.split('\t')
        if len(fields)!=17:raise ValueError('GAF2.2 requires17columns including trailing empty formID')
        canonical=fields[0]+':'+fields[1];parent_accession(canonical,allowed)
        subject=fields[16] or canonical;parent_accession(subject,allowed)
        if parent_accession(canonical,allowed)!=parent_accession(subject,allowed):raise ValueError('form/base source mismatch')
        subjects[subject]+=1;count+=1;forms+=bool(fields[16]);negated+='NOT' in fields[3].split('|')
    return dict(annotation_rows=count,form_rows=forms,negated_rows=negated,subject_counts=dict(subjects),version='2.2')

def retry_delay(status,attempt,retry_after):
    if attempt>=2 or status not in (408,429,500,502,503,504):raise ValueError('terminal/deferred source request')
    delay=[5,15][attempt]
    if retry_after is not None:
        try:delay=max(delay,int(retry_after))
        except (ValueError,TypeError):raise ValueError('unparsed Retry-After; defer instead of early retry')
    if delay>300:raise ValueError('service asks longer delay; defer finite job')
    return delay


def source_expression(groups,relations=False):
    return '|'.join(','.join((x['relation']+'('+x['db']+':'+x['id']+')') if relations else x['db']+':'+x['id'] for x in group['connectedXrefs']) for group in (groups or []))

def native_gaf_key(row):
    return (row['geneProductId'],row['qualifier'],row['goId'],row['reference'],row['goEvidence'],source_expression(row['withFrom']),row['date'],row['assignedBy'],source_expression(row['extensions'],True))

def gaf_semantic_keys(text):
    for line in text.splitlines():
        if not line or line.startswith('!'):continue
        f=line.split('\t')
        yield (f[16] or f[0]+':'+f[1],f[3],f[4],f[5],f[6],f[7],f[13],f[14],f[15])

def acquire_batch(accessions,request,snapshot,on_inventory=None):
    query_params(accessions,1)
    check_snapshot(request('annotation/about',{},'snapshot-before'),snapshot)
    probe=request('annotation/search',query_params(accessions,1,1),'count-probe')
    count=probe.get('numberOfHits')
    if not isinstance(count,int) or count<0:raise ValueError('invalid source count probe')
    total=max(1,math.ceil(count/200))
    if total>10000:raise ValueError('source query exceeds2000000rows; split accession batch')
    if on_inventory:on_inventory(dict(expected_rows=count,expected_json_pages=total,assets=['page-%05d.json'%i for i in range(1,total+1)]+['complete.gaf'],expected_byte_lengths='not supplied by generated API; per-response hashes/lengths recorded and raw budget enforced'))
    pages=[request('annotation/search',query_params(accessions,i),'page-%05d'%i) for i in range(1,total+1)]
    qc=summarize_pages(pages,accessions)
    if qc['annotation_rows']!=count:raise ValueError('source count changed after preflight')
    gaf=request('annotation/downloadSearch',query_params(accessions,1),'complete.gaf','text/gaf')
    gaf_qc=gaf_info(gaf,accessions)
    if gaf_qc['annotation_rows']!=count or gaf_qc['subject_counts']!=qc['subject_counts']:raise ValueError('GAF complete-export/native count or form mismatch')
    native=collections.Counter(native_gaf_key(r) for page in pages for r in page['results'])
    if native!=collections.Counter(gaf_semantic_keys(gaf)):raise ValueError('GAF/native evidence semantic mismatch')
    check_snapshot(request('annotation/about',{},'snapshot-after'),snapshot)
    return dict(native_QC=qc,GAF_QC=gaf_qc,source_snapshot=snapshot,available_annotations=count,quality_pass_annotations=count,task_admitted_annotations=0,full_batch_scan=True,all_original_JSON_fields_retained=True,GAF_native_semantic_multiset_equal=True,query_canonical_accessions=list(accessions),forms_are_distinct_source_subjects=True)
