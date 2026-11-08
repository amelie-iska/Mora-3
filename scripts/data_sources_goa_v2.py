"""GOA native evidence validation with full GPAD for null GO short-code assertions."""
import collections
import math
from data_sources_goa import query_params,parent_accession,check_snapshot,summarize_pages,gaf_info,native_gaf_key,gaf_semantic_keys,source_expression

def native_gpad_key(row):
    taxon=row.get('interactingTaxonId');taxon='' if taxon in (None,0) else 'taxon:'+str(taxon)
    return (row['geneProductId'],row['qualifier'],row['goId'],row['reference'],row['evidenceCode'],source_expression(row['withFrom']),taxon,row['date'],row['assignedBy'],source_expression(row['extensions'],True),'goEvidence='+(row['goEvidence'] or ''))

def gpad_keys(text,accessions):
    lines=text.splitlines()
    if '!gpa-version: 1.1' not in lines:raise ValueError('observed source GPAD1.1 required')
    for line in lines:
        if not line or line.startswith('!'):continue
        f=line.split('\t')
        if len(f)!=12:raise ValueError('GPAD1.1 requires12columns')
        subject=f[0]+':'+f[1];parent_accession(subject,set(accessions))
        yield (subject,f[2],f[3],f[4],f[5],f[6],f[7],f[8],f[9],f[10],f[11])

def crosscheck_exports(rows,accessions,gaf,gpad=None):
    representable=[r for r in rows if r['goEvidence']];unrepresentable=[r for r in rows if not r['goEvidence']]
    gqc=gaf_info(gaf,accessions);native=collections.Counter(native_gaf_key(r) for r in representable);export=collections.Counter(gaf_semantic_keys(gaf))
    if gqc['annotation_rows']!=len(representable) or native!=export:raise ValueError('GAFcount/fullsemantics do not equal every representable native assertion')
    gpad_match=False;gpad_count=None
    if unrepresentable and gpad is None:raise ValueError('null GOshortcode requires independent full-count GPAD/ECOcrosscheck')
    if gpad is not None:
        expected=collections.Counter(native_gpad_key(r) for r in rows);actual=collections.Counter(gpad_keys(gpad,accessions));gpad_count=sum(actual.values())
        if gpad_count!=len(rows) or actual!=expected:raise ValueError('GPADfullcount/ECO/qualifier/form/withfrom/extensions/evidenceproperty mismatch')
        gpad_match=True
    return dict(native_annotation_rows=len(rows),GAF_QC=gqc,GAF_representable_native_rows=len(representable),GAF_unrepresentable_native_rows=len(unrepresentable),GAF_unrepresentable_native_source_IDs=[r['id'] for r in unrepresentable],GAF_matches_every_representable_native_assertion=True,GAF_native_semantic_multiset_equal=not unrepresentable,GPAD_annotation_rows=gpad_count,GPAD_full_native_semantic_multiset_equal=gpad_match,all_native_count_and_semantics_crosschecked=True,independent_full_export='GPAD1.1ECO' if unrepresentable else 'GAF2.2',null_GO_shortcodes_preserved_without_inference=True)

def acquire_batch(accessions,request,snapshot,on_inventory=None):
    query_params(accessions,1);check_snapshot(request('annotation/about',{},'snapshot-before'),snapshot)
    probe=request('annotation/search',query_params(accessions,1,1),'count-probe');count=probe.get('numberOfHits')
    if not isinstance(count,int) or count<0:raise ValueError('invalidsourcecount')
    total=max(1,math.ceil(count/200))
    if total>10000:raise ValueError('query exceeds paginationcap; split')
    if on_inventory:on_inventory(dict(expected_rows=count,expected_json_pages=total,assets=['page-%05d.json'%i for i in range(1,total+1)]+['complete.gaf'],conditional_source_assets=['complete.gpad required whenever any source GOshortcode is missing'],expected_byte_lengths='generatedAPI lengthsunsupplied; perresponsehashes/lengths/budgets recorded'))
    pages=[request('annotation/search',query_params(accessions,i),'page-%05d'%i) for i in range(1,total+1)];qc=summarize_pages(pages,accessions)
    if qc['annotation_rows']!=count:raise ValueError('sourcecountdrift')
    rows=[r for page in pages for r in page['results']];gaf=request('annotation/downloadSearch',query_params(accessions,1),'complete.gaf','text/gaf');gpad=request('annotation/downloadSearch',query_params(accessions,1),'complete.gpad','text/gpad') if any(not r['goEvidence'] for r in rows) else None
    cross=crosscheck_exports(rows,accessions,gaf,gpad);check_snapshot(request('annotation/about',{},'snapshot-after'),snapshot)
    return dict(native_QC=qc,source_snapshot=snapshot,available_annotations=count,quality_pass_annotations=count,task_admitted_annotations=0,full_batch_scan=True,all_original_JSON_fields_retained=True,query_canonical_accessions=list(accessions),forms_are_distinct_source_subjects=True,validation_schema='mora3.GOA_native_GPAD_nullshortcode.v2',**cross)
