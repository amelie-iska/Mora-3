"""Finite full GO term collection, independently versioned from GOA annotations."""
import argparse
import collections
import datetime
import hashlib
import json
import math
import os
import pathlib
import time
import pyarrow as pa
import pyarrow.parquet as pq
from data_sources_goa_acquire_v2 import EfficientSkillClient
from data_sources_go_ontology import check_version,validate_page,project_term
from data_sources_goa_acquire import write_new,sha,now

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);args=parser.parse_args();pp=pathlib.Path(args.plan).resolve();root=pp.parent;plan=json.loads(pp.read_text());job=plan['job_id'];rawroot=root/'raw'/job;rawroot.mkdir(exist_ok=False);out=root/'curated'/job;out.mkdir(exist_ok=False)
    if plan['workers']!=1 or plan['aggregate_raw_max_bytes_including_new_ontology']>80*1024**3:raise ValueError('sourcebudget/concurrency gate')
    write_new(root/('lease-'+job+'.json'),dict(pid=os.getpid(),started_utc=now(),plan=str(pp),source_connections=1,ontology_timestamp=plan['ontology_timestamp']))
    client=EfficientSkillClient(root,plan);control=rawroot/'controls';control.mkdir();client.batch_dir=control;timestamp=plan['ontology_timestamp'];version=plan['ontology_version'];started=time.monotonic();seen=set();counts=collections.Counter();nodes=[];edges=[];nodefiles=[];edgefiles=[];sources=[];derived=0;parts=[];last_published_nodefile=0;last_published_edgefile=0
    ns=pa.schema([('go_id',pa.string()),('name',pa.string()),('aspect',pa.string()),('is_obsolete',pa.bool_()),('ontology_version',pa.string()),('source_raw_path',pa.string()),('source_result_index',pa.int64()),('native_term_sha256',pa.string()),('native_term_json',pa.large_string())]);es=pa.schema([('child_id',pa.string()),('relation',pa.string()),('parent_id',pa.string()),('ontology_version',pa.string()),('source_raw_path',pa.string()),('source_result_index',pa.int64())])
    live=root/('live-'+job+'.json')
    def status(phase,page=0,error=None):
        live.write_text(json.dumps(dict(updated_utc=now(),pid=os.getpid(),phase=phase,page=page,pages=plan['pages'],available_scanned_terms=len(seen),expected_terms=plan['full_term_count'],counts=dict(counts),raw_job_bytes=client.raw_bytes(),derived_bytes=derived,requests=client.request_count,elapsed_seconds=time.monotonic()-started,ontology_timestamp=timestamp,annotation_timestamp=plan['annotation_snapshot'],same_snapshot=False,complete_part_manifests=parts,error=error,source_connections=1),indent=2)+'\n')
    def flush():
        nonlocal derived
        for kind,rows,schema,files in [('nodes',nodes,ns,nodefiles),('current-child-edges',edges,es,edgefiles)]:
            if not rows:continue
            p=out/('%s-%04d.parquet'%(kind,len(files)));pq.write_table(pa.Table.from_pylist(rows,schema=schema),p,compression='zstd')
            if pq.read_table(p).to_pylist()!=rows:raise ValueError('ontology native projection roundtrip')
            files.append(dict(path=str(p),rows=len(rows),bytes=p.stat().st_size,sha256=sha(p),exact_decoded_roundtrip=True));derived+=p.stat().st_size;rows.clear()
        if derived>plan['derived_cap_bytes']:raise ValueError('ontologyderivedcap; originalspreserved')
    status('live finite native ontology acquisition')
    try:
        check_version(client.request('ontology/go/about',{},'ontology-before'),timestamp,version)
        for page in range(1,plan['pages']+1):
            group=(page-1)//50;directory=rawroot/('group-%04d'%group)
            if not directory.exists():directory.mkdir()
            client.batch_dir=directory
            if page==1:
                path=pathlib.Path(plan['reuse_first_page']['path'])
                if sha(path)!=plan['reuse_first_page']['sha256']:raise ValueError('seedterm page changed')
                response=json.loads(path.read_text())
            else:
                response=client.request(plan['page_endpoint'],dict(page=page),'page-%04d'%page);path=pathlib.Path(json.loads(pathlib.Path(client.receipts[-1]).read_text())['source_response_path'])
            validate_page(response,page,plan['full_term_count'],plan['page_size']);sources.append(dict(page=page,path=str(path),bytes=path.stat().st_size,sha256=sha(path),rows=len(response['results'])))
            for index,term in enumerate(response['results']):
                if term['id'] in seen:raise ValueError('duplicateGOIDacrosspages; no completeontologyclaim')
                seen.add(term['id']);node,current_edges=project_term(term,str(path),index,version);nodes.append(node);edges.extend(current_edges);counts['native_term_rows']+=1;counts['obsolete_term_rows']+=node['is_obsolete'];counts['missing_aspect_rows']+=node['aspect'] is None;counts['native_current_child_edges']+=len(current_edges);counts['native_history_entries']+=len(term.get('history') or [])
            if len(nodes)>=5000:flush()
            if page%50==0 or page==plan['pages']:
                check_version(client.request('ontology/go/about',{},'ontology-after-page-%04d'%page),timestamp,version)
                if len(nodefiles)>last_published_nodefile:
                    part=out/('part-%04d-manifest.json'%len(parts));write_new(part,dict(schema='mora3.GO_native_ontology_completed_part.v1',completed_utc=now(),ontology_timestamp=timestamp,ontology_version=version,annotation_snapshot=plan['annotation_snapshot'],same_snapshot=False,complete_through_page=page,nodes=nodefiles[last_published_nodefile:],current_child_edges=edgefiles[last_published_edgefile:],raw_sources=sources,source_history_not_current_edges=True,source_original_fields_retained=True,task_admitted_rows=0));parts.append(str(part));last_published_nodefile=len(nodefiles);last_published_edgefile=len(edgefiles)
                status('native ontology pages and source-version checks advancing',page);print('GO_ONTOLOGY_PROGRESS',page,len(seen),flush=True)
        flush();client.batch_dir=control;check_version(client.request('ontology/go/about',{},'ontology-final'),timestamp,version)
        if len(seen)!=plan['full_term_count']:raise ValueError('ontology uniqueID count incomplete')
        manifest=dict(schema='mora3.GO_native_ontology_full_source.v1',completed_utc=now(),source='EBIQuickGO ontologygo',ontology_timestamp=timestamp,ontology_version=version,annotation_snapshot=plan['annotation_snapshot'],same_version_as_annotation=False,license='CC-BY-4.0',source_review=str(root/'research/GO-native-ontology-source-review-v1.json'),native_GO_terms=len(seen),source_pages=plan['pages'],source_counts=dict(counts),all_source_terms_scanned=True,all_native_fields_preserved=True,current_children_edges_only=True,history_not_current_relations=True,nodes=nodefiles,current_child_edges=edgefiles,raw_sources=sources,raw_job_bytes=client.raw_bytes()+plan['reuse_first_page']['bytes'],derived_bytes=derived,source_version_checked_before_during_after=True,annotation_names_filled=False,task_admitted_rows=0,reader_support=False,admission_owner='dataset_construction')
        write_new(out/'manifest.json',manifest);status('finite full native ontology acquisition and QC completed',plan['pages']);print('GO_ONTOLOGY_FINISHED',len(seen),counts['native_current_child_edges'],flush=True)
    except Exception as exc:status('stopped; source pages and partial projections retained',page if 'page' in locals() else 0,type(exc).__name__+':'+str(exc));raise
if __name__=='__main__':main()
