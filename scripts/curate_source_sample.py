#!/usr/bin/env python3
"""Curate bounded, deterministic Parquet graph samples without changing sources.

This is an inspection/integration slice, not a full training-ready data release.
Original split labels and every selected source field are retained. Full source
license, annotation quality and cross-source leakage audits remain separate gates.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import pyarrow as pa
import pyarrow.parquet as pq


def fingerprint(path):
    stat=path.stat()
    with path.open('rb') as stream:
        header=stream.read(65536)
        stream.seek(max(0,stat.st_size-65536))
        footer=stream.read(65536)
    return dict(size_bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,
                header_footer_sha256=hashlib.sha256(header+footer).hexdigest(),
                fingerprint_kind='stat-and-first-last-64KiB; not full-file checksum')


def valid_graph(row):
    try:
        graph=json.loads(row['graph_json'])
        return (isinstance(graph,dict) and bool(row.get('id') or row.get('record_id') or graph.get('id')) and
                isinstance(graph.get('nodes'),list) and bool(graph['nodes']) and
                isinstance(graph.get('edges'),list))
    except (KeyError,TypeError,ValueError):
        return False


def curate(sources, output, *, rows_per_source=16, scan_rows=128, seed=0):
    if not sources or rows_per_source<=0 or scan_rows<rows_per_source:
        raise ValueError('Require sources and 0 < rows_per_source <= scan_rows')
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    manifest=dict(schema_version=1,seed=seed,rows_per_source=rows_per_source,
                  scan_rows_limit=scan_rows,training_ready=False,
                  outstanding_gates=['source license verification','UniProt annotation quality >=4 where applicable',
                                     'cross-source and similarity leakage audit','all required modality coverage',
                                     'Mora-3 schema conversion and target-mask validation'],sources=[])
    for index, source in enumerate(sources):
        source=Path(source).resolve(strict=True)
        before=fingerprint(source)
        parquet=pq.ParquetFile(source)
        rng=random.Random(seed+index)
        selected=[];seen=set();scanned=accepted=rejected=0
        for batch in parquet.iter_batches(batch_size=min(32,scan_rows)):
            for row in batch.to_pylist():
                if scanned>=scan_rows:
                    break
                offset=scanned;scanned+=1
                if not valid_graph(row):
                    rejected+=1;continue
                identity=row.get('id') or row.get('record_id') or json.loads(row['graph_json']).get('id')
                if identity in seen:
                    rejected+=1;continue
                seen.add(identity);accepted+=1
                item=(offset,row)
                if len(selected)<rows_per_source:
                    selected.append(item)
                else:
                    slot=rng.randrange(accepted)
                    if slot<rows_per_source:selected[slot]=item
            if scanned>=scan_rows:break
        after=fingerprint(source)
        if before!=after:raise RuntimeError('Source changed during sampling; sample is not ready')
        selected.sort(key=lambda item:item[0])
        target=output/('sample-%03d.parquet'%index)
        pq.write_table(pa.Table.from_pylist([row for _,row in selected],schema=parquet.schema_arrow),target)
        manifest['sources'].append(dict(source_path=str(source),source_fingerprint=before,
            source_unchanged=True,source_total_rows=parquet.metadata.num_rows,
            scanned_rows=scanned,accepted_rows=accepted,rejected_rows=rejected,
            selected_rows=len(selected),selected_source_row_offsets=[offset for offset,_ in selected],
            original_splits=sorted({str(row.get('split','unknown')) for _,row in selected}),
            original_sources=sorted({str(row.get('source') or row.get('dataset') or json.loads(row['graph_json']).get('source','unknown')) for _,row in selected}),
            sample_file=target.name,sample_sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    with (output/'manifest.json').open('x') as stream:
        json.dump(manifest,stream,indent=2);stream.write('\n')
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,action='append',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--rows-per-source',type=int,default=16)
    parser.add_argument('--scan-rows',type=int,default=128)
    parser.add_argument('--seed',type=int,default=0)
    args=parser.parse_args()
    result=curate(args.source,args.output,rows_per_source=args.rows_per_source,scan_rows=args.scan_rows,seed=args.seed)
    print(json.dumps({'manifest':str(args.output/'manifest.json'),
                      'selected_rows':sum(s['selected_rows'] for s in result['sources']),
                      'source_count':len(result['sources']),'training_ready':False}))


if __name__=='__main__':main()
