"""Candidates are filenames only; full source QC and raw semantic checks still apply."""
from pathlib import Path

def candidate_source_manifests(source_root):
    root=Path(source_root);fixed=[root/'handoff-manifest-sample-v1.json',root/'curated/manifest-pilot100-v1-batch-0000.json']
    legacy=sorted((root/'curated').glob('manifest-full-part-*.json'))
    parallel=sorted((root/'curated').glob('manifest-quickgo-parallel-v2-annotation-*-batch-*.json'))
    result=[p for p in fixed if p.is_file()]+legacy+parallel
    if len(result)!=len(set(result)):raise ValueError('duplicate source manifest path')
    return result
