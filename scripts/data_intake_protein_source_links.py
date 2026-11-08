"""Exact current-baseline to source-representative links; never cluster label transfer."""
import hashlib,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from data_intake_corpus import annotation_score

def representative_links(source,baseline):
    seq=source['sequence'];digest=hashlib.sha256(seq.encode()).hexdigest();links=[];quarantines=[]
    for accession in sorted(set(source['rep_accessions'] or [])):
        current=baseline.get(accession)
        if current is None:continue
        current_sha=hashlib.sha256(current['Sequence'].encode()).hexdigest()
        common=dict(qualified_accession=accession,source_cluster_id=source['id'],source_sequence_sha256=digest,current_qualified_sequence_sha256=current_sha,source_representative_member_id=source['rep_member_id'],source_seed_id=source['seed_id'],source_representative_is_seed=source['rep_is_seed'])
        try:
            score=annotation_score(current['Annotation'])
            if current['Reviewed']!='reviewed':raise ValueError('unreviewed')
        except ValueError:
            quarantines.append(dict(common,reason='current_baseline_reviewed_score_gate_failed'));continue
        if seq!=current['Sequence']:
            quarantines.append(dict(common,reason='representative_sequence_differs_from_current_qualified_accession'));continue
        links.append(dict(common,current_annotation_score=score,current_baseline_release='2026_03',underlying_source_UniRef_release=None,identity_scope='explicit_representative_accession_AND_exact_native_sequence;not_every_member_or_seed',annotation_transfer_allowed=False,new_protein_sequence_admitted=False,homology_split_safe=False,groups=['accession:'+accession,'sequence:'+digest,'source_UniRef_representative_reference:'+source['id']]))
    return links,quarantines
