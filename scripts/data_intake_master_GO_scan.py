"""Exact native GO-ID exposure summaries; proposal-only, never corpus mutation."""
import collections,re
from data_intake_heldout_GO_policy_v2 import GO_mentions,task_role_surfaces,GOA_projection_proposal
ACCESSION=re.compile(r'(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})')

def subject_scope(literal):
    """Classify literal subject syntax only; no aliases, parent or label transfer."""
    if not isinstance(literal,str):return 'missing_literal_subject'
    if re.search(r'(?:^|:)PRO_[0-9]+(?:$|[^0-9])',literal):return 'explicit_PRO_subject'
    tail=literal.split(':',1)[1] if literal.startswith('UniProtKB:') else literal
    if re.fullmatch(ACCESSION.pattern+r'-[0-9]+',tail):return 'explicit_isoform_subject'
    if ACCESSION.fullmatch(tail):return 'canonical_accession_syntax_only'
    return 'other_literal_entity_subject'

def row_exposure(packet,native_schema,source_family,protected):
    surfaces=task_role_surfaces(packet,native_schema)
    byrole=collections.defaultdict(set);hits=[]
    for role,surface in surfaces:
        for mention in GO_mentions(surface):
            byrole[role].add(mention['GO_ID'])
            if mention['GO_ID'] in protected:hits.append(dict(role=role,**mention))
    target=any(r.startswith('target') for r in [h['role'] for h in hits])
    condition=any(r.startswith('condition') for r in [h['role'] for h in hits])
    result=dict(GO_IDs_by_native_role={r:sorted(v) for r,v in byrole.items()},protected_native_role_mentions=hits,
        protected_target_exposure=target,protected_condition_exposure=condition,
        proposed_coarse_entire_view_exclusion=target or condition,proposal_only=True,policy_activated=False,
        lexical_ID_namespace_exposure_is_not_assertion_equivalence_or_actual_optimizer_exposure=True)
    if source_family=='UniProt-GOA-assertion-packets':
        raw=packet['targets']['source_GO_annotations'];typed=packet['targets']['typed_GO_annotations']
        by_raw={a['id']:a for a in raw};by_typed={a['source_annotation_id']:a for a in typed};raw_indices={a['id']:i for i,a in enumerate(raw)};typed_indices={a['source_annotation_id']:i for i,a in enumerate(typed)}
        if len(by_raw)!=len(raw) or len(by_typed)!=len(typed) or set(by_raw)!=set(by_typed):raise ValueError('GOA duplicate/missing raw-typed pairing')
        annotations=[]
        for annotation_id,a in by_raw.items():
            t=by_typed[annotation_id]
            if a['geneProductId']!=t['subject'] or a['goId']!=t['go_id']:raise ValueError('GOA raw/typed literal subject or root GO mismatch')
            matched={m['GO_ID'] for m in GO_mentions(a)+GO_mentions(t)}&protected
            if matched:
                annotations.append(dict(source_annotation_ID=annotation_id,literal_subject=a['geneProductId'],subject_scope=subject_scope(a['geneProductId']),root_GO_ID=a['goId'],root_GO_is_protected=a['goId'] in protected,
                    protected_IDs_in_entire_atomic_annotation=sorted(matched),relations=t.get('relations'),negated=t.get('negated'),aspect=t.get('aspect'),evidence_class=t.get('evidence_class'),GO_evidence=t.get('GO_evidence'),ECO_evidence=t.get('ECO_evidence'),extensions=t.get('extension_or_of_and'),with_from=t.get('with_from_or_of_and'),gene_product_form=t.get('gene_product_form'),
                    raw_target_index=raw_indices[annotation_id],typed_target_index=typed_indices[annotation_id]))
        result['GOA_atomic_source_annotations']=len(raw)
        result['protected_GOA_atomic_annotations']=annotations
        if target or condition:
            proposal=GOA_projection_proposal(packet,protected)
            remaining={m['GO_ID'] for m in GO_mentions(proposal['projected_target'])}&protected
            result['GOA_selective_proposal']={k:v for k,v in proposal.items() if k!='projected_target'}
            result['GOA_selective_proposal']['unresolved_protected_IDs_elsewhere_in_projected_targets']=sorted(remaining)
            result['GOA_selective_proposal']['retained_nonempty_view_candidate_only']=not remaining and not condition and proposal['retained_annotations']>0
        else:
            result['GOA_selective_proposal']=dict(target_annotations_projected_out=0,retained_annotations=len(raw),proposal_only=True,retained_nonempty_view_candidate_only=bool(raw))
    return result
