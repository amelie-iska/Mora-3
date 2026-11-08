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


def training_view_policy(exposure,source_split):
    """Keep ALL native split exposure; propose mixture actions ONLY for train."""
    from copy import deepcopy
    if source_split not in ['train','validation','test']:
        raise ValueError('unknown master split; refuse any mixture proposal')
    result=deepcopy(exposure);is_train=source_split=='train'
    any_hit=result.pop('proposed_coarse_entire_view_exclusion')
    audit=dict(protected_target_views=int(result['protected_target_exposure']),protected_condition_views=int(result['protected_condition_exposure']),protected_any_native_role_views=int(any_hit))
    train=dict(coarse_entire_view_exclusions=int(is_train and any_hit))
    policy=dict(scope='training_mixture_only;official_validation_and_test_audit_only',source_split=source_split,applies_to_this_split=is_train,coarse_entire_view_exclusion_proposed=bool(is_train and any_hit),selective_GOA_projection_proposed=False,validation_or_test_view_unchanged=not is_train,policy_activated=False,task_admission=False,model_use=False)
    if 'GOA_selective_proposal' in result:
        hypothetical=result.pop('GOA_selective_proposal')
        eligible=hypothetical.pop('retained_nonempty_view_candidate_only')
        for key in ['coarse_packet_exclusion','conditioned_view_requires_exclusion_or_separate_new_admission']:
            hypothetical.pop(key,None)
        audit.update(GOA_source_annotations_nonadditive_raw_typed=result['GOA_atomic_source_annotations'],GOA_atomic_annotations_with_protected_IDs=len(result['protected_GOA_atomic_annotations']),GOA_hypothetical_remaining_annotations=hypothetical['retained_annotations'])
        candidate=is_train and eligible
        project=candidate and hypothetical['target_annotations_projected_out']>0
        policy.update(selective_GOA_projection_proposed=bool(project),selective_GOA_nonempty_candidate=bool(candidate),alternative_requires_coarse_exclusion=bool(is_train and any_hit and not candidate),atomic_source_annotation_IDs_to_project_out=hypothetical.get('removed_source_annotation_IDs',[]) if project else [])
        train.update(GOA_nonempty_candidate_views_after_selective_projection=int(candidate),GOA_nonempty_affected_views_with_projection_proposed=int(project),GOA_atomic_annotations_proposed_for_selective_projection=hypothetical['target_annotations_projected_out'] if project else 0,GOA_atomic_annotations_lost_under_coarse_exclusion=result['GOA_atomic_source_annotations'] if is_train and any_hit else 0,GOA_alternative_requires_coarse_excluded_views=int(is_train and any_hit and not candidate),GOA_source_annotations_retained_after_selective_alternative=hypothetical['retained_annotations'] if candidate else 0)
        result['GOA_atomic_projection_hypothetical_audit']=hypothetical
    result.update(audit_role_counts=audit,training_proposal_counts=train,training_mixture_policy_proposal=policy,audit_any_protected_native_role_exposure=bool(any_hit))
    return result
