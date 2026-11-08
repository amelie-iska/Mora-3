"""Lossless, form-specific GOA annotation prediction projection; absent != negative."""
from pathlib import Path
import json,hashlib,sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from data_intake_corpus import CorpusTask,canonical
EXPERIMENTAL={'EXP','IDA','IPI','IMP','IGI','IEP','HTP','HDA','HMP','HGI','HEP'}

def annotation_projection(a):
    qualifier=a.get('qualifier');parts=qualifier.split('|') if isinstance(qualifier,str) and qualifier else []
    def expression(groups,relations=False):
        return [[dict(relation=x.get('relation'),identifier=x['db']+':'+x['id']) if relations else x['db']+':'+x['id'] for x in group['connectedXrefs']] for group in (groups or [])]
    return dict(source_annotation_id=a['id'],subject=a['geneProductId'],go_id=a['goId'],term_name=a.get('goName'),aspect=a.get('goAspect'),source_qualifier=qualifier,negated=('NOT' in parts) if parts else None,relations=[p for p in parts if p!='NOT'],GO_evidence=a.get('goEvidence'),ECO_evidence=a.get('evidenceCode'),evidence_class='source_experimental_evidence_code' if a.get('goEvidence') in EXPERIMENTAL else 'no_biological_data_placeholder' if a.get('goEvidence')=='ND' else 'source_inferred_or_other_evidence',reference=a.get('reference'),assigned_by=a.get('assignedBy'),date=a.get('date'),with_from_or_of_and=expression(a.get('withFrom')),extension_or_of_and=expression(a.get('extensions'),True),gene_product_form=a['geneProductId'],missing_is_negative=False,source_positive_GO_head_mask=bool(parts) and 'NOT' not in parts and a.get('goEvidence')!='ND',source_negative_GO_head_mask='NOT' in parts and a.get('goEvidence')!='ND',full_source_json=canonical(a).decode(),source_assertion_sha256=hashlib.sha256(canonical(a)).hexdigest())

def bind_subject(subject,baseline,forms,isoforms):
    if not subject.startswith('UniProtKB:'):raise ValueError('non-UniProtKB subject')
    sid=subject[len('UniProtKB:'):];parent=sid.split('-',1)[0]
    if parent not in baseline:raise ValueError('source subject outside qualified exact-entry baseline')
    seq=None;binding='unresolved_form_sequence';kind='unverified_form'
    if sid==parent:seq=baseline[parent];binding='exact_current_canonical_source';kind='canonical'
    elif '-PRO_' in sid:
        form=forms.get((parent,sid.split('-',1)[1]));seq=form.get('verified_sequence') if form else None;binding='exact_canonical_interval_primary_sequence' if seq is not None else 'unresolved_PRO_or_fuzzy_scope';kind='processed_product'
    elif sid in isoforms:seq=isoforms[sid];binding='exact_official_same_release_isoform_FASTA';kind='isoform'
    else:
        form=forms.get((parent,sid));seq=form.get('verified_sequence') if form else None
        if seq is not None:binding='sourceDisplayed_alias_exact_canonical_sequence';kind='displayed_isoform'
    return dict(parent_accession=parent,subject=subject,sequence=seq,sequence_binding=binding,subject_kind=kind,canonical_labels_inherited=False,GOA_historical_sequence_version='notexplicit;semanticannotationprioronly;notresiduecoordinatetruth')

def prediction_task(binding,annotations,*,source_snapshot,source_refs=None):
    if binding['sequence'] is None:return None
    projected=[annotation_projection(a) for a in annotations]
    if not any(p['source_positive_GO_head_mask'] or p['source_negative_GO_head_mask'] for p in projected):return None
    return CorpusTask('source_GOA_assertion_packet_prediction',dict(protein_sequence=binding['sequence']),dict(source_GO_annotations=annotations,typed_GO_annotations=projected),dict(source_snapshot=source_snapshot,source_refs=source_refs,binding={k:v for k,v in binding.items() if k!='sequence'},label_policy='observedsourceassertionsonly;NOTexplicit;missingunknown;NDnotbiologicalpositive;ORofANDpreserved',source_KB_sequence_release='2026_03;separatefromGOAJuly28snapshot',model_adapter_supported=False),['all107778GOAcompanioncoverage','pinned_GOterm_name_and_ontology_companion_when_source_null','validated_protein_homology_heldout','GOA_historical_subject_sequence_version','native_model_GOAdecoder'])
