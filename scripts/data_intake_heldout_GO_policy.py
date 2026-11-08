"""Native heldout-ID inventories and NON-APPLYING target projection proposals."""
from copy import deepcopy
import json,re
GO=re.compile(r'(?<![A-Za-z0-9_])GO:(\d{7})(?!\d)')
IRI=re.compile(r'https?://purl\.obolibrary\.org/obo/GO_(\d{7})(?!\d)')
ZERO=frozenset(['CL_val_zero_shot','eval_zero_shot','eval_zero_shot_hard'])


def native_split_GO_sets(rows):
    sets={};counts={}
    for row in rows:
        go=row['text_id'];split=row['split']
        if re.fullmatch(r'GO:\d{7}',go) is None:raise ValueError('native official GO ID required; no guessed namespace/padding')
        sets.setdefault(split,set()).add(go);counts[split]=counts.get(split,0)+1
    return dict(native_role_ID_sets={k:sorted(v) for k,v in sorted(sets.items())},native_role_relation_rows=counts,
        protected_native_zero_shot_roles=sorted(ZERO&sets.keys()),official_roles_unchanged=True,
        protected_union_is_only_a_proposed_cross_protocol_training_reservation=True)


def GO_mentions(value,*,path='',depth=0):
    """Exact lexical/explicit-db namespace exposure, not assertion equivalence."""
    if depth>64:raise ValueError('bounded native target traversal exceeded; no absence claim')
    out=[]
    def add(go,p,kind):out.append(dict(GO_ID=go,pointer=p,evidence_kind=kind))
    if isinstance(value,dict):
        db=next((value[k] for k in ['database','databaseName','db','source_database'] if k in value),None)
        if db in ['GO','Gene Ontology']:
            ident=next((value[k] for k in ['id','identifier','accession'] if k in value),None)
            if isinstance(ident,str) and re.fullmatch(r'\d{7}',ident):add('GO:'+ident,path,'explicit_GO_database_and_7digit_identifier')
        for key,child in value.items():
            pointer=path+'/'+str(key).replace('~','~0').replace('/','~1')
            for match in GO.finditer(str(key)):add('GO:'+match[1],pointer+'/@key','literal_GO_ID_dictionary_key')
            out.extend(GO_mentions(child,path=pointer,depth=depth+1))
    elif isinstance(value,list):
        for i,child in enumerate(value):out.extend(GO_mentions(child,path=path+'/'+str(i),depth=depth+1))
    elif isinstance(value,str):
        for match in GO.finditer(value):add('GO:'+match[1],path,'literal_GO_ID_string')
        for match in IRI.finditer(value):add('GO:'+match[1],path,'standard_GO_OBO_IRI')
        stripped=value.lstrip()
        if stripped.startswith(('{','[','"')):
            try:embedded=json.loads(value)
            except (ValueError,TypeError,RecursionError):embedded=None
            if embedded is not None and embedded!=value:out.extend(GO_mentions(embedded,path=path+'/@embedded_json',depth=depth+1))
    return out


def task_role_surfaces(packet,native_schema):
    """Native roles only. Audit/receipt fields never become modeled targets."""
    if native_schema=='mora3.CorpusTask/v1':
        return [('targets',packet['targets']),('conditions',packet['conditions'])]
    if native_schema=='mora3.ObservationalSciPlexTask/v1':
        return task_role_surfaces(packet['task']['sidecar'],'mora3.ArrayTask/v1')
    if native_schema=='mora3.ArrayTask/v1':
        roles=dict(packet['roles']);out=[]
        for array in packet['arrays']:
            role=roles[array['name']]
            if role not in ['target','predictor','condition','context','audit']:
                raise ValueError('unknown native array role; review before absence claim')
            if role=='audit':continue
            surface={k:v for k,v in array.items() if k!='values'}
            out.append(('target_array_coordinate' if role=='target' else 'condition_array_coordinate',surface))
            # Values are native numeric fields; validate type rather than scan
            # arbitrary numerical encodings as vocabulary text.
            values=array['values']
            if not isinstance(values,list):raise ValueError('unreviewed native array value encoding')
            if any(not isinstance(x,(int,float,type(None))) for x in values):
                out.append(('targets' if role=='target' else 'conditions',values))
        return out
    if native_schema=='mora3.RasterMorphologyTask/v1':
        if packet['direction'] not in ['image_to_mask','mask_to_image']:
            raise ValueError('unknown native raster direction')
        return [('numeric_raster_targets_no_text_vocabulary',{})]
    raise ValueError('unreviewed native schema; do not assume no phenotype target')


def GOA_projection_proposal(packet,protected):
    """What-if only: no rewrite, model use, admission or policy activation."""
    original=deepcopy(packet);raw=packet['targets']['source_GO_annotations'];typed=packet['targets']['typed_GO_annotations'];by_raw={x['id']:x for x in raw};by_typed={x['source_annotation_id']:x for x in typed}
    if len(by_raw)!=len(raw) or len(by_typed)!=len(typed) or set(by_raw)!=set(by_typed):raise ValueError('exact nonduplicated raw/typed source annotation pairing required')
    blocked=[]
    for annotation_id in by_raw:
        ids={x['GO_ID'] for x in GO_mentions(by_raw[annotation_id])+GO_mentions(by_typed[annotation_id])}
        if ids&protected:blocked.append(annotation_id)
    deny=set(blocked);projection=deepcopy(packet);projection['targets']['source_GO_annotations']=[x for x in raw if x['id'] not in deny];projection['targets']['typed_GO_annotations']=[x for x in typed if x['source_annotation_id'] not in deny];condition_hits={x['GO_ID'] for x in GO_mentions(packet['conditions'])}&protected
    assert packet==original
    return dict(proposal_only=True,source_or_old_shard_rewritten=False,task_admission=False,model_use=False,
        original_annotations=len(raw),target_annotations_projected_out=len(deny),retained_annotations=len(raw)-len(deny),
        removed_source_annotation_IDs=sorted(deny),projected_target=projection['targets'],
        coarse_packet_exclusion=bool(deny or condition_hits),
        protected_condition_IDs=sorted(condition_hits),conditioned_view_requires_exclusion_or_separate_new_admission=bool(condition_hits),
        original_full_source_and_audit_preserved=True,teacher_forcing_target_must_use_projected_target_not_just_zero_loss_mask=True,
        native_official_heldouts_unchanged=True,biological_negative_or_fact_equivalence_inferred=False)
