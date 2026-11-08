"""Lossless UniProtKB annotation projection with task-local evidence/roles.

No annotation is a default negative;reviewed/score4-5 is not experimental proof.
Raw complete entry JSON is retained independently of all typed/loss projections.
"""
import hashlib,json,re,math
try:
    from scripts.data_intake_corpus import CorpusTask,canonical,annotation_score
except ModuleNotFoundError:
    from data_intake_corpus import CorpusTask,canonical,annotation_score
EXPERIMENTAL_GO={'EXP','IDA','IPI','IMP','IGI','IEP','HTP','HDA','HMP','HGI','HEP'}
IDENTITY_KEYS={'primaryAccession','secondaryAccessions','uniProtkbId','entryType','entryAudit','annotationScore','sequence'}

def validate_entry(e,expected=None):
    if e.get('entryType')!='UniProtKB reviewed (Swiss-Prot)':raise ValueError('exact reviewed entry type required')
    score=annotation_score(e.get('annotationScore'));seq=e['sequence']['value'];acc=e['primaryAccession']
    if not isinstance(seq,str) or not seq or e['sequence']['length']!=len(seq):raise ValueError('canonical source sequence length mismatch')
    md5=e['sequence'].get('md5')
    if md5 and md5.upper()!=hashlib.md5(seq.encode()).hexdigest().upper():raise ValueError('canonical sequence MD5 mismatch')
    if expected is not None and (acc!=expected['Entry'] or seq!=expected['Sequence'] or score!=annotation_score(expected['Annotation'])):raise ValueError('expected accession/sequence/score mismatch')
    return dict(primary_accession=acc,annotation_score=score,sequence_sha256=hashlib.sha256(seq.encode()).hexdigest(),sequence_length=len(seq),entry_version=e.get('entryAudit',{}).get('entryVersion'),sequence_version=e.get('entryAudit',{}).get('sequenceVersion'))

def evidence_bindings(obj,pointer=''):
    rows=[]
    if isinstance(obj,dict):
        if 'evidences' in obj:rows.append(dict(source_pointer=pointer,evidences=obj['evidences']))
        for k,v in obj.items():
            if k!='evidences':rows+=evidence_bindings(v,pointer+'/'+str(k).replace('~','~0').replace('/','~1'))
    elif isinstance(obj,list):
        for i,v in enumerate(obj):rows+=evidence_bindings(v,pointer+'/'+str(i))
    return rows

def direct_experimental(obj):return any(e.get('evidenceCode')=='ECO:0000269' for e in obj.get('evidences',[]) if isinstance(e,dict))

def normalize_ec(value):
    text=str(value);parts=text.split('.');valid=len(parts)==4 and all(p=='-' or p.isdigit() for p in parts);prefixes=[]
    if valid:
        for i,p in enumerate(parts):
            if p=='-':break
            prefixes.append('.'.join(parts[:i+1]))
    return dict(identifier=text,format_status='parsed_EC_identifier' if valid else 'unparsed_source_identifier',known_prefixes=prefixes,complete=bool(valid and all(p.isdigit() for p in parts)),unknown_components=[i for i,p in enumerate(parts) if p=='-'] if valid else None,absence_is_negative=False)

def ec_assertions(obj,pointer=''):
    result=[]
    if isinstance(obj,dict):
        for key,value in obj.items():
            path=pointer+'/'+key
            if key=='ecNumbers' and isinstance(value,list):
                for i,item in enumerate(value):
                    if isinstance(item,dict) and 'value' in item:result.append(dict(source_pointer=path+'/'+str(i),normalized=normalize_ec(item['value']),source_value=item,source_original_value=item))
            elif key=='ecNumber':
                raw=value if isinstance(value,dict) else dict(value=value);v=raw.get('value')
                if v is not None:result.append(dict(source_pointer=path,normalized=normalize_ec(v),source_value=raw,source_original_value=value))
            else:result+=ec_assertions(value,path)
    elif isinstance(obj,list):
        for i,value in enumerate(obj):result+=ec_assertions(value,pointer+'/'+str(i))
    return result

def normalize_go(xref):
    props={p['key']:p.get('value') for p in xref.get('properties',[]) if isinstance(p,dict) and 'key' in p};term=props.get('GoTerm');aspect=None;name=term
    if isinstance(term,str) and len(term)>1 and term[1]==':':aspect={'F':'molecular_function','P':'biological_process','C':'cellular_component'}.get(term[0]);name=term[2:]
    ev=props.get('GoEvidenceType');code=ev.split(':',1)[0] if isinstance(ev,str) else None;qual=props.get('Qualifier',props.get('GoAnnotationQualifier'));neg=None;relation=props.get('Relation')
    if isinstance(qual,str):
        tokens=qual.split('|');neg='NOT' in tokens;relations=[v for v in tokens if v!='NOT'];relation=relation or (relations[0] if len(relations)==1 else None)
    return dict(go_id=xref.get('id'),term_name=name,aspect=aspect,source_evidence_code=code,source_evidence_detail=ev,qualifier_source=qual,negated=neg,relation=relation,closed_world_labels=False,complete_GOA_annotation=False,companion_qualifiers_forms_extensions_required=True,source_value_json=canonical(xref).decode())

def normalize_feature(feature,sequence,accession):
    loc=feature.get('location',{});s=loc.get('start',{});t=loc.get('end',{});a=s.get('value');b=t.get('value');ref=loc.get('sequence');scope='canonical_source_sequence' if ref in (None,accession) else 'unverified_isoform_reference';exact=type(a)is int and type(b)is int and s.get('modifier')=='EXACT' and t.get('modifier')=='EXACT';status='exact_source_bounds' if exact else 'fuzzy_or_unknown_source_bounds';eligible=exact and scope=='canonical_source_sequence' and 1<=a<=b<=len(sequence)
    if exact and not 1<=a<=b<=len(sequence):status='source_bounds_outside_canonical_sequence'
    if scope!='canonical_source_sequence':status='isoform_coordinate_mapping_unverified'
    original=feature.get('alternativeSequence',{}).get('originalSequence')
    if eligible and original is not None and sequence[a-1:b]!=original:eligible=False;status='reference_sequence_mismatch'
    endpoints=feature.get('type') in ('Disulfide bond','Cross-link');interval=[a-1,b] if eligible else None
    return dict(feature_type=feature.get('type'),scope=scope,source_location=loc,coordinate_status=status,coordinate_supervision=bool(eligible),coordinate_semantics='endpoint_pair' if endpoints else 'source_interval_or_residue',interval0=interval,endpoint_indices0=[a-1,b-1] if eligible and endpoints else None,range_supervision=bool(eligible and not endpoints),original_sequence_checked=original is not None and eligible,ligand=feature.get('ligand'),ligand_part=feature.get('ligandPart'),source_feature_json=canonical(feature).decode())

def physical_facts(e):
    facts=[];seq=e['sequence']
    for key,kind,unit in [('length','sequence_length','residue'),('molWeight','sequence_mass','Da')]:
        if key in seq:facts.append(dict(parameter_type=kind,value=seq[key],unit=unit,evidence_role='provider_computed_from_canonical_sequence',source_pointer='/sequence/'+key,source_value_json=canonical(seq[key]).decode(),measurement_context=None))
    for i,c in enumerate(e.get('comments',[])):
        pointer='/comments/'+str(i);typ=c.get('commentType')
        if typ=='BIOPHYSICOCHEMICAL PROPERTIES':
            kinetics=c.get('kineticParameters',{})
            for key,kind,valuekey in [('michaelisConstants','MichaelisConstant','constant'),('maximumVelocities','maximumVelocity','velocity')]:
                for j,p in enumerate(kinetics.get(key,[])):
                    facts.append(dict(parameter_type=kind,value=p.get(valuekey),unit=p.get('unit'),evidence_role='source_reported_kinetic_annotation',source_pointer=pointer+'/kineticParameters/'+key+'/'+str(j),source_value_json=canonical(p).decode(),substrate=p.get('substrate'),direct_ECO0000269=direct_experimental(p),measurement_context='not_bound_to_generic_entry_pH_or_temperature',not_converted_to_kcat=True))
            for key in ['absorption','phDependence','pHDependence','temperatureDependence','redoxPotential']:
                if key in c:facts.append(dict(parameter_type=key,value=None,unit=None,evidence_role='source_biophysical_annotation_unparsed_text_retained',source_pointer=pointer+'/'+key,source_value_json=canonical(c[key]).decode(),measurement_context='source_comment_context_only'))
        elif typ=='MASS SPECTROMETRY':
            facts.append(dict(parameter_type='mass_spectrometry_source_mass',value=c.get('mass'),unit=c.get('unit'),evidence_role='source_reported_mass_spectrometry_annotation',source_pointer=pointer,source_value_json=canonical(c).decode(),method=c.get('massSpectrometryMethod'),mass_error=c.get('massError'),measurement_context='source_method_preserved;unit_specification_review_if_not_explicit'))
    return facts

def full_source_table(entries):
    import pyarrow as pa
    return pa.Table.from_pylist([dict(primary_accession=e['primaryAccession'],full_source_json=canonical(e).decode(),source_structured_sha256=hashlib.sha256(canonical(e)).hexdigest()) for e in entries])

def annotation_packet(e):
    return {k:v for k,v in e.items() if k not in IDENTITY_KEYS}

def annotation_prediction_task(e,*,source_ref=None):
    identity=validate_entry(e);packet=annotation_packet(e)
    return CorpusTask('source_full_UniProt_annotation_prediction',dict(canonical_sequence=e['sequence']['value']),dict(annotation_packet=packet,provider_computed_sequence_properties=[f for f in physical_facts(e) if f['evidence_role']=='provider_computed_from_canonical_sequence']),dict(identity=identity,source_ref=source_ref,annotation_field_masks={k:bool(v) for k,v in packet.items()},label_policy='source_assertions_only;missing_not_negative;GOrelations/NOT/forms may require separateGOA',all_source_fields_ref='lossless_full_source_JSON;identity/version/sequencekeptoutsideannotationtargets',model_adapter_supported=False),['closed_world_annotation_negatives','complete_GOA_relations_qualifiers_extensions','validated_homology_heldout','native_model_annotation_decoder'])

def safe_generation_conditions(e):
    seq=e['sequence']['value'];functions=[];go=[];explicitnot=[];ec=[];kinetic=[]
    for c in e.get('comments',[]):
        if c.get('commentType')=='FUNCTION' and not c.get('molecule'):
            for text in c.get('texts',[]):
                value=text.get('value')
                if direct_experimental(text) and isinstance(value,str) and value and not re.search(r'\b[ACDEFGHIKLMNPQRSTVWYOU]{8,}\b',value):functions.append(value)
    for x in e.get('uniProtKBCrossReferences',[]):
        if x.get('database')!='GO' or x.get('isoformId'):continue
        normalized=normalize_go(x)
        if normalized['source_evidence_code'] in EXPERIMENTAL_GO:
            (explicitnot if normalized['negated'] is True else go).append(normalized['go_id'])
    for item in ec_assertions(e):
        if direct_experimental(item['source_value']):ec.append(item['normalized']['identifier'])
    for fact in physical_facts(e):
        if fact.get('direct_ECO0000269') and isinstance(fact.get('unit'),str) and fact['unit'] and type(fact.get('value')) in (int,float) and math.isfinite(fact['value']):
            kinetic.append(dict(parameter_type=fact['parameter_type'],source_value=fact['value'],source_unit=fact['unit'],substrate_label=fact.get('substrate'),assay_pH=None,assay_temperature=None))
    if not(functions or go or explicitnot or ec or kinetic):return None
    return dict(source_experimentally_supported_function_texts=sorted(set(functions)),source_experimental_associated_GO_ids=sorted(set(go)),source_explicit_experimental_NOT_GO_ids=sorted(set(explicitnot)),source_experimentally_supported_EC_ids=sorted(set(ec)),source_kinetic_conditions=kinetic,organism_taxon_id=e.get('organism',{}).get('taxonId'))

def annotation_conditioned_sequence_task(e,*,source_ref=None):
    identity=validate_entry(e);conditions=safe_generation_conditions(e)
    if conditions is None:return None
    return CorpusTask('source_UniProt_annotation_conditioned_canonical_sequence',conditions,dict(canonical_sequence=e['sequence']['value']),dict(identity=identity,source_ref=source_ref,condition_role='source_observed_annotations;notmodelgeneratedvalidation',excluded_condition_fields=['sequence','sequence_length_mass_checksum_pI','protein_gene_names','accessions_and_crossref_IDs_otherthanGO_EC','citations_evidence_referenceIDs','residue_features_variants_isoformsequences','inferred_only_conditions'],evidence_policy='directECO0000269forfunction_EC_kinetics;experimentalGOcodes;localbindingnotparentbroadcast',model_adapter_supported=False),['per_kinetic_assay_pH_temperature_protocol','validated_homology_heldout','all_unverified_isoform_label_transfer','native_model_annotation_conditioned_decoder'])

CATEGORY_KEYS={'proteinDescription':'names_and_EC','genes':'genes','organism':'taxonomy','keywords':'keywords','geneLocations':'gene_locations','proteinExistence':'protein_existence','comments':'comment_annotations','features':'residue_domain_features','uniProtKBCrossReferences':'linked_resources_and_GO','references':'citations','extraAttributes':'additional_source_attributes'}
def typed_assertions(e):
    seq=e['sequence']['value'];acc=e['primaryAccession'];result=[]
    for key,value in annotation_packet(e).items():
        items=value if isinstance(value,list) else [value]
        for i,item in enumerate(items):
            pointer='/'+key+('/'+str(i) if isinstance(value,list) else '');category=CATEGORY_KEYS.get(key,'unclassified_source_field');normalized=None
            if key=='features':normalized=normalize_feature(item,seq,acc)
            elif key=='uniProtKBCrossReferences' and item.get('database')=='GO':normalized=normalize_go(item);category='GO_KB_filtered_associations'
            elif key=='comments':category='comment:'+str(item.get('commentType','UNSPECIFIED'))
            elif key=='uniProtKBCrossReferences':category='crossref:'+str(item.get('database','UNSPECIFIED'))
            result.append(dict(primary_accession=acc,category=category,source_pointer=pointer,source_value_json=canonical(item).decode(),normalized_json=canonical(normalized).decode() if normalized is not None else None,evidence_bindings_json=canonical(evidence_bindings(item,pointer)).decode(),source_assertion_sha256=hashlib.sha256(canonical(item)).hexdigest()))
    for item in ec_assertions(e):result.append(dict(primary_accession=acc,category='EC_identifier',source_pointer=item['source_pointer'],source_value_json=canonical(item.get('source_original_value',item['source_value'])).decode(),normalized_json=canonical(item['normalized']).decode(),evidence_bindings_json=canonical(evidence_bindings(item['source_value'],item['source_pointer'])).decode(),source_assertion_sha256=hashlib.sha256(canonical(item.get('source_original_value',item['source_value']))).hexdigest()))
    for fact in physical_facts(e):result.append(dict(primary_accession=acc,category='physical:'+fact['parameter_type'],source_pointer=fact['source_pointer'],source_value_json=fact['source_value_json'],normalized_json=canonical(fact).decode(),evidence_bindings_json=canonical(evidence_bindings(json.loads(fact['source_value_json']),fact['source_pointer'])).decode(),source_assertion_sha256=hashlib.sha256(fact['source_value_json'].encode()).hexdigest()))
    return result
