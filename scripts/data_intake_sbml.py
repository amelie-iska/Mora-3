"""Source-specific static annotation utilities;no kinetic or causal inference.

Reviewed standards: SBML3V1Core4.11.3/4.11.4 and MIRIAM qualifier syntax at
https://sbml.org/documents/elaborations/miriam_annotation_syntax/ . Missing
stoichiometry stays unknown;hasPart/homology do not become exact identities.
"""
import math
RDF='{http://www.w3.org/1999/02/22-rdf-syntax-ns#}'
BIOLOGY='{http://biomodels.net/biology-qualifiers/}'


def biological_qualifiers(element):
    result={}
    for annotation in element:
        if annotation.tag.rsplit('}',1)[-1]!='annotation':continue
        for rdf in annotation:
            if rdf.tag!=RDF+'RDF':continue
            for description in rdf:
                if description.tag!=RDF+'Description':continue
                for qualifier in description:
                    if not qualifier.tag.startswith(BIOLOGY):continue
                    key=qualifier.tag.rsplit('}',1)[-1]
                    for collection in qualifier:
                        if collection.tag not in (RDF+'Bag',RDF+'Seq',RDF+'Alt'):continue
                        for item in collection:
                            resource=item.attrib.get(RDF+'resource')
                            if item.tag==RDF+'li' and resource:result.setdefault(key,set()).add(resource)
    return {key:sorted(values) for key,values in result.items()}


def exact_source_resources(qualifiers):
    return list(qualifiers.get('is',[]))


def participant(role,attributes):
    if role not in ('reactant','product','modifier') or not attributes.get('species'):
        raise ValueError('source role and participant species required')
    value=attributes.get('stoichiometry');numeric=None
    if value is not None:
        if role=='modifier':raise ValueError('modifier has no reactant/product stoichiometry')
        numeric=float(value)
        if not math.isfinite(numeric) or numeric<=0:raise ValueError('positive finite source stoichiometry required')
    return dict(role=role,species=attributes['species'],source_attributes=dict(attributes),stoichiometry_source=value,stoichiometry_numeric=numeric)



def qualified_biological_context(qualifiers):
    from urllib.parse import urlsplit
    allowed={'uniprot','chebi','ensembl','ncbigene','go','ec-code','rnacentral','pubchem.compound'}
    result={}
    for key in ('is','hasPart'):
        if key not in qualifiers:continue
        values=[]
        for url in qualifiers[key]:
            u=urlsplit(url);host=(u.hostname or '').lower();path=u.path.lstrip('/');namespace=None
            if host=='identifiers.org':namespace=path.split('/',1)[0].split(':',1)[0].lower()
            elif host in ('purl.uniprot.org','www.uniprot.org') and path.startswith('uniprot/'):namespace='uniprot'
            elif host in ('www.ebi.ac.uk','ebi.ac.uk') and path.startswith('chebi/'):namespace='chebi'
            elif host.endswith('ncbi.nlm.nih.gov') and path.startswith('gene/'):namespace='ncbigene'
            if namespace in allowed:values.append(url)
        result[key]=values
    return result

def masked_pathway_task(graph,masked_reactions,*,audit=None):
    """Known participant incidence with hidden source role/stoichiometry fields.

    Local source/reaction/reference IDs,role-encoding SBO values and explanatory
    reaction text remain excluded. Biological entity state labels and qualified
    reference parts are known node context;homology never becomes exact identity.
    """
    import hashlib,json
    try:
        from data_intake_corpus import CorpusTask
    except ModuleNotFoundError as exc:
        if exc.name!="data_intake_corpus":raise
        from scripts.data_intake_corpus import CorpusTask
    selected=set(masked_reactions)
    if not selected or any(type(i) is not int or not 0<=i<len(graph['reactions']) for i in selected):
        raise ValueError('nonempty source reaction subset required')
    ids=sorted(graph['species']);index={sid:i for i,sid in enumerate(ids)};compids=sorted(graph['compartments']);compindex={cid:i for i,cid in enumerate(compids)};species=[]
    for sid in ids:
        entity=graph['species'][sid];attrs=entity['attributes'];q=entity['qualifiers'];references=qualified_biological_context(q)
        species.append(dict(name=attrs.get('name'),sbo_term=attrs.get('sboTerm'),compartment_index=compindex.get(attrs.get('compartment')),qualified_reference_context=references))
    compartments=[dict(name=graph['compartments'][cid]['attributes'].get('name'),reference_qualifiers=graph['compartments'][cid]['qualifiers']) for cid in compids];reactions=[];targets=[]
    for i,reaction in enumerate(graph['reactions']):
        refs=reaction['participants'];order=sorted(range(len(refs)),key=lambda j:hashlib.sha256(f'20261031:{refs[j]["species"]}:{j}'.encode()).digest());condition=[];target=[]
        for j in order:
            ref=refs[j]
            if ref['species'] not in index:raise ValueError('missing source participant entity')
            if ref['role'] not in ('reactant','product','modifier'):raise ValueError('unknown source participant role')
            condition.append(dict(species_index=index[ref['species']],role=None if i in selected else ref['role'],stoichiometry_source=None if i in selected else ref['stoichiometry_source']))
            target.append(dict(role=ref['role'],stoichiometry_source=ref['stoichiometry_source']))
        reactions.append(dict(reversible_source=reaction['attributes'].get('reversible'),participants=condition));targets.append(target if i in selected else None)
    conditions=dict(species=species,compartments=compartments,reactions=reactions,known_participant_incidence=True)
    return CorpusTask('source_static_pathway_masked_participant_roles',conditions,dict(masked_reaction_participants=targets),audit or {},['kinetic_laws','rate_constants','physical_initial_concentrations','cell_specific_experimental_response','temporal_reaction_order','samecell_multimodal_pairing'])
