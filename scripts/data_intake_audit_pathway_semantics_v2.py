import pathlib,json,time,hashlib,tarfile,collections,xml.etree.ElementTree as ET,math,sys
import pyarrow as pa
import pyarrow.parquet as pq
R=pathlib.Path('/home/iska/Documents/amelie/bio/Mora-3');sys.path.insert(0,str(R/'scripts'));from data_intake_sbml import biological_qualifiers,participant
B=R/'outputs/data-intake/20261031-pathway-semantics-v2';B.mkdir(exist_ok=False);(B/'audits').mkdir();(B/'logs').mkdir();start=time.perf_counter();old=R/'outputs/data-intake/20261031-subagent/curated/reactome98-human-sbml-v2';manifest=json.loads((old/'manifest.json').read_text());source=pathlib.Path(manifest['source']['path']);assert hashlib.sha256(source.read_bytes()).hexdigest()==manifest['source']['sha256'];RDF='{http://www.w3.org/1999/02/22-rdf-syntax-ns#}';BQ='http://biomodels.net/biology-qualifiers/';counts=collections.Counter();qualcounts=collections.Counter();reaction_variants=collections.defaultdict(set);source_contexts=collections.defaultdict(list);anomalies=[];sizes=[];source_members=[]
def local(tag):return tag.rsplit('}',1)[-1]
def quals(el):return biological_qualifiers(el)
def notes(el):
 return [ET.tostring(x,encoding='unicode') for x in el if local(x.tag)=='notes']
def human_identity(q):return sorted({v.split('reactome:')[1] for v in q.get('is',[]) if 'reactome:R-HSA-' in v})
with (B/'source-native-pathway-models.jsonl').open('x') as output:
 with tarfile.open(source,'r|gz') as tar:
  for member in tar:
   if not member.isfile() or not member.name.endswith('.sbml'):continue
   assert member.size<=128*2**20;raw=tar.extractfile(member).read();root=ET.fromstring(raw);model=next(x for x in root if local(x.tag)=='model');ns=root.tag.split('}')[0]+'}';species={};compartments={};reactions=[];kinetic=0;params=[]
   for element in model:
    kind=local(element.tag)
    if kind=='listOfCompartments':
     for e in element:compartments[e.attrib['id']]=dict(attributes=dict(e.attrib),qualifiers=quals(e),notes=notes(e))
    elif kind=='listOfSpecies':
     for e in element:
      sid=e.attrib['id'];assert sid not in species;q=quals(e);qualcounts.update({k:len(v) for k,v in q.items()});species[sid]=dict(attributes=dict(e.attrib),qualifiers=q,notes=notes(e),source_human_reactome_identity=human_identity(q))
    elif kind in ('listOfParameters','listOfUnitDefinitions'):
     params.append(dict(kind=kind,source_xml=ET.tostring(element,encoding='unicode')))
    elif kind=='listOfReactions':
     for e in element:
      refs=[];rolecounts=collections.Counter()
      for child in e:
       k=local(child.tag)
       if k=='kineticLaw':kinetic+=1
       role={'listOfReactants':'reactant','listOfProducts':'product','listOfModifiers':'modifier'}.get(k)
       if not role:continue
       for ref in child:
        attrs=dict(ref.attrib);sid=attrs['species'];missing=sid not in species;stoich=attrs.get('stoichiometry');numeric=None
        if stoich is not None:
         try:numeric=float(stoich);assert math.isfinite(numeric) and numeric>0
         except (ValueError,AssertionError):anomalies.append(dict(member=member.name,reaction=e.attrib['id'],cause='invalid_stoichiometry',value=stoich))
        if missing:anomalies.append(dict(member=member.name,reaction=e.attrib['id'],cause='participant_entity_missing',value=sid))
        validated_participant=participant(role,attrs);assert validated_participant['stoichiometry_numeric']==numeric
        refs.append(dict(role=role,species=sid,source_attributes=attrs,source_qualifiers=quals(ref),stoichiometry_source=stoich,stoichiometry_numeric=numeric,source_symbolic_stoichiometry_xml=[ET.tostring(x,encoding='unicode') for x in ref]));rolecounts[role]+=1
      q=quals(e);stable=human_identity(q);key='|'.join(stable) if stable else 'local:'+e.attrib['id'];signature=dict(reversible=e.attrib.get('reversible'),compartment=e.attrib.get('compartment'),participants=[dict(role=x['role'],species=x['species'],stoichiometry_source=x['stoichiometry_source'],sbo=x['source_attributes'].get('sboTerm')) for x in refs]);sig=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest();reaction_variants[key].add(sig);source_contexts[key].append(member.name);reactions.append(dict(attributes=dict(e.attrib),qualifiers=q,notes=notes(e),participants=refs,source_human_reaction_identity=stable,source_semantic_signature=sig));counts.update(rolecounts);counts['reactions']+=1
   counts['models']+=1;counts['species']+=len(species);counts['compartments']+=len(compartments);counts['kinetic_laws']+=kinetic;sizes.append(dict(member=member.name,species=len(species),reactions=len(reactions),participant_edges=sum(len(r['participants']) for r in reactions)));source_members.append(dict(member=member.name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()));record=dict(schema='mora3.source_native_sbml_graph.v2',source_release='Reactome98',source_archive_sha256=manifest['source']['sha256'],source_member=member.name,source_member_sha256=hashlib.sha256(raw).hexdigest(),SBML_root_attributes=dict(root.attrib),model_attributes=dict(model.attrib),model_qualifiers=quals(model),model_notes=notes(model),compartments=compartments,species=species,reactions=reactions,parameters_units=params,task_admitted=False,evidence='source_curated_static_pathway_annotation;not measured response,flux or causaltrajectory',scope='source_is/hasPart/hasVersion/otherRDFqualifiersretained;not flattened identities');output.write(json.dumps(record,sort_keys=True,allow_nan=False)+'\n')
   if counts['models']%500==0:print(json.dumps(dict(stage='native_SBML_semantics',models=counts['models'],reactions=counts['reactions'],species=counts['species'],seconds=time.perf_counter()-start)),flush=True)
assert counts['models']==2889 and counts['species']==160266 and counts['reactions']==75233 and counts['kinetic_laws']==0
variants=[dict(reaction_identity=k,semantic_variants=len(v),semantic_signatures=sorted(v),occurrences=len(source_contexts[k]),source_members=source_contexts[k]) for k,v in reaction_variants.items()];pq.write_table(pa.Table.from_pylist(variants),B/'audits/reaction-identity-variants.parquet',compression='zstd');pq.write_table(pa.Table.from_pylist(sizes),B/'audits/pathway-context-sizes.parquet',compression='zstd')
if anomalies:pq.write_table(pa.Table.from_pylist(anomalies),B/'audits/source-semantic-anomalies.parquet',compression='zstd')
# Complete serialized native graph JSON readback;still source candidates,not trainingadmission.
readcount=readspecies=readreactions=0
with (B/'source-native-pathway-models.jsonl').open() as f:
 for line in f:
  d=json.loads(line);readcount+=1;readspecies+=len(d['species']);readreactions+=len(d['reactions'])
assert (readcount,readspecies,readreactions)==(2889,160266,75233)
p=B/'source-native-pathway-models.jsonl';audit=dict(schema='mora3.pathway_source_semantic_audit.v2',source=manifest['source'],counts=dict(counts),RDF_qualifier_resource_counts=dict(qualcounts),unique_source_reaction_identities=len(reaction_variants),reaction_identities_with_semantic_variants=sum(len(v)>1 for v in reaction_variants.values()),source_anomalies=len(anomalies),max_species_per_pathway=max(x['species'] for x in sizes),max_reactions_per_pathway=max(x['reactions'] for x in sizes),max_participant_edges_per_pathway=max(x['participant_edges'] for x in sizes),native_graphs_path=str(p.relative_to(R)),native_graphs_bytes=p.stat().st_size,native_graphs_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),all_native_graphs_full_JSON_readback=True,task_admitted=0,license='CC0-1.0',remaining_gates=['source_biological_qualifiers_must_not_flatten_hasPart_or_ortholog_versions_as_exactidentity','reaction participant semantics/masks/targetexclusion','nestedpathway overlap and semanticvariant treatment','staticcuratedprioronly;no kinetics,physicalconcentrations,dynamics orcausalresponse','modelreader/contextbudget pending'],no_raw_source_extraction_or_mutation=True,optimizer_steps=0,seconds=time.perf_counter()-start)
with (B/'semantic-audit.json').open('x') as f:json.dump(audit,f,indent=2)
print('SEMANTIC_AUDIT_COMPLETE',json.dumps(audit),flush=True)
