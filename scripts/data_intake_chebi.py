"""ChEBI reference classification for identity audits;not experimental labels."""


def formula_inventory(formula):
    """Element counts for formula QC;charge is compared separately.

    Source dot-adduct notation is expanded. Isotope labels are compared as their
    elements only in this formula check;isotopes remain in source structures and
    the separate full InChIKey consistency check,never erased from identities.
    Unsupported symbolic/polymeric formulae raise rather than guessing counts.
    """
    import re,collections
    value=re.sub(r'[+-][0-9]*$','',formula.replace(' ',''));result=collections.Counter()
    for component in re.split(r'[.·]',value):
        if not component:raise ValueError('empty formula component')
        coefficient=re.match(r'[0-9]+',component);factor=int(coefficient.group()) if coefficient else 1
        if coefficient:component=component[coefficient.end():]
        position=0
        while position<len(component):
            token=re.match(r'(?:\[([0-9]+)([A-Z][a-z]?)\]|([A-Z][a-z]?))([0-9]*)',component[position:])
            if not token:raise ValueError('unsupported source formula notation')
            element=token.group(2) or token.group(3)
            if element in ('D','T'):element='H'
            result[element]+=factor*int(token.group(4) or 1);position+=token.end()
    return dict(result)

def chebi_reference(record,requested):
    from rdkit import Chem
    import re
    from rdkit.Chem import rdMolDescriptors
    try:
        from data_intake_corpus import chemical_identity
    except ModuleNotFoundError as exc:
        if exc.name!='data_intake_corpus':raise
        from scripts.data_intake_corpus import chemical_identity
    primary=record.get('chebi_accession');structure=record.get('default_structure')
    result=dict(requested_id=requested,primary_id=primary,source_modified_on=record.get('modified_on'),source_stars=record.get('stars'),source_is_released=record.get('is_released'),source_smiles=None if not structure else structure.get('smiles'),canonical_isomeric_smiles=None,molecule_group=None,scaffold_group=None,fragment_groups=[],source_standard_inchi_key=None if not structure else structure.get('standard_inchi_key'),source_standard_inchi=None if not structure else structure.get('standard_inchi'),identity_representation_used=None,formula_charge_key_consistency=None)
    if requested!=primary and requested not in record.get('secondary_ids',[]):result['status']='requested_id_not_primary_or_explicit_secondary';return result
    if record.get('is_released') is not True:result['status']='source_not_released';return result
    if record.get('stars')!=3:result['status']='source_curation_less_than_three_stars';return result
    if not structure:result['status']='no_default_structure_specificity_unresolved';return result
    if structure.get('is_r_group') is True:result['status']='generic_R_group_not_single_compound';return result
    source_formula=(record.get('chemical_data') or {}).get('formula')
    if source_formula and re.search(r'\)[nmx]',source_formula):result['status']='generic_polymeric_formula_not_single_compound';return result
    smiles=structure.get('smiles')
    if not smiles:result['status']='no_supported_structure_identifier';return result
    mol=Chem.MolFromSmiles(smiles)
    fallback=False
    if mol is None:
        inchi=structure.get('standard_inchi');key=structure.get('standard_inchi_key');chemical=record.get('chemical_data') or {}
        mol=Chem.MolFromInchi(inchi) if inchi else None
        if mol is None or not key:
            result['status']='source_SMILES_and_supported_InChI_unavailable';return result
        formula=chemical.get('formula');charge=chemical.get('charge')
        try:
            consistent=(Chem.MolToInchiKey(mol)==key and formula is not None and charge is not None and formula_inventory(rdMolDescriptors.CalcMolFormula(mol))==formula_inventory(formula) and Chem.GetFormalCharge(mol)==charge)
        except ValueError as exc:
            result['status']='source_formula_notation_unresolved';result['error']=str(exc);return result
        result['formula_charge_key_consistency']=consistent
        if not consistent:result['status']='source_InChI_formula_charge_or_key_inconsistent';return result
        fallback=True
    if any(a.GetAtomicNum()==0 for a in mol.GetAtoms()):result['status']='wildcard_structure_not_single_compound';return result
    if not fallback:
        chemical=record.get('chemical_data') or {};checks={}
        if chemical.get('formula') is not None:
            try:checks['formula']=formula_inventory(rdMolDescriptors.CalcMolFormula(mol))==formula_inventory(chemical['formula'])
            except ValueError as exc:
                result['status']='source_formula_notation_unresolved';result['error']=str(exc);return result
        if chemical.get('charge') is not None:checks['charge']=Chem.GetFormalCharge(mol)==chemical['charge']
        if structure.get('standard_inchi_key'):checks['key']=Chem.MolToInchiKey(mol)==structure['standard_inchi_key']
        result['source_representation_consistency_checks']=checks
        if not all(checks.values()):result['status']='source_SMILES_formula_charge_or_key_inconsistent';return result
    result['identity_representation_used']='source_standard_inchi' if fallback else 'source_SMILES' 

    if any(a.GetAtomicNum()==0 for a in mol.GetAtoms()):result['status']='wildcard_structure_not_single_compound';return result
    try:
        result.update(chemical_identity(Chem.MolToSmiles(mol,isomericSmiles=True) if fallback else smiles));fragments=Chem.GetMolFrags(mol,asMols=True,sanitizeFrags=True);groups=[]
        for fragment in fragments:
            identity=chemical_identity(Chem.MolToSmiles(fragment,isomericSmiles=True));groups.extend([identity['molecule_group'],identity['scaffold_group']])
        result['fragment_groups']=sorted(set(groups));result['status']='curated_single_compound_InChI_reference_after_SMILES_failure' if fallback else 'curated_single_compound_structure';result['representation_limit']='InChI-derived reference is not a repair or proof of the literal original SMILES tautomer' if fallback else None;return result
    except Exception as exc:
        result['status']='derived_structure_identity_failed';result['error']=type(exc).__name__+': '+str(exc);result['molecule_group']=None;result['scaffold_group']=None;result['fragment_groups']=[];return result
