"""Original NCI source schema and explicit legacy CTAB syntax projection."""
import hashlib,json,math,re
from decimal import Decimal
from rdkit import Chem,rdBase
from rdkit.Chem import rdMolDescriptors

def parse_endpoint(row,endpoint):
    if endpoint not in ['EC50','IC50']:raise ValueError('explicit source endpoint required')
    fields={k.strip():v.strip() for k,v in row.items()}
    nsc=fields['NSC'];assert re.fullmatch('[0-9]+',nsc)
    def number(key):
        value=float(fields[key]);assert math.isfinite(value);return value
    high=number('Log10HiConc');value=number('Log10'+endpoint);sd=number('StdDev');num=int(fields['NumExp']);assert sd>=0 and num>0
    unit=fields['ConcUnit'];flag=fields['Flag'];assert flag in ['=','>','<']
    unit_known=unit in ['M','u'];source_gap=endpoint=='IC50' and (unit=='V' or flag=='<')
    context=[nsc,endpoint,unit,format(Decimal(fields['Log10HiConc']).normalize(),'f')]
    return dict(source_NSC=nsc,endpoint=endpoint,source_native_fields=dict(row),source_value=value,source_value_scale='log10_concentration_explicit_native_header',source_log10_highest_concentration=high,source_ConcUnit=unit,physical_unit_mapping_available=unit_known,unit_meaning='molar' if unit=='M' else 'micrograms_per_mL' if unit=='u' else 'unresolved_volumetric_definition',source_flag=flag,censoring_role='all_source_experiments_reached_endpoint' if flag=='=' else 'source_aggregate_flag_at_least_one_experiment;not_bound_of_aggregate_mean',point_target_available=(flag=='=' and unit_known and not source_gap),source_NumExp=num,source_stddev=sd,positive_uncertainty_target_available=(sd>0 and num>1 and flag=='=' and unit_known and not source_gap),zero_SD_is_not_perfect_precision=True,source_documentation_gap=source_gap,context_key=context,compound_highestdose_context_key=[nsc,unit,format(Decimal(fields['Log10HiConc']).normalize(),'f')],evidence_class='source_computed_summary_of_assay_measurements_not_prediction',same_cell_or_replicate_pair_asserted=False)

def parse_structure_record(text):
    lines=text.splitlines();assert len(lines)>=4
    property_start=next((i for i,line in enumerate(lines) if line.startswith('>')),len(lines)-int(lines[-1].strip()=='$$$$'))
    ct=lines[:property_start]
    while ct and not ct[-1].strip():ct.pop()
    assert len(ct)>=4
    n_atoms=int(ct[3][0:3]);n_bonds=int(ct[3][3:6]);assert n_atoms>0 and n_bonds>=0 and len(ct)>=4+n_atoms+n_bonds
    atoms=ct[4:4+n_atoms];bonds=ct[4+n_atoms:4+n_atoms+n_bonds];extra=ct[4+n_atoms+n_bonds:];known=['M  CHG','M  ISO','M  RAD','M  END']
    if any(line.strip() and not any(line.startswith(tag) for tag in known) for line in extra):raise ValueError('unsupported original CTAB extension retained in raw source')
    end_present=any(line.strip()=='M  END' for line in ct);adapted='\n'.join(ct+([] if end_present else ['M  END']))+'\n'
    with rdBase.BlockLogs():mol=Chem.MolFromMolBlock(adapted,sanitize=False,removeHs=False,strictParsing=True)
    if mol is None:raise ValueError('strict derived-syntax RDKit parse failed; source retained')
    assert mol.GetNumAtoms()==n_atoms and mol.GetNumBonds()==n_bonds
    periodic=Chem.GetPeriodicTable()
    for i,line in enumerate(atoms):
        symbol=line[31:34].strip();expected=1 if symbol in ['D','T'] else periodic.GetAtomicNumber(symbol);assert expected>0 and mol.GetAtomWithIdx(i).GetAtomicNum()==expected
    orders={1:1.0,2:2.0,3:3.0,4:1.5}
    for line in bonds:
        a,b,order=int(line[0:3])-1,int(line[3:6])-1,int(line[6:9]);assert order in orders;bond=mol.GetBondBetweenAtoms(a,b);assert bond is not None and bond.GetBondTypeAsDouble()==orders[order]
    with rdBase.BlockLogs():Chem.SanitizeMol(mol)
    # NCI explicitly states the Corina build used no stereochemical information.
    # Geometry-derived tags must never become source-observed stereochemical labels.
    copy=Chem.Mol(mol);Chem.RemoveStereochemistry(copy);copy.RemoveAllConformers()
    with rdBase.BlockLogs():heavy=Chem.RemoveHs(copy);smiles=Chem.MolToSmiles(heavy,canonical=True,isomericSmiles=True);back=Chem.MolFromSmiles(smiles)
    if back is None:raise ValueError('derived connectivity SMILES roundtrip failed')
    assert Chem.MolToSmiles(back,canonical=True,isomericSmiles=True)==smiles and Chem.GetFormalCharge(back)==Chem.GetFormalCharge(copy) and rdMolDescriptors.CalcMolFormula(back)==rdMolDescriptors.CalcMolFormula(copy)
    properties={};i=property_start
    while i<len(lines):
        match=re.match(r'^>.*<([^<>]+)>',lines[i])
        if match:
            key=match[1];i+=1;values=[]
            while i<len(lines) and lines[i].strip() and not lines[i].startswith('>') and lines[i].strip()!='$$$$':values.append(lines[i]);i+=1
            if key in properties:raise ValueError('duplicate native SDF property requires scope review')
            properties[key]='\n'.join(values)
        else:i+=1
    nsc=properties['NSC'].strip();assert re.fullmatch('[0-9]+',nsc)
    return dict(native_NSC=nsc,native_title=lines[0],native_program_header=lines[1:3],native_count_line=lines[3],native_atom_count=n_atoms,native_bond_count=n_bonds,native_atom_lines=atoms,native_bond_lines=bonds,native_extra_CTAB_lines=extra,native_properties=properties,native_record_sha256=hashlib.sha256(text.encode()).hexdigest(),derived_syntax_added_M_END=not end_present,all_literal_native_elements_bond_endpoints_and_orders_verified=True,derived_nonstereo_connectivity_smiles=smiles,derived_charge=Chem.GetFormalCharge(back),derived_formula=rdMolDescriptors.CalcMolFormula(back),source_stereochemistry_known=False,source_geometry_role='source_computed_Corina_no_optimization_not_observed',derived_graph_coordinates_used=False,source_hydrogen_axes_retained_in_native_tables=True,derived_SMILES_preserves_chemical_formula_charge_and_isotopes=True)
