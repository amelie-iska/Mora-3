"""Conservative source-native ADMET graph scope; no material/assay identity claim."""
import hashlib
from rdkit import Chem
from data_intake_chemical_projection_scope import project_source_graph


def source_graph_scope(native):
    projection=project_source_graph(native,source_unknown_stereo=False)
    mol=Chem.MolFromSmiles(native)
    if mol is None:raise ValueError('native source graph cannot parse; preserve without repair')
    enhanced=len(mol.GetStereoGroups());wildcards=sum(a.GetAtomicNum()==0 for a in mol.GetAtoms());name=mol.GetProp('_Name') if mol.HasProp('_Name') else None;coordinates=mol.GetNumConformers()>0;copy=Chem.Mol(mol)
    for atom in copy.GetAtoms():atom.SetAtomMapNum(0)
    copy=Chem.RemoveHs(copy);cx=Chem.MolToCXSmiles(copy)
    legacy_exact=projection['exact_projection_eligible'] and enhanced==0 and not wildcards and not name and not coordinates
    coarse=sorted({g[k] for _,g in projection['graphs'] for k in ['molecule_group','scaffold_group']})
    return dict(native_representation=native,source_original_unchanged=True,canonical_source_CXSMILES=cx,canonical_CXSMILES_SHA256=hashlib.sha256(cx.encode()).hexdigest(),source_enhanced_stereo_group_count=enhanced,source_wildcard_atom_count=wildcards,source_name_present=bool(name),source_coordinate_representation_present=coordinates,source_atom_maps_present=projection['source_atom_maps_present'],whole_legacy_isomeric_SMILES=projection['graphs'][0][1]['canonical_isomeric_smiles'],legacy_plain_SMILES_exact_comparison_eligible=bool(legacy_exact),legacy_plain_SMILES_scope='literal graph comparison only; enhanced stereo, source names, wildcard or coordinates block unqualified exact promotion',native_predictor_identity_candidate=not(projection['lossy_stereo_warning'] or wildcards or name or coordinates),conservative_leakage_groups=coarse,parser_messages=projection['parser_messages'],lossy_stereo_warning=projection['lossy_stereo_warning'],complete_material_identity_proven=False,assay_or_source_observation_join_authorized=False)
