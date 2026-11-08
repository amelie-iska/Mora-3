"""Lossless native SMILES plus separately derived standard SELFIES source candidates."""
import hashlib
from rdkit import Chem,rdBase
import selfies as sf

def _electronic_signature(mol):
    return sorted((a.GetAtomicNum(),a.GetIsotope(),a.GetFormalCharge(),a.GetNumRadicalElectrons()) for a in mol.GetAtoms())

def encode_molecule(native_smiles):
    result=dict(native_smiles=native_smiles,native_current_RDKit_parse_pass=False,source_canonical_isomeric_smiles=None,source_current_canonical_graph_sha256=None,derived_atom_count=None,derived_bond_count=None,derived_connected_components=None,derived_isotopic_atom_count=None,derived_formal_charge=None,derived_radical_electron_count=None,standard_selfies=None,generated_candidate_selfies=None,standard_selfies_token_count=None,decoded_canonical_isomeric_smiles=None,graph_roundtrip_pass=False,standard_selfies_target_present=False,status="missing_or_invalid_source_smiles",exception_class=None,coordinate_geometry_present=False,task_admitted=False,model_adapter_supported=False)
    if not isinstance(native_smiles,str) or not native_smiles:return result
    with rdBase.BlockLogs():mol=Chem.MolFromSmiles(native_smiles,sanitize=True)
    if mol is None:return result
    canonical=Chem.MolToSmiles(mol,canonical=True,isomericSmiles=True)
    result.update(native_current_RDKit_parse_pass=True,source_canonical_isomeric_smiles=canonical,source_current_canonical_graph_sha256=hashlib.sha256(canonical.encode()).hexdigest(),derived_atom_count=mol.GetNumAtoms(),derived_bond_count=mol.GetNumBonds(),derived_connected_components=len(Chem.GetMolFrags(mol)),derived_isotopic_atom_count=sum(a.GetIsotope()!=0 for a in mol.GetAtoms()),derived_formal_charge=sum(a.GetFormalCharge() for a in mol.GetAtoms()),derived_radical_electron_count=sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms()))
    candidate=None
    try:
        candidate=sf.encoder(native_smiles,strict=True)
        decoded=sf.decoder(candidate)
        with rdBase.BlockLogs():back=Chem.MolFromSmiles(decoded,sanitize=True)
        if back is None:
            result.update(status="decoded_graph_parse_failed",generated_candidate_selfies=candidate);return result
        backcanon=Chem.MolToSmiles(back,canonical=True,isomericSmiles=True)
        result["decoded_canonical_isomeric_smiles"]=backcanon
        if backcanon!=canonical or _electronic_signature(mol)!=_electronic_signature(back) or mol.GetNumBonds()!=back.GetNumBonds():
            result.update(status="graph_roundtrip_mismatch",generated_candidate_selfies=candidate);return result
        result.update(standard_selfies=candidate,standard_selfies_token_count=len(list(sf.split_selfies(candidate))),graph_roundtrip_pass=True,standard_selfies_target_present=True,status="standard_SELFIES_graph_roundtrip_pass")
    except (sf.EncoderError,sf.DecoderError,ValueError,RuntimeError,RecursionError) as exc:
        result.update(status="standard_SELFIES_encoding_or_decoding_quarantine",generated_candidate_selfies=candidate,exception_class=type(exc).__name__)
    return result
