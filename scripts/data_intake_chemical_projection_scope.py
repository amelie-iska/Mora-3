"""Capture lossy stereo parsing explicitly; source originals remain unchanged.

An eligible exact projection is a literal parsed graph comparison. It never
establishes complete stereochemical material, formulation, lot or assay identity.
"""
import logging
from rdkit import Chem,rdBase
from data_intake_corpus import chemical_identity

def project_source_graph(smiles,*,source_unknown_stereo=False):
    if not isinstance(smiles,str) or not smiles or type(source_unknown_stereo) is not bool:raise ValueError('literal source graph and explicit scope required')
    messages=[]
    class Capture(logging.Handler):
        def emit(self,record):messages.append(record.getMessage())
    rdBase.LogToPythonLogger();logger=logging.getLogger('rdkit');handler=Capture();logger.addHandler(handler)
    try:
        mol=Chem.MolFromSmiles(smiles)
        if mol is None:raise ValueError('native source graph unparseable; preserve source without repair')
        maps=any(a.GetAtomMapNum()!=0 for a in mol.GetAtoms());out=[('whole_source_graph',chemical_identity(smiles))]
        fragments=Chem.GetMolFrags(mol,asMols=True,sanitizeFrags=True)
        if len(fragments)>1:
            for fragment in fragments:out.append(('component_of_source_multicomponent_graph',chemical_identity(Chem.MolToSmiles(fragment,isomericSmiles=True))))
    finally:logger.removeHandler(handler)
    messages=list(dict.fromkeys(messages));lossy=any(any(key in text.lower() for key in ['stereo','conflicting','ambiguous']) for text in messages)
    scope='source_parser_stereo_warning_connectivity_guard_only' if lossy else 'source_unknown_stereo_connectivity_guard_only' if source_unknown_stereo else 'literal_isomeric_graph_identity_guard_only_no_material_or_assay_equivalence'
    return dict(source_smiles=smiles,source_atom_maps_present=maps,graphs=out,parser_messages=messages,lossy_stereo_warning=lossy,exact_projection_eligible=not(lossy or source_unknown_stereo),match_scope=scope,source_originals_unchanged=True)
