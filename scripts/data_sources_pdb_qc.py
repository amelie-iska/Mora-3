"""Preserve official PDB validation/chemical source roles; no admission thresholds."""
import xml.etree.ElementTree as ET
from data_sources_native import category_rows


def validation_record(xml_bytes, pdb_id):
    root = ET.fromstring(xml_bytes)
    entry = root.find("Entry")
    if entry is None or entry.attrib.get("pdbid", "").upper() != pdb_id.upper():
        raise ValueError("validation report PDB identity mismatch")
    def node(e):
        return {"tag": e.tag, "attributes": dict(e.attrib), "text": e.text,
                "children": [node(c) for c in e]}
    return {"pdb_id": pdb_id, "root_attributes": dict(root.attrib),
        "entry_attributes": dict(entry.attrib),
        "modelled_subgroups": [node(e) for e in root.findall("ModelledSubgroup")],
        "other_source_elements": [node(e) for e in root if e.tag not in ("Entry", "ModelledSubgroup")],
        "evidence_class": "official_experimental_structure_validation",
        "input_role": "quality/audit metadata, excluded from structure predictor conditions",
        "task_admitted": False}


def component_context(cif):
    entities = {r["id"]: r for r in category_rows(cif, "_entity.")}
    grouped = {}
    for atom in category_rows(cif, "_atom_site."):
        ccd = atom.get("label_comp_id")
        if atom.get("group_PDB") != "HETATM" or ccd == "HOH":
            continue
        key = ccd
        g = grouped.setdefault(key, {"ccd_id": ccd, "source_entity_ids": set(), "source_entity_types": set(), "source_asym_ids": set(), "native_atom_rows": 0})
        entity = atom.get("label_entity_id")
        g["source_entity_ids"].add(entity)
        g["source_entity_types"].add(entities.get(entity, {}).get("type", "unknown"))
        g["source_asym_ids"].add(atom.get("label_asym_id"))
        g["native_atom_rows"] += 1
    result = []
    for ccd, g in sorted(grouped.items()):
        for k in ("source_entity_ids", "source_entity_types", "source_asym_ids"):
            g[k] = sorted(g[k])
        g["binding_ligand_role"] = None
        g["role_note"] = "HETATM alone does not establish a binding ligand; modified polymers, ions, cofactors, solvent and glycans require distinct source roles."
        result.append(g)
    return result
