"""Read-only PaRoutes native formats and source-role QC; never admits tasks."""
from __future__ import annotations
import json
from functools import lru_cache
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

def iter_json_array(stream,chunk_chars=65536,max_item_chars=32*1024*1024):
    decoder=json.JSONDecoder();buf="";eof=False
    def more():
        nonlocal buf,eof
        chunk=stream.read(chunk_chars)
        if chunk:buf+=chunk
        else:eof=True
    while not buf.strip() and not eof:more()
    buf=buf.lstrip()
    if not buf.startswith("["):raise ValueError("source must be a JSON array")
    buf=buf[1:];may_end=True
    while True:
        while not buf.strip() and not eof:more()
        buf=buf.lstrip()
        if buf.startswith("]"):
            if not may_end:raise ValueError("trailing comma in source JSON array")
            buf=buf[1:]
            while not eof:more()
            if buf.strip():raise ValueError("trailing garbage after source array")
            return
        while True:
            try:
                item,end=decoder.raw_decode(buf)
                rest=buf[end:].lstrip()
                if not rest and not eof:
                    more();continue
                if not rest or rest[0] not in ",]":
                    if not eof and len(buf)<=max_item_chars:
                        more();continue
                    raise ValueError("missing source array delimiter")
                if end>max_item_chars:raise ValueError("bounded single source JSON item exceeded")
                delimiter=rest[0];buf=rest[1:];break
            except json.JSONDecodeError as exc:
                if eof:raise ValueError("incomplete/invalid source JSON array") from exc
                if len(buf)>max_item_chars:raise ValueError("bounded single source JSON item exceeded") from exc
                more()
        yield item
        if delimiter=="]":
            while not eof:more()
            if buf.strip():raise ValueError("trailing garbage after source array")
            return
        may_end=False

@lru_cache(maxsize=8192)
def smiles_parse(text):
    if not isinstance(text,str) or not text or len(text)>2*1024*1024:return False
    try:return Chem.MolFromSmiles(text) is not None
    except Exception:return False

@lru_cache(maxsize=8192)
def check_reaction(text):
    pieces=text.split(">") if isinstance(text,str) else []
    ok=(len(pieces)==3 and bool(pieces[0]) and bool(pieces[2])
        and all(smiles_parse(x) for x in [pieces[0],pieces[2]])
        and (not pieces[1] or smiles_parse(pieces[1])))
    return {"parse_pass":bool(ok),"source_roles":["reactants","reagents","products"],
            "native_components":pieces,
            "atom_mapping_status":"source_computed_not_independently_validated"}

def file_role(name):
    if name.startswith(("ref_routes","stock_","targets_")):
        return "EVALUATION_EXPOSURE_CONTROL_ONLY"
    if "template" in name or name=="uspto_lookup.json":
        return "COMPUTED_RULE_CONTEXT_NOT_INDEPENDENT_EXPERIMENT"
    return "CANDIDATE_REQUIRES_HELDOUT_AND_CHEMICAL_GROUP_EXCLUSION"

def check_route(route):
    reasons=[];mols=0;rxns=0;hashes=[]
    def visit(node,parent=None,depth=0):
        nonlocal mols,rxns
        if depth>256:reasons.append("source_tree_depth_limit");return
        if not isinstance(node,dict):reasons.append("invalid_source_node");return
        kind=node.get("type")
        if kind not in ("mol","reaction"):reasons.append("unknown_source_node_type")
        if kind==parent:reasons.append("nonalternating_source_tree")
        if kind=="mol":
            mols+=1
            if not smiles_parse(node.get("smiles")):reasons.append("native_molecule_SMILES_parse_failed")
        if kind=="reaction":
            rxns+=1;meta=node.get("metadata",{})
            if not isinstance(meta,dict):reasons.append("invalid_source_reaction_metadata");meta={}
            if meta.get("reaction_hash"):hashes.append(meta["reaction_hash"])
            text=meta.get("rsmi",meta.get("smiles"))
            if not check_reaction(text)["parse_pass"]:reasons.append("native_reaction_SMILES_parse_failed")
        children=node.get("children",[])
        if not isinstance(children,list):reasons.append("invalid_source_children");return
        for child in children:visit(child,kind,depth+1)
    visit(route)
    return {"molecule_nodes":mols,"reaction_nodes":rxns,"source_reaction_hashes":hashes,
            "native_root_smiles":route.get("smiles") if isinstance(route,dict) else None,
            "quality_reasons":sorted(set(reasons))}
