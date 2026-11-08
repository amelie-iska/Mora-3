"""Explicit subject/phenotype comparison; no inferred negative or label transfer."""
import ast,re

_ASPECT={'Process':'biological_process','Function':'molecular_function','Component':'cellular_component'}
def canonical_accession(subject):
    match=re.fullmatch(r'UniProtKB:([A-Z0-9]+)',subject)
    return match[1] if match else None

def source_relation_roles(literal):
    """Parse only bounded literal string lists; never evaluate source code."""
    if not isinstance(literal,str) or len(literal)>4096:return None
    try:value=ast.literal_eval(literal)
    except (ValueError,SyntaxError,MemoryError,RecursionError):return None
    if not isinstance(value,list) or not 1<=len(value)<=16 or any(not isinstance(x,str) for x in value):return None
    roles=[]
    for item in value:
        match=re.fullmatch(r'([a-z_]+)_(Process|Function|Component)',item)
        if not match:return None
        roles.append((match[1],_ASPECT[match[2]]))
    return frozenset(roles)

def potential_target_exposure(subject,go_id,procyon_accession,procyon_GO_id):
    return canonical_accession(subject)==procyon_accession and go_id==procyon_GO_id

def compatible_positive_role(procyon_relation,typed):
    roles=source_relation_roles(procyon_relation)
    if roles is None or typed.get('negated') is not False:return False
    return any((relation,typed.get('aspect')) in roles for relation in typed.get('relations',[]))
