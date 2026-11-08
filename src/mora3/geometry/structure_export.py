"""Bounded synthetic structure bytes and strict, separate profile reload.

No file/network IO, chemistry inference, geometry solver or authority grant.
CIF is an atom_site-only CIF1.1 projection, not deposition/dictionary validation.
"""
from dataclasses import dataclass,field,asdict
from enum import Enum
import hashlib
import json
import math
import re
import struct
from ..data import spatial_design_constraints as native_types

PROFILE="Mora3.TSE.SyntheticStructureRoundtrip/v1"
MAX_ATOMS=128
MAX_FILE_BYTES=1048576
MAX_LINE=2048
XYZ_COMMENT="Mora3 synthetic XYZ; mandatory identity sidecar"
_CANONICAL=dict(zip(
    ("ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE","LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"),
    "ARNDCQEGHILKMFPSTWYV"))
_CANONICAL.update({v:v for v in tuple(_CANONICAL.values())})
_NUMBER=re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")
_SYMBOL=re.compile(r"[A-Z][a-z]{0,2}")
CIF_TAGS=(
 "_atom_site.group_PDB","_atom_site.id","_atom_site.type_symbol",
 "_atom_site.label_atom_id","_atom_site.label_alt_id","_atom_site.label_comp_id",
 "_atom_site.label_asym_id","_atom_site.label_entity_id","_atom_site.label_seq_id",
 "_atom_site.pdbx_PDB_ins_code","_atom_site.Cartn_x","_atom_site.Cartn_y","_atom_site.Cartn_z",
 "_atom_site.occupancy","_atom_site.B_iso_or_equiv","_atom_site.pdbx_formal_charge",
 "_atom_site.auth_seq_id","_atom_site.auth_comp_id","_atom_site.auth_asym_id",
 "_atom_site.auth_atom_id","_atom_site.pdbx_PDB_model_num")

class Component(str,Enum):
    TS="transition_state_caller_declared"
    ENZYME="enzyme_caller_declared"
    COFACTOR="cofactor_caller_declared"

class SequenceScope(str,Enum):
    FULL="full_caller_declared_sequence"
    PARTIAL="partial_supplied_sequence"
    UNKNOWN="sequence_coverage_unknown"

class FastaStatus(str,Enum):
    SUPPORTED="SUPPORTED_CANONICAL"
    UNSUPPORTED="UNSUPPORTED"

def _text(x,label,limit=256):
    if type(x) is not str or not x or len(x)>limit or any(ord(c)<32 or ord(c)==127 for c in x):
        raise ValueError(label+" unsupported literal/control/length profile")
    try:x.encode("utf8")
    except UnicodeError:raise ValueError(label+" invalidUTF8") from None

def _integer(x,label,minimum=None):
    if type(x) is not int or abs(x)>2**63-1 or (minimum is not None and x<minimum):
        raise ValueError(label+" requires exact bounded integer")

def _float(x,label,minimum=None):
    if type(x) is not float or not math.isfinite(x) or (minimum is not None and x<minimum):
        raise ValueError(label+" requires exact finite float64 scalar")

def _typed(items,cls,label,limit=128):
    if type(items) is not tuple or len(items)>limit or any(type(x) is not cls for x in items):
        raise ValueError(label+" exact bounded tuple; atom profile128")
    for x in items:x.__post_init__()

def _native(atom):
    if type(atom) is not native_types.AtomKey:raise ValueError("exact original AtomKey")
    atom.__post_init__()

def _residue(res):
    if type(res) is not native_types.ResidueKey:raise ValueError("exact original ResidueKey")
    res.__post_init__()

def _key(atom):return atom.entity,atom.ordinal

def _bits(x):return struct.pack(">d",x).hex()

def _unbits(x):
    if type(x) is not str or re.fullmatch("[0-9a-f]{16}",x) is None:raise ValueError("exact float64 bit string")
    z=struct.unpack(">d",bytes.fromhex(x))[0];_float(z,"bitvalue");return z

def _sha(raw):return hashlib.sha256(raw).hexdigest()

def _duplicate_pairs(pairs):
    d={}
    for k,v in pairs:
        if k in d:raise ValueError("duplicate JSON key")
        d[k]=v
    return d

def _bad_constant(value):raise ValueError("nonfinite JSON constant")

def _json_load(raw,context=False):
    if type(raw) is bytes:
        if not raw or len(raw)>MAX_FILE_BYTES:raise ValueError("bounded nonempty sidecar required")
        text=raw.decode("utf8")
    elif type(raw) is str and len(raw.encode("utf8"))<=16384:text=raw
    else:raise ValueError("bounded exact JSON bytes/string")
    kwargs=dict(object_pairs_hook=_duplicate_pairs,parse_constant=_bad_constant)
    if context:kwargs.update(parse_float=str,parse_int=str)
    try:return json.loads(text,**kwargs)
    except (UnicodeError,ValueError,RecursionError) as e:raise ValueError("invalid bounded JSON: "+type(e).__name__) from None

def _json_bytes(value):
    z=(json.dumps(value,ensure_ascii=False,separators=(",",":"),allow_nan=False)+"\n").encode("utf8")
    if len(z)>MAX_FILE_BYTES:raise ValueError("sidecar profile extent exceeded")
    return z

@dataclass(frozen=True)
class ExportAtom:
    native:native_types.AtomKey
    xyz:tuple[float,float,float]
    component:Component
    group_pdb:str|None=None
    occupancy:float|None=None
    b_factor:float|None=None
    model_number:int|None=None
    def __post_init__(self):
        _native(self.native)
        if type(self.xyz) is not tuple or len(self.xyz)!=3:raise ValueError("three exact float64 positions")
        for x in self.xyz:_float(x,"coordinate")
        if type(self.component) is not Component:raise ValueError("explicit caller component role")
        if self.component is Component.ENZYME and self.native.residue is None:raise ValueError("enzyme native residue identity required")
        if self.component is Component.TS and self.native.residue is not None:raise ValueError("TS cannot alias protein residue")
        if self.group_pdb is not None and (type(self.group_pdb) is not str or self.group_pdb not in ("ATOM","HETATM")):
            raise ValueError("unknown/explicit group_PDB only")
        if self.occupancy is not None:
            _float(self.occupancy,"occupancy",0.)
            if self.occupancy>1.:raise ValueError("occupancy outside[0,1]")
        if self.b_factor is not None:_float(self.b_factor,"Bfactor",0.)
        if self.model_number is not None:_integer(self.model_number,"explicit modelnumber",1)

@dataclass(frozen=True)
class SequenceChain:
    native_chain:str
    identifier:str
    residues:tuple[native_types.ResidueKey,...]
    scope:SequenceScope
    def __post_init__(self):
        _text(self.native_chain,"native chain",128);_text(self.identifier,"FASTAidentifier",128)
        _typed(self.residues,native_types.ResidueKey,"literal sequence",4096)
        if type(self.scope) is not SequenceScope:raise ValueError("explicit sequence coverage")
        if self.scope is SequenceScope.FULL and not self.residues:raise ValueError("full sequence needs identities")
        if any(r.chain!=self.native_chain for r in self.residues):raise ValueError("exact native sequence chain")
        if len({(r.chain,r.index,r.insertion) for r in self.residues})!=len(self.residues):
            raise ValueError("duplicate/ambiguous residue positions")

@dataclass(frozen=True)
class BondFact:
    first:native_types.AtomKey
    second:native_types.AtomKey
    order:float|None
    provenance:str
    def __post_init__(self):
        _native(self.first);_native(self.second)
        if _key(self.first)==_key(self.second):raise ValueError("self bond unsupported")
        if self.order is not None:_float(self.order,"caller fractional bond order",0.)
        _text(self.provenance,"bond source/protocol declaration")

@dataclass(frozen=True)
class ExportPacket:
    identifier:str
    frame:str
    atoms:tuple[ExportAtom,...]
    chains:tuple[SequenceChain,...]
    total_charge:int|None=None
    spin_multiplicity:int|None=None
    saddle_order:int|None=None
    bonds:tuple[BondFact,...]|None=None
    context_json:str="{}"
    origin:str="synthetic_handcrafted"
    def __post_init__(self):
        _text(self.identifier,"packet identifier",128);_text(self.frame,"explicit frame")
        if type(self.origin) is not str or self.origin!="synthetic_handcrafted":raise ValueError("native/source origin unqualified")
        if type(self.atoms) is not tuple or not 1<=len(self.atoms)<=128:raise ValueError("unsupported full128 atom profile; never crop")
        _typed(self.atoms,ExportAtom,"complete ordered atom axis")
        if len({_key(a.native) for a in self.atoms})!=len(self.atoms):raise ValueError("duplicate native atom IDs")
        if not any(a.component is Component.TS for a in self.atoms):raise ValueError("explicit nonempty TS selection")
        _typed(self.chains,SequenceChain,"ordered chains",32)
        if len({c.native_chain for c in self.chains})!=len(self.chains) or len({c.identifier for c in self.chains})!=len(self.chains):
            raise ValueError("duplicate sequence chain/identifier")
        if self.total_charge is not None:_integer(self.total_charge,"totalcharge")
        if self.spin_multiplicity is not None:_integer(self.spin_multiplicity,"spin multiplicity",1)
        if self.saddle_order is not None:_integer(self.saddle_order,"saddle order",0)
        if self.bonds is not None:
            _typed(self.bonds,BondFact,"declared bonds",512)
            natives={a.native for a in self.atoms}
            if any(x.first not in natives or x.second not in natives for x in self.bonds):raise ValueError("unbound exact native bond identity")
            keys=[frozenset((_key(x.first),_key(x.second))) for x in self.bonds]
            if len(set(keys))!=len(keys):raise ValueError("duplicate ambiguous bond")
        _json_load(self.context_json,context=True)

@dataclass(frozen=True)
class ExportExpectation:
    identity_sha256:str
    sidecar_sha256:str
    profile:str=PROFILE
    def __post_init__(self):
        if type(self.profile) is not str or self.profile!=PROFILE:raise ValueError("exact profile revision")
        for x in (self.identity_sha256,self.sidecar_sha256):
            if type(x) is not str or re.fullmatch("[0-9a-f]{64}",x) is None:raise ValueError("exact external expected digest")

@dataclass(frozen=True)
class ExportBundle:
    xyz:bytes
    mmcif:bytes
    fasta:bytes|None
    sidecar:bytes
    expectation:ExportExpectation
    fasta_status:FastaStatus

@dataclass(frozen=True)
class XYZPayload:
    symbols:tuple[str,...]
    xyz:tuple[tuple[float,float,float],...]
    native_source_accepted:bool=field(default=False,init=False)

@dataclass(frozen=True)
class CifRow:
    symbol:str
    xyz:tuple[float,float,float]
    group_pdb:str|None
    insertion:str|None
    occupancy:float|None
    b_factor:float|None
    formal_charge:int|None
    auth_seq_id:str|None
    auth_comp_id:str|None
    auth_asym_id:str|None
    auth_atom_id:str|None
    model_number:int|None

@dataclass(frozen=True)
class CifPayload:
    identifier:str
    rows:tuple[CifRow,...]
    native_source_accepted:bool=field(default=False,init=False)

@dataclass(frozen=True)
class ReloadedPacket:
    packet:ExportPacket
    literal_roundtrip_verified:bool=field(default=True,init=False)
    native_source_accepted:bool=field(default=False,init=False)
    export_constraints_accepted:bool=field(default=False,init=False)
    def require_native_or_model_use(self):
        raise RuntimeError("roundtrip is not native/source/model/training authority")
    def require_export_constraint_acceptance(self):
        raise RuntimeError("requires real source authority AND numeric+freeze checks on RELOADED coords")

def _quote(s):
    _text(s,"CIFstring",128)
    if any(ord(c)>126 for c in s):raise ValueError("unsupportedCIF1.1 printableASCII profile")
    if "'" not in s:return "'"+s+"'"
    if '"' not in s:return '"'+s+'"'
    raise ValueError("unsupportedCIF profile both-quote string; no lossy repair")

def _optional_string(s):return "?" if s is None else _quote(s)

def _repr_float(x):
    _float(x,"serialized coordinate/value")
    return repr(x)

def _metadata(packet):
    return dict(identifier=packet.identifier,frame=packet.frame,units="angstrom",origin=packet.origin,
        atoms=[dict(native=asdict(a.native),xyz_bits=[_bits(x) for x in a.xyz],component=a.component.value,
            group_pdb=a.group_pdb,occupancy_bits=None if a.occupancy is None else _bits(a.occupancy),
            b_factor_bits=None if a.b_factor is None else _bits(a.b_factor),model_number=a.model_number) for a in packet.atoms],
        chains=[dict(native_chain=c.native_chain,identifier=c.identifier,residues=[asdict(x) for x in c.residues],scope=c.scope.value) for c in packet.chains],
        total_charge=packet.total_charge,spin_multiplicity=packet.spin_multiplicity,saddle_order=packet.saddle_order,
        bonds=None if packet.bonds is None else [dict(first=asdict(x.first),second=asdict(x.second),
            order_bits=None if x.order is None else _bits(x.order),provenance=x.provenance) for x in packet.bonds],
        context_json=packet.context_json)

def identity_digest(packet):
    if type(packet) is not ExportPacket:raise ValueError("exact literal packet")
    packet.__post_init__()
    return _sha(json.dumps(_metadata(packet),ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode("utf8"))

def _write_xyz(packet):
    selected=tuple(a for a in packet.atoms if a.component is Component.TS)
    lines=[str(len(selected)),XYZ_COMMENT]
    for a in selected:
        if _SYMBOL.fullmatch(a.native.element) is None:raise ValueError("unrepresentable explicitXYZ element label; no inferred symbol")
        lines.append(a.native.element+" "+" ".join(_repr_float(x) for x in a.xyz))
    return ("\n".join(lines)+"\n").encode("ascii")

def _write_fasta(packet):
    if not packet.chains:return None,FastaStatus.UNSUPPORTED
    lines=[]
    for c in packet.chains:
        if c.scope is not SequenceScope.FULL or re.fullmatch(r"[A-Za-z0-9_.:|+\-]{1,64}",c.identifier) is None:
            return None,FastaStatus.UNSUPPORTED
        if any(r.identity not in _CANONICAL for r in c.residues):return None,FastaStatus.UNSUPPORTED
        sequence="".join(_CANONICAL[r.identity] for r in c.residues)
        lines.append(">"+c.identifier)
        lines.extend(sequence[i:i+80] for i in range(0,len(sequence),80))
    return ("\n".join(lines)+"\n").encode("ascii"),FastaStatus.SUPPORTED

def _write_mmcif(packet):
    lines=["data_Mora3_synthetic","_entry.id "+_quote(packet.identifier),"#","loop_"]+list(CIF_TAGS)
    for i,a in enumerate(packet.atoms):
        n=a.native;r=n.residue
        if _SYMBOL.fullmatch(n.element) is None:raise ValueError("unsupported literal element symbol")
        insertion="?" if r is None else "." if r.insertion=="" else _quote(r.insertion)
        values=[_optional_string(a.group_pdb),str(i+1),_quote(n.element)]+["?"]*6+[insertion]+[_repr_float(x) for x in a.xyz]
        values += ["?" if a.occupancy is None else _repr_float(a.occupancy),
            "?" if a.b_factor is None else _repr_float(a.b_factor),
            "?" if n.formal_charge is None else str(n.formal_charge),
            "?" if r is None else _quote(str(r.index)),
            "?" if r is None else _quote(r.identity),
            "?" if r is None else _quote(r.chain),_quote(n.name),
            "?" if a.model_number is None else str(a.model_number)]
        row=" ".join(values)
        if len(row)>MAX_LINE:raise ValueError("unrepresentableCIF profile line extent")
        lines.append(row)
    lines.append("#")
    return ("\n".join(lines)+"\n").encode("ascii")

def export_bundle(packet):
    """Format bytes only. No paths, parser/network operations or authority receipt."""
    if type(packet) is not ExportPacket:raise ValueError("exact bounded synthetic packet")
    packet.__post_init__()
    xyz=_write_xyz(packet);mmcif=_write_mmcif(packet);fasta,status=_write_fasta(packet)
    for raw in (xyz,mmcif,fasta):
        if raw is not None and len(raw)>MAX_FILE_BYTES:raise ValueError("format extent profile")
    files={name:dict(bytes=None if raw is None else len(raw),sha256=None if raw is None else _sha(raw))
        for name,raw in (("xyz",xyz),("mmcif",mmcif),("fasta",fasta))}
    digest=identity_digest(packet)
    sidecar=_json_bytes(dict(schema=PROFILE,metadata=_metadata(packet),identity_sha256=digest,files=files,
        fasta_status=status.value,source_authority=None,model_authority=None,export_constraints_accepted=False))
    expected=ExportExpectation(digest,_sha(sidecar))
    return ExportBundle(xyz,mmcif,fasta,sidecar,expected,status)

# RELOAD: bounded lexical/data reconstruction, not calls to writer/format helpers.
def _lines(raw):
    if type(raw) is not bytes or not raw or len(raw)>MAX_FILE_BYTES:raise ValueError("bounded exact formatbytes")
    try:text=raw.decode("ascii")
    except UnicodeError:raise ValueError("unsupported nonASCIIprofile") from None
    if "\r" in text or not text.endswith("\n"):raise ValueError("strictLF complete record terminator")
    lines=text[:-1].split("\n")
    if any(len(x)>MAX_LINE or any(ord(c)<32 and c!="\t" for c in x) for x in lines):
        raise ValueError("unsupported record extent/control")
    return lines

def _number_value(text):
    if type(text) is not str or _NUMBER.fullmatch(text) is None:raise ValueError("strict numeric lexical form")
    try:x=float(text)
    except (ValueError,OverflowError):raise ValueError("invalidfloat64") from None
    _float(x,"decoded numeric");return x

def read_xyz_profile(raw):
    lines=_lines(raw)
    if len(lines)<3 or re.fullmatch(r"[1-9][0-9]*",lines[0]) is None:raise ValueError("exactXYZ count")
    n=int(lines[0])
    if not 1<=n<=128 or len(lines)!=n+2 or lines[1]!=XYZ_COMMENT:raise ValueError("XYZ extent/frame/profile mismatch")
    symbols=[];positions=[]
    for line in lines[2:]:
        parts=line.split()
        if len(parts)!=4 or _SYMBOL.fullmatch(parts[0]) is None:raise ValueError("strictXYZ atom row")
        symbols.append(parts[0]);positions.append(tuple(_number_value(x) for x in parts[1:]))
    return XYZPayload(tuple(symbols),tuple(positions))

def _lex_row(line):
    result=[];i=0
    while i<len(line):
        if line[i] in " \t":i+=1;continue
        c=line[i]
        if c in ("'",'"'):
            j=line.find(c,i+1)
            if j<0 or (j+1<len(line) and line[j+1] not in " \t"):raise ValueError("unsupported/truncated profile quote")
            result.append((line[i+1:j],True));i=j+1
        else:
            j=i
            while j<len(line) and line[j] not in " \t":j+=1
            word=line[i:j]
            if word.startswith(("#",";","_","$","[","]")) or word.lower().startswith(("data_","save_")) or word.lower() in ("loop_","stop_","global_"):
                raise ValueError("control/injection outside quoted row data")
            result.append((word,False));i=j
    return tuple(result)

def _string_token(token,nullable=True):
    value,quoted=token
    if not quoted:
        if nullable and value=="?":return None
        raise ValueError("expected quoted literal or unknown token")
    return value

def _float_token(token,nullable=False):
    value,quoted=token
    if not quoted and nullable and value=="?":return None
    if quoted:raise ValueError("quoted numeric is a string, not numeric value")
    return _number_value(value)

def _int_token(token,nullable=False,minimum=None):
    value,quoted=token
    if not quoted and nullable and value=="?":return None
    if quoted or re.fullmatch(r"-?(?:0|[1-9][0-9]*)",value) is None:raise ValueError("exact integer token")
    x=int(value);_integer(x,"integer token",minimum);return x

def read_mmcif_profile(raw):
    lines=_lines(raw);header_count=4+len(CIF_TAGS)
    if len(lines)<header_count+2 or lines[0]!="data_Mora3_synthetic" or lines[2:4]!=["#","loop_"] or lines[4:header_count]!=list(CIF_TAGS) or lines[-1]!="#":
        raise ValueError("exact single-block/tag/loop profile required")
    if not lines[1].startswith("_entry.id "):raise ValueError("entry scalar")
    values=_lex_row(lines[1][len("_entry.id "):])
    if len(values)!=1:raise ValueError("entry value")
    identifier=_string_token(values[0],False)
    _text(identifier,"entry identifier",128)
    rows=[]
    for i,line in enumerate(lines[header_count:-1]):
        ts=_lex_row(line)
        if len(ts)!=21 or _int_token(ts[1],minimum=1)!=i+1:raise ValueError("row width/native record order")
        if any(t!=("?",False) for t in ts[3:9]):raise ValueError("standardized identity namespaces unsupported; not invented")
        symbol=_string_token(ts[2],False)
        if _SYMBOL.fullmatch(symbol) is None:raise ValueError("type symbol")
        group=_string_token(ts[0])
        if group not in (None,"ATOM","HETATM"):raise ValueError("group_PDB")
        insertion="" if ts[9]==(".",False) else _string_token(ts[9])
        xyz=tuple(_float_token(t) for t in ts[10:13])
        occ=_float_token(ts[13],True);bf=_float_token(ts[14],True)
        if occ is not None and not 0.<=occ<=1.:raise ValueError("occupancy")
        if bf is not None and bf<0.:raise ValueError("Bfactor")
        row=CifRow(symbol,xyz,group,insertion,occ,bf,_int_token(ts[15],True),
            _string_token(ts[16]),_string_token(ts[17]),_string_token(ts[18]),_string_token(ts[19]),
            _int_token(ts[20],True,1))
        rows.append(row)
    if not 1<=len(rows)<=128:raise ValueError("unsupported128 atom profile")
    return CifPayload(identifier,tuple(rows))

def _read_fasta(raw):
    lines=_lines(raw);out=[];identifier=None;sequence=[]
    for line in lines:
        if line.startswith(">"):
            if identifier is not None:out.append((identifier,"".join(sequence)))
            identifier=line[1:];sequence=[]
            if re.fullmatch(r"[A-Za-z0-9_.:|+\-]{1,64}",identifier) is None:raise ValueError("FASTA identifier")
        else:
            if identifier is None or not 1<=len(line)<=80 or any(c not in "ARNDCQEGHILKMFPSTWYV" for c in line):
                raise ValueError("strict canonicalFASTA sequence")
            sequence.append(line)
    if identifier is not None:out.append((identifier,"".join(sequence)))
    if not out or any(not s for _,s in out) or len({i for i,_ in out})!=len(out):raise ValueError("FASTArecords")
    return tuple(out)

def _keys(d,keys):
    if type(d) is not dict or set(d)!=set(keys):raise ValueError("exact supported schema keys")

def _residue_data(d):
    if d is None:return None
    _keys(d,("chain","index","identity","insertion"))
    return native_types.ResidueKey(d["chain"],d["index"],d["identity"],d["insertion"])

def _native_data(d):
    _keys(d,("entity","ordinal","element","name","residue","isotope","formal_charge"))
    return native_types.AtomKey(d["entity"],d["ordinal"],d["element"],d["name"],_residue_data(d["residue"]),d["isotope"],d["formal_charge"])

def reload_bundle(files,expected):
    """Strict bytes/schema/profile reconstruction; no writer functions are called."""
    if type(files) is not ExportBundle or type(expected) is not ExportExpectation:raise ValueError("exact bundle/external expected binding")
    expected.__post_init__()
    if type(files.sidecar) is not bytes or _sha(files.sidecar)!=expected.sidecar_sha256:raise ValueError("sidecar expectedhash missing/mismatch")
    root=_json_load(files.sidecar)
    _keys(root,("schema","metadata","identity_sha256","files","fasta_status","source_authority","model_authority","export_constraints_accepted"))
    if root["schema"]!=PROFILE or root["source_authority"] is not None or root["model_authority"] is not None or root["export_constraints_accepted"] is not False:
        raise ValueError("unsupported profile or fabricated authority")
    _keys(root["files"],("xyz","mmcif","fasta"))
    for key,raw in (("xyz",files.xyz),("mmcif",files.mmcif),("fasta",files.fasta)):
        info=root["files"][key];_keys(info,("bytes","sha256"))
        if raw is None:
            if key!="fasta" or info!={"bytes":None,"sha256":None}:raise ValueError("missing mandatory bytes")
        elif type(raw) is not bytes or len(raw)>MAX_FILE_BYTES or type(info["bytes"]) is not int or info["bytes"]!=len(raw) or info["sha256"]!=_sha(raw):
            raise ValueError("format checksum/extent mismatch")
    xyz=read_xyz_profile(files.xyz);cif=read_mmcif_profile(files.mmcif)
    md=root["metadata"]
    _keys(md,("identifier","frame","units","origin","atoms","chains","total_charge","spin_multiplicity","saddle_order","bonds","context_json"))
    if md["units"]!="angstrom" or md["origin"]!="synthetic_handcrafted" or md["identifier"]!=cif.identifier:raise ValueError("identity/frame/profile metadata")
    if type(md["atoms"]) is not list or len(md["atoms"])!=len(cif.rows):raise ValueError("complete native axis metadata")
    atoms=[]
    for a,row in zip(md["atoms"],cif.rows):
        _keys(a,("native","xyz_bits","component","group_pdb","occupancy_bits","b_factor_bits","model_number"))
        n=_native_data(a["native"])
        if type(a["xyz_bits"]) is not list or len(a["xyz_bits"])!=3 or [_bits(x) for x in row.xyz]!=a["xyz_bits"]:
            raise ValueError("serialized float64 bits not exact source")
        for bit in a["xyz_bits"]:_unbits(bit)
        if (row.symbol,row.group_pdb,row.formal_charge,row.model_number)!=(n.element,a["group_pdb"],n.formal_charge,a["model_number"]):
            raise ValueError("native chemical/annotation projection mismatch")
        r=n.residue
        expected_auth=(None,None,None,None) if r is None else (str(r.index),r.identity,r.chain,r.insertion)
        if (row.auth_seq_id,row.auth_comp_id,row.auth_asym_id,row.insertion)!=expected_auth or row.auth_atom_id!=n.name:
            raise ValueError("native author labels/identities mismatch")
        if (None if row.occupancy is None else _bits(row.occupancy))!=a["occupancy_bits"] or (None if row.b_factor is None else _bits(row.b_factor))!=a["b_factor_bits"]:
            raise ValueError("occupancy/Bfactor bit mismatch")
        atoms.append(ExportAtom(n,row.xyz,Component(a["component"]),a["group_pdb"],row.occupancy,row.b_factor,a["model_number"]))
    selected=tuple(a for a in atoms if a.component is Component.TS)
    if xyz.symbols!=tuple(a.native.element for a in selected) or tuple(tuple(_bits(x) for x in p) for p in xyz.xyz)!=tuple(tuple(_bits(x) for x in a.xyz) for a in selected):
        raise ValueError("XYZ/CIF/native TS selection/bit mismatch")
    if type(md["chains"]) is not list or len(md["chains"])>32:raise ValueError("chain axis")
    chains=[]
    for c in md["chains"]:
        _keys(c,("native_chain","identifier","residues","scope"))
        if type(c["residues"]) is not list or len(c["residues"])>4096:raise ValueError("literal sequence extent")
        chains.append(SequenceChain(c["native_chain"],c["identifier"],tuple(_residue_data(d) for d in c["residues"]),SequenceScope(c["scope"])))
    if md["bonds"] is None:bonds=None
    else:
        if type(md["bonds"]) is not list or len(md["bonds"])>512:raise ValueError("bond facts extent")
        items=[]
        for z in md["bonds"]:
            _keys(z,("first","second","order_bits","provenance"))
            items.append(BondFact(_native_data(z["first"]),_native_data(z["second"]),None if z["order_bits"] is None else _unbits(z["order_bits"]),z["provenance"]))
        bonds=tuple(items)
    packet=ExportPacket(md["identifier"],md["frame"],tuple(atoms),tuple(chains),
        md["total_charge"],md["spin_multiplicity"],md["saddle_order"],bonds,md["context_json"],md["origin"])
    # This source identity check includes the complete literal native metadata
    # and sidecar-only electronic/context facts; caller expectedbinding is mandatory.
    actual=identity_digest(packet)
    if actual!=expected.identity_sha256 or root["identity_sha256"]!=actual:raise ValueError("external original identity binding mismatch")
    status=FastaStatus(root["fasta_status"])
    representable=bool(chains) and all(c.scope is SequenceScope.FULL and re.fullmatch(r"[A-Za-z0-9_.:|+\-]{1,64}",c.identifier) is not None and all(r.identity in _CANONICAL for r in c.residues) for c in chains)
    if status is FastaStatus.SUPPORTED:
        if not representable or files.fasta is None:raise ValueError("unsupported sequence cannot be canonicalized")
        parsed=_read_fasta(files.fasta)
        if parsed!=tuple((c.identifier,"".join(_CANONICAL[r.identity] for r in c.residues)) for c in chains):raise ValueError("FASTA literal identity/sequence mismatch")
    elif representable or files.fasta is not None:raise ValueError("FASTA unsupported disposition mismatch")
    if type(files.fasta_status) is not FastaStatus or files.fasta_status is not status:raise ValueError("exact FASTA disposition")
    return ReloadedPacket(packet)
