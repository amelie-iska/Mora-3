"""Synthetic enzyme-reference wire profile; no TS, source IO or authority grant.

Exact bounded reuse of structure_export v1 lexical/native/float/writer primitives.
The larger reference axis does not expand any request/math/composition profile.
"""
from dataclasses import dataclass,field,asdict
from enum import Enum
import hashlib,json,re
from . import structure_export as wire

PROFILE="Mora3.TSE.SyntheticProteinReference/v1"
REVISION="Mora3.TSE.ProteinReferenceWire/v1"
CODEC_SHA256="0eec7182b005287c5e5570f395c4ef13e5c64a7d6dca84001bc0fa5d6d2fc110"
native_types=wire.native_types
MAX_ATOMS=8192
MAX_SEQUENCE=8192
MAX_FILE_BYTES=8388608

class StructureScope(str,Enum):
    COMPLETE_SUPPLIED="complete_caller_supplied_atom_set_not_native_completeness"
    PARTIAL="partial_caller_supplied_structure"
    UNKNOWN="structure_coverage_unknown"

class ReferencePurpose(str,Enum):
    GUIDANCE="caller_supplied_guidance_reference"
    EVALUATION="evaluation_reference_never_conditioning"

def _count(x,label,maximum):
    if type(x) is not int or not 1<=x<=maximum:raise ValueError("exact positive bounded "+label)

def _items(values,cls,label,limit):
    if type(values) is not tuple or len(values)>limit or any(type(x) is not cls for x in values):
        raise ValueError("exact bounded "+label)
    for x in values:x.__post_init__()

def _sha(raw):return hashlib.sha256(raw).hexdigest()

def _json(value):
    return (json.dumps(value,ensure_ascii=False,separators=(",",":"),allow_nan=False)+"\n").encode("utf8")

@dataclass(frozen=True)
class ReferenceBudget:
    max_atoms:int
    max_sequence_residues:int
    max_chains:int
    max_format_bytes:int
    max_sidecar_bytes:int
    max_total_bytes:int
    def __post_init__(self):
        for x,label,limit in ((self.max_atoms,"atoms",MAX_ATOMS),(self.max_sequence_residues,"sequence residues",MAX_SEQUENCE),
            (self.max_chains,"chains",32),(self.max_format_bytes,"format bytes",MAX_FILE_BYTES),
            (self.max_sidecar_bytes,"sidecar bytes",MAX_FILE_BYTES),(self.max_total_bytes,"aggregate bytes",25165824)):
            _count(x,label,limit)
    def digest(self):
        self.__post_init__();return _sha(_json(asdict(self)))

@dataclass(frozen=True)
class ProteinReferencePacket:
    identifier:str
    frame:str
    atoms:tuple[wire.ExportAtom,...]
    chains:tuple[wire.SequenceChain,...]
    structure_scope:StructureScope
    purpose:ReferencePurpose
    role:native_types.Role
    missing_atoms:tuple[native_types.AtomKey,...]=()
    missing_residues:tuple[native_types.ResidueKey,...]=()
    total_charge:int|None=None
    spin_multiplicity:int|None=None
    bonds:tuple[wire.BondFact,...]|None=None
    context_json:str="{}"
    origin:str="synthetic_handcrafted"
    saddle_order:None=field(default=None,init=False)
    def __post_init__(self):
        wire._text(self.identifier,"reference identifier",128);wire._text(self.frame,"explicit reference frame")
        if type(self.origin) is not str or self.origin!="synthetic_handcrafted":raise ValueError("unqualified native/source origin")
        _items(self.atoms,wire.ExportAtom,"ordered complete supplied atom axis",MAX_ATOMS)
        if not self.atoms or not any(a.component is wire.Component.ENZYME for a in self.atoms):
            raise ValueError("nonempty caller enzyme reference required")
        if any(a.component is wire.Component.TS for a in self.atoms):raise ValueError("reference-only profile forbids TS/candidate roles")
        if len({wire._key(a.native) for a in self.atoms})!=len(self.atoms):raise ValueError("duplicate native atom IDs")
        _items(self.chains,wire.SequenceChain,"ordered literal supplied chains",32)
        if sum(len(c.residues) for c in self.chains)>MAX_SEQUENCE:raise ValueError("reference sequence profile")
        if len({c.native_chain for c in self.chains})!=len(self.chains) or len({c.identifier for c in self.chains})!=len(self.chains):
            raise ValueError("duplicate literal supplied chains")
        if type(self.structure_scope) is not StructureScope or type(self.purpose) is not ReferencePurpose or type(self.role) is not native_types.Role:
            raise ValueError("exact reference scope/purpose/role")
        expected=native_types.Role.USER_SUPPLIED if self.purpose is ReferencePurpose.GUIDANCE else native_types.Role.EVALUATION_REFERENCE
        if self.role is not expected:raise ValueError("no source/target/generated/eval role substitution")
        _items(self.missing_atoms,native_types.AtomKey,"explicit missing atom declarations",MAX_ATOMS)
        _items(self.missing_residues,native_types.ResidueKey,"explicit missing residue declarations",MAX_SEQUENCE)
        present={wire._key(a.native) for a in self.atoms}
        if len({wire._key(x) for x in self.missing_atoms})!=len(self.missing_atoms) or any(wire._key(x) in present for x in self.missing_atoms):
            raise ValueError("missing/present atom collision")
        present_residues={(a.native.residue.chain,a.native.residue.index,a.native.residue.insertion) for a in self.atoms if a.native.residue is not None}
        missing_slots=[(x.chain,x.index,x.insertion) for x in self.missing_residues]
        if len(set(missing_slots))!=len(missing_slots) or any(x in present_residues for x in missing_slots):raise ValueError("missing/present residue collision")
        if self.total_charge is not None:wire._integer(self.total_charge,"caller total charge")
        if self.spin_multiplicity is not None:wire._integer(self.spin_multiplicity,"caller spin",1)
        if self.bonds is not None:
            _items(self.bonds,wire.BondFact,"explicit provided bonds",512)
            natives={a.native for a in self.atoms}
            if any(x.first not in natives or x.second not in natives for x in self.bonds):raise ValueError("unbound literal bond")
            keys=[frozenset((wire._key(x.first),wire._key(x.second))) for x in self.bonds]
            if len(set(keys))!=len(keys):raise ValueError("duplicate bond declaration")
        wire._json_load(self.context_json,context=True)

def _preflight(packet,budget):
    if type(packet) is not ProteinReferencePacket or type(budget) is not ReferenceBudget:raise ValueError("exact reference packet/budget required")
    packet.__post_init__();budget.__post_init__()
    if len(packet.atoms)>budget.max_atoms or len(packet.missing_atoms)>budget.max_atoms:
        raise ValueError("declared reference atom budget, no crop")
    if sum(len(c.residues) for c in packet.chains)>budget.max_sequence_residues or len(packet.missing_residues)>budget.max_sequence_residues or len(packet.chains)>budget.max_chains:
        raise ValueError("declared reference sequence budget, no crop")

def _metadata(packet):
    return dict(literal=wire._metadata(packet),structure_scope=packet.structure_scope.value,
        purpose=packet.purpose.value,role=packet.role.value,
        missing_atoms=[asdict(x) for x in packet.missing_atoms],missing_residues=[asdict(x) for x in packet.missing_residues])

def identity_digest(packet):
    if type(packet) is not ProteinReferencePacket:raise ValueError("exact reference packet")
    packet.__post_init__()
    return _sha(_json(_metadata(packet)))

@dataclass(frozen=True)
class ReferenceExpectation:
    identity_sha256:str
    sidecar_sha256:str
    budget_sha256:str
    profile:str=PROFILE
    revision:str=REVISION
    def __post_init__(self):
        if type(self.profile) is not str or self.profile!=PROFILE or type(self.revision) is not str or self.revision!=REVISION:
            raise ValueError("distinct exact reference profile/revision")
        for x in (self.identity_sha256,self.sidecar_sha256,self.budget_sha256):
            if type(x) is not str or re.fullmatch("[0-9a-f]{64}",x) is None:raise ValueError("external expected SHA256")

@dataclass(frozen=True)
class ReferenceBundle:
    mmcif:bytes
    fasta:bytes|None
    sidecar:bytes
    expectation:ReferenceExpectation
    fasta_status:wire.FastaStatus

@dataclass(frozen=True)
class ReloadedReference:
    packet:ProteinReferencePacket
    literal_roundtrip_verified:bool=field(default=True,init=False)
    native_source_accepted:bool=field(default=False,init=False)
    export_constraints_accepted:bool=field(default=False,init=False)
    def require_native_or_model_use(self):raise RuntimeError("synthetic reference syntax is not source/native/model authority")
    def require_candidate_or_TS_use(self):raise RuntimeError("reference-only profile cannot substitute for a generated candidate/TS")
    def require_conditioning_use(self):raise RuntimeError("source/model/purpose/heldout authority separately pending")

def _extents(files,budget):
    if type(files) is not ReferenceBundle or type(budget) is not ReferenceBudget:raise ValueError("exact distinct reference bundle/budget")
    budget.__post_init__();total=0
    for label,raw,limit in (("mmcif",files.mmcif,budget.max_format_bytes),("fasta",files.fasta,budget.max_format_bytes),("sidecar",files.sidecar,budget.max_sidecar_bytes)):
        if raw is None and label=="fasta":continue
        if type(raw) is not bytes or not raw or len(raw)>limit:raise ValueError("declared "+label+" extent")
        total+=len(raw)
    if total>budget.max_total_bytes:raise ValueError("declared aggregate bytes")
    return total

def export_reference(packet,budget):
    """Byte-only reference formatting. No XYZ/TS, path, source or authority."""
    _preflight(packet,budget)
    cif=wire._write_mmcif(packet);fasta,status=wire._write_fasta(packet)
    files={k:dict(bytes=None if z is None else len(z),sha256=None if z is None else _sha(z)) for k,z in (("mmcif",cif),("fasta",fasta))}
    identity=identity_digest(packet)
    sidecar=_json(dict(schema=PROFILE,revision=REVISION,budget_sha256=budget.digest(),metadata=_metadata(packet),
        identity_sha256=identity,files=files,fasta_status=status.value,source_authority=None,model_authority=None,export_constraints_accepted=False))
    expectation=ReferenceExpectation(identity,_sha(sidecar),budget.digest())
    result=ReferenceBundle(cif,fasta,sidecar,expectation,status);_extents(result,budget);return result

def _reference_lines(raw,budget):
    if type(raw) is not bytes or not raw or len(raw)>budget.max_format_bytes:raise ValueError("bounded exact reference format")
    try:text=raw.decode("ascii")
    except UnicodeError:raise ValueError("existing profile is ASCII only") from None
    if "\r" in text or not text.endswith("\n"):raise ValueError("strict LF complete terminator")
    lines=text[:-1].split("\n")
    if any(len(x)>wire.MAX_LINE or any(ord(c)<32 and c!="\t" for c in x) for x in lines):raise ValueError("record/control extent")
    return lines

def read_reference_mmcif(raw,*,budget):
    if type(budget) is not ReferenceBudget:raise ValueError('explicit reference read budget')
    budget.__post_init__()
    lines=_reference_lines(raw,budget);header_count=4+len(wire.CIF_TAGS)
    if len(lines)<header_count+2 or lines[0]!="data_Mora3_synthetic" or lines[2:4]!=["#","loop_"] or lines[4:header_count]!=list(wire.CIF_TAGS) or lines[-1]!="#":
        raise ValueError("exact single-block/tag/loop profile required")
    if not lines[1].startswith("_entry.id "):raise ValueError("entry scalar")
    values=wire._lex_row(lines[1][len("_entry.id "):])
    if len(values)!=1:raise ValueError("entry value")
    identifier=wire._string_token(values[0],False)
    wire._text(identifier,"entry identifier",128)
    rows=[]
    for i,line in enumerate(lines[header_count:-1]):
        ts=wire._lex_row(line)
        if len(ts)!=21 or wire._int_token(ts[1],minimum=1)!=i+1:raise ValueError("row width/native record order")
        if any(t!=("?",False) for t in ts[3:9]):raise ValueError("standardized identity namespaces unsupported; not invented")
        symbol=wire._string_token(ts[2],False)
        if wire._SYMBOL.fullmatch(symbol) is None:raise ValueError("type symbol")
        group=wire._string_token(ts[0])
        if group not in (None,"ATOM","HETATM"):raise ValueError("group_PDB")
        insertion="" if ts[9]==(".",False) else wire._string_token(ts[9])
        xyz=tuple(wire._float_token(t) for t in ts[10:13])
        occ=wire._float_token(ts[13],True);bf=wire._float_token(ts[14],True)
        if occ is not None and not 0.<=occ<=1.:raise ValueError("occupancy")
        if bf is not None and bf<0.:raise ValueError("Bfactor")
        row=wire.CifRow(symbol,xyz,group,insertion,occ,bf,wire._int_token(ts[15],True),
            wire._string_token(ts[16]),wire._string_token(ts[17]),wire._string_token(ts[18]),wire._string_token(ts[19]),
            wire._int_token(ts[20],True,1))
        rows.append(row)
    if not 1<=len(rows)<=budget.max_atoms:raise ValueError("declared reference atom budget")
    return wire.CifPayload(identifier,tuple(rows))

def _sidecar(raw,budget):
    if type(raw) is not bytes or not raw or len(raw)>budget.max_sidecar_bytes:raise ValueError("bounded reference sidecar")
    try:return json.loads(raw.decode("utf8"),object_pairs_hook=wire._duplicate_pairs,parse_constant=wire._bad_constant)
    except (UnicodeError,ValueError,RecursionError):raise ValueError("strict reference JSON") from None

def reload_reference(files,expected,*,budget):
    """Reconstruct only bytes bound by EXTERNAL original expectation."""
    if type(expected) is not ReferenceExpectation:raise ValueError("distinct external reference expectation")
    expected.__post_init__();_extents(files,budget)
    if expected.budget_sha256!=budget.digest():raise ValueError("changed original declared budget")
    if _sha(files.sidecar)!=expected.sidecar_sha256:raise ValueError("external original sidecar mismatch")
    root=_sidecar(files.sidecar,budget)
    wire._keys(root,("schema","revision","budget_sha256","metadata","identity_sha256","files","fasta_status","source_authority","model_authority","export_constraints_accepted"))
    if root["schema"]!=PROFILE or root["revision"]!=REVISION or root["budget_sha256"]!=budget.digest() or root["source_authority"] is not None or root["model_authority"] is not None or root["export_constraints_accepted"] is not False:
        raise ValueError("reference schema/budget/fabricated authority")
    wire._keys(root["files"],("mmcif","fasta"))
    for key,raw in (("mmcif",files.mmcif),("fasta",files.fasta)):
        d=root["files"][key];wire._keys(d,("bytes","sha256"))
        if raw is None:
            if key!="fasta" or d!={"bytes":None,"sha256":None}:raise ValueError("missing format")
        elif type(d["bytes"]) is not int or d["bytes"]!=len(raw) or d["sha256"]!=_sha(raw):raise ValueError("reference format integrity")
    cif=read_reference_mmcif(files.mmcif,budget=budget)
    envelope=root["metadata"]
    wire._keys(envelope,("literal","structure_scope","purpose","role","missing_atoms","missing_residues"))
    md=envelope["literal"]
    wire._keys(md,("identifier","frame","units","origin","atoms","chains","total_charge","spin_multiplicity","saddle_order","bonds","context_json"))
    if md["units"]!="angstrom" or md["origin"]!="synthetic_handcrafted" or md["identifier"]!=cif.identifier or md["saddle_order"] is not None:
        raise ValueError("reference identity/frame/profile")
    if type(md["atoms"]) is not list or len(md["atoms"])!=len(cif.rows):raise ValueError("complete ordered reference atom metadata")
    atoms=[]
    for a,row in zip(md["atoms"],cif.rows):
        wire._keys(a,("native","xyz_bits","component","group_pdb","occupancy_bits","b_factor_bits","model_number"))
        n=wire._native_data(a["native"])
        if type(a["xyz_bits"]) is not list or len(a["xyz_bits"])!=3 or [wire._bits(x) for x in row.xyz]!=a["xyz_bits"]:
            raise ValueError("serialized float64 bits not exact source")
        for bit in a["xyz_bits"]:wire._unbits(bit)
        if (row.symbol,row.group_pdb,row.formal_charge,row.model_number)!=(n.element,a["group_pdb"],n.formal_charge,a["model_number"]):
            raise ValueError("native chemical/annotation projection mismatch")
        r=n.residue
        expected_auth=(None,None,None,None) if r is None else (str(r.index),r.identity,r.chain,r.insertion)
        if (row.auth_seq_id,row.auth_comp_id,row.auth_asym_id,row.insertion)!=expected_auth or row.auth_atom_id!=n.name:
            raise ValueError("native author labels/identities mismatch")
        if (None if row.occupancy is None else wire._bits(row.occupancy))!=a["occupancy_bits"] or (None if row.b_factor is None else wire._bits(row.b_factor))!=a["b_factor_bits"]:
            raise ValueError("occupancy/Bfactor bit mismatch")
        atoms.append(wire.ExportAtom(n,row.xyz,wire.Component(a["component"]),a["group_pdb"],row.occupancy,row.b_factor,a["model_number"]))
    if type(md["chains"]) is not list or len(md["chains"])>budget.max_chains:raise ValueError("chain axis budget")
    chains=[]
    for c in md["chains"]:
        wire._keys(c,("native_chain","identifier","residues","scope"))
        if type(c["residues"]) is not list or len(c["residues"])>4096:raise ValueError("unchanged per-chain profile")
        chains.append(wire.SequenceChain(c["native_chain"],c["identifier"],tuple(wire._residue_data(x) for x in c["residues"]),wire.SequenceScope(c["scope"])))
    bonds=None
    if md["bonds"] is not None:
        if type(md["bonds"]) is not list or len(md["bonds"])>512:raise ValueError("unchanged explicit bond profile")
        items=[]
        for d in md["bonds"]:
            wire._keys(d,("first","second","order_bits","provenance"))
            items.append(wire.BondFact(wire._native_data(d["first"]),wire._native_data(d["second"]),None if d["order_bits"] is None else wire._unbits(d["order_bits"]),d["provenance"]))
        bonds=tuple(items)
    if type(envelope["missing_atoms"]) is not list or len(envelope["missing_atoms"])>budget.max_atoms:raise ValueError("missing atom axis")
    if type(envelope["missing_residues"]) is not list or len(envelope["missing_residues"])>budget.max_sequence_residues:raise ValueError("missing residue axis")
    packet=ProteinReferencePacket(md["identifier"],md["frame"],tuple(atoms),tuple(chains),
        StructureScope(envelope["structure_scope"]),ReferencePurpose(envelope["purpose"]),native_types.Role(envelope["role"]),
        tuple(wire._native_data(x) for x in envelope["missing_atoms"]),tuple(wire._residue_data(x) for x in envelope["missing_residues"]),
        md["total_charge"],md["spin_multiplicity"],bonds,md["context_json"],md["origin"])
    _preflight(packet,budget)
    identity=identity_digest(packet)
    if identity!=expected.identity_sha256 or root["identity_sha256"]!=identity:raise ValueError("external original reference identity mismatch")
    status=wire.FastaStatus(root["fasta_status"])
    representable=bool(chains) and all(c.scope is wire.SequenceScope.FULL and re.fullmatch(r"[A-Za-z0-9_.:|+\-]{1,64}",c.identifier) is not None and all(r.identity in wire._CANONICAL for r in c.residues) for c in chains)
    if status is wire.FastaStatus.SUPPORTED:
        if not representable or files.fasta is None:raise ValueError("unrepresentable sequence")
        if wire._read_fasta(files.fasta)!=tuple((c.identifier,"".join(wire._CANONICAL[r.identity] for r in c.residues)) for c in chains):raise ValueError("FASTA identity mismatch")
    elif representable or files.fasta is not None:raise ValueError("FASTA unsupported disposition")
    if type(files.fasta_status) is not wire.FastaStatus or files.fasta_status is not status:raise ValueError("exact FASTA disposition")
    return ReloadedReference(packet)
