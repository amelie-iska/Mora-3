"""Frozen UMA/eSEN RPC contracts; standard-library import boundary.

Unapplied/source-only. No model loading, checkpoint deserialization, network,
process launcher or credential lookup. The original model owner implements the
isolated safe service. Metadata equality is necessary, not a launch approval.
"""
from dataclasses import asdict, dataclass, fields
from pathlib import Path
import hashlib
import io
import json
import math
import socket
import struct
import time
import zipfile

FAIRCHEM_REVISION = "ac3561498abfb5d869a9379b1afb44b065047e58"
REPOSITORIES = {
    "UMA": ("facebook/UMA", "1f43bb7ff24084b72e8253f9e16fbac19320adae"),
    "ESEN": ("facebook/OMol25", "9a8827c74c0da5292f74344ed3d50d726408cce1"),
}
KB_EV_PER_K = 8.617333262145e-5


class TeacherRefusal(ValueError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode("utf8")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def hex_sha(value):
    return type(value) is str and len(value) == 64 and not set(value)-set("0123456789abcdef")


def text(value):
    return type(value) is str and bool(value.strip())


def finite_float(value):
    return type(value) is float and math.isfinite(value)


def _keys(data, cls):
    if type(data) is not dict or set(data) != {f.name for f in fields(cls) if f.init}:
        raise TeacherRefusal("exact typed RPC fields required: " + cls.__name__)


def _tuple3(values):
    if type(values)not in (list,tuple) or any(type(row)not in (list,tuple) or len(row)!=3 for row in values):
        raise TeacherRefusal("explicit full ordered triples required")
    return tuple(tuple(row) for row in values)


def _determinant(cell):
    a,b,c=cell
    return (a[0]*(b[1]*c[2]-b[2]*c[1])
        -a[1]*(b[0]*c[2]-b[2]*c[0])+a[2]*(b[0]*c[1]-b[1]*c[0]))


def coordinates_sha256(xyz):
    if type(xyz) is not tuple or not xyz:
        raise TeacherRefusal("complete ordered coordinate axis required")
    if any(type(row) is not tuple or len(row)!=3
           or any(not finite_float(x) for x in row) for row in xyz):
        raise TeacherRefusal("finite float64 full N by3 angstrom coordinates required")
    return sha(b"".join(struct.pack("<d", x) for row in xyz for x in row))


@dataclass(frozen=True, slots=True)
class SystemSpec:
    system_id: str
    topology_id: str
    atom_ids: tuple[str, ...]
    atomic_numbers: tuple[int, ...]
    formal_charges: tuple[int | None, ...]
    total_charge: int
    spin_multiplicity: int
    pbc: tuple[bool, bool, bool]
    cell_angstrom: tuple | None
    topology_json: str
    topology_sha256: str

    def __post_init__(self):
        if type(self.atom_ids)is not tuple:
            raise TeacherRefusal("complete ordered atom IDs required")
        n=len(self.atom_ids)
        if (not text(self.system_id) or not text(self.topology_id)
                or type(self.atom_ids) is not tuple or not n
                or any(not text(k) for k in self.atom_ids) or len(set(self.atom_ids))!=n
                or type(self.atomic_numbers) is not tuple or len(self.atomic_numbers)!=n
                or any(type(z)is not int or not 1<=z<=118 for z in self.atomic_numbers)
                or type(self.formal_charges)is not tuple or len(self.formal_charges)!=n
                or any(q is not None and type(q)is not int for q in self.formal_charges)
                or type(self.total_charge)is not int or not -100<=self.total_charge<=100
                or type(self.spin_multiplicity)is not int or not 1<=self.spin_multiplicity<=100
                or type(self.pbc)is not tuple or len(self.pbc)!=3
                or any(type(v)is not bool for v in self.pbc)):
            raise TeacherRefusal("explicit complete system charge/spin/PBC/ordered atoms required")
        electrons=sum(self.atomic_numbers)-self.total_charge
        if electrons<0 or self.spin_multiplicity-1>electrons or (electrons-(self.spin_multiplicity-1))%2:
            raise TeacherRefusal("charge/spin incompatible with explicit electron count")
        if all(q is not None for q in self.formal_charges) and sum(self.formal_charges)!=self.total_charge:
            raise TeacherRefusal("known formal charges do not sum to supplied total charge")
        if self.cell_angstrom is None:
            if any(self.pbc):raise TeacherRefusal("periodic system requires explicit row-vector cell")
        elif (type(self.cell_angstrom)is not tuple or len(self.cell_angstrom)!=3
                or any(type(row)is not tuple or len(row)!=3
                    or any(not finite_float(x) for x in row) for row in self.cell_angstrom)
                or not math.isfinite(_determinant(self.cell_angstrom))
                or _determinant(self.cell_angstrom)<=0):
            raise TeacherRefusal("proper finite positive-volume row-vector cell required")
        if not text(self.topology_json) or not hex_sha(self.topology_sha256):
            raise TeacherRefusal("exact canonical construction topology bytes required")
        payload=json.loads(self.topology_json)
        if type(payload)is not dict:
            raise TeacherRefusal("construction topology JSON object required")
        if canonical(payload).decode("utf8")!=self.topology_json or sha(self.topology_json.encode("utf8"))!=self.topology_sha256:
            raise TeacherRefusal("topology canonical bytes or SHA mismatch")
        expected_system_fields={"system_id","topology_id","atoms","bonds","formal_charges",
            "total_charge","spin_multiplicity","protonation","cofactor_ids","chain_links",
            "cell_angstrom","pbc","chemistry_context_json","source_refs","sequence_chains","cofactor_atom_map"}
        expected_atom_fields={"key","entity","ordinal","element","atomic_number","name",
            "component","residue","isotope"}
        atoms=payload.get("atoms")
        if (type(payload)is not dict or set(payload)!=expected_system_fields
                or type(atoms)is not list
                or any(type(a)is not dict or set(a)!=expected_atom_fields for a in atoms)):
            raise TeacherRefusal("complete pinned construction SystemSpec/AtomSpec fields required")
        if (payload.get("system_id")!=self.system_id or payload.get("topology_id")!=self.topology_id
                or type(atoms)is not list or len(atoms)!=n
                or tuple(a["key"] for a in atoms)!=self.atom_ids
                or tuple(a["atomic_number"] for a in atoms)!=self.atomic_numbers):
            raise TeacherRefusal("topology and teacher full atom maps differ")
        for key in ("formal_charges","total_charge","spin_multiplicity","pbc","cell_angstrom"):
            expected=getattr(self,key)
            if canonical(payload.get(key))!=canonical(expected):
                raise TeacherRefusal("topology and teacher electronic/cell declarations differ")

    @property
    def sha256(self):return sha(canonical(asdict(self)))

    @classmethod
    def from_mapping(cls,data):
        _keys(data,cls);d=dict(data)
        for k in ("atom_ids","atomic_numbers","formal_charges","pbc"):
            if type(d[k])not in (list,tuple):
                raise TeacherRefusal("explicit complete arrays required; unknown is not defaulted")
            d[k]=tuple(d[k])
        if d["cell_angstrom"] is not None:d["cell_angstrom"]=_tuple3(d["cell_angstrom"])
        return cls(**d)

    @classmethod
    def from_export_mapping(cls,payload):
        # Actual construction SystemSpec canonical fields are preserved, never
        # replaced with a cropped atom list or invented bond/sequence metadata.
        if (type(payload)is not dict or type(payload.get("atoms"))is not list
                or type(payload.get("formal_charges"))not in (list,tuple)):
            raise TeacherRefusal("complete atom/per-atom charge availability declaration required")
        raw=canonical(payload)
        return cls(system_id=payload["system_id"],topology_id=payload["topology_id"],
            atom_ids=tuple(a["key"] for a in payload["atoms"]),
            atomic_numbers=tuple(a["atomic_number"] for a in payload["atoms"]),
            formal_charges=tuple(payload["formal_charges"]),
            total_charge=payload["total_charge"],spin_multiplicity=payload["spin_multiplicity"],
            pbc=tuple(payload["pbc"]) if payload["pbc"] is not None else (),
            cell_angstrom=_tuple3(payload["cell_angstrom"]) if payload["cell_angstrom"] is not None else None,
            topology_json=raw.decode("utf8"),topology_sha256=sha(raw))


@dataclass(frozen=True, slots=True)
class TeacherSpec:
    teacher_id: str
    family: str
    repository_id: str
    repository_revision: str
    fairchem_revision: str
    checkpoint_path: str
    checkpoint_sha256: str
    task: str
    position_unit: str
    energy_unit: str
    force_unit: str
    energy_reference_scheme: str
    atom_reference_sha256: str | None
    formation_reference_sha256: str | None
    inference_settings_json: str
    execution_fingerprint_sha256: str
    seed: int

    def __post_init__(self):
        if (not text(self.teacher_id) or self.family not in REPOSITORIES
                or (self.repository_id,self.repository_revision)!=REPOSITORIES[self.family]
                or self.fairchem_revision!=FAIRCHEM_REVISION
                or not text(self.checkpoint_path) or not Path(self.checkpoint_path).is_absolute()
                or not hex_sha(self.checkpoint_sha256)
                or self.task!="omol"
                or (self.position_unit,self.energy_unit,self.force_unit)!=("angstrom","eV","eV/angstrom")
                or not hex_sha(self.execution_fingerprint_sha256) or not text(self.inference_settings_json)
                or type(self.seed)is not int or not 0<=self.seed<2**63):
            raise TeacherRefusal("exact pinned UMA/eSEN local checkpoint/task/units/runtime required")
        if self.energy_reference_scheme=="raw-local-no-reference-restoration":
            if self.atom_reference_sha256 is not None or self.formation_reference_sha256 is not None:
                raise TeacherRefusal("raw-local energy cannot silently restore reference assets")
        elif self.energy_reference_scheme=="explicit-pinned-reference-assets":
            if (self.atom_reference_sha256 is None and self.formation_reference_sha256 is None
                    or any(v is not None and not hex_sha(v) for v in
                           (self.atom_reference_sha256,self.formation_reference_sha256))):
                raise TeacherRefusal("explicit reference assets must be exactly pinned")
        else:raise TeacherRefusal("unsupported energy-reference convention")
        settings=json.loads(self.inference_settings_json)
        if type(settings)is not dict or canonical(settings).decode("utf8")!=self.inference_settings_json:
            raise TeacherRefusal("canonical explicit inference settings required")

    @property
    def sha256(self):return sha(canonical(asdict(self)))

    @classmethod
    def from_mapping(cls,data):
        _keys(data,cls);return cls(**data)


@dataclass(frozen=True, slots=True)
class Evaluation:
    frame_id: str
    xyz_sha256: str
    status: str
    energy_ev: float | None
    forces_ev_per_angstrom: tuple | None
    energy_unit: str
    force_unit: str
    position_unit: str
    execution_fingerprint_sha256: str | None
    teacher_trainable_parameter_count: int | None
    teacher_training_mode: bool | None
    force_relation: str
    conservative_evidence_sha256: str | None
    refusal_reason: str | None

    def __post_init__(self):
        if (not text(self.frame_id) or not hex_sha(self.xyz_sha256)
                or self.status not in ("ok","refused","unsupported","error")
                or (self.position_unit,self.energy_unit,self.force_unit)!=("angstrom","eV","eV/angstrom")):
            raise TeacherRefusal("exact finite teacher evaluation envelope required")
        if self.status!="ok":
            if self.energy_ev is not None or self.forces_ev_per_angstrom is not None or not text(self.refusal_reason):
                raise TeacherRefusal("refusal must not carry successful energy/force values")
            return
        if (not finite_float(self.energy_ev) or self.refusal_reason is not None
                or not hex_sha(self.execution_fingerprint_sha256)
                or type(self.teacher_trainable_parameter_count)is not int
                or self.teacher_trainable_parameter_count!=0
                or self.teacher_training_mode is not False):
            raise TeacherRefusal("finite successful energy with actual frozen eval-mode teacher required")
        if self.forces_ev_per_angstrom is None:
            if self.force_relation!="not_requested" or self.conservative_evidence_sha256 is not None:
                raise TeacherRefusal("force missingness must be explicit")
        else:
            coordinates_sha256(self.forces_ev_per_angstrom)
            if self.force_relation=="verified_conservative_same_energy":
                if not hex_sha(self.conservative_evidence_sha256):
                    raise TeacherRefusal("same-energy conservative force evidence required")
            elif self.force_relation!="returned_numeric_unverified" or self.conservative_evidence_sha256 is not None:
                raise TeacherRefusal("returned numeric forces cannot imply external gradients")

    @classmethod
    def from_mapping(cls,data):
        _keys(data,cls);d=dict(data)
        if d["forces_ev_per_angstrom"] is not None:d["forces_ev_per_angstrom"]=_tuple3(d["forces_ev_per_angstrom"])
        return cls(**d)


@dataclass(frozen=True, slots=True)
class SocketEndpoint:
    socket_path: str
    timeout_seconds: float
    max_request_bytes: int
    max_response_bytes: int
    expected_peer_pid: int
    expected_peer_uid: int
    expected_peer_start_ticks: int

    def __post_init__(self):
        if (not text(self.socket_path) or not Path(self.socket_path).is_absolute()
                or not finite_float(self.timeout_seconds) or not 0<self.timeout_seconds<=120
                or any(type(v)is not int or not 1<=v<=67108864
                    for v in (self.max_request_bytes,self.max_response_bytes))
                or type(self.expected_peer_pid)is not int or self.expected_peer_pid<=0
                or type(self.expected_peer_uid)is not int or self.expected_peer_uid<0
                or type(self.expected_peer_start_ticks)is not int or self.expected_peer_start_ticks<=0):
            raise TeacherRefusal("ROOT-bound finite local service endpoint required")

    def exchange(self,payload):
        if type(payload)is not bytes or len(payload)>self.max_request_bytes:
            raise TeacherRefusal("teacher request exceeds exact byte budget")
        if not hasattr(socket,"SO_PEERCRED"):
            raise TeacherRefusal("actual Linux local peer identity verification unsupported")
        deadline=time.monotonic()+self.timeout_seconds
        with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as stream:
            def remaining():
                value=deadline-time.monotonic()
                if value<=0:raise TeacherRefusal("finite teacher RPC deadline exhausted")
                stream.settimeout(value)
            remaining();stream.connect(self.socket_path)
            pid,uid,_=struct.unpack("3i",stream.getsockopt(socket.SOL_SOCKET,socket.SO_PEERCRED,12))
            stat=Path(f"/proc/{pid}/stat").read_text()
            start=int(stat.rsplit(")",1)[1].split()[19])
            if (pid,uid,start)!=(self.expected_peer_pid,self.expected_peer_uid,self.expected_peer_start_ticks):
                raise TeacherRefusal("teacher peer PID/UID/start identity changed")
            remaining();stream.sendall(struct.pack(">Q",len(payload))+payload)
            def read_exact(n):
                pieces=[];left=n
                while left:
                    remaining();part=stream.recv(left)
                    if not part:raise TeacherRefusal("teacher response truncated")
                    pieces.append(part);left-=len(part)
                return b"".join(pieces)
            size=struct.unpack(">Q",read_exact(8))[0]
            if not 0<size<=self.max_response_bytes:raise TeacherRefusal("teacher response byte budget exceeded")
            return read_exact(size)


class FrozenTeacherClient:
    def __init__(self,spec,endpoint):
        if type(spec)is not TeacherSpec or type(endpoint)is not SocketEndpoint:
            raise TeacherRefusal("real typed frozen teacher and isolated endpoint required")
        self.spec=spec;self.endpoint=endpoint
        self._initial_teacher_sha256=spec.sha256

    def evaluate(self,system,xyz_angstrom,*,frame_id,request_id,return_forces=False):
        return self.evaluate_batch(system,((frame_id,xyz_angstrom),),request_id=request_id,
            return_forces=return_forces,path_provenance="assessment_only")[0]

    def evaluate_batch(self,system,frames,*,request_id,return_forces,path_provenance):
        if type(system)is not SystemSpec or self.spec.sha256!=self._initial_teacher_sha256:
            raise TeacherRefusal("teacher revision changed; new target/checkpoint/logZ required")
        if (not text(request_id) or type(return_forces)is not bool
                or path_provenance not in ("generative_dag_not_physical_time","assessment_only")
                or type(frames)is not tuple or not frames
                or any(type(item)is not tuple or len(item)!=2 for item in frames)):
            raise TeacherRefusal("bounded exact self-generated/assessment frame scope required")
        queries=[]
        for frame_id,xyz in frames:
            if not text(frame_id) or type(xyz)is not tuple or len(xyz)!=len(system.atom_ids):
                raise TeacherRefusal("complete teacher frame and exact atom maps required")
            queries.append(dict(frame_id=frame_id,xyz_angstrom=xyz,xyz_sha256=coordinates_sha256(xyz)))
        if len({q["frame_id"] for q in queries})!=len(queries):
            raise TeacherRefusal("unique frame IDs required")
        request=dict(schema="Mora3.FrozenTeacherRequest/v1",request_id=request_id,
            teacher=asdict(self.spec),teacher_sha256=self.spec.sha256,
            system=asdict(system),system_sha256=system.sha256,return_forces=return_forces,
            frames=queries,path_provenance=path_provenance)
        raw=self.endpoint.exchange(canonical(request))
        reply=json.loads(raw)
        if canonical(reply)!=raw:
            raise TeacherRefusal("teacher response must preserve exact canonical JSON bytes")
        if (type(reply)is not dict
                or set(reply)!={"schema","request_id","teacher_sha256","system_sha256","evaluations"}
                or reply["schema"]!="Mora3.FrozenTeacherResponse/v1"
                or reply["request_id"]!=request_id or reply["teacher_sha256"]!=self.spec.sha256
                or reply["system_sha256"]!=system.sha256
                or type(reply["evaluations"])is not list or len(reply["evaluations"])!=len(queries)):
            raise TeacherRefusal("teacher response identity/order/target mismatch")
        result=[]
        for query,value in zip(queries,reply["evaluations"]):
            evaluation=Evaluation.from_mapping(value)
            if (evaluation.frame_id!=query["frame_id"] or evaluation.xyz_sha256!=query["xyz_sha256"]):
                raise TeacherRefusal("teacher returned different/reordered atom frame")
            if evaluation.status!="ok":
                raise TeacherRefusal(evaluation.status+":"+evaluation.refusal_reason)
            if evaluation.execution_fingerprint_sha256!=self.spec.execution_fingerprint_sha256:
                raise TeacherRefusal("teacher runtime changed; target/checkpoint/logZ revision required")
            if return_forces:
                if evaluation.forces_ev_per_angstrom is None or len(evaluation.forces_ev_per_angstrom)!=len(system.atom_ids):
                    raise TeacherRefusal("full requested numeric forces missing")
            elif evaluation.forces_ev_per_angstrom is not None:
                raise TeacherRefusal("unrequested force field is not silently consumed")
            result.append(evaluation)
        return tuple(result)


def load_numeric_npz(path,*,expected_sha256,array_manifest,max_uncompressed_bytes):
    """Optional audited numeric codec; RPCv1 initial transport remains JSON."""
    if (not hex_sha(expected_sha256) or type(array_manifest)is not dict or not array_manifest
            or type(max_uncompressed_bytes)is not int or max_uncompressed_bytes<=0):
        raise TeacherRefusal("exact NPZ checksum/numeric manifest/byte budget required")
    for name,row in array_manifest.items():
        if (not text(name) or "/" in name or type(row)is not dict
                or set(row)!={"shape","dtype"} or type(row["shape"])not in (list,tuple)
                or any(type(n)is not int or n<0 for n in row["shape"])
                or row["dtype"] not in ("<f8","<f4","<i8","<i4","<u8","<u4","|b1")):
            raise TeacherRefusal("numeric nonobject NPZ declaration required")
    path=Path(path)
    if path.stat().st_size>max_uncompressed_bytes+1048576:
        raise TeacherRefusal("NPZ compressed-file read budget exceeded")
    data=path.read_bytes()
    if sha(data)!=expected_sha256:raise TeacherRefusal("NPZ byte checksum mismatch")
    with zipfile.ZipFile(io.BytesIO(data)) as zipped:
        if (set(zipped.namelist())!={k+".npy" for k in array_manifest}
                or sum(v.file_size for v in zipped.infolist())>max_uncompressed_bytes):
            raise TeacherRefusal("NPZ names/uncompressed byte budget differ")
    import numpy as np
    # Validate numeric NPY headers before allocating arrays from untrusted shapes.
    with zipfile.ZipFile(io.BytesIO(data)) as zipped:
        for name,row in array_manifest.items():
            with zipped.open(name+".npy") as stream:
                version=np.lib.format.read_magic(stream)
                if version==(1,0):
                    shape,fortran,dtype=np.lib.format.read_array_header_1_0(stream,max_header_size=10000)
                elif version==(2,0):
                    shape,fortran,dtype=np.lib.format.read_array_header_2_0(stream,max_header_size=10000)
                else:raise TeacherRefusal("unsupported numeric NPY header version")
                if (dtype.hasobject or dtype.fields is not None or dtype.str!=row["dtype"]
                        or tuple(shape)!=tuple(row["shape"]) or fortran
                        or math.prod(shape)*dtype.itemsize>max_uncompressed_bytes):
                    raise TeacherRefusal("NPZ numeric C-order header/shape budget mismatch")
    with np.load(io.BytesIO(data),allow_pickle=False) as archive:
        result={}
        for name,row in array_manifest.items():
            value=archive[name]
            if (value.dtype.hasobject or value.dtype.fields is not None
                    or value.dtype.str!=row["dtype"] or tuple(value.shape)!=tuple(row["shape"])
                    or value.nbytes>max_uncompressed_bytes
                    or value.dtype.kind=="f" and not np.isfinite(value).all()):
                raise TeacherRefusal("NPZ physical dtype/shape/finite contract mismatch")
            result[name]=value.copy()
    return result
