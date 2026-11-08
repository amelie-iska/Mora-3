"""Fixed-system configurational Boltzmann target, explicit normalized box base.

No physical MD clock, variable-composition chemical potential, teacher gradient,
reward clipping or equilibrium-quality certificate. Student data-free means no
reference ensemble targets; a frozen learned teacher still has training data.
"""
from dataclasses import asdict, dataclass
import math
import json
from .teacher_protocol import (SystemSpec, TeacherSpec, TeacherRefusal,
    KB_EV_PER_K, canonical, sha, hex_sha, finite_float, text)


@dataclass(frozen=True, slots=True)
class BoxSupport:
    system_sha256: str
    free_atom_indices: tuple[int, ...]
    fixed_xyz_angstrom: tuple
    lower_angstrom: tuple[float, ...]
    upper_angstrom: tuple[float, ...]
    reference_frame: str
    properness_evidence_sha256: str

    def __post_init__(self):
        if (not hex_sha(self.system_sha256) or not text(self.reference_frame)
                or not hex_sha(self.properness_evidence_sha256)
                or type(self.free_atom_indices)is not tuple or not self.free_atom_indices
                or any(type(i)is not int or i<0 for i in self.free_atom_indices)
                or tuple(sorted(set(self.free_atom_indices)))!=self.free_atom_indices
                or type(self.fixed_xyz_angstrom)is not tuple or not self.fixed_xyz_angstrom
                or self.free_atom_indices[-1]>=len(self.fixed_xyz_angstrom)):
            raise TeacherRefusal("explicit fixed-system free/frozen axes and properness evidence required")
        free=set(self.free_atom_indices)
        for i,row in enumerate(self.fixed_xyz_angstrom):
            if i in free:
                if row is not None:raise TeacherRefusal("free atoms cannot acquire hidden reference targets")
            elif (type(row)is not tuple or len(row)!=3 or any(not finite_float(v) for v in row)):
                raise TeacherRefusal("every frozen atom requires exact caller-supplied XYZ")
        d=3*len(free)
        if (type(self.lower_angstrom)is not tuple or type(self.upper_angstrom)is not tuple
                or len(self.lower_angstrom)!=d or len(self.upper_angstrom)!=d
                or any(not finite_float(v) for v in self.lower_angstrom+self.upper_angstrom)
                or any(not lo<hi or not math.isfinite(hi-lo)
                    for lo,hi in zip(self.lower_angstrom,self.upper_angstrom))):
            raise TeacherRefusal("finite positive-width full declared Cartesian box required")

    @property
    def dimension(self):return len(self.lower_angstrom)

    @property
    def log_volume(self):return sum(math.log(hi-lo) for lo,hi in zip(self.lower_angstrom,self.upper_angstrom))

    @property
    def sha256(self):return sha(canonical(asdict(self)))

    def bind_system(self,system):
        if type(system)is not SystemSpec or system.sha256!=self.system_sha256 or len(system.atom_ids)!=len(self.fixed_xyz_angstrom):
            raise TeacherRefusal("support composition/atom/topology/charge/spin changed")
        if any(system.pbc):
            cell=system.cell_angstrom
            if any(cell[i][j]!=0.0 for i in range(3) for j in range(3) if i!=j):
                raise TeacherRefusal("nonorthogonal periodic support requires separate fractional-measure adapter")
            for k,_ in enumerate(self.free_atom_indices):
                for axis in range(3):
                    lo,hi=self.lower_angstrom[3*k+axis],self.upper_angstrom[3*k+axis]
                    if lo!=0.0 or hi!=cell[axis][axis]:
                        raise TeacherRefusal("periodic box must be explicit complete orthorhombic fundamental cell")
            for row in self.fixed_xyz_angstrom:
                if row is not None and any(not 0.0<=row[j]<cell[j][j] for j in range(3)):
                    raise TeacherRefusal("frozen periodic atom outside declared fundamental cell")
        return self

    def complete_positions(self,free_xyz):
        if type(free_xyz)is not tuple or len(free_xyz)!=self.dimension or any(not finite_float(v) for v in free_xyz):
            raise TeacherRefusal("complete finite physical free-coordinate vector required")
        if any(not lo<v<hi for v,lo,hi in zip(free_xyz,self.lower_angstrom,self.upper_angstrom)):
            raise TeacherRefusal("physical sample outside open declared support; no clipping")
        values=list(self.fixed_xyz_angstrom)
        for k,i in enumerate(self.free_atom_indices):values[i]=tuple(free_xyz[3*k:3*k+3])
        return tuple(values)


@dataclass(frozen=True, slots=True)
class BoltzmannTarget:
    system: SystemSpec
    teacher: TeacherSpec
    support: BoxSupport
    temperature_kelvin: float
    energy_reference_ev: float
    teacher_energy_lower_bound_ev: float
    properness_evidence_sha256: str
    target_revision: str
    optional_stop: bool

    def __post_init__(self):
        if (type(self.system)is not SystemSpec or type(self.teacher)is not TeacherSpec
                or type(self.support)is not BoxSupport
                or not finite_float(self.temperature_kelvin) or self.temperature_kelvin<=0
                or not finite_float(self.energy_reference_ev)
                or not finite_float(self.teacher_energy_lower_bound_ev)
                or not hex_sha(self.properness_evidence_sha256)
                or not text(self.target_revision) or type(self.optional_stop)is not bool):
            raise TeacherRefusal("explicit frozen teacher/fixed-system/T/Eref/proper target required")
        self.support.bind_system(self.system)
        topology=json.loads(self.system.topology_json)
        for chain in topology.get("sequence_chains") or ():
            if chain.get("redesign_residue_ids"):
                raise TeacherRefusal("variable sequence/chemistry requires composition prior and chemical potentials")
            if chain.get("region_policies"):
                raise TeacherRefusal("uncomposed sequence/region/RMSD constraints unsupported; never silently relax")
        # A digest alone does not prove lowerboundedness: original owner/ROOT
        # must review its actual domain/teacher evidence before runtime. This
        # explicit contract permits a mathematically proper normalized box base.

    @property
    def sha256(self):
        return sha(canonical(dict(schema="Mora3.FixedSystemBoltzmannTarget/v1",
            system=asdict(self.system),teacher=asdict(self.teacher),support=asdict(self.support),
            temperature_kelvin=self.temperature_kelvin,energy_reference_ev=self.energy_reference_ev,
            teacher_energy_lower_bound_ev=self.teacher_energy_lower_bound_ev,
            properness_evidence_sha256=self.properness_evidence_sha256,
            target_revision=self.target_revision,optional_stop=self.optional_stop,
            base_measure="normalized-uniform-Cartesian-box",
            variable_composition=False)))

    def require_same_system(self,system):
        if type(system)is not SystemSpec or system.sha256!=self.system.sha256:
            raise TeacherRefusal("variable composition/sequence/charge/spin/topology/support unsupported; explicit prior and chemical potentials required")

    def energy_log_reward(self,energy_ev):
        if not finite_float(energy_ev):
            raise TeacherRefusal("finite returned energy required; invalid is not clipped")
        if energy_ev<self.teacher_energy_lower_bound_ev:
            raise TeacherRefusal("teacher energy violates pinned lower-bound/domain contract")
        return boltzmann_log_reward(energy_ev,
            temperature_kelvin=self.temperature_kelvin,energy_reference_ev=self.energy_reference_ev)

    def target_manifest(self):
        return dict(schema="Mora3.FixedSystemBoltzmannTargetManifest/v1",
            system=asdict(self.system),teacher=asdict(self.teacher),support=asdict(self.support),
            temperature_kelvin=self.temperature_kelvin,energy_reference_ev=self.energy_reference_ev,
            teacher_energy_lower_bound_ev=self.teacher_energy_lower_bound_ev,
            properness_evidence_sha256=self.properness_evidence_sha256,
            target_revision=self.target_revision,optional_stop=self.optional_stop,
            base_measure="normalized-uniform-Cartesian-box",
            target_variable_composition_supported=False)

    def checkpoint_binding(self):
        return dict(schema="Mora3.BoltzmannCheckpointTarget/v1",target_sha256=self.sha256,
            system_sha256=self.system.sha256,teacher_sha256=self.teacher.sha256,
            support_sha256=self.support.sha256,target_revision=self.target_revision,
            temperature_kelvin=self.temperature_kelvin,energy_reference_ev=self.energy_reference_ev,
            logZ_measure="normalized-box-base-and-normalized-terminal-layer",
            evidence_role="frozen-MLIP-predicted-not-experimental-or-abinitio-groundtruth")

    def validate_checkpoint_binding(self,binding):
        if binding!=self.checkpoint_binding():
            raise TeacherRefusal("target/teacher/T/reference/support changed; new checkpoint and logZ required")


def boltzmann_log_reward(energy_ev,*,temperature_kelvin,energy_reference_ev):
    """Exact energy term logR=-(U-Eref)/(kB*T); no energy/force conversion."""
    if (not finite_float(energy_ev) or not finite_float(temperature_kelvin)
            or temperature_kelvin<=0 or not finite_float(energy_reference_ev)):
        raise TeacherRefusal("explicit finite eV energy/reference and positive kelvin T required")
    value=-(energy_ev-energy_reference_ev)/(KB_EV_PER_K*temperature_kelvin)
    if not math.isfinite(value):
        raise TeacherRefusal("unrepresentable Boltzmann log reward; no silent clipping")
    return value
