"""Frozen-geometry MLIP interaction feedback with explicit participant systems."""
from dataclasses import asdict, dataclass
import json
import math

from .teacher_protocol import (
    FrozenTeacherClient, REPOSITORIES, SystemSpec, TeacherRefusal, canonical, coordinates_sha256,
    finite_float, hex_sha, sha, text,
)


def _project_links(parent, fragments, atom_owner, field):
    expected = [[] for _ in fragments]
    links = parent[field]
    if type(links) is not list:
        raise TeacherRefusal(f'Explicit complete complex {field} required')
    for link in links:
        if type(link) is not dict or not {'first', 'second'} <= link.keys():
            raise TeacherRefusal(f'Exact complex {field} endpoint schema required')
        left, right = link['first'], link['second']
        if type(left) is not str or type(right) is not str or left not in atom_owner or right not in atom_owner:
            raise TeacherRefusal('Topology endpoint outside complete complex')
        owner = atom_owner[left]
        if owner != atom_owner[right]:
            raise TeacherRefusal('Cross-participant covalent or coordination link requires another reference protocol')
        expected[owner].append(canonical(link))
    for topology, links in zip(fragments, expected, strict=True):
        actual = topology[field]
        if type(actual) is not list or sorted(map(canonical, actual)) != sorted(links):
            raise TeacherRefusal(f'Exact projected participant {field} required')


@dataclass(frozen=True, slots=True)
class ComplexPartition:
    """One complete nonperiodic complex and an exact, explicitly charged atom partition."""

    complex_system: SystemSpec
    participants: tuple[SystemSpec, ...]

    def __post_init__(self):
        whole = self.complex_system
        if (type(whole) is not SystemSpec or type(self.participants) is not tuple
                or len(self.participants) < 2 or any(type(p) is not SystemSpec for p in self.participants)):
            raise TeacherRefusal('Complete typed complex and at least two typed participant systems required')
        systems = (whole, *self.participants)
        if (len({p.system_id for p in systems}) != len(systems)
                or any(p.cell_angstrom is not None or any(p.pbc) for p in systems)):
            raise TeacherRefusal('Distinct complete nonperiodic complex and participant identities required')
        ids = tuple(atom for p in self.participants for atom in p.atom_ids)
        if len(ids) != len(set(ids)) or set(ids) != set(whole.atom_ids):
            raise TeacherRefusal('Participant atoms must partition the complete complex exactly once')
        if sum(p.total_charge for p in self.participants) != whole.total_charge:
            raise TeacherRefusal('Complex and participant electronic charges must balance')
        spins = tuple(p.spin_multiplicity - 1 for p in self.participants)
        upper = sum(spins)
        lower = max(upper % 2, 2 * max(spins) - upper)
        spin = whole.spin_multiplicity - 1
        if not lower <= spin <= upper or (upper - spin) % 2:
            raise TeacherRefusal('Complex and participant spin declarations have no compatible spin coupling')
        parent = json.loads(whole.topology_json)
        fragments = tuple(json.loads(p.topology_json) for p in self.participants)
        parent_atoms = {a['key']: a for a in parent['atoms']}
        parent_charges = dict(zip(whole.atom_ids, whole.formal_charges, strict=True))
        for part, topology in zip(self.participants, fragments, strict=True):
            for atom, charge in zip(topology['atoms'], part.formal_charges, strict=True):
                source_atom = parent_atoms[atom['key']]
                if (any(atom[k] != source_atom[k] for k in atom if k != 'ordinal')
                        or charge != parent_charges[atom['key']]):
                    raise TeacherRefusal('Participant atom chemistry and charge availability must match the complex')
        atom_owner = {atom: i for i, part in enumerate(self.participants) for atom in part.atom_ids}
        _project_links(parent, fragments, atom_owner, 'bonds')
        _project_links(parent, fragments, atom_owner, 'chain_links')

    @property
    def sha256(self):
        return sha(canonical((self.complex_system.sha256, tuple(p.sha256 for p in self.participants))))


@dataclass(frozen=True, slots=True)
class MLIPInteractionFeedback:
    """One teacher-specific electronic interaction assessment in eV."""

    partition_sha256: str
    complex_xyz_sha256: str
    teacher_sha256: str
    teacher_family: str
    energy_reference_scheme: str
    complex_energy_ev: float
    participant_energies_ev: tuple[float, ...]
    evaluation_sha256: tuple[str, ...]

    def __post_init__(self):
        if (not all(hex_sha(x) for x in (self.partition_sha256,self.complex_xyz_sha256,self.teacher_sha256))
                or self.teacher_family not in REPOSITORIES
                or self.energy_reference_scheme not in ('raw-local-no-reference-restoration','explicit-pinned-reference-assets')
                or not finite_float(self.complex_energy_ev)
                or type(self.participant_energies_ev) is not tuple or len(self.participant_energies_ev)<2
                or any(not finite_float(x) for x in self.participant_energies_ev)
                or type(self.evaluation_sha256) is not tuple
                or len(self.evaluation_sha256) != 1 + len(self.participant_energies_ev)
                or not all(hex_sha(x) for x in self.evaluation_sha256)
                or len(set(self.evaluation_sha256)) != len(self.evaluation_sha256)):
            raise TeacherRefusal('Complete finite same-teacher interaction evidence required')
        _ = self.interaction_energy_ev

    @property
    def interaction_energy_ev(self):
        try:
            value = math.fsum((self.complex_energy_ev, *(-e for e in self.participant_energies_ev)))
        except OverflowError as exc:
            raise TeacherRefusal('Nonfinite interaction energy') from exc
        if not math.isfinite(value):
            raise TeacherRefusal('Nonfinite interaction energy')
        return value

    @property
    def sha256(self):
        return sha(canonical(asdict(self)))


def evaluate_interaction_feedback(partition, xyz_angstrom, teacher, *, request_id):
    """Assess the complex and frozen participants in the same geometry and energy convention."""
    if type(partition) is not ComplexPartition or type(teacher) is not FrozenTeacherClient or not text(request_id):
        raise TeacherRefusal('Exact complex partition, frozen teacher and request identity required')
    partition.__post_init__()
    xyz_sha = coordinates_sha256(xyz_angstrom)
    whole = partition.complex_system
    if len(xyz_angstrom) != len(whole.atom_ids):
        raise TeacherRefusal('Complete complex coordinates required')
    positions = dict(zip(whole.atom_ids, xyz_angstrom, strict=True))
    systems = (whole,*partition.participants)
    coords = (xyz_angstrom,*(tuple(positions[a] for a in part.atom_ids) for part in partition.participants))
    specification = teacher.spec
    assessments = tuple(
        teacher.evaluate(system,frame,frame_id=f'{request_id}:{i}',request_id=f'{request_id}:{i}')
        for i,(system,frame) in enumerate(zip(systems,coords,strict=True))
    )
    if teacher.spec.sha256 != specification.sha256:
        raise TeacherRefusal('Teacher identity changed across complex and participant evaluations')
    return MLIPInteractionFeedback(
        partition.sha256,xyz_sha,specification.sha256,specification.family,
        specification.energy_reference_scheme,assessments[0].energy_ev,
        tuple(a.energy_ev for a in assessments[1:]),tuple(sha(canonical(asdict(a))) for a in assessments),
    )
