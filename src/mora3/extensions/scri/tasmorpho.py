"""Read the audited tasmorpho CSV interfaces without importing its scripts.

The pinned revision exports aggregate BETSE cases, not cell states or graphs.
The count table is a reconstruction of published experimental counts; fitted
and predicted probabilities are different evidence and are not accepted here.
No pickle/dill payload, simulator, fitted model, or pharmacology mapping runs
when this module is imported or an artifact is read.
"""
from dataclasses import asdict, dataclass
from hashlib import sha256
import csv
import io
import math
import re

from .contracts import canonical, digest, number, text
from .curation import regular_file


SOURCE_REPOSITORY = 'https://github.com/amelie-iska/tasmorpho'
SUPPORTED_SOURCE_COMMIT = 'ab09864214ca36787a2144bb89425c8b9610bd6f'
SOURCE_BASE = f'{SOURCE_REPOSITORY}/blob/{SUPPORTED_SOURCE_COMMIT}/'
SUMMARY_SOURCE_PATH = 'scripts/betse_tas_example_run/betse_summary.csv'
SUMMARY_HEADER = ('case', 'E_mem', 'E_gj', 'E_pump', 'E_total',
                  'd_mean_vmem_mV', 'd_wound_gj')
CASES = ('control', 'na', 'k', 'gj', 'na_k', 'na_gj', 'k_gj')
COUNT_HEADER = ('label', 'successes', 'trials', 'proportion', 'role', 'description')
# These labels are deliberately opaque. In particular, the README's chemical
# expansion of 8OH conflicts with the primary planarian papers' 1-octanol.
COUNT_SPECS = {
    '8OH_immediate_DH': (143, 573, 'primary_fit_target', 'treated_cohort'),
    '8OH_challenge_DH_among_immediate_SH':
        (36, 155, 'primary_fit_target', 'immediate_SH'),
    'nigericin_immediate_DH': (17, 132, 'condition_setting', 'treated_cohort'),
    'monensin_immediate_DH': (11, 89, 'condition_setting', 'treated_cohort'),
    '8OH_challenge_DH_among_immediate_DH':
        (100, 100, 'consistency_check_only', 'immediate_DH'),
}
MAX_CSV_BYTES = 1024 * 1024


class UnsupportedTasmorphoSemantics(ValueError):
    """An aggregate artifact cannot supply the requested biological mapping."""


@dataclass(frozen=True)
class TasmorphoProvenance:
    source_commit: str
    artifact_sha256: str
    run_id: str
    source_locator: str
    betse_version: str | None = None
    random_seed: int | None = None
    geometry_sha256: str | None = None

    def __post_init__(self):
        if self.source_commit != SUPPORTED_SOURCE_COMMIT:
            raise ValueError('unaudited tasmorpho source revision; explicit schema review required')
        digest(self.artifact_sha256, 'artifact_sha256')
        text(self.run_id, 'run_id')
        text(self.source_locator, 'source_locator')
        if self.betse_version is not None:
            text(self.betse_version, 'betse_version')
        if self.random_seed is not None and (
                type(self.random_seed) is not int or self.random_seed < 0):
            raise ValueError('random_seed must be an explicit nonnegative integer or None')
        if self.geometry_sha256 is not None:
            digest(self.geometry_sha256, 'geometry_sha256')

    @property
    def partition_group(self):
        """All cases sharing a simulator run and control stay in one partition."""
        return 'tasmorpho-run:' + sha256(canonical({
            'repository': SOURCE_REPOSITORY, 'commit': self.source_commit,
            'run_id': self.run_id,
        })).hexdigest()

    def wire(self):
        return {**asdict(self), 'source_repository': SOURCE_REPOSITORY,
                'software_license_id': 'MIT', 'data_license_id': None,
                'provider_run_independently_verified': False}


@dataclass(frozen=True)
class SimulationCase:
    case: str
    E_mem: float
    E_gj: float
    E_pump: float
    E_total: float
    d_mean_vmem_mV: float
    d_wound_gj: float

    def __post_init__(self):
        if self.case not in CASES:
            raise ValueError('unknown native BETSE case')
        for name in SUMMARY_HEADER[1:]:
            number(getattr(self, name), name)
        energies = (self.E_mem, self.E_gj, self.E_pump, self.E_total)
        if any(value < 0 for value in energies):
            raise ValueError('integrated squared-flux components must be nonnegative')
        if not math.isclose(self.E_total, math.fsum(energies[:3]),
                            rel_tol=1e-9, abs_tol=1e-15):
            raise ValueError('E_total differs from the sum of native flux components')
        if self.case == 'control' and any(getattr(self, name) != 0
                                          for name in SUMMARY_HEADER[1:]):
            raise ValueError('native control-relative summary must have an exactly zero control')


@dataclass(frozen=True)
class GroundingMatrices:
    Q: tuple[tuple[float, float], tuple[float, float]]
    R: tuple[tuple[float, float], tuple[float, float]]
    K: tuple[tuple[float, float], tuple[float, float]]
    Q_condition_number: float

    def wire(self):
        return {
            **asdict(self),
            'perturbation_coordinates': ['native_na_case_contrast', 'native_gj_case_contrast'],
            'response_coordinates': ['mean_membrane_voltage_shift', 'wound_gj_proxy_contrast'],
            'response_units': ['V', 'native_gj_proxy_arbitrary_unit'],
            'Q_unit': 's',
            'R_row_units': ['V', 'native_gj_proxy_arbitrary_unit'],
            'K_element_units': [
                ['V^2/s', 'V*native_gj_proxy_arbitrary_unit/s'],
                ['V*native_gj_proxy_arbitrary_unit/s', 'native_gj_proxy_arbitrary_unit^2/s'],
            ],
            'coordinate_scaling': 'source case contrasts, not physical drug doses',
            'morphology_axis_identified': False,
            'eigenaxis_exported': False,
        }


def _grounding(cases):
    by_case = {row.case: row for row in cases}
    a = by_case['na'].E_total
    c = by_case['gj'].E_total
    b = 0.5 * (by_case['na_gj'].E_total - a - c)
    # Scale before computing a determinant; a singular/indefinite fitted
    # quadratic form is not silently regularized into an apparent metric.
    scale = max(abs(a), abs(b), abs(c))
    if scale == 0 or a <= 0 or c <= 0:
        raise ValueError('Q must be positive definite')
    an, bn, cn = a / scale, b / scale, c / scale
    det = an * cn - bn * bn
    if det <= 0:
        raise ValueError('Q must be positive definite')
    lambda_max = 0.5 * (an + cn + math.hypot(an - cn, 2 * bn))
    lambda_min = det / lambda_max
    condition = lambda_max / lambda_min
    if not math.isfinite(condition) or condition > 1e12:
        raise ValueError('Q is too ill-conditioned for an auditable inverse')
    inverse = ((cn / det / scale, -bn / det / scale),
               (-bn / det / scale, an / det / scale))
    R = ((by_case['na'].d_mean_vmem_mV / 1000,
          by_case['gj'].d_mean_vmem_mV / 1000),
         (by_case['na'].d_wound_gj, by_case['gj'].d_wound_gj))
    K = tuple(tuple(math.fsum(R[i][u] * inverse[u][v] * R[j][v]
                             for u in range(2) for v in range(2))
                    for j in range(2)) for i in range(2))
    for matrix in (inverse, R, K):
        for row in matrix:
            for value in row:
                number(value, 'derived matrix element')
    return GroundingMatrices(((a, b), (b, c)), R, K, condition)


@dataclass(frozen=True)
class TasmorphoSummary:
    provenance: TasmorphoProvenance
    cases: tuple[SimulationCase, ...]
    grounding: GroundingMatrices

    def wire(self):
        return {
            'schema': 'Mora3.TasmorphoSummary/v1',
            'source': self.provenance.wire(),
            'evidence_kind': 'simulation',
            'observation_scope': 'aggregate_simulation_case',
            'partition_group': self.provenance.partition_group,
            'case_order': list(CASES),
            'cases': [asdict(row) for row in self.cases],
            'native_units': {
                'E_mem': 's', 'E_gj': 's', 'E_pump': 's', 'E_total': 's',
                'd_mean_vmem_mV': 'mV', 'd_wound_gj': 'native_gj_proxy_arbitrary_unit',
            },
            'energy_definition': 'time integral of mean squared control-RMS-normalized flux difference; not joules',
            'time_unit_basis': 'pinned BETSE configuration uses seconds; CSV has no time samples',
            'native_drug_mapping': None,
            'coordinates': None,
            'cell_ids': None,
            'cell_graph': None,
            'grounding': self.grounding.wire(),
            'training_admitted': False,
            'causal_claim': False,
            'admission_blockers': [
                'aggregate cases do not contain cell states, cell geometry, or trajectories',
                'simulation labels are not observed phenotype outcomes or pharmacological identities',
                'shared-control cases require one run-level split; seven cases are not independent biological replicates',
                'source run identity, configuration, and data rights require independent evidence',
            ],
        }

    def to_state_records(self):
        raise UnsupportedTasmorphoSemantics(
            'aggregate BETSE CSV has no cell observations, coordinates, time samples, or experimentally paired outcomes')

    def to_cell_graph(self):
        raise UnsupportedTasmorphoSemantics(
            'aggregate BETSE CSV has no cell IDs, geometry, edge endpoints, or edge units')


def _csv_rows(payload, expected_sha256, header):
    if type(payload) is not bytes or len(payload) > MAX_CSV_BYTES:
        raise ValueError('a native CSV byte artifact of at most 1 MiB is required')
    digest(expected_sha256, 'expected_sha256')
    if sha256(payload).hexdigest() != expected_sha256:
        raise ValueError('artifact SHA256 mismatch')
    try:
        decoded = payload.decode('utf-8')
        if '\x00' in decoded:
            raise ValueError('NUL bytes are not permitted in CSV')
        rows = list(csv.reader(io.StringIO(decoded, newline=''), strict=True))
    except (UnicodeDecodeError, csv.Error) as exc:
        raise ValueError('strict UTF-8 CSV required') from exc
    if not rows or tuple(rows[0]) != header:
        raise ValueError('exact native CSV header required; extra, reordered, or duplicate columns are rejected')
    if any(len(row) != len(header) for row in rows[1:]):
        raise ValueError('incomplete or extra CSV cells')
    return [dict(zip(header, row)) for row in rows[1:]]


def _float(value, field):
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f'{field} must contain a finite native number') from exc
    number(result, field)
    return result


def parse_summary_bytes(payload, *, source_commit, run_id, expected_sha256,
                        source_locator, betse_version=None, random_seed=None,
                        geometry_sha256=None):
    """Verify one complete, native seven-case aggregate summary.

    A commit pin and a byte digest assert the caller's source declaration;
    they do not prove the artifact was actually generated by that source.
    Unknown generator metadata stays None rather than being fabricated.
    """
    provenance = TasmorphoProvenance(source_commit, expected_sha256, run_id,
                                   source_locator, betse_version, random_seed,
                                   geometry_sha256)
    rows = _csv_rows(payload, expected_sha256, SUMMARY_HEADER)
    if len(rows) != len(CASES) or {row['case'] for row in rows} != set(CASES):
        raise ValueError('exactly one row for each of the seven native cases is required')
    cases_by_name = {row['case']: SimulationCase(
        row['case'], *(_float(row[name], name) for name in SUMMARY_HEADER[1:]))
        for row in rows}
    cases = tuple(cases_by_name[name] for name in CASES)
    return TasmorphoSummary(provenance, cases, _grounding(cases))


def load_summary(path, *, source_commit, run_id, expected_sha256,
                 source_locator=None, betse_version=None, random_seed=None,
                 geometry_sha256=None):
    """Read a safe CSV; this never deserializes a BETSE pickle or runs a script."""
    path = regular_file(path, max_bytes=MAX_CSV_BYTES)
    with path.open('rb') as stream:
        payload = stream.read(MAX_CSV_BYTES + 1)
    return parse_summary_bytes(
        payload, source_commit=source_commit, run_id=run_id,
        expected_sha256=expected_sha256,
        source_locator=source_locator or path.absolute().as_uri(),
        betse_version=betse_version, random_seed=random_seed,
        geometry_sha256=geometry_sha256)


@dataclass(frozen=True)
class ReconstructedCount:
    label: str
    successes: int
    trials: int
    proportion: float
    role: str
    description: str
    denominator_population: str


@dataclass(frozen=True)
class ReconstructedCounts:
    provenance: TasmorphoProvenance
    counts: tuple[ReconstructedCount, ...]
    primary_source_locator: str | None

    def wire(self):
        return {
            'schema': 'Mora3.TasmorphoReconstructedCounts/v1',
            'source': self.provenance.wire(),
            'evidence_kind': 'experimental',
            'observation_scope': 'published_aggregate_count_reconstruction',
            'counts': [asdict(row) for row in self.counts],
            'primary_source_locator': self.primary_source_locator,
            'primary_source_counts_independently_verified': False,
            'native_drug_mapping': None,
            'cell_ids': None,
            'training_admitted': False,
            'causal_claim': False,
            'admission_blockers': [
                'reconstructed counts require verification against primary experimental reports and data rights',
                '8OH chemical identity conflicts between repository README and primary literature; preserve the source label',
                'challenge among immediate SH and challenge among immediate DH have different conditional denominators',
                'nigericin and monensin challenge probabilities are model predictions, not observed counts',
                'count cohorts have no specimen identifiers and cannot establish individual paired transitions',
            ],
        }


def parse_reconstructed_counts_bytes(payload, *, source_commit, run_id,
                                    expected_sha256, source_locator,
                                    primary_source_locator=None):
    """Read the exact five-row table produced by pinned script03.

    The table is generated from constants in the audited script, not included
    as a CSV in the pinned repository. No generator is executed here. A changed
    count table must receive a separate source review instead of masquerading
    as this revision's reconstruction. Prediction-table schemas are rejected.
    """
    provenance = TasmorphoProvenance(source_commit, expected_sha256, run_id, source_locator)
    if primary_source_locator is not None:
        text(primary_source_locator, 'primary_source_locator')
    rows = _csv_rows(payload, expected_sha256, COUNT_HEADER)
    if len(rows) != len(COUNT_SPECS) or {row['label'] for row in rows} != set(COUNT_SPECS):
        raise ValueError('exactly the five native reconstructed count labels are required')
    parsed = {}
    for row in rows:
        label = row['label']
        if any(re.fullmatch(r'[0-9]+', row[field]) is None
               for field in ('successes', 'trials')):
            raise ValueError('successes and trials must be literal nonnegative integers')
        successes, trials = int(row['successes']), int(row['trials'])
        expected_successes, expected_trials, role, denominator = COUNT_SPECS[label]
        if (successes, trials, row['role']) != (expected_successes, expected_trials, role):
            raise ValueError('counts and roles differ from the pinned native reconstruction')
        proportion = _float(row['proportion'], 'proportion')
        if not math.isclose(proportion, successes / trials, rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError('proportion differs from the count numerator and denominator')
        text(row['description'], 'description')
        parsed[label] = ReconstructedCount(label, successes, trials, proportion,
                                           role, row['description'], denominator)
    return ReconstructedCounts(provenance, tuple(parsed[label] for label in COUNT_SPECS),
                               primary_source_locator)


def load_reconstructed_counts(path, *, source_commit, run_id, expected_sha256,
                              source_locator=None, primary_source_locator=None):
    path = regular_file(path, max_bytes=MAX_CSV_BYTES)
    with path.open('rb') as stream:
        payload = stream.read(MAX_CSV_BYTES + 1)
    return parse_reconstructed_counts_bytes(
        payload, source_commit=source_commit, run_id=run_id,
        expected_sha256=expected_sha256,
        source_locator=source_locator or path.absolute().as_uri(),
        primary_source_locator=primary_source_locator)
