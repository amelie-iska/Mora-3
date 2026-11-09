"""Typed simulation provenance, intervention families and independent replay checks.

BETSEE is the GUI environment; BETSE is the numerical engine. Code licensing
never turns simulations into experimental measurements or validates third-party
calibration data. No upstream scripts, pickle files or simulator are executed.
"""
from dataclasses import dataclass,asdict
from hashlib import sha256
from importlib.metadata import version,PackageNotFoundError
import math
import re
from .contracts import canonical,digest,number,text
from .tasmorpho import SUPPORTED_SOURCE_COMMIT,SUMMARY_HEADER


@dataclass(frozen=True)
class Intervention:
    kind: str
    target: str
    direction: str
    value: float
    unit: str
    start: float
    stop: float
    time_unit: str

    def __post_init__(self):
        if self.kind not in ('gap_junction','ion_channel','membrane_permeability'):
            raise ValueError('explicit supported simulator intervention kind required')
        text(self.target,'native simulator target')
        if self.direction not in ('increase','decrease','set'):
            raise ValueError('explicit intervention direction required')
        for x in (self.value,self.start,self.stop):number(x,'native intervention coordinate')
        if self.start<0 or self.stop<=self.start or self.time_unit!='s':
            raise ValueError('strictly ordered nonnegative interval in native seconds required')
        if self.unit not in ('multiplier','percent-blocked'):
            raise ValueError('native multiplier or block percentage required; not a drug dose')
        if self.unit=='multiplier' and self.value<=0 or self.unit=='percent-blocked' and not 0<=self.value<=100:
            raise ValueError('native intervention value outside its domain')


@dataclass(frozen=True)
class SimulationBinding:
    source_revision: str
    base_configuration_sha256: str
    geometry_sha256: str
    source_lineage_sha256: str
    engine_version: str
    seed: int
    biological_system: str
    time_unit: str
    control_arm: str
    interventions: tuple[Intervention,...]
    evidence_kind: str
    independent_experimental_support_sha256: str | None

    def __post_init__(self):
        if type(self.source_revision)is not str or re.fullmatch('[0-9a-f]{40}',self.source_revision)is None:
            raise ValueError('exact source revision required')
        for name in ('base_configuration_sha256','geometry_sha256','source_lineage_sha256'):
            digest(getattr(self,name),name)
        for name in ('engine_version','biological_system','control_arm'):text(getattr(self,name),name)
        if type(self.seed)is not int or self.seed<0 or self.time_unit!='s' or self.evidence_kind!='simulation':
            raise ValueError('explicit seed, native time and simulation-only evidence required')
        if type(self.interventions)is not tuple or any(type(v)is not Intervention for v in self.interventions):
            raise ValueError('typed immutable simulator interventions required')
        for v in self.interventions:v.__post_init__()
        if self.independent_experimental_support_sha256 is not None:
            digest(self.independent_experimental_support_sha256,'independent experimental support')


def simulation_group(binding):
    """All seeds and near-duplicate actions share the original simulator lineage.

    Deliberately excludes seed/action values; it is conservative rather than a
    nearest-neighbor similarity assertion. Cross-lineage equivalence still needs
    an explicit construction-owner link. Engine changes do not split a family.
    """
    if type(binding)is not SimulationBinding:raise ValueError('typed simulation binding required')
    binding.__post_init__()
    return sha256(canonical({'schema':'SCRISimulationFamily/v1',
        'lineage':binding.source_lineage_sha256,'base_configuration':binding.base_configuration_sha256,
        'geometry':binding.geometry_sha256,'system':binding.biological_system,'control':binding.control_arm})).hexdigest()


def compare_summary(expected,actual,*,atol,rtol):
    """Compare case-matched native fields using predeclared tolerances, not rounding."""
    for value in (atol,rtol):
        number(value,'tolerance')
        if value<0:raise ValueError('nonnegative predeclared tolerance required')
    def index(rows):
        if type(rows) not in (list,tuple) or not rows:raise ValueError('nonempty complete summary required')
        result={}
        for row in rows:
            if type(row)is not dict or set(row)!=set(SUMMARY_HEADER):raise ValueError('exact native summary fields required')
            text(row['case'],'case')
            if row['case']in result:raise ValueError('duplicate simulation case')
            for name in SUMMARY_HEADER[1:]:number(row[name],name)
            result[row['case']]=row
        return result
    x,y=index(expected),index(actual)
    if set(x)!=set(y):raise ValueError('identical case support required')
    failures=[];maximum=0.
    for case in sorted(x):
        for name in SUMMARY_HEADER[1:]:
            delta=abs(x[case][name]-y[case][name]);maximum=max(maximum,delta)
            if delta>atol+rtol*abs(x[case][name]):failures.append({'case':case,'field':name,'absolute_error':delta})
    return dict(within_tolerance=not failures,failures=failures,max_absolute_error=maximum,
        atol=atol,rtol=rtol,cases=len(x),independent_simulation_execution_established=False,
        experimental_validation=False)


def installed_simulation_readiness():
    installed={}
    for package in ('betse','betsee','PySide2'):
        try:installed[package]=version(package)
        except PackageNotFoundError:installed[package]=None
    return dict(installed=installed,engine_available=installed['betse']is not None,
        gui_available=installed['betsee']is not None and installed['PySide2']is not None,
        dependency_install_performed=False,simulator_execution_performed=False,
        historical_seed_known=False,historical_engine_version_known=False,
        exact_historical_reproduction_qualified=False,
        blockers=['historical example seed and engine version are not recorded',
                  *(['BETSE absent in selected environment']if installed['betse']is None else [])])


def tasmorpho_interventions(case):
    """Pinned script 01 parameters, NOT a mapping to pharmacological compounds."""
    known={'control':(), 'na':('na',),'k':('k',),'gj':('gj',),
           'na_k':('na','k'),'na_gj':('na','gj'),'k_gj':('k','gj')}
    if case not in known:raise ValueError('unknown pinned tasmorpho case')
    return tuple(Intervention('gap_junction','native-gj-network','decrease',20.,'percent-blocked',0.,.01,'s')
        if name=='gj' else Intervention('membrane_permeability',name,'increase',1.3,'multiplier',0.,.01,'s')
        for name in known[case])
