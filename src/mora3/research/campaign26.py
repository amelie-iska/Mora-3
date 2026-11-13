"""The user's exact 24-proxy + 2 full-target run limit.

This is accounting and immutable selection binding, not source/model admission.
The existing exclusive model worker owns actual execution. A reservation consumes
its slot even if execution fails. Interrupted/partial entries are retained and
block continuation; no retry, orphan adoption, reset, daemon or scheduler exists.
The local controller is trusted: hashes are integrity checks, not authentication
against someone authorized to rewrite the entire project.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
import fcntl
import json
import os
from pathlib import Path
import re
import stat
import uuid

from mora3.data.admet import canonical
from .mup_sweep_selection import FrozenProxySelection, planned_proxy_trials, select_proxy_transfer

TARGET_PARAMETER_BOUNDS=(79_000_000,81_000_000)
TERMINAL_STATES=('completed','wall_time_censored','insufficient_exposure','refused','oom',
                 'diverged','interrupted','missing_metrics','failed')


def _digest(value):
    if type(value)is not str or re.fullmatch('[0-9a-f]{64}',value)is None:
        raise ValueError('Exact SHA256 commitment required')
    return value


def _positive(value):
    if type(value)is not int or value<1:raise ValueError('Explicit positive integer bound required')
    return value


@dataclass(frozen=True)
class Campaign26:
    directive_sha256: str
    source_snapshot_sha256: str
    proxy_campaign_sha256: str
    target_seeds: tuple[int,int]=(2026111001,2026111002)
    target_specification_sha256: str | None=None

    def __post_init__(self):
        for value in (self.directive_sha256,self.source_snapshot_sha256,self.proxy_campaign_sha256):_digest(value)
        if self.target_specification_sha256 is not None:_digest(self.target_specification_sha256)
        if (type(self.target_seeds)is not tuple or len(self.target_seeds)!=2
                or any(type(x)is not int or x<0 for x in self.target_seeds)
                or len(set(self.target_seeds))!=2):
            raise ValueError('Exactly two distinct target-training seeds required')

    @property
    def run_ids(self):
        return tuple(t.trial_id for t in planned_proxy_trials())+tuple(f'full80m-{s}'for s in self.target_seeds)

    def wire(self):
        self.__post_init__()
        result=dict(schema='Mora3.Campaign26/v1',directive_sha256=self.directive_sha256,
            source_snapshot_sha256=self.source_snapshot_sha256,proxy_campaign_sha256=self.proxy_campaign_sha256,
            maximum_training_runs=26,proxy_runs=24,full_target_runs=2,run_ids=self.run_ids,
            target_seeds=self.target_seeds,target_nominal_parameters=80_000_000,
            target_parameter_bounds=TARGET_PARAMETER_BOUNDS,target_parameterization='muP',
            standard_parameterization_runs=0,intermediate_training_runs=0,
            failed_attempts_consume_slots=True,automatic_retry=False,admits_training=False)
        if self.target_specification_sha256 is not None:
            result.update(schema='Mora3.Campaign26/v2',target_specification_sha256=self.target_specification_sha256)
        return result

    @property
    def sha256(self):return sha256(canonical(self.wire())).hexdigest()


def _safe(path):
    path=Path(path).absolute()
    if any(p.is_symlink()for p in (path,*path.parents)):
        raise ValueError('Nonsymlink controller paths required')
    return path


def _read(path):
    _safe(path)
    try:
        fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    except OSError as exc:raise ValueError('Readable regular controller artifact required')from exc
    with os.fdopen(fd,'rb')as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):raise ValueError('Regular controller artifact required')
        raw=stream.read(1_048_577)
    if len(raw)>1_048_576:raise ValueError('Controller artifact exceeds bounded size')
    try:doc=json.loads(raw)
    except (ValueError,UnicodeError)as exc:raise ValueError('Complete controller JSON required')from exc
    if type(doc)is not dict or canonical(doc)!=raw:raise ValueError('Exact canonical controller bytes required')
    return doc


def _sync(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:os.fsync(fd)
    finally:os.close(fd)


def _publish(path,document):
    _safe(path);pending=path.with_name(path.name+'.pending')
    with pending.open('xb')as stream:
        stream.write(canonical(document));stream.flush();os.fsync(stream.fileno())
    _sync(path.parent)
    os.link(pending,path,follow_symlinks=False)
    _sync(path.parent)
    pending.unlink()  # this call's successfully published temporary only
    _sync(path.parent)


def _process():
    # Bind the handle to this actual process, including PID-reuse protection.
    text=Path('/proc/self/stat').read_text()
    return (os.getpid(),int(text[text.rfind(')')+2:].split()[19]))


@dataclass(frozen=True)
class RunClaim:
    store: object
    run_id: str
    nonce: str
    process_identity: tuple[int,int]

    def enter(self):
        """Exactly once immediately before the qualified worker executes this run."""
        self.store._enter(self)


class RunBudget26:
    def __init__(self,directory,plan):
        if type(plan)is not Campaign26:raise ValueError('Exact 26-run campaign required')
        plan.__post_init__();self.directory=_safe(directory);self.plan=plan

    @classmethod
    def create(cls,directory,plan):
        obj=cls(directory,plan);root=obj.directory
        root.mkdir();_sync(root.parent)
        (root/'runs').mkdir()
        with(root/'lock').open('xb')as f:f.flush();os.fsync(f.fileno())
        _publish(root/'genesis.json',plan.wire())
        obj.snapshot();return obj

    @contextmanager
    def _locked(self):
        root=_safe(self.directory)
        try:fd=os.open(root/'lock',os.O_RDWR|os.O_NOFOLLOW|os.O_NONBLOCK)
        except OSError as exc:raise ValueError('Existing regular budget lock required')from exc
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):raise ValueError('Regular budget lock required')
            fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
            yield
        finally:os.close(fd)

    def _load(self):
        root=_safe(self.directory);runs=_safe(root/'runs')
        allowed={'genesis.json','runs','lock','selection.json'}
        if not root.is_dir()or not runs.is_dir()or not {'genesis.json','runs','lock'}<=set(p.name for p in root.iterdir())<=allowed:
            raise ValueError('Incomplete or unknown controller entry')
        if canonical(_read(root/'genesis.json'))!=canonical(self.plan.wire()):
            raise ValueError('Frozen 26-run campaign changed')
        entries=list(runs.iterdir());count=len(entries)
        if count>26 or set(p.name for p in entries)!=set(self.plan.run_ids[:count]):
            raise ValueError('Exact sequential prefix of at most26 run slots required')
        selection=None
        if (root/'selection.json').exists():
            selection=_read(root/'selection.json')
            if (set(selection)!={'schema','campaign_sha256','selection_sha256'}
                    or selection['schema']!='Mora3.Campaign26Selection/v1'
                    or selection['campaign_sha256']!=self.plan.sha256):
                raise ValueError('Exact frozen transfer selection commitment required')
            _digest(selection['selection_sha256'])
            if count<24:raise ValueError('Selection before complete proxy phase')
        states=[]
        for index,rid in enumerate(self.plan.run_ids[:count]):
            path=_safe(runs/rid)
            if not path.is_dir():raise ValueError('Regular run directory required')
            names={p.name for p in path.iterdir()}
            if 'reservation.json'not in names or not names<={'reservation.json','entered.json','terminal.json'}:
                raise ValueError('Incomplete or unknown run receipt; explicit reconciliation required')
            claim=_read(path/'reservation.json')
            keys={'schema','campaign_sha256','run_id','index','nonce','process_identity','selection_sha256'}
            if (set(claim)!=keys or claim['schema']!='Mora3.Campaign26Reservation/v1'
                    or claim['campaign_sha256']!=self.plan.sha256 or claim['run_id']!=rid
                    or type(claim['index'])is not int or claim['index']!=index
                    or type(claim['nonce'])is not str or re.fullmatch('[0-9a-f]{32}',claim['nonce'])is None
                    or type(claim['process_identity'])is not list or len(claim['process_identity'])!=2
                    or any(type(x)is not int or x<1 for x in claim['process_identity'])):
                raise ValueError('Exact campaign run reservation required')
            expected_selection=None if index<24 else None if selection is None else selection['selection_sha256']
            if index>=24 and selection is None or claim['selection_sha256']!=expected_selection:
                raise ValueError('Run is not bound to the frozen transfer selection')
            entered='entered.json'in names;terminal=None
            if entered:
                if _read(path/'entered.json')!=dict(schema='Mora3.Campaign26Entered/v1',reservation_sha256=sha256(canonical(claim)).hexdigest()):
                    raise ValueError('Execution entry does not match its reserved run')
            if 'terminal.json'in names:
                terminal=_read(path/'terminal.json')
                if (not entered or set(terminal)!={'schema','reservation_sha256','status','artifact_sha256'}
                        or terminal['schema']!='Mora3.Campaign26Terminal/v1'
                        or terminal['reservation_sha256']!=sha256(canonical(claim)).hexdigest()
                        or terminal['status']not in TERMINAL_STATES):
                    raise ValueError('Exact entered terminal disposition required')
                _digest(terminal['artifact_sha256'])
            if index<count-1 and terminal is None:raise ValueError('Nonterminal earlier run forbids later execution')
            states.append((claim,entered,terminal))
        if selection is not None and any(row[2]is None for row in states[:24]):
            raise ValueError('Selection requires every proxy run to be terminal')
        return states,selection

    def snapshot(self):
        with self._locked():
            states,selection=self._load()
            return dict(campaign_sha256=self.plan.sha256,reserved_runs=len(states),maximum_runs=26,
                entered_runs=sum(x[1]for x in states),terminal_runs=sum(x[2]is not None for x in states),
                failed_runs=sum(x[2]is not None and x[2]['status']not in ('completed','wall_time_censored')for x in states),
                selection_sha256=None if selection is None else selection['selection_sha256'],training_admission=False)

    def reserve(self,run_id):
        with self._locked():
            states,selection=self._load();index=len(states)
            if index>=26 or run_id!=self.plan.run_ids[index]:raise ValueError('Only next unused slot of the26-run campaign may start')
            if states and states[-1][2]is None:raise ValueError('Unfinished attempt requires explicit reconciliation; no retry')
            if index>=24 and selection is None:raise ValueError('Both target runs require a frozen validation selection')
            nonce=uuid.uuid4().hex;identity=_process();path=self.directory/'runs'/run_id
            path.mkdir();_sync(path.parent)
            _publish(path/'reservation.json',dict(schema='Mora3.Campaign26Reservation/v1',campaign_sha256=self.plan.sha256,
                run_id=run_id,index=index,nonce=nonce,process_identity=identity,
                selection_sha256=None if index<24 else selection['selection_sha256']))
            self._load();return RunClaim(self,run_id,nonce,identity)

    def _claim(self,claim,states):
        if (type(claim)is not RunClaim or claim.store is not self or claim.process_identity!=_process()
                or not states or states[-1][0]['run_id']!=claim.run_id
                or states[-1][0]['nonce']!=claim.nonce
                or tuple(states[-1][0]['process_identity'])!=claim.process_identity):
            raise ValueError('Exact current-process live run claim required')
        return states[-1]

    def _enter(self,claim):
        with self._locked():
            states,_=self._load();doc,entered,terminal=self._claim(claim,states)
            if entered or terminal is not None:raise ValueError('Run already entered; no repeat execution')
            _publish(self.directory/'runs'/claim.run_id/'entered.json',dict(schema='Mora3.Campaign26Entered/v1',reservation_sha256=sha256(canonical(doc)).hexdigest()))

    def finish(self,claim,*,status,artifact_sha256):
        if status not in TERMINAL_STATES:raise ValueError('Explicit terminal disposition required')
        _digest(artifact_sha256)
        with self._locked():
            states,_=self._load();doc,entered,terminal=self._claim(claim,states)
            if not entered or terminal is not None:raise ValueError('Only one entered run can publish one terminal result')
            _publish(self.directory/'runs'/claim.run_id/'terminal.json',dict(schema='Mora3.Campaign26Terminal/v1',
                reservation_sha256=sha256(canonical(doc)).hexdigest(),status=status,artifact_sha256=artifact_sha256))

    def bind_selection(self,selection_sha256):
        """Bind a separately verified native selector artifact, not user-supplied HPs."""
        _digest(selection_sha256)
        with self._locked():
            states,old=self._load()
            if len(states)!=24 or any(row[2]is None for row in states)or old is not None:
                raise ValueError('One immutable selection only after all24 terminal proxies')
            _publish(self.directory/'selection.json',dict(schema='Mora3.Campaign26Selection/v1',
                campaign_sha256=self.plan.sha256,selection_sha256=selection_sha256))


def target_run_specs(plan,selection,outcomes,*,source_snapshot_sha256,target_width,target_parameter_count,
        complete_owner_sha256,width_family_sha256,runtime_profile_sha256,maximum_updates,maximum_seconds,
        selection_publication=None):
    """Freeze two target specifications using only the existing native selector.

    Counts/commitments must subsequently be checked against the actual complete
    source-bound owner. This constructor does not authenticate those authorities.
    """
    if type(plan)is not Campaign26 or type(selection)is not FrozenProxySelection:
        raise ValueError('Exact campaign and native selection required')
    plan.__post_init__();selection.__post_init__()
    if source_snapshot_sha256!=plan.source_snapshot_sha256 or selection.campaign_manifest_sha256!=plan.proxy_campaign_sha256:
        raise ValueError('Same source snapshot and proxy campaign required')
    for x in(complete_owner_sha256,width_family_sha256,runtime_profile_sha256):_digest(x)
    _positive(target_width);_positive(target_parameter_count);_positive(maximum_updates);_positive(maximum_seconds)
    if target_width%2 or not TARGET_PARAMETER_BOUNDS[0]<=target_parameter_count<=TARGET_PARAMETER_BOUNDS[1]:
        raise ValueError('Complete approximately80M model count and even native width required')
    selected=select_proxy_transfer(selection,tuple(outcomes));winner=selected['frozen_transfer_hyperparameters']
    if winner is None:raise ValueError('No complete eligible paired-seed proxy winner')
    if type(selection_publication)is not bytes or len(selection_publication)>1_048_576:
        raise ValueError('Exact native pinned selection publication bytes required; bare result hashes are not interchangeable')
    try:publication=json.loads(selection_publication)
    except (ValueError,UnicodeError)as exc:raise ValueError('Complete native selection publication required')from exc
    if (type(publication)is not dict or set(publication)!={'schema','terminal_pin_prefix_sha256','selection'}
            or publication['schema']!='Mora3.PinnedProxyTransferSelection/v1'
            or canonical(publication)!=selection_publication
            or canonical(publication['selection'])!=canonical(selected)):
        raise ValueError('Published selection must exactly match the replayed native24-result selector')
    _digest(publication['terminal_pin_prefix_sha256'])
    commitment=sha256(selection_publication).hexdigest()
    return tuple(dict(schema='Mora3.Transferred80MTrainingSpec/v2',run_id=run_id,seed=seed,
        campaign26_sha256=plan.sha256,source_snapshot_sha256=source_snapshot_sha256,
        selection_sha256=commitment,selection_result_sha256=sha256(canonical(selected)).hexdigest(),
        terminal_pin_prefix_sha256=publication['terminal_pin_prefix_sha256'],
        target_specification_sha256=plan.target_specification_sha256,
        hyperparameters=dict(winner),parameterization='muP',
        target_width=target_width,target_parameter_count=target_parameter_count,
        complete_owner_sha256=complete_owner_sha256,width_family_sha256=width_family_sha256,
        runtime_profile_sha256=runtime_profile_sha256,maximum_updates=maximum_updates,maximum_seconds=maximum_seconds,
        target_tuning_allowed=False,launch_authorized=False)
        for run_id,seed in zip(plan.run_ids[24:],plan.target_seeds,strict=True))
