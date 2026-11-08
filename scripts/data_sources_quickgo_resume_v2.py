"""Strict immutable QuickGO response reuse and finite-worker shared resource gates."""
import hashlib,json,pathlib,time
BASE='https://www.ebi.ac.uk/QuickGO/services'
PATHS={'annotation/search','annotation/downloadSearch','annotation/about','ontology/go/terms','ontology/go/about'}
ACCEPTS={'application/json','text/gaf','text/gpad'}
def sha(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def request_key(path,params,accept):
    if path not in PATHS or accept not in ACCEPTS:raise ValueError('unreviewed QuickGO endpoint or response format')
    if path!='annotation/downloadSearch' and accept!='application/json':raise ValueError('native JSON required for endpoint')
    return json.dumps([BASE,path,params,accept],sort_keys=True,separators=(',',':'))
def certify_receipt(path):
    path=pathlib.Path(path).resolve();r=json.loads(path.read_text());q=r.get('query',{});accept=q.get('headers',{}).get('Accept');key=request_key(q.get('path'),q.get('params'),accept)
    if q.get('base_url')!=BASE or r.get('http',{}).get('status')!=200 or not r.get('skill_result',{}).get('ok'):raise ValueError('source receipt not exact public successful HTTP200')
    p=pathlib.Path(r['source_response_path']).resolve()
    if not p.is_file() or p.stat().st_size!=r.get('source_bytes') or sha(p)!=r.get('source_sha256'):raise ValueError('incomplete or changed source bytes')
    if accept=='application/json':json.loads(p.read_text())
    return dict(request_key=key,path=str(p),bytes=p.stat().st_size,sha256=sha(p),receipt=str(path),receipt_sha256=sha(path),query=q)
def load_cached(cert,path,params,accept):
    if cert['request_key']!=request_key(path,params,accept):raise ValueError('cache scope/request mismatch')
    if sha(cert['receipt'])!=cert['receipt_sha256']:raise ValueError('source receipt changed')
    p=pathlib.Path(cert['path'])
    if p.stat().st_size!=cert['bytes'] or sha(p)!=cert['sha256']:raise ValueError('cached source bytes changed')
    return json.loads(p.read_text()) if accept=='application/json' else p.read_text()
def conservation(completed_groups,tasks,baseline):
    complete=[x for group in completed_groups for x in group];pending=[x for task in tasks for x in task['canonical_accessions']]
    if len(complete)!=len(set(complete)) or len(pending)!=len(set(pending)) or set(complete)&set(pending) or set(complete)|set(pending)!=set(baseline):raise ValueError('query duplication, omission or foreign baselineID')
    return dict(completed=len(complete),remaining=len(pending),union=len(set(complete)|set(pending)),complete_pending_disjoint=True,exact_baseline_conserved=True,remaining_set_sha256=hashlib.sha256(('\n'.join(sorted(pending))+'\n').encode()).hexdigest())

class SharedLimits:
    def __init__(self,ctx,starts_per_second,budgets):
        self.lock=ctx.RLock();self.next_start=ctx.Value('d',0,lock=False);self.starts_per_second=starts_per_second;self.stop=ctx.Value('b',False,lock=False);self.reason=ctx.Array('c',128,lock=False)
        self.budgets={k:dict(cap=int(v[0]),used=ctx.Value('q',int(v[1]),lock=False),reserved=ctx.Value('q',0,lock=False)) for k,v in budgets.items()};self.starts=ctx.Value('q',0,lock=False)
    def ensure_open(self):
        if self.stop.value:raise ValueError('shared source stop: '+self.reason.value.decode())
    def block(self,reason):
        with self.lock:self.stop.value=True;self.reason.value=str(reason).encode()[:127]
    def try_start(self,current=None):
        with self.lock:
            self.ensure_open();t=time.monotonic() if current is None else current
            if t<self.next_start.value:return False,self.next_start.value-t
            # Admit at the ACTUAL wake/admission time. Future waiters never book
            # tickets that could expire together while the processes are delayed.
            self.next_start.value=t+1/self.starts_per_second;self.starts.value+=1;return True,0
    def wait_for_start(self,clock=None,sleeper=None):
        clock=time.monotonic if clock is None else clock;sleeper=time.sleep if sleeper is None else sleeper
        while True:
            with self.lock:
                # Read the real admission clock only AFTER taking the shared
                # lock; a wait for this lock cannot make the timestamp stale.
                actual=clock();allowed,delay=self.try_start(actual)
                if allowed:return actual
            sleeper(delay)
    def reserve_bytes(self,category,n,wait=True):
        deadline=time.monotonic()+60
        while True:
            with self.lock:
                self.ensure_open();b=self.budgets[category]
                if b['used'].value+n>b['cap']:raise ValueError('shared remaining source budget insufficient')
                if b['used'].value+b['reserved'].value+n<=b['cap']:b['reserved'].value+=n;return
            if not wait or time.monotonic()>deadline:raise ValueError('finite shared reservation timeout')
            time.sleep(.1)
    def consume(self,category,n,reserved_credit=0):
        with self.lock:
            b=self.budgets[category]
            if n<0 or reserved_credit<0 or reserved_credit>n or reserved_credit>b['reserved'].value:raise ValueError('invalid shared accounting')
            b['reserved'].value-=reserved_credit;b['used'].value+=n
            if b['used'].value+b['reserved'].value>b['cap']:self.block('source raw cap');raise ValueError('source cap; preserve received bytes in quarantine')
    def release_bytes(self,category,n):
        with self.lock:
            b=self.budgets[category]
            if n<0 or n>b['reserved'].value:raise ValueError('invalid reservation release')
            b['reserved'].value-=n
    def snapshot(self):
        with self.lock:return dict({k:dict(cap=v['cap'],used=v['used'].value,reserved=v['reserved'].value) for k,v in self.budgets.items()},request_starts=self.starts.value,blocked=bool(self.stop.value),reason=self.reason.value.decode())


class CheckpointClaim:
    """Checkpoint-hash flock inherited by forked workers; no lockfile deletion.
    A resumed attempt must review/recompute its remaining-query checkpoint. The
    claim permits normal acquisition after all recorded prior owners are gone.
    """
    def __init__(self,root,checkpoint,run_id):
        import fcntl,os,re,datetime
        if re.fullmatch(r'[A-Za-z0-9_.-]+',run_id) is None:raise ValueError('unsafe checkpoint runID')
        self.fd=None;self.key=sha(checkpoint);directory=pathlib.Path(root)/'audits/checkpoint-execution-locks';directory.mkdir(parents=True,exist_ok=True);self.lock_path=directory/(self.key+'.lock');self.prior_owners_checked=[]
        fd=os.open(self.lock_path,os.O_RDWR|os.O_CREAT,0o600)
        try:
            try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise ValueError('checkpoint already owned by another coordinator/worker')
            for p in sorted(directory.glob(self.key+'-*.claim.json')):
                old=json.loads(p.read_text());owners=[old['owner']];children=pathlib.Path(old['children_receipt'])
                if children.exists():owners+=json.loads(children.read_text())['children']
                for owner in owners:
                    if self.owner_alive(owner):raise ValueError('prior checkpoint owner remains alive; explicit recovery proof required')
                self.prior_owners_checked.append(dict(receipt=str(p),sha256=sha(p),recorded_owner_and_workers_exited=True))
            self.receipt=directory/(self.key+'-'+run_id+'.claim.json');self.children_receipt=directory/(self.key+'-'+run_id+'.children.json')
            record=dict(schema='mora3.exclusive_checkpoint_execution_claim.v2',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),checkpoint=str(pathlib.Path(checkpoint).resolve()),checkpoint_sha256=self.key,run_id=run_id,owner=self.owner_ref(os.getpid()),children_receipt=str(self.children_receipt),prior_owners_checked=self.prior_owners_checked,lockfile_preserved=True,lock_inherited_by_forked_workers=True)
            with self.receipt.open('x') as f:json.dump(record,f,indent=2);f.flush();os.fsync(f.fileno())
            self.fd=fd
        except BaseException:os.close(fd);raise
    @staticmethod
    def owner_ref(pid):
        p=pathlib.Path('/proc/'+str(pid)+'/stat')
        if not p.exists():return dict(pid=pid,start_ticks=None,already_exited=True)
        s=p.read_text().split();return dict(pid=pid,start_ticks=int(s[21]))
    @staticmethod
    def owner_alive(owner):
        p=pathlib.Path('/proc/'+str(owner['pid'])+'/stat')
        if not p.exists() or owner.get('start_ticks') is None:return False
        s=p.read_text().split();return int(s[21])==owner['start_ticks'] and s[2]!='Z'
    def record_children(self,pids):
        import os
        with self.children_receipt.open('x') as f:json.dump(dict(children=[self.owner_ref(p) for p in pids]),f,indent=2);f.flush();os.fsync(f.fileno())
    def close(self):
        import os
        if self.fd is not None:
            # Never LOCK_UN: forked workers share this open-file description.
            # Closing only our descriptor preserves their lock after parentexit.
            os.close(self.fd);self.fd=None
