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
    def reserve_start(self,current=None):
        with self.lock:
            self.ensure_open();t=time.monotonic() if current is None else current;at=max(t,self.next_start.value);self.next_start.value=at+1/self.starts_per_second;self.starts.value+=1;return at
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
