from pathlib import Path
import datetime,hashlib,json,shlex,subprocess,time
B=Path('outputs/data-intake/20261101-nucleotide-corpus-v1/model-handoff-v1');prior=B/'current-v14';o=B/'current-v14-validated';o.mkdir(exist_ok=False)
def ref(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
d=json.loads((prior/'executable-model-handoff.json').read_text());d['supersedes']=ref(prior/'executable-model-handoff.json');d['correction']='Inspect command uses the actual parser argument --limit 2; original --limit2 typo retained in prior artifact.';d['created_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
receipts=[]
for task in d['tasks']:
 for key in ['command','inspect_command']:
  if key not in task:continue
  command=task[key].replace('--limit2','--limit 2');task[key]=command;start=time.perf_counter();r=subprocess.run(shlex.split(command),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True);p=o/(task['id']+'.log');p.open('x').write(r.stdout);receipts.append(dict(task=task['id'],argv=shlex.split(command),exit_code=r.returncode,seconds=time.perf_counter()-start,log=ref(p)));assert r.returncode==0,r.stdout
(o/'executable-model-handoff.json').open('x').write(json.dumps(d,indent=2)+'\n')
r=json.loads((prior/'reader-readiness.json').read_text());r['validated_executable_handoff']=str(o/'executable-model-handoff.json');r['supersedes_readiness']=ref(prior/'reader-readiness.json');(o/'reader-readiness.json').open('x').write(json.dumps(r,indent=2)+'\n');(o/'HANDOFF.md').open('x').write((prior/'HANDOFF.md').read_text()+'\nThe two diagnostic commands in the accompanying handoff were executed successfully; exact commands, logs and bounds are in executable-command-verification.json. The earlier inspect flag typo is corrected in this new version.\n');(o/'executable-command-verification.json').open('x').write(json.dumps(dict(schema='mora3.ExecutableDataHandoffCheck/v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),checks=receipts,optimizer_steps=0,model_or_source_edits=False),indent=2)+'\n');print('PASS',o)
