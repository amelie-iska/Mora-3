"""Read-only API/source snapshot; never import models or assert neural coverage."""
import ast,datetime,hashlib


def inventory(root,master_ref,families):
    modules=[];families=set(families)
    for path in sorted((root/'src/mora3/data').glob('*.py')):
        raw=path.read_bytes();entry=dict(path=str(path.relative_to(root)),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),code_only_not_runtime_or_training_support=True)
        try:
            tree=ast.parse(raw);apis=[]
            candidates=[('',node) for node in tree.body]
            for owner in tree.body:
                if isinstance(owner,ast.ClassDef) and not owner.name.startswith('_'):
                    candidates.extend((owner.name+'.',node) for node in owner.body)
            for prefix,node in candidates:
                if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and not node.name.startswith('_'):
                    args=node.args;apis.append(dict(name=prefix+node.name,line=node.lineno,scope='class method' if prefix else 'module function',argument_names=[x.arg for x in args.posonlyargs+args.args+args.kwonlyargs],vararg=args.vararg.arg if args.vararg else None,kwarg=args.kwarg.arg if args.kwarg else None))
            entry.update(public_candidate_API=apis,exact_known_source_family_literals=sorted({n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str) and n.value in families}),stable_bytes_before_after=path.read_bytes()==raw)
        except (SyntaxError,UnicodeError) as exc:
            entry.update(parse_status='active source snapshot not parseable; no model defect or support claim inferred',exception_type=type(exc).__name__,stable_bytes_before_after=path.read_bytes()==raw)
        modules.append(entry)
    return dict(schema='mora3.CurrentNativeReaderSourceAPIInventory/v2',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),master=master_ref,modules=modules,scope='read-only source hashes and public candidate argument names only; no imports, native invocation, head/objective, neural input-exclusion or optimizer verification; absence of literal/API is not proof of missing support',families=sorted(families))
