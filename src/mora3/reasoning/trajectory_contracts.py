"""Complete, evidence-typed reasoning trajectories with prefix-local information.

A computational dependency is not a biological causal arrow. A temporal ordering
is not an intervention. State identities include reasoning layer/history: revisiting
the same latent vector does not create a reverse-time transition. Merges require
an explicit equivalence artifact and only facts available along EVERY incoming
history may remain visible. Hashes bind artifacts; they do not verify causality.

This finite data contract supports complete multi-branch/path analogies. Its
bounds reject overflow rather than silently reducing a trajectory to endpoints.
"""
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import re
from .directed_topology import Arrow, Square, DirectedComplex, DirectedFunctor


def _text(x):
    if type(x) is not str or not x.strip() or len(x) > 4096:
        raise ValueError('Bounded nonempty identifier required')


def _sha(x):
    if type(x) is not str or re.fullmatch('[0-9a-f]{64}', x) is None:
        raise ValueError('Immutable evidence digest required')


def _tuple_text(x):
    if type(x) is not tuple or len(set(x)) != len(x):
        raise ValueError('Immutable unique identifier tuple required')
    for v in x:
        _text(v)


def _json(raw):
    if type(raw) is not str or len(raw.encode()) > 131072:
        raise ValueError('Bounded native fact JSON required')
    def pairs(items):
        obj = {}
        for k, v in items:
            if k in obj: raise ValueError('Duplicate native JSON key')
            obj[k] = v
        return obj
    def constant(_): raise ValueError('Nonfinite native JSON')
    try:
        value = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
        json.dumps(value, allow_nan=False)
    except (ValueError, TypeError, RecursionError) as exc:
        raise ValueError('Finite well-formed native fact JSON required') from exc
    return value


def _canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


@dataclass(frozen=True)
class TraceFact:
    name: str
    value_json: str
    role: str
    origin_node: str | None
    depends_on: tuple[str, ...]
    evidence_kind: str
    evidence_sha256: str

    def __post_init__(self):
        _text(self.name); _json(self.value_json); _tuple_text(self.depends_on); _sha(self.evidence_sha256)
        if self.role not in ('condition', 'desired', 'derived', 'target', 'evaluation'):
            raise ValueError('Explicit condition/desired/derived/target/evaluation role required')
        if self.evidence_kind not in ('observed', 'computed', 'model_generated', 'assumed', 'synthetic_fixture'):
            raise ValueError('Explicit native evidence class required')
        if self.origin_node is not None: _text(self.origin_node)
        if self.role in ('condition', 'desired') and (self.origin_node is not None or self.depends_on):
            raise ValueError('Initial conditions are not future derived facts')
        if self.role == 'derived' and (self.origin_node is None or not self.depends_on):
            raise ValueError('Derived facts need their production state and explicit dependencies')
        if self.name in self.depends_on:
            raise ValueError('A fact cannot justify itself')


@dataclass(frozen=True)
class TraceNode:
    key: str
    layer: int
    visible_facts: tuple[str, ...]
    status: str
    merge_evidence_sha256: str | None = None

    def __post_init__(self):
        _text(self.key); _tuple_text(self.visible_facts)
        if type(self.layer) is not int or not 0 <= self.layer <= 64:
            raise ValueError('Explicit bounded nonnegative reasoning layer required')
        if self.status not in ('open', 'terminal', 'censored', 'invalid'):
            raise ValueError('Explicit terminal versus censored/invalid status required')
        if self.merge_evidence_sha256 is not None: _sha(self.merge_evidence_sha256)


@dataclass(frozen=True)
class TraceStep:
    key: str
    source: str
    target: str
    action: str
    direction_kind: str
    requires: tuple[str, ...]
    evidence_sha256: str
    intervention_evidence_sha256: str | None = None

    def __post_init__(self):
        for x in (self.key, self.source, self.target, self.action): _text(x)
        _tuple_text(self.requires); _sha(self.evidence_sha256)
        if self.direction_kind not in ('computational', 'temporal', 'observational', 'interventional'):
            raise ValueError('Explicit computational/temporal/observational/interventional direction required')
        if self.direction_kind == 'interventional':
            _sha(self.intervention_evidence_sha256)
        elif self.intervention_evidence_sha256 is not None:
            raise ValueError('Intervention artifact must not silently upgrade another direction class')


@dataclass(frozen=True)
class ReasoningTrace:
    identity: str
    source_sha256: str
    context_sha256: str
    group_policy_sha256: str
    split: str
    groups: tuple[str, ...]
    nodes: tuple[TraceNode, ...]
    steps: tuple[TraceStep, ...]
    facts: tuple[TraceFact, ...]
    squares: tuple[Square, ...]

    def __post_init__(self):
        _text(self.identity)
        for value in (self.source_sha256, self.context_sha256, self.group_policy_sha256): _sha(value)
        _tuple_text(self.groups)
        if not self.groups or self.split not in ('train', 'validation', 'test'):
            raise ValueError('Original split and nonempty leakage-group identities required')
        for values, kind, maximum in ((self.nodes, TraceNode, 128), (self.steps, TraceStep, 512),
                                       (self.facts, TraceFact, 2048), (self.squares, Square, 256)):
            if type(values) is not tuple or len(values) > maximum or any(type(v) is not kind for v in values):
                raise ValueError('Complete typed trajectory exceeds its declared finite bound')
            for value in values: value.__post_init__()
        if not self.nodes or len({n.key for n in self.nodes}) != len(self.nodes):
            raise ValueError('Unique nonempty trajectory states required')
        if len({e.key for e in self.steps}) != len(self.steps) or len({f.name for f in self.facts}) != len(self.facts):
            raise ValueError('Unique step and fact identities required')
        if sum(len(f.value_json.encode()) for f in self.facts) > 2_000_000:
            raise ValueError('Complete trace payload exceeds byte budget')
        ns = {n.key:n for n in self.nodes}; fs = {f.name:f for f in self.facts}
        incoming = {n.key:[] for n in self.nodes}; outgoing = {n.key:[] for n in self.nodes}
        for e in self.steps:
            if e.source not in ns or e.target not in ns or ns[e.source].layer >= ns[e.target].layer:
                raise ValueError('Strictly forward known-layer trajectory required; no reversed/cyclic shortcut')
            if ns[e.source].status != 'open':
                raise ValueError('Terminal/censored/invalid states must not expand')
            if not set(e.requires) <= set(ns[e.source].visible_facts):
                raise ValueError('Action depends on information unavailable at its source prefix')
            incoming[e.target].append(e.source); outgoing[e.source].append(e.target)
        roots = [n for n in self.nodes if not incoming[n.key]]
        if len(roots) != 1 or roots[0].layer != 0:
            raise ValueError('One explicit layer-zero root per trace required')
        root = roots[0].key; dom = {}
        for n in sorted(self.nodes, key=lambda n:n.layer):
            parents = incoming[n.key]
            if n.key == root: dom[n.key] = {root}
            else:
                if not parents or any(p not in dom for p in parents):
                    raise ValueError('Every trace state must be reachable from its declared root')
                dom[n.key] = set.intersection(*(dom[p] for p in parents)) | {n.key}
            if len(parents) > 1 and n.merge_evidence_sha256 is None:
                raise ValueError('Merged histories require explicit sufficient-state equivalence evidence')
            if not outgoing[n.key] and n.status == 'open':
                raise ValueError('Unexpanded leaf must be marked censored, invalid or genuinely terminal')
            if any(name not in fs for name in n.visible_facts):
                raise ValueError('Visible fact is absent from the native fact table')
            for name in n.visible_facts:
                fact = fs[name]
                if fact.role in ('target', 'evaluation'):
                    raise ValueError('Targets/evaluation fields cannot enter prefix policy inputs')
                if fact.origin_node is not None and fact.origin_node not in dom[n.key]:
                    raise ValueError('Future or sibling fact is not available on every incoming path')
                if not set(fact.depends_on) <= set(n.visible_facts):
                    raise ValueError('Derived fact hides an unavailable dependency')
        # Dependencies must be acyclic and available before production, not merely
        # present at a later state after the outcome has been observed.
        visiting, done = set(), set()
        def check_fact(name):
            if name in visiting: raise ValueError('Cyclic fact justification')
            if name in done: return
            visiting.add(name); fact = fs[name]
            if fact.origin_node is not None and fact.origin_node not in ns:
                raise ValueError('Fact names an unknown production state')
            for dep in fact.depends_on:
                if dep not in fs or fs[dep].role in ('target','evaluation'):
                    raise ValueError('Unknown or target-derived information used as justification')
                check_fact(dep)
                if fact.role == 'derived':
                    parents = incoming[fact.origin_node]
                    if not parents or any(dep not in ns[p].visible_facts for p in parents):
                        raise ValueError('Derived fact dependency was not available before its step')
            visiting.remove(name); done.add(name)
        for name in fs: check_fact(name)
        self.directed_complex()  # validate every supplied relation square, no auto filling

    def directed_complex(self):
        return DirectedComplex(tuple(n.key for n in self.nodes),
            tuple(Arrow(e.key,e.source,e.target,e.action,1.,e.evidence_sha256) for e in self.steps),
            self.squares, 'computational-step-count')

    def maximal_paths(self, *, max_paths=4096):
        if type(max_paths) is not int or not 1 <= max_paths <= 65536:
            raise ValueError('Explicit finite complete-path budget required')
        ns={n.key:n for n in self.nodes}; incoming={e.target for e in self.steps}
        root=next(n.key for n in self.nodes if n.key not in incoming)
        outgoing={n.key:[] for n in self.nodes}
        for e in self.steps: outgoing[e.source].append(e)
        result=[]; stack=[(root,())]
        while stack:
            key,path=stack.pop()
            if not outgoing[key]:
                if len(result)>=max_paths: raise ValueError('Complete trajectory path budget exceeded; never truncate branches')
                result.append((path,ns[key].status)); continue
            for e in reversed(outgoing[key]): stack.append((e.target,path+(e.key,)))
            if len(stack)+len(result)>max_paths+len(self.steps):
                raise ValueError('Complete trajectory frontier exceeds budget')
        return tuple(result)

    def complete_paths(self, *, max_paths=4096):
        return tuple(path for path,status in self.maximal_paths(max_paths=max_paths) if status=='terminal')


def _path_nodes(trace, path):
    if type(trace) is not ReasoningTrace or type(path) is not tuple:
        raise ValueError('Exact trace and immutable prefix path required')
    trace.__post_init__(); incoming={e.target for e in trace.steps}
    root=next(n.key for n in trace.nodes if n.key not in incoming)
    graph=trace.directed_complex(); graph.endpoint(root,path)
    lookup={e.key:e for e in trace.steps}
    return (root,)+tuple(lookup[key].target for key in path)


def prefix_record(trace, path):
    """Serialize policy inputs from this prefix alone, never terminal gold or siblings."""
    keys=_path_nodes(trace,path); ns={n.key:n for n in trace.nodes}; es={e.key:e for e in trace.steps}
    node=ns[keys[-1]]; fs={f.name:f for f in trace.facts}
    state=dict(layer=node.layer,actions=[es[k].action for k in path],
        direction_kinds=[es[k].direction_kind for k in path],
        step_dependencies=[es[k].requires for k in path],
        facts=[dict(name=name,role=fs[name].role,value=_json(fs[name].value_json),
                    depends_on=fs[name].depends_on,evidence_kind=fs[name].evidence_kind) for name in node.visible_facts])
    return dict(schema='Mora3.ReasoningPrefix/v1',state=state,
        condition_sha256=sha256(_canonical(state)).hexdigest(),
        context_sha256=trace.context_sha256,source_sha256=trace.source_sha256,
        original_split=trace.split,groups=trace.groups,
        direction_evidence=[es[k].direction_kind for k in path],
        biological_causality_verified=False,training_admission=False)


def prefix_attention_mask(trace):
    """Dominance mask: identity/ancestors common to every history, not siblings.

    This prepares a mask; callers must actually use it in a compatible sequence
    attention path. It does not retrofit a dense native graph encoder by metadata.
    """
    if type(trace) is not ReasoningTrace: raise ValueError('Exact native trace required')
    trace.__post_init__(); dom={}; incoming={n.key:[] for n in trace.nodes}
    for e in trace.steps:incoming[e.target].append(e.source)
    for n in sorted(trace.nodes,key=lambda n:n.layer):
        parents=incoming[n.key]
        dom[n.key]=({n.key} if not parents else set.intersection(*(dom[p] for p in parents))|{n.key})
    return tuple(tuple(k.key in dom[n.key] for k in trace.nodes) for n in trace.nodes)


def check_trace_partitions(traces):
    if type(traces) not in (tuple,list) or not traces or len(traces)>65536:
        raise ValueError('Nonempty bounded native trace collection required')
    ids=set(); groups={}; policies=set()
    for t in traces:
        if type(t) is not ReasoningTrace:raise ValueError('Exact native trace required')
        t.__post_init__()
        if t.identity in ids:raise ValueError('Duplicate trajectory identity')
        ids.add(t.identity);policies.add(t.group_policy_sha256)
        for group in t.groups:
            if group in groups and groups[group]!=t.split:raise ValueError('Connected reasoning-problem group crosses original splits')
            groups[group]=t.split
    if len(policies)!=1:raise ValueError('Incompatible leakage grouping policies')
    return dict(traces=len(traces),groups=len(groups),cross_split_groups=0,
        group_completeness_verified=False,source_rights_admitted=False)


def verify_trajectory_analogy(source,target,functor,*,max_paths=4096,max_states=20000):
    """Verify every complete source path and intermediate map, not just endpoints.

    Source and target presentations may differ in path length: each source step
    maps to an explicit target subtrajectory. All required source branches are
    checked. This is not an automatic discovery or biological validation oracle.
    """
    if type(source) is not ReasoningTrace or type(target) is not ReasoningTrace or type(functor) is not DirectedFunctor:
        raise ValueError('Native complete source/target traces and explicit path functor required')
    source.__post_init__();target.__post_init__()
    if source.split!='train' or source.group_policy_sha256!=target.group_policy_sha256 or set(source.groups)&set(target.groups):
        raise ValueError('Analogy exemplar must be train-only and group-disjoint under the same policy')
    if functor.source!=source.directed_complex() or functor.target!=target.directed_complex():
        raise ValueError('Functor does not bind these complete native trajectory presentations')
    source_paths=source.complete_paths(max_paths=max_paths);target.maximal_paths(max_paths=max_paths)
    if not source_paths:raise ValueError('No complete source reasoning trajectory')
    incoming_s={e.target for e in source.steps};incoming_t={e.target for e in target.steps}
    root_s=next(n.key for n in source.nodes if n.key not in incoming_s)
    root_t=next(n.key for n in target.nodes if n.key not in incoming_t)
    if functor.objects[root_s]!=root_t:raise ValueError('Analogy must start at the corresponding complete problem root')
    targets={n.key:n for n in target.nodes};ssteps={e.key:e for e in source.steps};tsteps={e.key:e for e in target.steps}
    mapped=[]
    for path in source_paths:
        target_path=functor.map_path(root_s,path)
        endpoint=functor.target.endpoint(root_t,target_path)
        if targets[endpoint].status!='terminal':raise ValueError('Complete source solution mapped to a censored/incomplete target')
        for key in path:
            if not functor.arrows[key]:raise ValueError('A reasoning step cannot silently disappear in a trajectory analogy')
            # A computational analogy never licenses a causal evidence upgrade.
            for image in functor.arrows[key]:
                if tsteps[image].direction_kind=='interventional' and ssteps[key].direction_kind!='interventional':
                    raise ValueError('Analogy cannot upgrade procedural ordering into intervention evidence')
        mapped.append(dict(source_path=path,target_path=target_path))
    proof=functor.verify(max_states=max_states)
    return dict(status='proved-in-declared-presentations' if proof['status']=='proved' else proof['status'],
        checked_complete_source_paths=len(source_paths),checked_source_steps=len(source.steps),
        mapped_trajectories=tuple(mapped),endpoint_only_comparison=False,
        source_censored_paths=sum(s=='censored'for _,s in source.maximal_paths(max_paths=max_paths)),
        biological_causality_transferred=False,source_target_semantic_alignment_validated=False,
        training_admission=False)


def prefix_native_graph(trace,path):
    """Prepare the entire allowed prefix as native typed graph/task inputs.

    Facts retain their value JSON (no unit or modality inference). Physical source
    geometry belongs to its separate native geometry adapter. An unavailable
    next-state request supplies the required task target but contains no label.
    """
    from mora3.data.schema import GraphRecord,Node,Edge,Measurement
    from mora3.data.provenance import Provenance
    from mora3.data.tasks import ConditionalTask
    keys=_path_nodes(trace,path);ns={n.key:n for n in trace.nodes};fs={f.name:f for f in trace.facts}
    es={e.key:e for e in trace.steps};nodes=[];roles=[]
    for i,key in enumerate(keys):
        fields=[Measurement('reasoning_layer',ns[key].layer,'reasoning-step','observed',True)]
        node_roles=[(('node',i,'reasoning_layer'),'condition')]
        for name in ns[key].visible_facts:
            fact=fs[name];field='fact:'+name
            status='observed'if fact.evidence_kind=='observed'else'predicted'
            fields.extend((Measurement(field,fact.value_json,None,status,True),
                           Measurement('evidence:'+name,fact.evidence_kind,None,'observed',True),
                           Measurement('depends_on:'+name,_canonical(fact.depends_on).decode(),None,'observed',True)))
            node_roles.extend(((('node',i,field),'desired'if fact.role=='desired'else'condition'),
                               (('node',i,'evidence:'+name),'condition'),
                               (('node',i,'depends_on:'+name),'condition')))
        if i==len(keys)-1:
            fields.append(Measurement('requested_next_state',None,None,'missing',False))
            node_roles.append((('node',i,'requested_next_state'),'target'))
        nodes.append(Node('reasoning-prefix-state',tuple(fields)));roles.extend(node_roles)
    edges=[]
    for i,key in enumerate(path):
        e=es[key]
        edges.append(Edge(i,i+1,e.action,True,(Measurement('direction_evidence',e.direction_kind,None,'observed',True),
            Measurement('requires',_canonical(e.requires).decode(),None,'observed',True))))
        roles.extend(((('edge',i,'direction_evidence'),'condition'),(('edge',i,'requires'),'condition')))
    receipt=prefix_record(trace,path)
    graph=GraphRecord('prefix:'+receipt['condition_sha256'],tuple(nodes),tuple(edges),
        (Provenance('native-reasoning-prefix','v1',trace.identity,trace.source_sha256,'source-rights-not-granted-by-adapter'),))
    return graph,ConditionalTask(tuple(roles),topology_known=True),dict(receipt,
        native_nodes=len(nodes),native_edges=len(edges),observed_biological_trajectory=False)
