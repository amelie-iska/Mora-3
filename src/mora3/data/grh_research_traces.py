"""Pinned GRH numerical DAGs as complete native computation-reasoning references.

A replay of a trained checkpoint is still computed evidence, not new training,
biological data, a proof of GRH, or an on-policy GFlowNet sample. These imports
are evaluation-only. A whole execution schedule and an ancestor-only operation
prefix are explicitly different views of the retained dependency DAG.
"""
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import json
import math
import re

REVISION = 'dbd954eede479a5129831e4e7c30537389ae973d'
MANIFEST_SHA256 = 'f4d2e685a423cfb0bc99c29019f0edabb3b26157a50309e5b9706f953d1e4904'
KINDS = ('trained-checkpoint-replay', 'fixed-weight-reference', 'computed-carrier')


def _wire(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf8')


def _text(value):
    if type(value) is not str or not value.strip() or len(value) > 4096:
        raise ValueError('Bounded nonempty UTF-8 source text required')
    try:
        value.encode('utf8')
    except UnicodeError as error:
        raise ValueError('Invalid UTF-8 source text') from error


def _digest(value, length):
    if type(value) is not str or re.fullmatch('[0-9a-f]{'+str(length)+'}', value) is None:
        raise ValueError('Exact source digest or revision required')


def _json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate source JSON key')
            result[key] = value
        return result
    def refuse(_):
        raise ValueError('Nonfinite source JSON')
    try:
        value = json.loads(raw, object_pairs_hook=unique, parse_constant=refuse)
        _wire(value)  # Reject overflowed exponents in any metadata as well.
        return value
    except (UnicodeError, TypeError, RecursionError, OverflowError) as error:
        raise ValueError('Finite bounded native JSON required') from error


def _tensor(tensor, maximum):
    if type(tensor) is not dict:
        raise ValueError('Exact tensor object required')
    shape, axes, dtype = tensor.get('shape'), tensor.get('axes'), tensor.get('dtype')
    if (type(shape) is not list or len(shape) > 8
            or any(type(n) is not int or not 1 <= n <= 1_000_000 for n in shape)
            or type(axes) is not list or len(axes) != len(shape)):
        raise ValueError('Complete explicit tensor shape and axes required')
    for axis in axes:
        _text(axis)
    if dtype not in ('bool', 'int64', 'float32', 'float64', 'complex64', 'complex128'):
        raise ValueError('Unsupported tensor dtype')
    complex_value = dtype.startswith('complex')
    if complex_value:
        if 'real' not in tensor or 'imag' not in tensor or 'values' in tensor:
            raise ValueError('Both full complex parts required without an alternative value field')
        parts = (tensor['real'], tensor['imag'])
    else:
        if ('values' in tensor) == ('real' in tensor) or 'imag' in tensor:
            raise ValueError('One unambiguous real/boolean/integer tensor field required')
        parts = (tensor.get('values', tensor.get('real')),)
    count = math.prod(shape) * len(parts)
    if count > maximum:
        raise ValueError('Complete tensor exceeds element budget; never truncate')

    def check(value, dimensions):
        if dimensions:
            if type(value) is not list or len(value) != dimensions[0]:
                raise ValueError('Tensor dimensions disagree with native values')
            for child in value:
                check(child, dimensions[1:])
        elif dtype == 'bool':
            if type(value) is not bool:
                raise ValueError('Boolean tensor requires boolean values')
        elif dtype == 'int64':
            if type(value) is not int or not -(2**63) <= value < 2**63:
                raise ValueError('Exact int64 tensor values required')
        else:
            limit = 3.4028234663852886e38 if dtype in ('float32', 'complex64') else 1.7976931348623157e308
            if type(value) not in (int, float) or abs(value) > limit or not math.isfinite(value):
                raise ValueError('Finite dtype-representable real components required; booleans are not scalars')
    for part in parts:
        check(part, shape)
    return count


@dataclass(frozen=True)
class GRHNumericTrace:
    raw: bytes
    sha256: str
    revision: str
    max_bytes: int = 16 * 2**20
    max_nodes: int = 512
    max_elements: int = 2_000_000

    def __post_init__(self):
        _digest(self.sha256, 64)
        _digest(self.revision, 40)
        if (any(type(n) is not int or n < 1 for n in (self.max_bytes, self.max_nodes, self.max_elements))
                or self.max_bytes > 64 * 2**20 or self.max_nodes > 4096 or self.max_elements > 8_000_000
                or type(self.raw) is not bytes or not 0 < len(self.raw) <= self.max_bytes
                or sha256(self.raw).hexdigest() != self.sha256):
            raise ValueError('Exact immutable source bytes and finite budgets required')
        data = _json(self.raw)
        if (type(data) is not dict or type(data.get('schema_version')) is not int
                or data['schema_version'] != 1 or type(data.get('provenance')) is not dict
                or data['provenance'].get('kind') not in KINDS
                or type(data.get('nodes')) is not list or not 1 <= len(data['nodes']) <= self.max_nodes):
            raise ValueError('Declared complete GRH numerical trace schema required')
        _text(data.get('id'))
        seen = set()
        count = 0
        for node in data['nodes']:
            if type(node) is not dict:
                raise ValueError('Exact source computation node required')
            key, inputs = node.get('id'), node.get('inputs')
            _text(key)
            if (key in seen or type(inputs) is not list
                    or any(type(x) is not str for x in inputs) or len(set(inputs)) != len(inputs)
                    or not set(inputs) <= seen):
                raise ValueError('Unique source nodes and prior acyclic dependencies required')
            _text(node.get('formula'))
            count += _tensor(node.get('tensor'), self.max_elements - count)
            seen.add(key)
        checkpoint = data['provenance'].get('checkpoint')
        if checkpoint is not None:
            if type(checkpoint) is not dict:
                raise ValueError('Source checkpoint identity must be explicit')
            _digest(checkpoint.get('sha256'), 64)

    def document(self):
        """Fresh mutable view; modifying it never changes the retained source."""
        self.__post_init__()
        return _json(self.raw)

    @property
    def training_admitted(self):
        return False

    def require_training_admission(self):
        raise ValueError('GRH numerical references are evaluation-only, not campaign training admission')


def _read(root, relative, maximum):
    root = Path(root).absolute()
    rel = Path(relative)
    if rel.is_absolute() or '..' in rel.parts:
        raise ValueError('Exact relative source path required')
    path = root / rel
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
        raise ValueError('Original regular source file required; no symlink traversal')
    if path.stat().st_size > maximum:
        raise ValueError('Source file exceeds declared byte budget')
    with path.open('rb') as stream:
        raw = stream.read(maximum + 1)
    if len(raw) > maximum:
        raise ValueError('Source file grew beyond its budget')
    return raw


def load_pinned_trace(root, trace_id):
    """Only exact entries of the pinned public numerical manifest, no repository code."""
    _text(trace_id)
    raw = _read(root, 'manifest.json', 2**20)
    if sha256(raw).hexdigest() != MANIFEST_SHA256:
        raise ValueError('Pinned GRH numerical manifest changed')
    manifest = _json(raw)
    entries = [row for row in manifest['traces'] if row['id'] == trace_id]
    if len(entries) != 1:
        raise ValueError('Exactly one original manifest trace required')
    entry = entries[0]
    source = GRHNumericTrace(_read(root, entry['file'], 16*2**20), entry['sha256'], REVISION)
    data = source.document()
    if (data['id'] != entry['id'] or len(data['nodes']) != entry['nodes']
            or data['provenance']['kind'] != entry['kind'] or len(data.get('checks', {})) != entry['checks']):
        raise ValueError('Actual source trace differs from pinned manifest identity')
    return source


def as_reasoning_trace(source, *, target_node_id=None):
    """Complete known-program execution schedule, or a target's entire ancestor DAG.

    Execution-order states are not source dataflow vertices or searched FoT
    branches. Original dataflow edges remain fact dependencies. No source tensor
    is visible before its production step. The ancestor-only mode additionally
    excludes unrelated siblings, weights and output heads.
    """
    from mora3.reasoning.trajectory_contracts import TraceFact, TraceNode, TraceStep, ReasoningTrace
    if type(source) is not GRHNumericTrace:
        raise ValueError('Exact pinned source trace required')
    data = source.document()
    original = data['nodes']
    lookup = {n['id']: n for n in original}
    if target_node_id is not None:
        _text(target_node_id)
        if target_node_id not in lookup or not lookup[target_node_id]['inputs']:
            raise ValueError('Known derived target operation required')
        selected = {target_node_id}
        frontier = [target_node_id]
        while frontier:
            for parent in lookup[frontier.pop()]['inputs']:
                if parent not in selected:
                    selected.add(parent)
                    frontier.append(parent)
        original = [n for n in original if n['id'] in selected]
    derived = [n for n in original if n['inputs']]
    if len(derived) > 64:
        raise ValueError('Complete computation exceeds native reasoning depth; no truncated trace')
    program = dict(semantics='given-numerical-program-not-unknown-target-topology',
        operations=[dict(id=n['id'], inputs=n['inputs'], formula=n['formula']) for n in original])
    program_raw = _wire(program)
    facts = [TraceFact('program', program_raw.decode(), 'condition', None, (), 'computed', sha256(program_raw).hexdigest())]
    visible = ['program']
    for node in original:
        if not node['inputs']:
            value = _wire(node['tensor'])
            facts.append(TraceFact('tensor:'+node['id'], value.decode(), 'condition', None, (), 'computed', sha256(value).hexdigest()))
            visible.append('tensor:'+node['id'])
    context = sha256(_wire([dict(name=f.name, value=_json(f.value_json)) for f in facts])).hexdigest()
    nodes = [TraceNode('execution:0', 0, tuple(visible), 'open' if derived else 'terminal')]
    steps = []
    for i, node in enumerate(derived, 1):
        key = 'execution:'+str(i)
        value = _wire(node['tensor'])
        dependencies = tuple('tensor:'+p for p in node['inputs'])
        facts.append(TraceFact('tensor:'+node['id'], value.decode(), 'derived', key, dependencies,
            'computed', sha256(value).hexdigest()))
        steps.append(TraceStep('operation:'+str(i), nodes[-1].key, key, node['formula'],
            'computational', ('program',)+dependencies, sha256(_wire(node)).hexdigest()))
        visible.append('tensor:'+node['id'])
        nodes.append(TraceNode(key, i, tuple(visible), 'terminal' if i == len(derived) else 'open'))
    groups = ['GRH-research-revision:'+source.revision]
    checkpoint = data['provenance'].get('checkpoint')
    if checkpoint:
        groups.append('GRH-checkpoint:'+checkpoint['sha256'])
    return ReasoningTrace('GRH-reference:'+data['id']+(':'+target_node_id if target_node_id else ':complete'),
        source.sha256, context, sha256(b'GRH-all-source-revision-reference-only-v1').hexdigest(),
        'test', tuple(groups), tuple(nodes), tuple(steps), tuple(facts), ())


def native_operation_prefix(source, target_node_id):
    """Existing native GraphRecord/ConditionalTask for a whole ancestor-only prefix.

    The requested output tensor is absent. No source program is executed and no
    tensor/checkpoint is evaluated as code. This is a given-program reference task.
    """
    from mora3.reasoning.trajectory_contracts import prefix_native_graph
    trace = as_reasoning_trace(source, target_node_id=target_node_id)
    path = tuple(step.key for step in trace.steps[:-1])
    graph, task, evidence = prefix_native_graph(trace, path)
    return graph, task, dict(evidence, original_trace_sha256=source.sha256,
        target_operation=target_node_id, scope='entire-ancestor-subgraph-before-target',
        source_reference_only=True, new_training_run=False)


def verify_tropical_reference(source, *, tolerance=1e-12):
    """Independent full masked max-plus reduction and all-tie winner verification."""
    import numpy as np
    if type(source) is not GRHNumericTrace or type(tolerance) not in (int, float) or not 0 <= tolerance <= 1e-8:
        raise ValueError('Exact source and bounded explicit comparison tolerance required')
    data = source.document()
    if data.get('category') not in ('classical-tropical-attention', 'classical-tropical-ring'):
        raise ValueError('Declared tropical reference required, not ordinary/hybrid attention')
    nodes = {n['id']: n['tensor'] for n in data['nodes']}
    def array(name, dtype):
        value = nodes[name]
        if value['dtype'] != dtype:
            raise ValueError('Original tensor dtype changed')
        return np.asarray(value.get('values', value.get('real')), dtype=bool if dtype=='bool' else np.float64)
    score, value = array('layer0.scores', 'float64'), array('layer0.value', 'float64')
    legal = array('layer0.legal', 'bool')
    actual = array('layer0.context', 'float64')
    winners = array('layer0.winners', 'bool')
    if (score.ndim != 4 or value.ndim != 4 or legal.shape != score.shape
            or score.shape[:2] != value.shape[:2] or score.shape[-1] != value.shape[-2]
            or actual.shape != (*score.shape[:3], value.shape[-1])
            or winners.shape != (*score.shape, value.shape[-1])):
        raise ValueError('Full source score/value/query/key/context axes required')
    if math.prod(winners.shape) > source.max_elements:
        raise ValueError('Complete reference calculation exceeds allocation budget')
    terms = score[..., None] + value[:, :, None, :, :]
    if not np.isfinite(terms).all():
        raise ValueError('Nonfinite reference sum')
    masked = np.where(legal[..., None], terms, -np.inf)
    maximum = masked.max(axis=-2)
    expected = np.where(legal.any(axis=-1)[..., None], maximum, 0.)
    expected_winners = legal[..., None] & (terms == maximum[..., None, :])
    error = float(np.max(np.abs(actual - expected)))
    return dict(schema='Mora3.GRHIndependentTropicalReference/v1', source_sha256=source.sha256,
        source_revision=source.revision, full_query_key_channel_shape=list(winners.shape),
        context_max_absolute_error=error, context_pass=error <= tolerance,
        winner_mask_exact=bool(np.array_equal(winners, expected_winners)),
        masked_keys_ignored=True, empty_queries_zero=bool(np.all(actual[~legal.any(axis=-1)]==0)),
        declared_source_checks_not_used_as_proof=True, mora_training_performed=False,
        biological_causality_verified=False, training_admission=False)
