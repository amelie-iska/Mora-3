"""Canonical finite construction of ordered rooted forests.

ADD_CHILD appends one vertex to the current DFS stack; CLOSE completes its top.
The condition-root list is given, never inferred from hidden target topology.
Every admissible ordered forest has one construction trace, of length 2*N-R.
Finite support plus normalized action kernels defines a normalized forest law;
this is not a measure on unlabelled graph isomorphism classes or biological causes.
"""
from dataclasses import dataclass

CLOSE = 0
ADD_CHILD = 1


@dataclass(frozen=True)
class TopologyBounds:
    max_nodes: int = 128
    max_depth: int = 127
    max_children: int = 127

    def __post_init__(self):
        if (type(self.max_nodes) is not int or not 1 <= self.max_nodes <= 512
                or type(self.max_depth) is not int or not 0 <= self.max_depth < self.max_nodes
                or type(self.max_children) is not int or not 0 <= self.max_children < self.max_nodes):
            raise ValueError('Exact finite node, depth and child support bounds required')


@dataclass(frozen=True)
class TopologyPrefix:
    root_count: int
    actions: tuple[int, ...]
    parents: tuple[int, ...]
    depths: tuple[int, ...]
    children: tuple[int, ...]
    closed: tuple[int, ...]
    stack: tuple[int, ...]
    completed_roots: int
    allowed_actions: tuple[int, ...]

    @property
    def complete(self):
        return self.completed_roots == self.root_count


@dataclass(frozen=True)
class CanonicalForest:
    actions: tuple[int, ...]
    parents: tuple[int, ...]
    canonical_to_original: tuple[int, ...]
    root_count: int


def prefix_state(bounds, root_count, actions):
    """Replay an exact prefix; no tensors, source values, randomness or side effects."""
    if type(bounds) is not TopologyBounds:
        raise ValueError('Exact topology support specification required')
    bounds.__post_init__()
    if (type(root_count) is not int or not 1 <= root_count <= bounds.max_nodes
            or type(actions) is not tuple or len(actions) > 2*bounds.max_nodes-root_count
            or any(type(a) is not int or a not in (CLOSE, ADD_CHILD) for a in actions)):
        raise ValueError('Exact given root count and bounded action prefix required')
    parents=[-1]*root_count;depths=[0]*root_count;children=[0]*root_count
    closed=[];stack=[0];done=0

    def allowed():
        if done == root_count:
            return ()
        top=stack[-1]
        can_add=(len(parents) < bounds.max_nodes and depths[top] < bounds.max_depth
                 and children[top] < bounds.max_children)
        return (CLOSE, ADD_CHILD) if can_add else (CLOSE,)

    for action in actions:
        if action not in allowed():
            raise ValueError('Invalid action, exceeded support, or action after completion')
        if action == ADD_CHILD:
            parent=stack[-1];children[parent]+=1
            parents.append(parent);depths.append(depths[parent]+1);children.append(0)
            stack.append(len(parents)-1)
        else:
            closed.append(stack.pop())
            if not stack:
                done+=1
                if done < root_count:
                    stack.append(done)
    return TopologyPrefix(root_count,actions,tuple(parents),tuple(depths),tuple(children),
        tuple(closed),tuple(stack),done,allowed())


def maximum_forest_nodes(bounds, root_count):
    """Exact attainable node maximum for the full declared ordered support.

    R roots have at most R*C**j vertices at depth j. Summing through D and
    capping at M is attainable by a complete bounded-branch tree followed by
    leaf removal. Saturation keeps every intermediate integer bounded by M.
    """
    prefix_state(bounds,root_count,())
    total=frontier=root_count
    for _ in range(bounds.max_depth):
        if total == bounds.max_nodes or frontier == 0:
            break
        frontier=min(bounds.max_nodes-total,frontier*bounds.max_children)
        total+=frontier
    return total


def canonical_forest_actions(bounds, parents, *, root_count):
    """Keep every node; preserve source root/sibling order under a DFS reindexing.

    The caller supplies the known number of condition roots independently of
    targets. The returned bijection permits exact alignment of all source payloads.
    Arbitrary merged DAGs, cycles, unknown future parents or overflow are refused.
    """
    prefix_state(bounds,root_count,())
    if (type(parents) is not tuple or not parents or len(parents) > bounds.max_nodes
            or any(type(p) is not int or not -1 <= p < i for i,p in enumerate(parents))):
        raise ValueError('Complete bounded topologically ordered original forest required')
    roots=tuple(i for i,p in enumerate(parents) if p == -1)
    if len(roots) != root_count:
        raise ValueError('Source root count differs from the given condition-root contract')
    adjacency=[[] for _ in parents]
    for i,p in enumerate(parents):
        if p >= 0:
            adjacency[p].append(i)
    order=list(roots);actions=[]
    # Iterative DFS avoids recursion limits even at the supported 512-node cap.
    for root in roots:
        frames=[(root,0)]
        while frames:
            vertex,next_child=frames[-1]
            if next_child < len(adjacency[vertex]):
                child=adjacency[vertex][next_child]
                frames[-1]=(vertex,next_child+1);frames.append((child,0))
                order.append(child);actions.append(ADD_CHILD)
            else:
                frames.pop();actions.append(CLOSE)
    state=prefix_state(bounds,root_count,tuple(actions))
    if not state.complete or sorted(order) != list(range(len(parents))):
        raise ValueError('Complete original-to-canonical node bijection required')
    inverse={old:new for new,old in enumerate(order)}
    expected=tuple(-1 if parents[old] == -1 else inverse[parents[old]] for old in order)
    if state.parents != expected:
        raise ValueError('Canonical source edge conservation failed')
    return CanonicalForest(tuple(actions),state.parents,tuple(order),root_count)
