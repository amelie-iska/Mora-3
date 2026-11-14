"""Byte-exact long-payload token windows for a future explicit continuation decoder.

This is a data representation, NOT an autoregressive neural decoder. A finite
4096-slot learned decoder cannot represent arbitrary full source payloads in one
call. Continuation windows require separately qualified prefix-conditioned neural
execution and training; repeated fixed-state projections are not such execution.
No parameter axis, control vocabulary, source role or training gate changes here.
"""
from dataclasses import dataclass
from hashlib import sha256

from mora3.reasoning.flow_forest_refinement import ForestState, forest_fingerprint
from mora3.tokenization.forest_tokens import ForestTokenAxis

MAX_WINDOW_COUNT = 65536
MAX_TOTAL_TOKENS = 64_000_000
MAX_PAYLOAD_BYTES = 64 * 1024 ** 2


@dataclass(frozen=True)
class ForestTokenWindow:
    node: int
    index: int
    tokens: tuple[int, ...]


@dataclass(frozen=True)
class ForestTokenWindowPlan:
    axis_sha256: str
    forest_sha256: str
    context_sha256: str
    coordinate_sha256: str
    max_slots: int
    max_total_tokens: int
    max_payload_bytes: int
    payload_sha256: tuple[tuple[int, str], ...]
    windows: tuple[ForestTokenWindow, ...]
    training_admission: bool = False


def _bound_inputs(axis, forest, slots, tokens, payload_bytes):
    if type(axis) is not ForestTokenAxis or type(forest) is not ForestState:
        raise ValueError('Exact native forest token axis and state required')
    axis.__post_init__()
    identity = forest_fingerprint(forest)
    if not forest.is_complete:
        raise ValueError('Complete native forest required')
    for value, maximum in ((slots, 4096), (tokens, MAX_TOTAL_TOKENS),
                           (payload_bytes, MAX_PAYLOAD_BYTES)):
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError('Explicit bounded long-output resource budgets required')
    return identity


def compile_forest_token_windows(axis, forest, payloads, *, max_slots,
                                  max_total_tokens=262144,
                                  max_payload_bytes=4 * 1024 ** 2):
    """Canonical exact windows over free-node bytes; no hidden truncation.

    Every nonterminal window has exactly max_slots payload-token IDs. Exactly
    one final window ends in the already registered NODE_END control. A
    multiple-of-max_slots payload therefore needs a separate stop-only window.
    The index is EXTERNAL address metadata, not a learned slot ID; later neural
    continuation must explicitly condition on it and on preceding payloads.
    """
    identity = _bound_inputs(axis, forest, max_slots, max_total_tokens, max_payload_bytes)
    if type(payloads) is not tuple or len(payloads) != len(forest.parents):
        raise ValueError('One explicit optional payload per native node required')
    tok = axis.tokenizer()
    stop = tok.control_id('<MORA:NODE_END>')
    windows = []
    pins = []
    total_bytes = total_tokens = 0
    for node, raw in enumerate(payloads):
        if raw is None:
            continue
        if node in forest.clamped:
            raise ValueError('Known clamped source inputs cannot become token supervision')
        if type(raw) is not bytes:
            raise ValueError('Only exact source bytes or absent labels may be represented')
        total_bytes += len(raw)
        if total_bytes > max_payload_bytes:
            raise ValueError('Complete payload exceeds declared byte budget; no truncation')
        encoded = tuple(tok.encode(raw))
        if total_tokens + len(encoded) + 1 > max_total_tokens:
            raise ValueError('Complete token sequence exceeds declared budget; no truncation')
        count = len(encoded) // max_slots + 1
        if len(windows) + count > MAX_WINDOW_COUNT:
            raise ValueError('Complete window sequence exceeds declared count budget')
        for index in range(count):
            start = index * max_slots
            values = encoded[start:start + max_slots]
            if index == count - 1:
                values = values + (stop,)
            windows.append(ForestTokenWindow(node, index, values))
        pins.append((node, sha256(raw).hexdigest()))
        total_tokens += len(encoded) + 1
    return ForestTokenWindowPlan(axis.sha256, identity, forest.context_sha256,
                                 forest.coordinate_sha256, max_slots, max_total_tokens,
                                 max_payload_bytes, tuple(pins), tuple(windows), False)


def decode_forest_token_windows(axis, forest, plan):
    """Validate structural, source, axis and canonical token identity before readback."""
    if type(plan) is not ForestTokenWindowPlan:
        raise ValueError('Exact complete window representation required')
    identity = _bound_inputs(axis, forest, plan.max_slots,
                             plan.max_total_tokens, plan.max_payload_bytes)
    if (plan.training_admission is not False or plan.axis_sha256 != axis.sha256
            or plan.forest_sha256 != identity
            or plan.context_sha256 != forest.context_sha256
            or plan.coordinate_sha256 != forest.coordinate_sha256
            or type(plan.windows) is not tuple or type(plan.payload_sha256) is not tuple
            or len(plan.windows) > MAX_WINDOW_COUNT):
        raise ValueError('Window source/axis/forest binding mismatch')
    tok = axis.tokenizer()
    stop = tok.control_id('<MORA:NODE_END>')
    grouped = {}
    total = 0
    for window in plan.windows:
        if (type(window) is not ForestTokenWindow
                or type(window.node) is not int
                or not 0 <= window.node < len(forest.parents)
                or window.node in forest.clamped
                or type(window.index) is not int or window.index < 0
                or type(window.tokens) is not tuple
                or not 1 <= len(window.tokens) <= plan.max_slots
                or any(type(t) is not int for t in window.tokens)):
            raise ValueError('Malformed bounded free-node window')
        total += len(window.tokens)
        if total > plan.max_total_tokens:
            raise ValueError('Complete window sequence exceeds token budget')
        rows = grouped.setdefault(window.node, [])
        if window.index != len(rows):
            raise ValueError('Missing, duplicate or reordered continuation window')
        rows.append(window.tokens)
    payloads = [None] * len(forest.parents)
    used_bytes = 0
    for node, rows in grouped.items():
        ids = []
        for n, tokens in enumerate(rows):
            last = n == len(rows) - 1
            if last:
                if tokens[-1] != stop or stop in tokens[:-1]:
                    raise ValueError('Missing or premature terminal stop')
                data = tokens[:-1]
            else:
                if len(tokens) != plan.max_slots or stop in tokens:
                    raise ValueError('Incomplete or prematurely terminated continuation')
                data = tokens
            if any(t < 0 or t >= len(tok.vocabulary) for t in data):
                raise ValueError('Control IDs cannot be injected as literal payload')
            ids.extend(data)
        raw = tok.decode(ids)
        used_bytes += len(raw)
        if used_bytes > plan.max_payload_bytes:
            raise ValueError('Decoded payload exceeds byte budget')
        if tuple(tok.encode(raw)) != tuple(ids):
            raise ValueError('Noncanonical learned-token payload sequence')
        payloads[node] = raw
    output = tuple(payloads)
    canonical = compile_forest_token_windows(
        axis, forest, output, max_slots=plan.max_slots,
        max_total_tokens=plan.max_total_tokens,
        max_payload_bytes=plan.max_payload_bytes)
    if canonical != plan:
        raise ValueError('Window identifiers, digests or segmentation disagree with exact source')
    return output
