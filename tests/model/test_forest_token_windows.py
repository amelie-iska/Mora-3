"""Pure lossless multiwindow label envelope; fixtures are not training records."""
from dataclasses import replace
from hashlib import sha256
import pytest
import torch
from test_forest_flow import skeleton
from test_forest_tokens import base
from mora3.tokenization.forest_tokens import ForestTokenAxis
from mora3.model.forest_token_windows import (
    ForestTokenWindowPlan, compile_forest_token_windows, decode_forest_token_windows,
)


def axis():
    return ForestTokenAxis.from_annotation(base())


def test_long_source_payload_roundtrips_exactly_without_new_token_axis_or_truncation():
    a = axis(); f = skeleton(); data = b'0' * 4096 + bytes(range(256)) * 23 + b'\x00\xff'
    payloads = (None, data, b'', None, b'ends')
    original_rng = torch.get_rng_state().clone()
    plan = compile_forest_token_windows(a, f, payloads, max_slots=4096, max_total_tokens=30000)
    assert type(plan) is ForestTokenWindowPlan and plan.training_admission is False
    assert plan.axis_sha256 == a.sha256 and plan.forest_sha256
    assert all(1 <= len(w.tokens) <= 4096 for w in plan.windows)
    assert len([w for w in plan.windows if w.node == 1]) >= 3
    assert decode_forest_token_windows(a, f, plan) == payloads
    assert torch.equal(original_rng, torch.get_rng_state())


@pytest.mark.parametrize('length', [0, 1, 3, 4, 7, 8, 9, 4096, 8192])
def test_boundary_windows_are_unique_final_stop_only(length):
    a = axis(); f = skeleton(); data = bytes(range(256)) * (length // 256) + bytes(range(length % 256))
    plan = compile_forest_token_windows(a, f, (None, data, None, None, None),
                                        max_slots=4, max_total_tokens=20000)
    assert decode_forest_token_windows(a, f, plan)[1] == data
    rows = [w for w in plan.windows if w.node == 1]
    stop = a.tokenizer().control_id('<MORA:NODE_END>')
    assert [w.index for w in rows] == list(range(len(rows)))
    assert all(len(w.tokens) == 4 and stop not in w.tokens for w in rows[:-1])
    assert rows[-1].tokens[-1] == stop and stop not in rows[-1].tokens[:-1]


def test_refuses_clamped_supervision_and_budget_overflow_without_partial_output():
    a = axis(); f = skeleton()
    with pytest.raises(ValueError, match='clamped'):
        compile_forest_token_windows(a, f, (b'leak', None, None, None, None), max_slots=4)
    with pytest.raises(ValueError, match='budget'):
        compile_forest_token_windows(a, f, (None, b'abcdef', None, None, None),
                                     max_slots=4, max_total_tokens=4)
    with pytest.raises(ValueError):
        compile_forest_token_windows(a, f, (None, b'ok', None, None, None), max_slots=True)
    with pytest.raises(ValueError):
        compile_forest_token_windows(a, f, (None, b'ok', None, None, None),
                                     max_slots=4, max_payload_bytes=1)


@pytest.mark.parametrize('tamper', ['missing', 'index', 'control', 'early-stop', 'digest', 'slot', 'axis'])
def test_readback_rejects_tampering_and_noncanonical_or_incomplete_segments(tamper):
    a = axis(); f = skeleton()
    p = compile_forest_token_windows(a, f, (None, b'abcdefghij', None, None, None), max_slots=4)
    if tamper == 'missing':
        p = replace(p, windows=p.windows[:-1])
    elif tamper == 'index':
        p = replace(p, windows=(replace(p.windows[0], index=1),) + p.windows[1:])
    elif tamper == 'control':
        p = replace(p, windows=(replace(p.windows[0], tokens=(a.tokenizer().control_id('<MORA:PASS>'),) + p.windows[0].tokens[1:]),) + p.windows[1:])
    elif tamper == 'early-stop':
        p = replace(p, windows=(replace(p.windows[0], tokens=p.windows[0].tokens[:-1] + (a.tokenizer().control_id('<MORA:NODE_END>'),)),) + p.windows[1:])
    elif tamper == 'digest':
        p = replace(p, payload_sha256=((1, '0'*64),))
    elif tamper == 'slot':
        p = replace(p, max_slots=2)
    elif tamper == 'axis':
        p = replace(p, axis_sha256='0'*64)
    with pytest.raises(ValueError):
        decode_forest_token_windows(a, f, p)


def test_source_template_drift_and_unknown_free_node_do_not_silently_bind():
    a = axis(); f = skeleton()
    p = compile_forest_token_windows(a, f, (None, b'a', None, None, None), max_slots=4)
    with pytest.raises(ValueError):
        decode_forest_token_windows(a, replace(f, directions=('root','temporal','computational','computational','computational')), p)
    assert decode_forest_token_windows(a, f, p) == (None, b'a', None, None, None)
