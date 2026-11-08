"""Synthetic correctness cases; real coverage is measured separately."""
import importlib
import itertools
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))


def tokenizer(*args, **kwargs):
    assert Path('src/mora3/tokenization/shortest_path.py').exists(), 'shortest-path component missing'
    return importlib.import_module('mora3.tokenization.shortest_path').ByteTokenizer(*args, **kwargs)


@pytest.mark.parametrize('payload', [b'', bytes(range(256)), 'שָׁלוֹם\r\n🙂'.encode(), b'[C][O][Branch1][C]', b'<pad>\x00\xff'])
def test_exact_byte_roundtrip(payload):
    t = tokenizer((b'[C]', b'\xd7\xa9', b'<pad>'), controls=('pad',))
    ids = t.encode(payload)
    assert t.decode(ids) == payload
    assert t.control_id('pad') not in ids


def test_shortest_path_beats_longest_match():
    t = tokenizer((b'ab', b'abc', b'cde'))
    assert t.encode(b'abcde') == (256, 258)


def test_equation_one_against_exhaustive_segmentations():
    vocab = (b'ab', b'ba', b'aba', b'aa')
    t = tokenizer(vocab)
    def all_lengths(data):
        if not data:
            return [0]
        return [1 + n for token in tuple(bytes([b]) for b in range(256)) + vocab
                if data.startswith(token) for n in all_lengths(data[len(token):])]
    total = 0
    oracle = 0
    for n in range(7):
        for chars in itertools.product(b'ab', repeat=n):
            data = bytes(chars)
            ids = t.encode(data)
            assert t.decode(ids) == data
            assert len(ids) == min(all_lengths(data))
            total += len(ids)
            oracle += min(all_lengths(data))
    assert total == oracle


def test_tie_prefers_longer_first_token_and_stable_ids():
    t = tokenizer((b'ab', b'bc'))
    assert t.encode(b'abc') == (256, ord('c'))
    assert t.encode(b'abc') == t.encode(b'abc')
    assert t.vocabulary[:256] == tuple(bytes([i]) for i in range(256))


def test_payload_boundaries_are_not_merged():
    t = tokenizer((b'ab',))
    assert t.encode(b'a') + t.encode(b'b') != t.encode(b'ab')


@pytest.mark.parametrize('learned', [(b'',), (b'a',), (b'ab', b'ab'), ('abc',)])
def test_invalid_learned_vocabulary(learned):
    with pytest.raises(ValueError):
        tokenizer(learned)


@pytest.mark.parametrize('controls', [('pad', 'pad'), ('',), (1,)])
def test_invalid_controls(controls):
    with pytest.raises(ValueError):
        tokenizer(controls=controls)


@pytest.mark.parametrize('ids', [(-1,), (256,), (True,), (1.0,)])
def test_decode_rejects_invalid_or_control_ids(ids):
    t = tokenizer(controls=('pad',))
    with pytest.raises(ValueError):
        t.decode(ids)


def test_control_ids_and_input_validation():
    t = tokenizer((b'ab',), controls=('pad', 'node'))
    assert t.control_id('pad') == 257
    assert t.control_id('node') == 258
    assert t.size == 259
    with pytest.raises(KeyError):
        t.control_id('unknown')
    with pytest.raises(TypeError):
        t.encode('not bytes')
