"""Small independent optima catch missing flow, capacity and budget constraints."""
import importlib
import itertools
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from mora3.tokenization.shortest_path import ByteTokenizer


def component():
    assert Path('src/mora3/tokenization/convex_vocab.py').exists(), 'LP learner missing'
    return importlib.import_module('mora3.tokenization.convex_vocab')


@pytest.mark.parametrize('budget', [0, 1, 2, 3])
def test_relaxation_brackets_exhaustive_integer_optimum(budget):
    corpus = (b'ababa', b'baaab', b'ababa')
    candidates = (b'ab', b'ba', b'aa')
    r = component().fit_vocabulary(corpus, candidates, vocab_size=256 + budget)
    optimum = min(sum(len(ByteTokenizer(v).encode(s)) for s in corpus)
                  for k in range(budget + 1)
                  for v in itertools.combinations(candidates, k))
    assert r.lp_objective <= optimum + 1e-7
    assert r.rounded_objective >= optimum
    assert r.max_constraint_violation < 1e-7
    assert abs(r.lp_objective - r.dual_objective) < 1e-7
    assert len(r.learned) <= budget


def test_shared_colour_capacity_is_per_occurrence_not_sum():
    r = component().fit_vocabulary((b'abab', b'ab'), (b'ab',), vocab_size=257)
    assert r.lp_objective == pytest.approx(3)
    assert r.learned == (b'ab',)


def test_corpus_frequency_changes_optimal_choice_and_order_is_stable():
    corpus = (b'ab', b'ab', b'cd')
    r = component().fit_vocabulary(corpus, (b'cd', b'ab'), vocab_size=258, controls=('pad',))
    assert r.learned == (b'ab',)
    assert r.rounded_objective == 4
    reverse = component().fit_vocabulary(corpus[::-1], (b'ab', b'cd'), vocab_size=258, controls=('pad',))
    assert r == reverse


def test_candidates_do_not_cross_field_boundaries():
    assert component().substring_candidates((b'ab', b'cd'), max_length=3) == (b'ab', b'cd')


def test_paper_rounding_with_deterministic_byte_ties():
    m = component()
    scores = {b'long': .8, b'aa': .6, b'bb': .6, b'zz': 1.0}
    assert m.round_vocabulary(scores, 2, method='det') == (b'long', b'zz')
    assert m.round_vocabulary(scores, 2, method='bias') == (b'aa', b'zz')
    assert m.round_vocabulary(scores, 2, method='int') == (b'zz',)


@pytest.mark.parametrize('kwargs', [dict(vocab_size=255), dict(vocab_size=True),
    dict(vocab_size=256, controls=('pad',)), dict(vocab_size=258, controls=('x', 'x'))])
def test_invalid_reserved_budget(kwargs):
    with pytest.raises(ValueError):
        component().fit_vocabulary((b'ab',), (b'ab',), **kwargs)


def test_no_candidates_or_empty_fields_keep_byte_fallback():
    r = component().fit_vocabulary((b'', b'abc'), (), vocab_size=256)
    assert r.lp_objective == 3
    assert r.rounded_objective == 3
    with pytest.raises(ValueError):
        component().fit_vocabulary((b'',), (), vocab_size=256)


def test_resource_bounds_fail_explicitly_without_truncating():
    with pytest.raises(ValueError, match='byte'):
        component().fit_vocabulary((b'abcd',), (), vocab_size=256, max_bytes=3)
    with pytest.raises(ValueError, match='edge'):
        component().fit_vocabulary((b'abcd',), (), vocab_size=256, max_edges=3)
    with pytest.raises(ValueError, match='candidate'):
        component().substring_candidates((b'abcd',), max_length=3, max_candidates=1)


@pytest.mark.parametrize('candidates', [(b'a',), (b'ab', b'ab'), ('ab',)])
def test_invalid_candidates(candidates):
    with pytest.raises(ValueError):
        component().fit_vocabulary((b'ab',), candidates, vocab_size=257)


def test_fractional_optimum_is_not_claimed_as_integer_optimum():
    corpus = (b'aaabb', b'babab', b'bbbab')
    candidates = component().substring_candidates(corpus, max_length=3)
    r = component().fit_vocabulary(corpus, candidates, vocab_size=258)
    integer_opt = min(sum(len(ByteTokenizer(v).encode(s)) for s in corpus)
                      for v in itertools.combinations(candidates, 2))
    assert r.lp_objective == pytest.approx(8)
    assert integer_opt == 9
    assert r.rounded_objective >= 9
    assert any(0 < score < 1 for _, score in r.scores)



def test_explicit_interior_point_keeps_fractional_certificate_and_integer_bounds():
    corpus=(b'aaabb',b'babab',b'bbbab')
    candidates=component().substring_candidates(corpus,max_length=3)
    fits=[component().fit_vocabulary(corpus,candidates,vocab_size=258,lp_method=method)
          for method in ('highs-ds','highs-ipm')]
    integer_optimum=min(sum(len(ByteTokenizer(v).encode(s))for s in corpus)
        for v in itertools.combinations(candidates,2))
    assert integer_optimum==9
    for fit in fits:
        assert fit.lp_objective==pytest.approx(8.)
        assert fit.lp_objective==pytest.approx(fit.dual_objective)
        assert fit.max_constraint_violation<1e-7 and fit.dual_residual<1e-7
        assert fit.rounded_objective>=integer_optimum
        assert any(0<score<1 for _,score in fit.scores)


@pytest.mark.parametrize('method',[None,True,'simplex','unknown'])
def test_unknown_lp_method_refuses_explicitly(method):
    with pytest.raises(ValueError,match='LP method'):
        component().fit_vocabulary((b'ab',),(b'ab',),vocab_size=257,lp_method=method)
