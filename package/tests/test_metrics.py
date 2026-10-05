import math

import pytest

from stabilization_uplift import stabilization_score, stabilization_uplift


def test_score_is_one_without_auc_change():
    assert stabilization_score(0.8, 0.8, 0.3) == pytest.approx(1.0)


def test_score_formula():
    expected = 1 - 0.1 / (1 + math.log(1 + 0.2 + 1e-5))
    assert stabilization_score(0.8, 0.7, 0.2) == pytest.approx(expected)


def test_score_is_symmetric_in_auc_change():
    assert stabilization_score(0.8, 0.7, 0.2) == pytest.approx(stabilization_score(0.7, 0.8, 0.2))


def test_score_larger_shift_softens_penalty():
    assert stabilization_score(0.8, 0.7, 0.9) > stabilization_score(0.8, 0.7, 0.1)


def test_auc_below_half_is_flipped():
    assert stabilization_score(0.2, 0.3, 0.2) == pytest.approx(stabilization_score(0.8, 0.7, 0.2))


@pytest.mark.parametrize("auc", [0.0, -0.1, 1.1])
def test_invalid_auc_raises(auc):
    with pytest.raises(ValueError):
        stabilization_score(auc, 0.7, 0.2)
    with pytest.raises(ValueError):
        stabilization_uplift(0.8, 0.7, auc, 0.7, 0.2)


def test_uplift_when_b_holds_up_under_shock():
    assert stabilization_uplift(0.80, 0.70, 0.80, 0.81, 0.2) > 0.5


def test_small_uplift_when_b_also_drops():
    assert 0 < stabilization_uplift(0.80, 0.70, 0.81, 0.78, 0.2) < 0.1


def test_no_uplift_when_b_is_worse():
    uplift = stabilization_uplift(0.81, 0.78, 0.80, 0.70, 0.2)
    assert uplift == 0.0
    assert math.copysign(1, uplift) == 1  # not -0.0


def test_no_uplift_for_identical_models():
    assert stabilization_uplift(0.8, 0.7, 0.8, 0.7, 0.2) == pytest.approx(0.0, abs=1e-12)


def test_uplift_flips_b_aucs_independently_of_a():
    # B's AUCs below 0.5 must be flipped using B's own values.
    assert stabilization_uplift(0.80, 0.70, 0.19, 0.22, 0.2) == pytest.approx(
        stabilization_uplift(0.80, 0.70, 0.81, 0.78, 0.2)
    )


def test_uplift_returns_python_float():
    assert type(stabilization_uplift(0.80, 0.70, 0.81, 0.78, 0.2)) is float
