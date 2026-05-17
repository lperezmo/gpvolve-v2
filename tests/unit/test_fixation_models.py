"""Unit tests for the built-in fixation models."""

from __future__ import annotations

import gpvolve
import numpy as np
import pytest
from gpvolve.fixation import (
    bloom_dms,
    get_fixation_model,
    list_fixation_models,
    mccandlish,
    moran,
    register_fixation_model,
    strong_selection_weak_mutation,
    validate_params,
    weak_mutation,
)


class TestRegistry:
    def test_builtin_models_registered(self) -> None:
        names = list_fixation_models()
        for expected in ["sswm", "moran", "mccandlish", "bloom_dms", "weak_mutation"]:
            assert expected in names

    def test_mcclandish_alias_resolves(self) -> None:
        model = get_fixation_model("mcclandish")
        assert model.name == "mccandlish"
        assert model is mccandlish

    def test_sswm_alias_resolves(self) -> None:
        model = get_fixation_model("strong_selection_weak_mutation")
        assert model is strong_selection_weak_mutation

    def test_unknown_model_raises(self) -> None:
        with pytest.raises(gpvolve.ModelError, match="unknown fixation model"):
            get_fixation_model("nonexistent")

    def test_validate_params_missing(self) -> None:
        with pytest.raises(gpvolve.ModelError, match="population_size"):
            validate_params(moran, {})

    def test_validate_params_ok(self) -> None:
        validate_params(moran, {"population_size": 100.0})
        validate_params(strong_selection_weak_mutation, {})

    def test_protocol_metadata(self) -> None:
        assert moran.name == "moran"
        assert moran.bounded_unit_interval is True
        assert "population_size" in moran.required_params
        assert weak_mutation.bounded_unit_interval is False


class TestSSWM:
    def test_neutral_returns_zero(self) -> None:
        out = strong_selection_weak_mutation(np.array([1.0]), np.array([1.0]))
        assert out[0] == pytest.approx(0.0)

    def test_beneficial_positive(self) -> None:
        out = strong_selection_weak_mutation(np.array([1.0]), np.array([2.0]))
        assert 0.0 < out[0] < 1.0
        assert out[0] == pytest.approx(1.0 - np.exp(-1.0))

    def test_deleterious_zero(self) -> None:
        out = strong_selection_weak_mutation(np.array([2.0]), np.array([1.0]))
        assert out[0] == pytest.approx(0.0)

    def test_vectorized(self) -> None:
        fi = np.array([1.0, 1.0, 2.0])
        fj = np.array([2.0, 0.5, 2.0])
        out = strong_selection_weak_mutation(fi, fj)
        assert out.shape == (3,)
        assert out[0] > 0
        assert out[1] == pytest.approx(0.0)
        assert out[2] == pytest.approx(0.0)

    def test_invalid_fitness(self) -> None:
        with pytest.raises(ValueError, match="must be > 0"):
            strong_selection_weak_mutation(np.array([0.0]), np.array([1.0]))


class TestMoran:
    def test_neutral_returns_inverse_n(self) -> None:
        """Moran fixation of a neutral mutant in pop N is 1/N (matches v1 averaging)."""
        out = moran(np.array([1.0]), np.array([1.0]), population_size=10.0)
        assert out[0] == pytest.approx(0.1, rel=1e-3)

    def test_beneficial_higher_than_neutral(self) -> None:
        neutral = moran(np.array([1.0]), np.array([1.0]), population_size=100.0)
        beneficial = moran(np.array([1.0]), np.array([1.1]), population_size=100.0)
        assert beneficial[0] > neutral[0]

    def test_deleterious_lower_than_neutral(self) -> None:
        neutral = moran(np.array([1.0]), np.array([1.0]), population_size=100.0)
        deleterious = moran(np.array([1.0]), np.array([0.9]), population_size=100.0)
        assert deleterious[0] < neutral[0]
        assert deleterious[0] > 0

    def test_population_one_returns_one(self) -> None:
        out = moran(np.array([1.0]), np.array([2.0]), population_size=1.0)
        assert out[0] == pytest.approx(1.0)

    def test_invalid_population_size(self) -> None:
        with pytest.raises(ValueError, match="population_size"):
            moran(np.array([1.0]), np.array([1.0]), population_size=0.5)


class TestMcCandlish:
    def test_neutral_returns_inverse_n(self) -> None:
        out = mccandlish(np.array([1.0]), np.array([1.0]), population_size=10.0)
        assert out[0] == pytest.approx(0.1, rel=1e-3)

    def test_beneficial_higher_than_neutral(self) -> None:
        neutral = mccandlish(np.array([1.0]), np.array([1.0]), population_size=50.0)
        beneficial = mccandlish(np.array([1.0]), np.array([1.05]), population_size=50.0)
        assert beneficial[0] > neutral[0]

    def test_population_one_returns_one(self) -> None:
        out = mccandlish(np.array([1.0]), np.array([0.5]), population_size=1.0)
        assert out[0] == pytest.approx(1.0)


class TestWeakMutation:
    def test_neutral_zero(self) -> None:
        out = weak_mutation(np.array([1.0]), np.array([1.0]))
        assert out[0] == pytest.approx(0.0)

    def test_beneficial_positive(self) -> None:
        out = weak_mutation(np.array([1.0]), np.array([1.5]))
        assert out[0] == pytest.approx(0.5)

    def test_deleterious_zero(self) -> None:
        out = weak_mutation(np.array([2.0]), np.array([1.0]))
        assert out[0] == pytest.approx(0.0)

    def test_not_bounded_in_metadata(self) -> None:
        assert weak_mutation.bounded_unit_interval is False


class TestBloomDms:
    def test_with_indices_returns_table_values(self) -> None:
        pi_table = np.array([[0.0, 0.4, 0.2], [0.1, 0.0, 0.5], [0.3, 0.2, 0.0]])
        fi = np.array([1.0, 1.0, 1.0])
        fj = np.array([1.0, 1.0, 1.0])
        i_idx = np.array([0, 1, 2], dtype=np.int64)
        j_idx = np.array([1, 2, 0], dtype=np.int64)
        out = bloom_dms(fi, fj, pi_table=pi_table, indices=(i_idx, j_idx))
        assert out.tolist() == [0.4, 0.5, 0.3]

    def test_without_indices_falls_back_to_sswm(self) -> None:
        fi = np.array([1.0])
        fj = np.array([2.0])
        pi_table = np.zeros((2, 2))
        out = bloom_dms(fi, fj, pi_table=pi_table, indices=None)
        expected = strong_selection_weak_mutation(fi, fj)
        assert out[0] == pytest.approx(expected[0])


class TestRegisterCustom:
    def test_custom_model_registers(self) -> None:
        @register_fixation_model(
            name="test_custom_model",
            bounded_unit_interval=True,
            required_params=frozenset({"alpha"}),
        )
        def my_model(fi: np.ndarray, fj: np.ndarray, /, *, alpha: float, **_: object) -> np.ndarray:
            return np.full(fi.shape, alpha, dtype=np.float64)

        try:
            retrieved = get_fixation_model("test_custom_model")
            assert retrieved is my_model
            out = retrieved(np.array([1.0, 2.0]), np.array([1.0, 2.0]), alpha=0.42)
            assert out.tolist() == [0.42, 0.42]
        finally:
            # Clean up so other tests do not see this registration.
            from gpvolve.fixation.protocol import _REGISTRY

            _REGISTRY.pop("test_custom_model", None)

    def test_duplicate_registration_raises(self) -> None:
        with pytest.raises(gpvolve.ModelError, match="already registered"):

            @register_fixation_model(
                name="sswm",
                bounded_unit_interval=True,
                required_params=frozenset(),
            )
            def dup(fi: np.ndarray, fj: np.ndarray, /, **_: object) -> np.ndarray:
                return fi
