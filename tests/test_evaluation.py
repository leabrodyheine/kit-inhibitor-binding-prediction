"""Unit tests for src/evaluation.py (Phase 4 step 4: held-out evaluation metrics)."""

import numpy as np
import pytest

from evaluation import evaluate_regression


class TestEvaluateRegression:
    def test_perfect_predictions(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        metrics = evaluate_regression(y, y.copy())
        assert metrics["rmse"] == pytest.approx(0.0, abs=1e-9)
        assert metrics["r2"] == pytest.approx(1.0, abs=1e-9)
        assert metrics["spearman"] == pytest.approx(1.0, abs=1e-9)

    def test_returns_plain_python_floats(self):
        y = np.array([1.0, 2.0, 3.0])
        metrics = evaluate_regression(y, y.copy())
        assert set(metrics.keys()) == {"rmse", "r2", "spearman"}
        for value in metrics.values():
            assert isinstance(value, float)

    def test_mean_baseline_has_zero_r2(self):
        # Predicting the mean for every point is the textbook R²=0 case, and
        # RMSE should equal the population std of y_true.
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mean_pred = np.full_like(y, y.mean())
        metrics = evaluate_regression(y, mean_pred)
        assert metrics["r2"] == pytest.approx(0.0, abs=1e-9)
        assert metrics["rmse"] == pytest.approx(y.std(), abs=1e-9)

    def test_rmse_matches_manual_calculation(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.5, 1.5, 3.5, 5.0])
        expected_rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        metrics = evaluate_regression(y_true, y_pred)
        assert metrics["rmse"] == pytest.approx(expected_rmse)

    def test_worse_than_mean_gives_negative_r2(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        wildly_wrong_pred = np.array([5.0, 1.0, 5.0, 1.0, 5.0])
        metrics = evaluate_regression(y_true, wildly_wrong_pred)
        assert metrics["r2"] < 0

    def test_monotonic_nonlinear_relationship_has_perfect_spearman_but_imperfect_r2(self):
        # y_pred is a monotonic (cubic) but non-linear transform of y_true:
        # rank order is perfectly preserved (Spearman=1) even though the
        # predictions are far from the true values on an absolute scale
        # (R² << 1). This is exactly why Design Doc §5.5 treats Spearman as a
        # meaningfully different metric from R², not a redundant one.
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = y_true**3
        metrics = evaluate_regression(y_true, y_pred)
        assert metrics["spearman"] == pytest.approx(1.0, abs=1e-9)
        assert metrics["r2"] < 0.9
