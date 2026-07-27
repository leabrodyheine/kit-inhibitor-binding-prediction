"""Unit tests for src/models.py (Phase 4 steps 2-3: XGBoost-on-ECFP baseline
and MLP-on-ChemBERTa comparison model)."""

import numpy as np
import pytest

from models import train_mlp_chemberta, train_xgboost_ecfp


@pytest.fixture
def linear_regression_data():
    # A trivial, noise-free relationship a tree ensemble should fit almost
    # exactly: y = sum of the first 3 feature columns. Lets tests assert on
    # fit quality without depending on real ECFP/potency data.
    rng = np.random.RandomState(0)
    X = rng.randint(0, 2, size=(200, 2048)).astype(np.uint8)
    y = X[:, :3].sum(axis=1).astype(float)
    return X, y


class TestTrainXgboostEcfp:
    def test_returns_fitted_model_that_predicts_right_shape(self, linear_regression_data):
        X, y = linear_regression_data
        model = train_xgboost_ecfp(X, y)
        preds = model.predict(X)
        assert preds.shape == y.shape

    def test_fits_a_learnable_relationship_well(self, linear_regression_data):
        X, y = linear_regression_data
        model = train_xgboost_ecfp(X, y)
        preds = model.predict(X)
        ss_res = np.sum((y - preds) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        assert r2 > 0.9

    def test_deterministic_with_default_random_state(self, linear_regression_data):
        X, y = linear_regression_data
        model_a = train_xgboost_ecfp(X, y)
        model_b = train_xgboost_ecfp(X, y)
        assert np.allclose(model_a.predict(X), model_b.predict(X))

    def test_kwargs_override_defaults(self, linear_regression_data):
        X, y = linear_regression_data
        shallow = train_xgboost_ecfp(X, y, n_estimators=2, max_depth=1)
        deep = train_xgboost_ecfp(X, y, n_estimators=300, max_depth=6)
        # A near-degenerate model (2 shallow trees) should fit far worse than
        # the default-strength model on this easy, low-noise relationship.
        shallow_r2 = 1 - np.sum((y - shallow.predict(X)) ** 2) / np.sum((y - y.mean()) ** 2)
        deep_r2 = 1 - np.sum((y - deep.predict(X)) ** 2) / np.sum((y - y.mean()) ** 2)
        assert deep_r2 > shallow_r2


@pytest.fixture
def continuous_regression_data():
    # 768-dim to mirror real ChemBERTa embeddings; y is a noise-free linear
    # combination of the first 5 dims, learnable by a small MLP. 1000 samples
    # (vs. real data's 4452) keeps the internal early-stopping validation
    # split (validation_fraction=0.1) large enough for a reliable signal --
    # 200 samples made that split too small and noisy, stopping training
    # before it actually converged.
    rng = np.random.RandomState(0)
    X = rng.normal(size=(1000, 768)).astype(np.float32)
    y = X[:, :5].sum(axis=1).astype(float)
    return X, y


class TestTrainMlpChemberta:
    def test_returns_fitted_pipeline_that_predicts_right_shape(self, continuous_regression_data):
        X, y = continuous_regression_data
        model = train_mlp_chemberta(X, y)
        preds = model.predict(X)
        assert preds.shape == y.shape

    def test_fits_a_learnable_relationship_well(self, continuous_regression_data):
        X, y = continuous_regression_data
        model = train_mlp_chemberta(X, y)
        preds = model.predict(X)
        r2 = 1 - np.sum((y - preds) ** 2) / np.sum((y - y.mean()) ** 2)
        assert r2 > 0.8

    def test_deterministic_with_default_random_state(self, continuous_regression_data):
        X, y = continuous_regression_data
        model_a = train_mlp_chemberta(X, y)
        model_b = train_mlp_chemberta(X, y)
        assert np.allclose(model_a.predict(X), model_b.predict(X))

    def test_kwargs_override_defaults(self, continuous_regression_data):
        X, y = continuous_regression_data
        undertrained = train_mlp_chemberta(X, y, max_iter=1, early_stopping=False)
        default = train_mlp_chemberta(X, y)
        undertrained_r2 = 1 - np.sum((y - undertrained.predict(X)) ** 2) / np.sum((y - y.mean()) ** 2)
        default_r2 = 1 - np.sum((y - default.predict(X)) ** 2) / np.sum((y - y.mean()) ** 2)
        assert default_r2 > undertrained_r2

    def test_inputs_are_standardized(self, continuous_regression_data):
        # Confirm the pipeline actually includes a StandardScaler fit to the
        # training data, not just that training happens to converge.
        X, y = continuous_regression_data
        model = train_mlp_chemberta(X, y)
        scaler = model.named_steps["standardscaler"]
        assert np.allclose(scaler.mean_, X.mean(axis=0), atol=1e-6)
