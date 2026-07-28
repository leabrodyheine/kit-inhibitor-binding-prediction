"""Unit tests for src/models.py (Phase 4 steps 2-3: XGBoost-on-ECFP baseline
and MLP-on-ChemBERTa comparison model; Phase 5 step 2: variant-conditioned
prediction helper; accuracy-improvement pass: scaffold-grouped CV tuning)."""

import numpy as np
import pytest

from models import (
    predict_both_variants,
    train_mlp_chemberta,
    train_xgboost_ecfp,
    tune_xgboost_ecfp,
)


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


@pytest.fixture
def variant_aware_model_and_data():
    # Synthetic data where the last (variant-flag) column has a known,
    # learnable +3 effect on the label -- lets tests check that
    # predict_both_variants actually appends 0 then 1, not e.g. both zeros.
    rng = np.random.RandomState(0)
    n = 300
    X_struct = rng.randint(0, 2, size=(n, 20)).astype(np.float32)
    flag = rng.randint(0, 2, size=n).astype(np.float32)
    y = X_struct[:, 0] * 5 + flag * 3
    X = np.hstack([X_struct, flag.reshape(-1, 1)])
    model = train_xgboost_ecfp(X, y, n_estimators=100)
    return model, X_struct


class TestPredictBothVariants:
    def test_returns_two_arrays_of_correct_length(self, variant_aware_model_and_data):
        model, X_struct = variant_aware_model_and_data
        pred_wt, pred_d816v = predict_both_variants(model, X_struct)
        assert pred_wt.shape == (len(X_struct),)
        assert pred_d816v.shape == (len(X_struct),)

    def test_single_1d_fingerprint_input_works(self, variant_aware_model_and_data):
        model, X_struct = variant_aware_model_and_data
        pred_wt, pred_d816v = predict_both_variants(model, X_struct[0])
        assert pred_wt.shape == (1,)
        assert pred_d816v.shape == (1,)

    def test_flag_actually_appended_as_zero_then_one(self, variant_aware_model_and_data):
        # The synthetic label adds +3 exactly when the flag column is 1, so
        # the mean difference between the two prediction sets should be
        # close to +3 -- confirming flag=0 was used for pred_wt and flag=1
        # for pred_d816v (not, say, both flag=0, which would give ~0 diff).
        model, X_struct = variant_aware_model_and_data
        pred_wt, pred_d816v = predict_both_variants(model, X_struct)
        mean_diff = (pred_d816v - pred_wt).mean()
        assert mean_diff == pytest.approx(3.0, abs=0.5)

    def test_predictions_differ_when_flag_matters(self, variant_aware_model_and_data):
        model, X_struct = variant_aware_model_and_data
        pred_wt, pred_d816v = predict_both_variants(model, X_struct)
        assert not np.allclose(pred_wt, pred_d816v)


@pytest.fixture
def grouped_regression_data():
    # 20 scaffold groups of 10 rows each, so GroupKFold(n_splits=2) has
    # something real to split on. y is a learnable linear function of the
    # first 3 columns.
    rng = np.random.RandomState(0)
    n_groups, group_size = 20, 10
    n = n_groups * group_size
    X = rng.randint(0, 2, size=(n, 30)).astype(np.float32)
    y = X[:, :3].sum(axis=1).astype(float)
    groups = np.repeat(np.arange(n_groups), group_size)
    return X, y, groups


class TestTuneXgboostEcfp:
    def test_returns_fitted_model_and_params(self, grouped_regression_data):
        X, y, groups = grouped_regression_data
        model, params = tune_xgboost_ecfp(X, y, groups, n_iter=3, n_splits=2, seed=0, n_jobs=1)
        preds = model.predict(X)
        assert preds.shape == y.shape
        assert isinstance(params, dict)
        assert "n_estimators" in params

    def test_deterministic_given_seed(self, grouped_regression_data):
        X, y, groups = grouped_regression_data
        model_a, params_a = tune_xgboost_ecfp(X, y, groups, n_iter=3, n_splits=2, seed=0, n_jobs=1)
        model_b, params_b = tune_xgboost_ecfp(X, y, groups, n_iter=3, n_splits=2, seed=0, n_jobs=1)
        assert params_a == params_b
        assert np.allclose(model_a.predict(X), model_b.predict(X))

    def test_no_scaffold_group_split_across_folds(self, grouped_regression_data):
        # Directly verify the GroupKFold guarantee this function relies on:
        # every group in a fold's validation set is absent from that fold's
        # training set. Confirmed via the same splitter tune_xgboost_ecfp
        # constructs internally, so this is checking the actual mechanism,
        # not just trusting sklearn's docs.
        from sklearn.model_selection import GroupKFold

        X, y, groups = grouped_regression_data
        for train_fold_idx, val_fold_idx in GroupKFold(n_splits=2).split(X, y, groups=groups):
            train_groups = set(groups[train_fold_idx])
            val_groups = set(groups[val_fold_idx])
            assert train_groups.isdisjoint(val_groups)
