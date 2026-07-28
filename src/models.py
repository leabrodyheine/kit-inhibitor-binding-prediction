"""Model definitions and training helpers (Phase 4)."""

import numpy as np
from scipy.stats import randint, uniform
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


def train_xgboost_ecfp(X_train, y_train, **xgb_kwargs):
    """Train an XGBoost regressor on ECFP fingerprint features to predict
    potency (pIC50/pKi/pKd, on the common p_value scale from Phase 2).

    Design Doc §5.3: gradient-boosted trees on fingerprints are a realistic,
    strong baseline for a small-to-medium dataset and shouldn't be skipped in
    favor of jumping straight to the ChemBERTa-based model.

    Any keyword in `xgb_kwargs` overrides the corresponding default below.
    """
    params = dict(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=0,
        n_jobs=-1,
    )
    params.update(xgb_kwargs)
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    return model


def train_mlp_chemberta(X_train, y_train, **mlp_kwargs):
    """Train a small MLP head on frozen ChemBERTa embeddings to predict
    potency (pIC50/pKi/pKd, on the common p_value scale from Phase 2).

    Design Doc §5.3: a small MLP on frozen embeddings, compared directly
    against the XGBoost-on-ECFP baseline rather than assumed to win.
    Embeddings are standardized first (`StandardScaler`) since, unlike the
    tree-based ECFP baseline, MLP training is sensitive to input scale.

    Any keyword in `mlp_kwargs` overrides the corresponding MLPRegressor
    default below.
    """
    params = dict(
        hidden_layer_sizes=(128, 32),
        activation="relu",
        alpha=1e-3,
        learning_rate_init=1e-3,
        max_iter=2000,
        early_stopping=True,
        n_iter_no_change=20,
        validation_fraction=0.1,
        random_state=0,
    )
    params.update(mlp_kwargs)
    model = make_pipeline(StandardScaler(), MLPRegressor(**params))
    model.fit(X_train, y_train)
    return model


def tune_xgboost_ecfp(X_train, y_train, scaffold_groups, n_iter=80, n_splits=5, seed=0, n_jobs=8):
    """Hyperparameter-tune an XGBoost-on-ECFP model via scaffold-grouped
    cross-validation, restricted entirely to `X_train`/`y_train` -- the test
    set is never touched here or anywhere in this search.

    Design Doc §5.4's whole reason for a scaffold split (same-scaffold
    compounds shouldn't leak across train/test) applies just as much to
    *inner* CV folds used for tuning: plain KFold CV would let near-identical
    analogues split across folds and give an overoptimistic CV score, the
    same leakage problem the outer split exists to avoid. `scaffold_groups`
    (one Bemis-Murcko scaffold string per training row, e.g. from
    `data_utils.bemis_murcko_scaffold`) is passed to `GroupKFold` so no
    scaffold appears in both a fold's train and validation split.

    Returns (best_estimator, best_params) from `RandomizedSearchCV`, fit on
    the full `X_train` with the best-found hyperparameters.
    """
    param_dist = {
        "n_estimators": randint(100, 900),
        "max_depth": randint(3, 10),
        "learning_rate": uniform(0.01, 0.19),
        "subsample": uniform(0.5, 0.5),
        "colsample_bytree": uniform(0.2, 0.8),
        "min_child_weight": randint(1, 11),
        "reg_alpha": uniform(0, 2),
        "reg_lambda": uniform(0.5, 9.5),
        "gamma": uniform(0, 2),
    }

    search = RandomizedSearchCV(
        XGBRegressor(random_state=seed, n_jobs=1),
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=GroupKFold(n_splits=n_splits),
        n_jobs=n_jobs,
        random_state=seed,
    )
    search.fit(X_train, y_train, groups=scaffold_groups)
    return search.best_estimator_, search.best_params_


def predict_both_variants(model, fingerprints):
    """Given a variant-aware model (trained on features from
    `featurization.add_variant_indicator`) and one or more compounds'
    *raw* structural fingerprints (without the variant flag column), predict
    potency under both KIT variants by appending flag=0 (WT) and flag=1
    (D816V) respectively (Phase 5 step 2).

    `fingerprints` may be a single compound's 1D feature vector or a 2D
    (n_compounds, n_features) array. Returns (pred_wt, pred_d816v) arrays,
    one prediction per compound.
    """
    fingerprints = np.atleast_2d(fingerprints)
    n = fingerprints.shape[0]
    X_wt = np.hstack([fingerprints, np.zeros((n, 1), dtype=np.float32)])
    X_d816v = np.hstack([fingerprints, np.ones((n, 1), dtype=np.float32)])
    return model.predict(X_wt), model.predict(X_d816v)
