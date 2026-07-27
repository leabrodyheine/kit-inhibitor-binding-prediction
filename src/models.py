"""Model definitions and training helpers (Phase 4)."""

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
