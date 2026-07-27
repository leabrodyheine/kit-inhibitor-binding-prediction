"""Model definitions and training helpers (Phase 4)."""

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
