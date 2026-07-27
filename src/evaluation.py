"""Evaluation metrics and diagnostic plotting helpers (Phase 4)."""

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, r2_score


def evaluate_regression(y_true, y_pred):
    """Compute the three held-out metrics from Design Doc §5.5 for a
    regression model's predictions against ground-truth potency (p_value):
    RMSE, R², and Spearman rank correlation (the secondary metric the design
    doc calls out as often mattering more in practice than exact value
    prediction, since getting the *relative ranking* of compounds right is
    usually what matters for prioritization).

    Returns a dict with keys "rmse", "r2", "spearman".
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    spearman = float(spearmanr(y_true, y_pred).statistic)
    return {"rmse": rmse, "r2": r2, "spearman": spearman}


def bootstrap_compare(y_true, preds_a, preds_b, n_boot=1000, seed=0):
    """Bootstrap-resample the held-out set to check whether an apparent gap
    between two models' metrics is a robust pattern or could plausibly be
    noise from one fixed test set (Design Doc §5.3: don't assume the more
    sophisticated model wins without actually checking).

    For each of `n_boot` resamples (with replacement, same size as `y_true`),
    computes RMSE/R²/Spearman for both `preds_a` and `preds_b` and records
    whether A had the better score (lower RMSE, higher R²/Spearman).

    Returns a dict with keys "rmse", "r2", "spearman", each the fraction of
    resamples where model A beat model B on that metric -- e.g. 0.98 means A
    was better in 98% of resamples, a robust win; ~0.5 means the two are
    indistinguishable given this data.
    """
    y_true = np.asarray(y_true)
    preds_a = np.asarray(preds_a)
    preds_b = np.asarray(preds_b)
    n = len(y_true)
    rng = np.random.RandomState(seed)

    wins = {"rmse": 0, "r2": 0, "spearman": 0}
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        metrics_a = evaluate_regression(y_true[idx], preds_a[idx])
        metrics_b = evaluate_regression(y_true[idx], preds_b[idx])
        wins["rmse"] += metrics_a["rmse"] < metrics_b["rmse"]
        wins["r2"] += metrics_a["r2"] > metrics_b["r2"]
        wins["spearman"] += metrics_a["spearman"] > metrics_b["spearman"]

    return {key: value / n_boot for key, value in wins.items()}
