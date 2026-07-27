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
