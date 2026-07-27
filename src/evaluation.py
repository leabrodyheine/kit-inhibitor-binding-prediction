"""Evaluation metrics and diagnostic plotting helpers (Phase 4)."""

import matplotlib

matplotlib.use("Agg")  # this module only ever saves figures, never shows them interactively

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, r2_score

# Okabe-Ito colorblind-safe qualitative pair (validated: scripts/validate_palette.js
# "#0072B2,#D55E00" --mode light -- all checks pass). Fixed order, one color per
# model, reused across every panel so identity never gets reshuffled.
_MODEL_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]


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


def plot_diagnostics(y_true, predictions, save_path=None):
    """Generate predicted-vs-actual and residual diagnostic plots (Design Doc
    §5.5 / IMPLEMENTATION_PLAN Phase 4 step 7) for one or more models.

    `predictions` is a dict mapping model name -> array of predicted p_value.
    Each model gets its own column: top row is predicted vs. actual (with a
    dashed y=x parity line), bottom row is residuals vs. predicted (with a
    dashed zero line). Models keep a fixed color across both panels (Okabe-Ito
    colorblind-safe pair), and since each panel plots a single model, no
    legend is needed -- the panel title names it.

    Returns the matplotlib Figure; also saves it to `save_path` if given.
    """
    y_true = np.asarray(y_true)
    model_names = list(predictions.keys())
    n_models = len(model_names)

    fig, axes = plt.subplots(2, n_models, figsize=(5 * n_models, 8.5), squeeze=False)

    for i, name in enumerate(model_names):
        y_pred = np.asarray(predictions[name])
        color = _MODEL_COLORS[i % len(_MODEL_COLORS)]

        ax = axes[0, i]
        ax.scatter(y_true, y_pred, s=16, alpha=0.35, color=color, edgecolors="none")
        lo = min(y_true.min(), y_pred.min())
        hi = max(y_true.max(), y_pred.max())
        ax.plot([lo, hi], [lo, hi], color="0.6", linewidth=1, linestyle="--", zorder=0)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linewidth=0.5, color="0.85")
        ax.set_axisbelow(True)
        ax.set_xlabel("Actual p_value")
        ax.set_ylabel("Predicted p_value")
        ax.set_title(f"{name}\npredicted vs. actual")

        ax = axes[1, i]
        residuals = y_true - y_pred
        ax.scatter(y_pred, residuals, s=16, alpha=0.35, color=color, edgecolors="none")
        ax.axhline(0, color="0.6", linewidth=1, linestyle="--", zorder=0)
        ax.grid(True, linewidth=0.5, color="0.85")
        ax.set_axisbelow(True)
        ax.set_xlabel("Predicted p_value")
        ax.set_ylabel("Residual (actual − predicted)")
        ax.set_title(f"{name}\nresiduals")

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
