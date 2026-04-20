"""
Custom LightGBM callback for the degradation regression model.

This is a regression problem — Precision/Recall/F1 don't belong here.
Those live in the cliff detector (Stage 2, train_cliff.py).

Metrics reported each block:
  MAE        — primary loss proxy (Huber target)
  RMSE       — penalises large errors more; useful for spotting cliff-lap outliers
  Within ±Xs — % of predictions inside a tolerance window (operationally useful:
               a strategy engineer cares "how often is the model off by more than Xs?")
  Overestimate / Underestimate rate — directional bias check
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Tolerance windows for the "within" accuracy metric
TIGHT_TOL = 0.3   # s — tight: good model should hit this on most laps
LOOSE_TOL = 0.5   # s — loose: acceptable engineering tolerance


def _box(lines: list[str], width: int = 56) -> str:
    top    = "╔" + "═" * width + "╗"
    div    = "╠" + "═" * width + "╣"
    bottom = "╚" + "═" * width + "╝"
    rows = [top]
    for item in lines:
        if item == "---":
            rows.append(div)
        else:
            padded = item.ljust(width)
            rows.append(f"║ {padded[:width - 1]}║")
    rows.append(bottom)
    return "\n".join(rows)


def _row(label: str, value: str, width: int = 54) -> str:
    gap = width - len(label) - len(value) - 2
    return f"  {label}{'·' * max(gap, 1)}{value}"


class EpochMetricsCallback:
    """
    Logs a formatted regression metrics block every `log_every` boosting rounds.
    Attach to the final model training only — not to Optuna trials.

    Parameters
    ----------
    X_train, y_train : training set  (used for overfitting gap)
    X_val,   y_val   : held-out val  (primary tracking signal)
    log_every        : rounds between printouts
    total_rounds     : shown in header so you can track progress
    """

    def __init__(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        log_every: int = 100,
        total_rounds: int = 2000,
    ) -> None:
        self.X_train = np.asarray(X_train)
        self.y_train = np.asarray(y_train)
        self.X_val   = np.asarray(X_val)
        self.y_val   = np.asarray(y_val)
        self.log_every    = log_every
        self.total_rounds = total_rounds

    def __call__(self, env) -> None:
        iteration = env.iteration + 1
        if iteration % self.log_every != 0:
            return

        tr = env.model.predict(self.X_train)
        vl = env.model.predict(self.X_val)

        tr_err = tr - self.y_train
        vl_err = vl - self.y_val

        # Core regression metrics
        tr_mae  = mean_absolute_error(self.y_train, tr)
        vl_mae  = mean_absolute_error(self.y_val,   vl)
        tr_rmse = np.sqrt(mean_squared_error(self.y_train, tr))
        vl_rmse = np.sqrt(mean_squared_error(self.y_val,   vl))

        # Within-tolerance accuracy (operationally meaningful)
        tr_tight = np.mean(np.abs(tr_err) < TIGHT_TOL)
        vl_tight = np.mean(np.abs(vl_err) < TIGHT_TOL)
        tr_loose = np.mean(np.abs(tr_err) < LOOSE_TOL)
        vl_loose = np.mean(np.abs(vl_err) < LOOSE_TOL)

        # Directional bias — are we systematically under/over-predicting?
        vl_over  = np.mean(vl_err > 0)   # predicted worse than reality
        vl_under = np.mean(vl_err < 0)   # predicted better than reality

        gap_mae = vl_mae - tr_mae   # positive = overfitting

        lines = [
            f"  ROUND {iteration:>4} / {self.total_rounds}",
            "---",
            _row("Train MAE",            f"{tr_mae:.4f} s"),
            _row("  Val MAE",            f"{vl_mae:.4f} s"),
            _row("  Gap (val−train)",    f"{gap_mae:+.4f} s"),
            "---",
            _row("Train RMSE",           f"{tr_rmse:.4f} s"),
            _row("  Val RMSE",           f"{vl_rmse:.4f} s"),
            "---",
            _row(f"Train within ±{TIGHT_TOL}s", f"{tr_tight * 100:5.1f} %"),
            _row(f"  Val within ±{TIGHT_TOL}s", f"{vl_tight * 100:5.1f} %"),
            _row(f"  Val within ±{LOOSE_TOL}s", f"{vl_loose * 100:5.1f} %"),
            "---",
            _row("Val overestimates",    f"{vl_over  * 100:5.1f} %"),
            _row("Val underestimates",   f"{vl_under * 100:5.1f} %"),
        ]
        print("\n" + _box(lines))
