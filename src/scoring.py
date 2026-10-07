"""Local approximation of the PHME 2026 score (Data_Challenge_2026.pdf, section 2).

The PDF leaves two details open, so this is a best reading, not the official script:
- how Precision (a percentage, higher is better) is passed through f(m). Here the
  shortfall (100 - precision) is used so that perfect precision maps to 1.
- the direction of PH. The text says values near 1 mean predictions stay within
  20% almost to end of life, so PH = t_alpha / t_EOF is used, with t_alpha the first
  index whose relative error exceeds 20% (PH = 1 if that never happens).
"""
import numpy as np

A, B, C, D = 4443.76, 1.53, 4443.76, 0.0
ALPHA = 2.0


def f_norm(m):
    return A / (m ** B + C) + D


def score_run(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rel = np.abs(y_true - y_pred) / y_true
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    precision = 100.0 * float(np.mean(rel <= 0.1))
    bad = np.flatnonzero(rel > 0.2)
    ph = 1.0 if len(bad) == 0 else bad[0] / len(y_true)
    rmse_n = f_norm(rmse)
    prec_n = f_norm(100.0 - precision)
    score = (rmse_n + prec_n + ALPHA * ph) / (2 + ALPHA)
    return {"rmse": rmse, "precision": precision, "ph": ph,
            "rmse_n": rmse_n, "prec_n": prec_n, "score": score}


def score_runs(runs):
    """runs: iterable of (y_true, y_pred). Returns the mean of each component."""
    rows = [score_run(t, p) for t, p in runs]
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
