"""Features of a run seen only up to some cycle, and training examples built by cutting complete runs.

Everything here is causal: a feature at cut-off cycle t uses files of cycles <= t only.
"""
import numpy as np
import pandas as pd

from health import detect_shocks, health_series, total_cycles

FAIL_LEVEL = 19.0   # 10% of the nominal closing position of 190
DROP = 9.0          # typical drop per shock, from the exploratory analysis
EXCLUDED_RUNS = (5, 30)  # training runs that end well above the failure level


def prefix_features(hi, slow):
    """Describe a health series that stops at its last index (the cut-off cycle)."""
    t = int(hi.index[-1])
    s = detect_shocks(hi)
    k = len(s)
    hi_now = float(np.median(hi.to_numpy()[-5:]))
    f = {"t": t, "slow": int(slow), "hi_now": hi_now, "n_shocks": k,
         "needed": max((hi_now - FAIL_LEVEL) / DROP, 0.5),
         "first_shock": np.nan, "since_last": np.nan, "since_first": np.nan,
         "gap_mean": np.nan, "gap_last": np.nan, "gap_min": np.nan,
         "drop_mean": np.nan, "extrap_life": np.nan}
    if k >= 1:
        f["first_shock"] = float(s.cycle.iloc[0])
        f["since_last"] = float(t - s.cycle.iloc[-1])
        f["since_first"] = float(t - s.cycle.iloc[0])
        f["drop_mean"] = float((s.before - s.after).mean())
    if k >= 2:
        gaps = s.gap.iloc[1:].to_numpy(dtype=float)
        f["gap_mean"], f["gap_last"], f["gap_min"] = gaps.mean(), gaps[-1], gaps.min()
        # physical extrapolation: shocks still needed times the mean interval seen so far
        f["extrap_life"] = float(s.cycle.iloc[-1] + f["needed"] * gaps.mean())
    return f


def build_examples(features_df, slow_by_run, step=20, t_min=100, t_margin=10):
    """Cut every complete run at regular points and compute prefix features for each cut."""
    rows = []
    for run, d in features_df.groupby("run"):
        hi = health_series(d)
        life = total_cycles(d)
        for t in range(t_min, int(hi.index[-1]) - t_margin, step):
            f = prefix_features(hi[hi.index <= t], slow_by_run[run])
            f.update(run=run, life=life, remaining=life - t)
            rows.append(f)
    return pd.DataFrame(rows)


def stage_of(n_shocks):
    n = np.asarray(n_shocks)
    return np.select([n == 0, n == 1, n == 2, n <= 4], ["0 shocks", "1 shock", "2 shocks", "3-4 shocks"], "5+ shocks")


STAGES = ["0 shocks", "1 shock", "2 shocks", "3-4 shocks", "5+ shocks"]


def session_signature(features_df, first=20, last=100):
    """Mean of every per-file feature over an early window of cycles, one row per run.

    Columns are prefixed Cl_ (Closing files) and Op_ (Opening files).
    """
    skip = ("run", "cycle", "kind", "rul", "n_rows", "cond")
    cols = [c for c in features_df.columns if c not in skip]
    early = features_df[features_df["cycle"].between(first, last)]
    parts = [early[early["kind"] == kind].groupby("run")[cols].mean().add_prefix(kind[:2] + "_")
             for kind in ("Closing", "Opening")]
    return pd.concat(parts, axis=1)
