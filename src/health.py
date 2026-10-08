"""Health indicator and shock detection for the door runs.

The health indicator is the position the door reached on its last closing move.
An Opening file starts with the door sitting at that position, so the maximum
position feedback in the Opening file is used. It is 188 to 190 on a healthy
door and falls in steps towards about 15 at failure.
"""
import numpy as np
import pandas as pd

MIN_DROP = 3.0   # position units; real shocks are 3 units or more, jitter is 1 to 2
CONFIRM = 5      # cycles a new level must hold before it counts as a shock
SMOOTH = 5       # rolling-maximum window applied before detection
INIT = 20        # cycles used to establish the starting level


def health_series(run_df):
    """Per-cycle health indicator of one run, indexed by cycle number."""
    o = run_df[run_df["kind"] == "Opening"].sort_values("cycle")
    return pd.Series(o["pos_fbk_max"].to_numpy(), index=o["cycle"].to_numpy(), name="hi")


def detect_shocks(hi):
    """Find the step drops in a health series.

    Returns a DataFrame with one row per shock: cycle, level before, level after,
    percentage drop, and cycles since the previous shock (or since start).

    A late recording trigger can only make the indicator read low, never high, so
    the series is first replaced by its rolling maximum over the last SMOOTH cycles.
    A shock is then a drop of at least MIN_DROP that holds for CONFIRM cycles.
    Everything is causal: a shock at cycle c is confirmed with data up to
    c + CONFIRM - 1, and the rolling maximum delays it by at most SMOOTH - 1 cycles.
    """
    cyc = hi.index.to_numpy()
    x = hi.rolling(SMOOTH, min_periods=1).max().to_numpy()
    level = float(np.median(x[:INIT]))
    rows, last = [], 0
    i = CONFIRM
    while i <= len(x) - CONFIRM:
        new = float(np.median(x[i:i + CONFIRM]))
        if new <= level - MIN_DROP and x[i] <= level - MIN_DROP + 1:
            rows.append({"cycle": int(cyc[i]), "before": level, "after": new,
                         "pct": 100 * (1 - new / level), "gap": int(cyc[i] - last)})
            level, last = new, int(cyc[i])
            i += CONFIRM
        else:
            i += 1
    return pd.DataFrame(rows, columns=["cycle", "before", "after", "pct", "gap"])


def total_cycles(run_df):
    """Life of a run in cycles, using the organisers' convention (file pairs counted from the end)."""
    return int(np.ceil(len(run_df) / 2))
