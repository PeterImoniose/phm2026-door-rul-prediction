"""Health indicator and shock detection for the door runs.

The health indicator is the position the door reached on its last closing move.
An Opening file starts with the door sitting at that position, so the maximum
position feedback in the Opening file is used. It is 188 to 190 on a healthy
door and falls in steps towards about 15 at failure.
"""
import numpy as np
import pandas as pd

MIN_DROP = 2.0   # position units; the feedback itself jitters by +-1
CONFIRM = 5      # cycles a new level must hold before it counts as a shock


def health_series(run_df):
    """Per-cycle health indicator of one run, indexed by cycle number."""
    o = run_df[run_df["kind"] == "Opening"].sort_values("cycle")
    return pd.Series(o["pos_fbk_max"].to_numpy(), index=o["cycle"].to_numpy(), name="hi")


def detect_shocks(hi):
    """Find the step drops in a health series.

    Returns a DataFrame with one row per shock: cycle, level before, level after,
    percentage drop, and cycles since the previous shock (or since start).
    Only the cycles up to and including the confirmation window are used, so the
    detection at cycle c needs data up to c + CONFIRM - 1.
    """
    x = hi.to_numpy()
    cyc = hi.index.to_numpy()
    level = float(np.median(x[:CONFIRM]))
    rows, last = [], 0
    i = 1
    while i <= len(x) - CONFIRM:
        new = float(np.median(x[i:i + CONFIRM]))
        if new <= level - MIN_DROP and x[i] <= level - MIN_DROP:
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
