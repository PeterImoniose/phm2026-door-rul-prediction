"""Summarise every measurement file into one row of features.

Output: data/features_train.parquet and data/features_test.parquet, one row per
measurement file, in submission order (cycle ascending, Closing before Opening).
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / "Train" / "Train"
TEST_DIR = ROOT / "phm2026data-test_date" / "Test" / "Test"
OUT_DIR = ROOT / "data"

COLS = ["t", "pos_ref", "pos_fbk", "vel_ref", "vel_fbk", "hall", "enc", "vbus",
        "mot_temp", "cur_a", "cur_b", "cur_c", "drv_temp", "vol_a", "vol_b", "vol_c"]
FILE_RE = re.compile(r"F_\d+_(\d+)_(Closing|Opening)\.csv$")


def list_run(run_dir):
    """Measurement files of one run, sorted cycle ascending, Closing before Opening."""
    files = []
    for p in run_dir.iterdir():
        m = FILE_RE.match(p.name)
        if m:
            files.append((int(m.group(1)), m.group(2), p))
    files.sort(key=lambda x: (x[0], x[1]))
    return files


def summarise(path):
    a = pd.read_csv(path, sep=";", header=None).to_numpy(dtype=float)
    out = {"n_rows": len(a)}
    for j, c in enumerate(COLS):
        if c in ("t", "enc", "mot_temp"):
            continue
        x = a[:, j]
        out[f"{c}_first"] = x[0]
        out[f"{c}_last"] = x[-1]
        out[f"{c}_min"] = x.min()
        out[f"{c}_max"] = x.max()
        out[f"{c}_mean"] = x.mean()
        out[f"{c}_std"] = x.std()
    # sample index at which the door first moves and the hall state first changes
    moved = np.flatnonzero(a[:, 2] != a[0, 2])
    out["move_idx"] = moved[0] if len(moved) else len(a)
    hall = np.flatnonzero(a[:, 5] != a[0, 5])
    out["hall_idx"] = hall[0] if len(hall) else len(a)
    out["hall_changes"] = int((np.diff(a[:, 5]) != 0).sum())
    out["cur_rms"] = float(np.sqrt((a[:, 9:12] ** 2).mean()))
    out["track_err"] = float(np.abs(a[:, 1] - a[:, 2]).mean())
    return out


def process_run(run_dir, has_rul):
    run_id = int(run_dir.name.split("_")[1])
    files = list_run(run_dir)
    rows = []
    for cycle, kind, p in files:
        r = summarise(p)
        r.update(run=run_id, cycle=cycle, kind=kind)
        rows.append(r)
    df = pd.DataFrame(rows)
    if has_rul:
        rul = pd.read_csv(next(run_dir.glob("F_*_RUL.csv")), header=None)[0].to_numpy()
        if len(rul) != len(df):
            raise ValueError(f"{run_dir.name}: {len(rul)} RUL rows vs {len(df)} files")
        df["rul"] = rul
    return df


def build(base, has_rul, out_name):
    runs = sorted((d for d in base.iterdir() if d.is_dir()),
                  key=lambda d: int(d.name.split("_")[1]))
    parts = Parallel(n_jobs=-2, verbose=5)(delayed(process_run)(d, has_rul) for d in runs)
    df = pd.concat(parts, ignore_index=True)
    OUT_DIR.mkdir(exist_ok=True)
    df.to_parquet(OUT_DIR / out_name)
    print(out_name, df.shape, "runs:", df["run"].nunique())


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "train"):
        build(TRAIN_DIR, True, "features_train.parquet")
    if which in ("all", "test"):
        build(TEST_DIR, False, "features_test.parquet")
