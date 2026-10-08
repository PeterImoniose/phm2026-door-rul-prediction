"""Life-prediction models and the leave-one-run-out harness.

Every model maps prefix features (see prefix.py) to a predicted total life in cycles.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LinearRegression

from scoring import score_run

DEGRADATION_FEATURES = ["t", "slow", "hi_now", "n_shocks", "needed", "first_shock", "since_last",
                        "since_first", "gap_mean", "gap_last", "gap_min", "drop_mean", "extrap_life"]

# Averages over cycles 20 to 100, tested as a possible early signal and not used in the main model (notebook 2, section 2.6)
SESSION_FEATURES = ["Cl_vel_ref_max", "Cl_hall_mean", "Op_hall_first", "Cl_drv_temp_mean",
                    "Cl_vbus_mean", "Cl_cur_rms", "Op_cur_rms"]

STAGED_FEATURES = {
    "0 shocks": ["log_t", "slow"],
    "1 shock": ["log_t", "slow", "log_since_last", "log_needed", "log_first_shock"],
    "2 shocks": ["log_t", "slow", "log_since_last", "log_needed", "log_gap_mean", "log_first_shock"],
    "3-4 shocks": ["slow", "log_since_last", "log_needed", "log_gap_mean", "log_gap_last"],
    "5+ shocks": ["slow", "log_since_last", "log_needed", "log_gap_mean", "log_gap_last"],
}


def run_weights(df):
    """Each run counts once, however many cut points it contributes."""
    return 1.0 / df.groupby("run")["run"].transform("size")


def predict_prior(test_df, lives, first_shocks):
    """Survival prior: median life of training runs that were in the same situation.

    Before any shock, that means runs whose first shock came later than cycle t.
    After a shock, runs that simply lasted longer than t.
    """
    after_plateau = float(np.median(lives - first_shocks))
    out = []
    for t, k in zip(test_df["t"], test_df["n_shocks"]):
        pool = lives[first_shocks > t] if k == 0 else lives[lives > t]
        if len(pool) >= 5:
            out.append(np.median(pool))
        elif k == 0:
            # plateau longer than almost any training run: assume the first shock is about to happen
            out.append(t + after_plateau)
        else:
            pool = lives[lives > t]
            out.append(np.median(pool) if len(pool) else 1.2 * t)
    return np.maximum(np.array(out, dtype=float), test_df["t"].to_numpy() + 1)


def predict_extrapolation(test_df, prior):
    """Shocks still needed times the mean interval seen so far; the prior when fewer than two shocks."""
    ext = test_df["extrap_life"].to_numpy()
    return np.where(np.isnan(ext), prior, np.maximum(ext, test_df["t"].to_numpy() + 1))


def _with_logs(df):
    d = df.copy()
    d["log_t"] = np.log(d["t"])
    d["log_needed"] = np.log(d["needed"])
    d["log_since_last"] = np.log1p(d["since_last"])
    d["log_first_shock"] = np.log(d["first_shock"])
    d["log_gap_mean"] = np.log(d["gap_mean"])
    d["log_gap_last"] = np.log(d["gap_last"])
    return d


def fit_staged(train_df):
    """One small log-linear regression of remaining life per degradation stage."""
    d = _with_logs(train_df)
    models = {}
    for stage, feats in STAGED_FEATURES.items():
        a = d[d["stage"] == stage]
        models[stage] = LinearRegression().fit(a[feats], np.log(a["remaining"]), sample_weight=run_weights(a))
    return models


def predict_staged(models, test_df):
    d = _with_logs(test_df)
    out = pd.Series(np.nan, index=d.index)
    for stage, feats in STAGED_FEATURES.items():
        b = d[d["stage"] == stage]
        if len(b):
            out[b.index] = b["t"] + np.exp(models[stage].predict(b[feats]))
    return out.to_numpy()


def fit_gbm(train_df, features, seed=0):
    m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=8, min_child_samples=30,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5,
                          verbose=-1, random_state=seed)
    return m.fit(train_df[features], np.log(train_df["remaining"]), sample_weight=run_weights(train_df))


def predict_gbm(model, test_df, features):
    return test_df["t"].to_numpy() + np.exp(model.predict(test_df[features]))


def predict_all(train_df, test_df, lives, first_shocks):
    """Fit every model on train_df and predict life for test_df. Returns a DataFrame of predictions."""
    out = pd.DataFrame(index=test_df.index)
    out["prior"] = predict_prior(test_df, lives, first_shocks)
    out["extrapolation"] = predict_extrapolation(test_df, out["prior"].to_numpy())
    out["staged"] = predict_staged(fit_staged(train_df), test_df)
    out["gbm"] = predict_gbm(fit_gbm(train_df, DEGRADATION_FEATURES), test_df, DEGRADATION_FEATURES)
    feats = DEGRADATION_FEATURES + SESSION_FEATURES
    out["gbm_session"] = predict_gbm(fit_gbm(train_df, feats), test_df, feats)
    # final model: nothing beats the prior before the first shock; boosted trees after it
    no_shock = (test_df["n_shocks"] == 0).to_numpy()
    out["final"] = np.where(no_shock, out["prior"], out["gbm"])
    # experimental variant: session features only where no shock has been seen yet
    out["final_session"] = np.where(no_shock, out["gbm_session"], out["gbm"])
    return out


def leave_one_run_out(examples, lives, first_shocks):
    """Predict every cut of every run with models that never saw that run."""
    parts = []
    for run in sorted(examples["run"].unique()):
        held = examples[examples["run"] == run]
        rest = examples[examples["run"] != run]
        others = lives.index[lives.index != run]
        parts.append(predict_all(rest, held, lives[others], first_shocks[others]))
    return pd.concat(parts).loc[examples.index]


def local_score(life, t_seen, predicted_life):
    """Challenge score (local approximation) of a countdown from predicted_life over the files seen."""
    i = np.arange(2 * int(t_seen))
    true = np.ceil((2 * life - i) / 2)
    pred = np.maximum(np.ceil((2 * predicted_life - i) / 2), 1)
    return score_run(true, pred)["score"]
