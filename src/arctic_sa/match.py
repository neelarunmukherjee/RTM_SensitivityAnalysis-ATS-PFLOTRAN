"""Emulator fit on the Latin hypercube, Sobol indices, history matching (NROY), Pareto front and candidate selection (notebook section 6)."""
import numpy as np, pandas as pd
from . import emulator as EM
from .outputs import RATE_OUT
from .parameters import LHS_PARAMETERS, NAMES, PARAMS
from .templates import base_values

EM_COLS = ["CO2(aq)_logratio", "CH4(aq)_logratio", "DOC_logratio", "Fe2+_logratio", "Fe(III)_logratio", "pH_diff"] + RATE_OUT + ["aq_score", "rate_score"]
HM_COLS = ["CO2(aq)_logratio", "CH4(aq)_logratio", "DOC_logratio", "Fe2+_logratio", "Fe(III)_logratio", "pH_diff"] + RATE_OUT

def fit_lhs(design, out, names=None, seed=0):
    """fit one GP per output (and log1p(clamped cells)); returns dict(gps, q2, cv, U, Y, names)"""
    names = names or LHS_PARAMETERS; out = out.loc[out.index.intersection(design.index)]
    U = EM.unit_matrix(design.loc[out.index], names); Y = out[EM_COLS].copy(); Y["log_clamped"] = np.log1p(out["clamped"])
    ok = (Y.notna().all(axis=1) & np.isfinite(Y).all(axis=1)).to_numpy(); U, Y = U[ok], Y[ok]
    gps, q2, cv = EM.fit_all(U, Y, seed=seed); return dict(gps=gps, q2=q2, cv=cv, U=U, Y=Y, names=names, n_left_out=int((~ok).sum()))

def sobol_scores(fit, N=4096, seed=0): return {c: EM.sobol(fit["gps"][c], len(fit["names"]), N=N, seed=seed) for c in ("aq_score", "rate_score")}

def history_match(fit, cfg, n=40000, seed=5, max_clamped=2):
    """implausibility over `n` random points: NROY = second-largest I < 3 and predicted clamped cells < `max_clamped`; Pareto front of aq_score vs rate_score"""
    gps, names = fit["gps"], fit["names"]; Up = np.random.default_rng(seed).random((n, len(names)))
    I = EM.implausibility(gps, Up, HM_COLS, cfg); I2 = np.sort(I, 1)[:, -2]
    feasible = gps["log_clamped"].predict(Up) < np.log(max_clamped); NROY = (I2 < 3) & feasible
    PA = EM.predict(gps, Up, ["aq_score", "rate_score"]); idx = np.where(feasible)[0]; srt = idx[np.argsort(PA[idx, 0])]; front = []; best = np.inf
    for i in srt:
        if PA[i, 1] < best: front.append(i); best = PA[i, 1]
    return dict(U=Up, I=I, NROY=NROY, feasible=feasible, pred=PA, front=np.array(front))

def pick_candidates(fit, hm, cfg, n=6):
    """n points spread evenly along the Pareto front, as full 23-parameter rows (the non-sampled parameters at baseline) -> DataFrame cand1..candn"""
    front = hm["front"]
    if len(front) < n: raise ValueError(f"Pareto front has only {len(front)} points")
    pick = front[np.unique(np.linspace(0, len(front) - 1, n).round().astype(int))]; phys = EM.to_physical(hm["U"][pick], fit["names"])
    full = pd.DataFrame([base_values(cfg, PARAMS)] * len(pick))[NAMES]
    for nm in fit["names"]: full[nm] = phys[nm].to_numpy()
    full.index = [f"cand{i + 1}" for i in range(len(pick))]; return full
