"""Gaussian-process emulator, Sobol indices and history matching for the Latin-hypercube ensemble (theory: notebook section 6)."""
import warnings
import numpy as np, pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as CK
from sklearn.model_selection import KFold
from .parameters import to_unit, from_unit

def unit_matrix(df, names): return np.column_stack([to_unit(df[n].to_numpy(float), n) for n in names])
def to_physical(U, names):
    df = pd.DataFrame(index=range(len(U)), columns=names, dtype=float)
    for j, n in enumerate(names): df[n] = from_unit(U[:, j], n)
    return df
def fit_gp(U, y, seed=0):
    k = CK(1.0, (1e-2, 1e2)) * Matern(length_scale=np.ones(U.shape[1]), length_scale_bounds=(0.05, 20.0), nu=2.5) + WhiteKernel(1e-2, (1e-6, 1.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore"); return GPR(k, normalize_y=True, n_restarts_optimizer=2, random_state=seed).fit(U, y)
def fit_all(U, Y, nfold=5, seed=0):
    """one GP per output; returns {col: gp}, cross-validated Q2 (1 = perfect, 0 = no better than the mean) and the CV predictions"""
    gps, q2, cv = {}, {}, {}
    for c in Y.columns:
        y = Y[c].to_numpy(float); pred = np.full(len(y), np.nan)
        for tr, te in KFold(nfold, shuffle=True, random_state=seed).split(U): pred[te] = fit_gp(U[tr], y[tr], seed).predict(U[te])
        q2[c] = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2); cv[c] = pred; gps[c] = fit_gp(U, y, seed)
    return gps, pd.Series(q2, name="Q2"), pd.DataFrame(cv, index=Y.index)
def sobol(gp, d, N=4096, seed=0, nboot=100):
    """first-order S1 (Saltelli 2010) and total-order ST (Jansen 1999) Sobol indices of the emulator mean; bootstrap 90 % intervals"""
    rng = np.random.default_rng(seed); A, B = rng.random((N, d)), rng.random((N, d)); fA, fB = gp.predict(A), gp.predict(B); fAB = []
    for i in range(d):
        ABi = A.copy(); ABi[:, i] = B[:, i]; fAB.append(gp.predict(ABi))
    fAB = np.array(fAB)
    def est(ix):
        a, b, ab = fA[ix], fB[ix], fAB[:, ix]; V = np.var(np.r_[a, b]); return np.mean(b * (ab - a), 1) / V, 0.5 * np.mean((a - ab) ** 2, 1) / V
    S1, ST = est(np.arange(N)); bs = [est(rng.integers(0, N, N)) for _ in range(nboot)]
    return S1, ST, np.percentile([b[0] for b in bs], [5, 95], 0), np.percentile([b[1] for b in bs], [5, 95], 0)
def sigma_obs(col, cfg):
    tol = np.log10(cfg.tolerance)
    return cfg.sigma_ph if col == "pH_diff" else np.log10(cfg.rate_factor) / 2 if col.startswith("rate_") else tol / 2
def implausibility(gps, U, cols, cfg):
    """I(x) = |target - E[f(x)]| / sqrt(var_emulator + var_observation/discrepancy); the target is 0 (model = field) for every output"""
    I = np.empty((len(U), len(cols)))
    for j, c in enumerate(cols):
        m, s = gps[c].predict(U, return_std=True); I[:, j] = np.abs(m) / np.sqrt(s ** 2 + sigma_obs(c, cfg) ** 2)
    return I
def predict(gps, U, cols): return np.column_stack([gps[c].predict(U) for c in cols])
