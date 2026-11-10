"""Morris trajectories, Latin hypercube and control designs (DataFrames: one row per member, one column per parameter, physical values)."""
import numpy as np, pandas as pd
from .parameters import PARAMS, NAMES, SPEC, from_unit
from .templates import base_values

def to_phys(u):
    """unit-cube point(s) -> physical values (rows) in PARAMS order"""
    u = np.atleast_2d(u); out = np.empty_like(u, dtype=float)
    for j, sp in enumerate(PARAMS): out[:, j] = from_unit(u[:, j], sp[0])
    return out

def morris_design(r=10, levels=4, candidates=300, seed=1):
    """Morris trajectories on a `levels`-level grid (step 2/3); the r trajectories with the largest mutual distance among `candidates`."""
    rng = np.random.default_rng(seed); p = len(PARAMS); delta = levels / (2.0 * (levels - 1)); grid = np.arange(levels // 2) / (levels - 1)
    def traj():
        x = rng.choice(grid, p); order = rng.permutation(p); sign = rng.choice([-1, 1], p); T = [x.copy()]
        for j in order:
            s = sign[j] if 0 <= x[j] + sign[j] * delta <= 1 else -sign[j]
            x = x.copy(); x[j] += s * delta; T.append(x.copy())
        return np.array(T)
    cand = [traj() for _ in range(candidates)]
    def dist(a, b): return np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)).sum()
    chosen = [0]; D = np.array([[dist(a, b) for b in cand] for a in cand])
    while len(chosen) < r: chosen.append(int(np.argmax([D[c][chosen].sum() if c not in chosen else -1 for c in range(len(cand))])))
    U = np.vstack([cand[c] for c in chosen]); ids = [f"m{t:02d}_{s:02d}" for t in range(r) for s in range(p + 1)]
    return pd.DataFrame(to_phys(U), columns=NAMES, index=ids)

def lhs_design(cfg, n=None, names=None, seed=None):
    """Latin hypercube (random-cd optimised) on the parameters in `names`; the others stay at their baseline value."""
    from scipy.stats import qmc
    from .parameters import LHS_PARAMETERS
    n = n or cfg.lhs["n"]; seed = cfg.lhs["seed"] if seed is None else seed; use = names or LHS_PARAMETERS
    s = qmc.LatinHypercube(d=len(use), optimization="random-cd", seed=seed).random(n); b = base_values(cfg, PARAMS)
    df = pd.DataFrame(index=[f"l{i:03d}" for i in range(n)], columns=NAMES, dtype=float)
    for nm in NAMES:
        df[nm] = b[nm]
        if nm in use: df[nm] = from_unit(s[:, use.index(nm)], nm)
    return df

def control_design(cfg, n=2):
    b = base_values(cfg, PARAMS); return pd.DataFrame([b] * n, index=[f"control{i + 1}" for i in range(n)])[NAMES]
