"""Morris elementary effects, mu*, mu, sigma with bootstrap intervals, and the ranking rule (Morris 1991; Campolongo et al. 2007)."""
import warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)     # steps excluded because of clamped members
from .parameters import PARAMS, NAMES, to_unit
from .outputs import AQ_OUT, RATE_OUT

GROUP = {"R1": "DOM aerobic respiration", "R3": "DOM aerobic respiration", "R2": "Fermentation", "R4": "H2 / CH4 oxidation", "R5": "H2 / CH4 oxidation",
         "R6": "Methanogenesis", "R7": "Methanogenesis", "R8": "Anaerobic acetate oxidation", "R9": "Iron cycle", "R10": "Iron cycle",
         "CO2cap": "CO2 cap", "logK_FeOH3": "Iron cycle", "logK_SOM": "DOC supply (SOM)", "SOM_mineral": "DOC supply (SOM)", "FeOH3_mineral": "Iron cycle"}
def group_of(nm):
    t = nm.split("_")[0]
    return GROUP[t] if t in GROUP else next(g for k, g in GROUP.items() if nm.startswith(k) and not (k[0] == "R" and k[1:].isdigit()))

def unit(design):
    u = pd.DataFrame(index=design.index, columns=NAMES, dtype=float)
    for n in NAMES: u[n] = to_unit(design[n].to_numpy(float), n)
    return u
def elementary_effects(design, out, ok=None):
    """EE[trajectory, parameter, output]; NaN when either member of the step is missing / failed / excluded by `ok` (dict member -> bool)"""
    U = unit(design); p = len(NAMES); r = len(design) // (p + 1); cols = list(out.columns); EE = np.full((r, p, len(cols)), np.nan)
    for t in range(r):
        ids = list(design.index[t * (p + 1):(t + 1) * (p + 1)])
        for s in range(1, p + 1):
            a, b = ids[s - 1], ids[s]; du = U.loc[b] - U.loc[a]; j = int(np.argmax(np.abs(du.values)))
            if a in out.index and b in out.index and (ok is None or (ok.get(a, True) and ok.get(b, True))):
                EE[t, j, :] = (out.loc[b].values.astype(float) - out.loc[a].values.astype(float)) / du.values[j]
    return EE, cols
def morris_table(EE, cols, nboot=1000, seed=0):
    rng = np.random.default_rng(seed); r, p, q = EE.shape; mu_s = np.nanmean(np.abs(EE), 0); mu = np.nanmean(EE, 0); sig = np.nanstd(EE, 0, ddof=1)
    bs = np.array([np.nanmean(np.abs(EE[rng.integers(0, r, r)]), 0) for _ in range(nboot)]); lo, hi = np.nanpercentile(bs, 5, 0), np.nanpercentile(bs, 95, 0)
    return mu_s, mu, sig, lo, hi
def rank_parameters(mu_s, cols, threshold=0.1):
    """normalised mu* per output; aqueous importance = mean over the 6 concentrations + pH, rate importance = mean over the 5 rates, overall = 0.7 aq + 0.3 rate"""
    ci = {c: i for i, c in enumerate(cols)}; norm = mu_s / np.nanmax(mu_s, 0, keepdims=True)
    aq = np.nanmean(norm[:, [ci[c] for c in AQ_OUT]], 1); rt = np.nanmean(norm[:, [ci[c] for c in RATE_OUT]], 1)
    keep = np.nanmax(norm[:, [ci[c] for c in AQ_OUT + RATE_OUT + ["aq_score", "rate_score"]]], 1) >= threshold
    return pd.DataFrame({"process": [group_of(n) for n in NAMES], "aqueous": aq, "rate": rt, "overall": 0.7 * aq + 0.3 * rt, "keep_for_LHS": keep}, index=NAMES).sort_values("overall", ascending=False), norm
def run_morris(design, out, cfg, exclude_clamped=True):
    """full Morris analysis of a finished Morris ensemble; returns dict(ranking, mu_star, mu, sigma, lo, hi, cols, normalised)"""
    okmap = {m: bool(out.loc[m, "clamped"] == 0) for m in out.index} if exclude_clamped else None
    cols_keep = AQ_OUT + RATE_OUT + [c for c in out.columns if "_bin" in c] + ["aq_score", "rate_score"]
    EE, cols = elementary_effects(design, out[cols_keep], okmap); MU_S, MU, SIG, LO, HI = morris_table(EE, cols)
    rank, norm = rank_parameters(MU_S, cols, cfg.morris_keep_threshold)
    return dict(ranking=rank, mu_star=MU_S, mu=MU, sigma=SIG, lo=LO, hi=HI, cols=cols, normalised=norm, EE=EE)
