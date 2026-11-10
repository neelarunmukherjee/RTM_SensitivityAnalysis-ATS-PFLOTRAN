"""One row of scored outputs per finished member (log10 model/field ratios, pH difference, rate ratios, aggregate scores), cached as CSV."""
import glob, os
import h5py, numpy as np, pandas as pd
from . import observations as O
from .observations import TOT, NTOP, Z, SPECIES, PH
from .run import Run
from .rates import water_basis_factor, model_rate_rows, lit_select, _habitat, KEEP_HAB

BINS = [(.05, .10), (.10, .15), (.15, .20), (.20, .25), (.25, .35)]
RATE_KEYS = ["methanogenesis", "Fe3_reduction", "anaerobic_CO2", "DOM_aerobic", "aerobic_CO2"]; RATE_KINDS = ["CH4prod", "FeII", "CO2an", "CO2aer", "CO2aer"]
AQ_OUT = [f"{l}_logratio" for l in ("CO2(aq)", "CH4(aq)", "O2(aq)", "DOC", "Fe2+", "Fe(III)")] + ["pH_diff"]; RATE_OUT = [f"rate_{k}_logratio" for k in RATE_KEYS]

def _fieldbin(col, lo, hi): return O.FIELD[(O.FIELD.depth_m >= lo) & (O.FIELD.depth_m < hi)][col].mean()

def member_outputs(h5file, infile, cfg):
    """all outputs of one finished member"""
    r = Run(h5file, "member", None); o = {}; T = TOT
    for lab, fc, key in SPECIES + [PH]:
        m = r.profile_mean(key)[:NTOP].mean(); fm = O.FIELD[fc].dropna().mean()
        o[f"{lab}_logratio" if lab != "pH" else "pH_diff"] = np.log10(max(m, 1e-30) / fm) if lab != "pH" else m - fm
    for s, fc in (("CO2(aq)", "CO2"), ("CH4(aq)", "CH4")):
        pr = r.profile_mean(T + s)
        for lo, hi in BINS: o[f"{fc}_bin{int(lo * 100)}-{int(hi * 100)}_logratio"] = np.log10(max(pr[(Z >= lo) & (Z < hi)].mean(), 1e-30) / _fieldbin(fc, lo, hi))
    fac = water_basis_factor(r); rows = model_rate_rows(r, infile); dist = []
    for (lab, v, kind), nm in zip(rows, RATE_KEYS):
        d = lit_select(kind); d = d[d.site.map(_habitat).isin(KEEP_HAB)]; x = d.avg.to_numpy() * fac; m = v.mean()
        o[f"rate_{nm}_logratio"] = np.log10(max(m, 1e-30) / np.exp(np.log(x).mean())); dist.append(max(0.0, np.log10(x.min() / max(m, 1e-30)), np.log10(max(m, 1e-30) / x.max())))
    tol = np.log10(cfg.tolerance); aq = [o[f"{l}_logratio"] / tol for l in ("CO2(aq)", "CH4(aq)", "DOC", "Fe2+", "Fe(III)")] + [o["pH_diff"] / (2 * cfg.sigma_ph)]
    o["aq_score"] = float(np.sqrt(np.mean(np.square(aq)))); o["rate_score"] = float(np.sqrt(np.mean(np.square(dist))))
    ch = r.profile_mean(T + "CH4(aq)")[:NTOP].mean()
    with h5py.File(h5file, "r") as f: prev = np.mean([f[T + "CH4(aq)"][str(d)][:NTOP, 0].mean() * 1e6 for d in r.days if r.last - 730 < d <= r.last - 365 and d > 0])
    o["CH4_drift_pct"] = 100 * (ch / prev - 1); o["clamped"] = r.clamped_cells(); return o

def load_outputs(folder, cfg, cache=None):
    """score every finished member in `folder` (new ones only; earlier rows come from the cache CSV)"""
    cache = cache or f"{folder}_outputs.csv"; old = pd.read_csv(cache, index_col=0) if os.path.exists(cache) else pd.DataFrame()
    done = sorted(os.path.basename(f).replace("_ats_vis_data.h5", "") for f in glob.glob(f"{folder}/*_ats_vis_data.h5")); new = {}
    for mid in done:
        if mid not in old.index:
            try: new[mid] = member_outputs(f"{folder}/{mid}_ats_vis_data.h5", f"{folder}/{mid}.in", cfg)
            except Exception as e: print("outputs failed for", mid, e)
    out = pd.concat([old, pd.DataFrame(new).T]) if new else old
    if len(out): out.to_csv(cache)
    return out
