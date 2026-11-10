"""Observations (field concentrations, incubation rates, soil-air gases) and the fixed definitions that connect them to ATS output datasets.
Call load(cfg) once; FIELD and LITDF are then available as module attributes (observations.FIELD / observations.LITDF)."""
import csv, re
import numpy as np, pandas as pd

TOT = "total_component_concentration."          # prefix of the ATS component-concentration datasets (mol/L)
NTOP = 35                                        # model cells compared with the field: top 35 cm (1 cm cells)
Z = np.arange(93) * 0.01 + 0.005                 # cell-centre depth [m]
N_COL = 93
# (label, field column, model dataset): the six dissolved species + pH, in the order used by every plot
SPECIES = [("CO2(aq)", "CO2", TOT + "CO2(aq)"), ("CH4(aq)", "CH4", TOT + "CH4(aq)"), ("O2(aq)", "O2", TOT + "O2(aq)"),
           ("DOC", "DOC", TOT + "DOM1"), ("Fe2+", "FeII", TOT + "Fe++"), ("Fe(III)", "FeIII", TOT + "Fe+++")]
PH = ("pH", "pH", "pH.0")
PANELS = SPECIES + [PH]
# the three soil-air samples of 18 Aug 2026 (depth m, pCO2 uatm, pCH4 uatm); the two 8 cm samples read the same model cell
GAS_DEPTHS = [0.05, 0.08, 0.08]
GAS_LABELS = ["5 cm", "8 cm (a)", "8 cm (b)"]
GAS_OBS = {"CO2(gas)": [1385.0, 2282.0, 6070.0], "CH4(gas)": [7.7, 3.4, 5.1]}
GAS_ATM = {"CO2(gas)": 425.0, "CH4(gas)": 1.9}
_MM = {"c": 12.011, "fe": 55.845, "o2": 31.998}; _PRE = {"ng": 1e-3, "pg": 1e-6, "ug": 1.0}      # molar masses [g/mol]; prefix -> ug

FIELD = None; LITDF = None

def load_lit(path):
    """measured incubation rates from the literature table, converted to umol per g soil per day"""
    rows, paper, site = [], "", ""
    for r in csv.reader(open(path)):
        r = (r + [""] * 8)[:8]
        if r[0] == "Paper" or not any(r): continue
        paper = r[0] or paper; site = r[1] or site
        tok = r[3].lower().replace("/", " ").replace("-1", " ").split()                          # e.g. ['ng','c','g','soil','h']
        f = _PRE[tok[0]] / _MM[tok[1]] * (24.0 if tok[-1] in ("h", "hr") else 1.0)               # -> umol per g per day
        rows.append(dict(paper=re.sub(r" \(.*", "", paper), site=site, type=r[2], avg=float(r[4]) * f, err=float(r[5]) * f, temp=r[6], notes=r[7]))
    return pd.DataFrame(rows)

def load_field(path):
    """field samples shallower than 0.35 m, all concentrations in uM"""
    f = pd.read_csv(path); f = f[f["depth_m"] <= 0.35].copy(); f["O2"] = f["DO_mgL"] / 32.0 * 1e3; return f      # DO mg/L -> uM

def load(cfg):
    global FIELD, LITDF
    FIELD = load_field(cfg.field_chem); LITDF = load_lit(cfg.lit_rates); return FIELD, LITDF
