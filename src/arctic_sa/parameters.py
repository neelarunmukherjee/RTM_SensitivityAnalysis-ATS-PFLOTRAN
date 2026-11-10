"""The 23 uncertain parameters of the config-8 study (name, kind, reaction no., species, (low, high)); log10-uniform unless kind starts with 'logK'.
Ranges: baseline /30 .. x30, widened to include the Sulman et al. 2022 constant.  alpha_CO2 / alpha_CH4 are never varied."""
import numpy as np
PARAMS = [
 ("R1_k_DOM_aerobic",     "k", 1, None, 2.7e-11, 5.4e-7), ("R2_k_fermentation",   "k", 2, None, 5.3e-12, 4.5e-8),
 ("R3_k_acetate_aerobic", "k", 3, None, 1e-7, 9e-5),      ("R4_k_H2_oxidation",   "k", 4, None, 5e-8, 4.5e-5),
 ("R5_k_CH4_oxidation",   "k", 5, None, 3.3e-12, 3e-9),   ("R6_k_H2_methanogenesis", "k", 6, None, 2.4e-10, 2.2e-7),
 ("R7_k_acetoclastic",    "k", 7, None, 2.5e-11, 4.5e-8), ("R8_k_anaerobic_acetate_ox", "k", 8, None, 3e-7, 2.7e-4),
 ("R9_k_Fe2_oxidation",   "k", 9, None, 1e-11, 4.5e-6),   ("R10_k_Fe3_reduction", "k", 10, None, 7.5e-9, 8.1e-6),
 ("R10_K_acetate",   "monod", 10, "Acetate-", 1e-3, 0.1), ("R8_K_acetate", "monod", 8, "Acetate-", 3e-5, 3e-2),
 ("R7_K_acetate",    "monod", 7, "Acetate-", 3e-5, 3e-2), ("R6_K_H2", "monod", 6, "H2(aq)", 1e-8, 1e-5),
 ("R6_Ki_Fe3",       "inhib", 6, "Fe+++", 3e-7, 3e-4),    ("R2_Ki_O2", "inhib", 2, "O2(aq)", 1e-8, 1e-5),
 ("R6_Ki_O2",        "inhib", 6, "O2(aq)", 1e-7, 1e-4),
 ("CO2cap_threshold", "cap_thr", None, None, 2.1e-3, 3.9e-3), ("CO2cap_steepness", "cap_scale", None, None, 4e3, 4e5),
 ("logK_FeOH3", "logK_FeOH3", None, None, 9.7, 10.7), ("logK_SOM_DOC_ceiling", "logK_SOM", None, None, -3.0093, -2.7093),
 ("SOM_mineral_rate_k", "xml_som", None, None, 7e-3 / 30, 7e-3 * 30), ("FeOH3_mineral_rate_k", "xml_fe", None, None, 1e-7 / 30, 1e-7 * 30),
]
NAMES = [p[0] for p in PARAMS]
LINEAR = ("logK_FeOH3", "logK_SOM")                 # kinds sampled linearly; all others log10-uniformly
SPEC = {p[0]: p for p in PARAMS}
# the 17 parameters retained by the Morris screening (15 flagged + R6_Ki_Fe3, R7_K_acetate); the other 6 stay at baseline in the Latin hypercube
LHS_PARAMETERS = ["R1_k_DOM_aerobic", "R2_k_fermentation", "R3_k_acetate_aerobic", "R5_k_CH4_oxidation", "R6_k_H2_methanogenesis", "R8_k_anaerobic_acetate_ox",
        "R9_k_Fe2_oxidation", "R10_k_Fe3_reduction", "R10_K_acetate", "R8_K_acetate", "R2_Ki_O2", "CO2cap_threshold", "CO2cap_steepness",
        "logK_FeOH3", "logK_SOM_DOC_ceiling", "R6_Ki_Fe3", "R7_K_acetate"]
assert len(LHS_PARAMETERS) == 17 and all(k in NAMES for k in LHS_PARAMETERS)

def is_linear(kind): return kind in LINEAR
def to_unit(x, name):
    _, kind, _, _, lo, hi = SPEC[name]; x = np.asarray(x, float)
    return (x - lo) / (hi - lo) if is_linear(kind) else (np.log10(x) - np.log10(lo)) / (np.log10(hi) - np.log10(lo))
def from_unit(u, name):
    _, kind, _, _, lo, hi = SPEC[name]; u = np.asarray(u, float)
    return lo + u * (hi - lo) if is_linear(kind) else 10 ** (np.log10(lo) + u * (np.log10(hi) - np.log10(lo)))
