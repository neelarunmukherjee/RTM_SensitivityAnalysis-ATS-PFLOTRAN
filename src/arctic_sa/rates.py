"""Reaction rates rebuilt from ATS output with PFLOTRAN's rate law, and the model quantities compared with measured incubation rates."""
import h5py, numpy as np
from . import observations as O
from .observations import NTOP, N_COL

def read_reactions(infile):
    """Parse the MICROBIAL_REACTION blocks of a PFLOTRAN .in: stoichiometry, RATE_CONSTANT, MONOD and INHIBITION terms."""
    rxns, cur, sub = [], None, None
    for raw in open(infile):
        t = raw.split("#")[0].strip()
        if not t: continue
        w = t.split()
        if t == "MICROBIAL_REACTION": cur = {"monod": [], "inhib": []}; continue
        if cur is None: continue
        if w[0] == "REACTION":
            lhs, rhs = t[len("REACTION"):].split("->"); st = {}
            for side, sg in ((lhs, -1), (rhs, 1)):
                for term in side.split(" + "):
                    q = term.split(); n, sp = (float(q[0]), q[1]) if len(q) == 2 else (1.0, q[0])
                    if sp != "H2O": st[sp] = st.get(sp, 0) + sg * n
            cur["stoich"] = st
        elif w[0] == "RATE_CONSTANT": cur["k"] = float(w[1])
        elif w[0] == "MONOD": sub = {"kind": "monod"}; cur["monod"].append(sub)
        elif w[0] == "INHIBITION": sub = {"kind": "inhib"}; cur["inhib"].append(sub)
        elif sub is not None and w[0] == "SPECIES_NAME": sub["sp"] = w[1]
        elif sub is not None and w[0] == "HALF_SATURATION_CONSTANT": sub["K"] = float(w[1])
        elif sub is not None and w[0] == "THRESHOLD_CONCENTRATION": sub["Cth"] = float(w[1])
        elif sub is not None and w[0] == "TYPE": sub["type"] = w[1]; sub["scale"] = float(w[2]) if len(w) > 2 else None
        elif sub is not None and w[0] == "INHIBITION_CONSTANT": sub["C"] = float(w[1])
        elif t == "/" and sub is not None: sub = None
        elif t == "/" and sub is None: rxns.append(cur); cur = None
    return rxns

def reaction_rates(run, infile, day=None):
    """Rate of every microbial reaction in every cell [mol per L water per s], rebuilt from the output with PFLOTRAN's rate law
    (rate = k * prod(monod) * prod(inhibition), reaction_microbial.F90); free-ion concentrations as molar."""
    d = str(day or run.last); out = []
    with h5py.File(run.file, "r") as f:
        C = lambda sp: f["primary_free_ion_concentration." + sp][d][:, 0]
        for r in read_reactions(infile):
            rate = np.full(N_COL, r["k"])
            for m in r["monod"]:
                x = C(m["sp"]) - m.get("Cth", 0.0); rate = rate * x / (m["K"] + x)
            for h in r["inhib"]:
                x = C(h["sp"])
                if h["type"] == "MONOD": rate = rate * h["C"] / (h["C"] + x)
                elif h["type"] == "THRESHOLD": rate = rate * (0.5 + np.sign(h["C"]) * np.arctan((x - abs(h["C"])) * h["scale"]) / np.pi)
            out.append((r["stoich"], rate))
        wet = f["porosity"][d][:, 0] * f["saturation_liquid"][d][:, 0]               # pore-water fraction of the bulk volume
    return out, wet

def water_basis_factor(run, day=None):
    """mol per L of pore water per s for 1 umol per g dry soil per day, with the model's own top-35 cm bulk density
    (porosity-based: (1 - phi) x density_rock) and water content (phi x saturation), averaged over the cells."""
    d = str(day or run.last)
    with h5py.File(run.file, "r") as f:
        phi = f["porosity"][d][:NTOP, 0]; s = f["saturation_liquid"][d][:NTOP, 0]; rho = f["density_rock"][d][:NTOP, 0]
    bd = (1 - phi) * rho / 1000.0                                  # g per cm3 of bulk soil
    return float(np.mean(bd * 1000.0 / (phi * s) * 1e-6 / 86400.0))


RATE_NAMES = ["DOM aerobic resp.", "fermentation", "acetate aerobic resp.", "H2 oxidation", "CH4 oxidation", "hydrogenotrophic methanogenesis",
              "acetoclastic methanogenesis", "anaerobic acetate ox. (S1)", "Fe(II) oxidation", "Fe(III) reduction"]

def model_rate_rows(run, infile):
    """The model quantities compared with measurements: (label, per-cell values [mol/L water/s] of the cells used, measured-rate group, see lit_select)."""
    rx, wet = reaction_rates(run, infile); names = RATE_NAMES if len(rx) == len(RATE_NAMES) else [n for n in RATE_NAMES if n != "CH4 oxidation"]   # CH4 oxidation block commented out (A26)
    R = dict(zip(names, rx)); top = slice(0, NTOP)
    P = lambda n, sp: R[n][0].get(sp, 0) * R[n][1]                                  # production of sp by reaction n [mol/L water/s]
    with h5py.File(run.file, "r") as f: o2 = f["primary_free_ion_concentration.O2(aq)"][str(run.last)][:NTOP, 0]
    anox, ox = o2 < 1e-6, o2 > 1e-5                                                # anoxic (< 1 uM) and oxic (> 10 uM) cells of the top 35 cm
    co2_an = sum(P(n, "CO2(aq)") for n in names if "O2(aq)" not in R[n][0])
    co2_ox = P("DOM aerobic resp.", "CO2(aq)") + P("acetate aerobic resp.", "CO2(aq)")
    ch4 = P("hydrogenotrophic methanogenesis", "CH4(aq)") + P("acetoclastic methanogenesis", "CH4(aq)")
    return [("Methanogenesis\n(CH$_4$ production)", ch4[top], "CH4prod"),
            ("Fe(III) reduction\n(Fe(II) production)", P("Fe(III) reduction", "Fe++")[top], "FeII"),
            (f"Anaerobic CO$_2$ production\n(anoxic cells, {anox.sum()} of {NTOP})", co2_an[top][anox], "CO2an"),
            ("DOM aerobic respiration\n(CO$_2$, top 35 cm mean)", co2_ox[top], "CO2aer"),
            (f"Aerobic CO$_2$ production\n(oxic cells, {ox.sum()} of {NTOP})", co2_ox[top][ox], "CO2aer")]

def lit_select(kind):
    """Rows of the literature table (umol per g soil per day) that correspond to one model quantity."""
    d, t, n = O.LITDF, O.LITDF["type"], O.LITDF["notes"]
    if kind == "CH4prod": m = t.str.contains("CH4 production") & (d.avg > 0) & ~n.str.contains("compressed air")     # N2-purged jars
    elif kind == "FeII":  m = t.str.startswith("Net dissolved Fe")
    elif kind == "CO2an": m = t.str.startswith("CO2 from respiration") & n.str.contains("N2")                       # N2-purged jars
    elif kind == "CO2aer": m = (t.str.startswith("CO2 from respiration") & (n.str.contains("compressed air") | (n == ""))) | t.str.startswith("O2 consumption")
    return d[m]

def _habitat(site): return "o" if "oil water" in site else "D" if "ussock" in site else "s" if "et sedge" in site else "^" if "irch" in site else "o"
KEEP_HAB = ("s", "o")                      # measured points used: wet sedge and soil water only
