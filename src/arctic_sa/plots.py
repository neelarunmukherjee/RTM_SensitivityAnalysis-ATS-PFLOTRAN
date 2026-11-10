"""Figures (no gridlines anywhere).  Every function takes a Run (see run.py); observations come from observations.load(cfg)."""
import h5py, numpy as np, pandas as pd, matplotlib.pyplot as plt
from . import observations as O
from .observations import TOT, NTOP, Z, PANELS, SPECIES, PH, GAS_OBS, GAS_DEPTHS, GAS_LABELS, GAS_ATM
from .rates import water_basis_factor, model_rate_rows, lit_select, _habitat, KEEP_HAB

PAPER_COL = {"Judd & Kling 2002": "#2ca02c", "Zak and Kling 2006": "#9467bd", "Trusiak et al. 2019": "#ff7f0e", "Romanowicz et al. 2021": "#1f77b4"}
HAB_NAME = {"D": "tussock", "s": "wet sedge", "^": "birch-willow", "o": "soil water (tussock + wet sedge)"}


def plot_rate_comparison(run, infile, title=None):
    """Model rate (red dot = mean of the cells used, bar reaches the most active cell) against the measured rates of data/lit_rates.csv on a log axis.
    Colour = paper, marker = habitat, bar = +-SE or +-SD as given in the table. Per-gram rates are converted with the model's own bulk density and water content."""
    fac = water_basis_factor(run); rows = model_rate_rows(run, infile); n = len(rows)
    fig, a = plt.subplots(figsize=(11.5, 1.2 * n + 2.2))
    for i, (lab, v, kind) in enumerate(rows):
        y = n - 1 - i; vp = v[v > 0]; m = v.mean()
        a.plot([m, vp.max()], [y, y], color="C3", lw=6, alpha=0.3, solid_capstyle="round", zorder=2)
        a.scatter(m, y, s=150, color="C3", edgecolor="white", zorder=4)
        d = lit_select(kind); d = d[d.site.map(_habitat).isin(KEEP_HAB)]; x = d.avg.to_numpy() * fac; e = d.err.to_numpy() * fac
        for (_, r), xi, ei in zip(d.iterrows(), x, e):
            col = PAPER_COL.get(r.paper, "k")
            a.errorbar(xi, y, xerr=[[min(ei, xi * 0.999)], [ei]], color=col, lw=1.3, capsize=3, zorder=3)
            a.scatter(xi, y, s=120, marker=_habitat(r.site), color=col, edgecolor="k", zorder=5)
        a.text(m, y + 0.2, f"model = {m / x.max():.2g}" + (f"-{m / x.min():.2g}" if len(x) > 1 else "") + " x measured", ha="left", fontsize=8.5, fontweight="bold")
    a.set_yticks(range(n)); a.set_yticklabels([r[0] for r in rows][::-1], fontsize=9.5); a.set_xscale("log"); a.set_ylim(-0.6, n - 0.4)
    a.set_xlabel("rate  [mol per L of pore water per s]   (left = slower)"); a.spines[["top", "right"]].set_visible(False)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", color="C3", ls="", ms=9, label="model mean (bar: most active cell)")]
    h += [Line2D([], [], marker="o", color=c_, ls="", ms=8, mec="k", label=p) for p, c_ in PAPER_COL.items()]
    h += [Line2D([], [], marker=m_, color="w", mfc="0.6", mec="k", ls="", ms=8, label=HAB_NAME[m_]) for m_ in ("s", "o")]
    a.legend(handles=h, loc="upper center", bbox_to_anchor=(0.45, -0.12), ncol=4, fontsize=8, frameon=False)
    fig.text(0.01, 0.01, f"incubation rates (2.8-15 C) per g soil, taken as dry soil, converted with the model's own top-35 cm bulk density and water content: 1 µmol g$^{{-1}}$ d$^{{-1}}$ = {fac:.2e} mol L$^{{-1}}$ s$^{{-1}}$;  bars = ±SE or ±SD as listed", fontsize=7.5, color="0.35")
    a.set_title("Model reaction rates vs measured rates", loc="left", fontweight="bold")
    fig.suptitle(title or f"{run.label}: reaction rates at day {run.last} vs measured rates", fontsize=12); plt.tight_layout(rect=(0, 0.035, 1, 1)); plt.show()


def plot_profiles(run, compare=(), title=None):
    fig, axes = plt.subplots(1, 7, figsize=(20, 6), sharey=True)
    for ax, (lab, fc, key) in zip(axes, PANELS):
        fm = O.FIELD.groupby("depth_m")[fc].mean()
        for g, r in zip(["0.55", "0.75", "0.35"], compare): ax.plot(r.profile(key), Z, color=g, lw=2.2, zorder=2.5, label=r.label)
        lg = lab not in ("O2(aq)", "pH"); m_ = run.profile_mean(key)          # yearly mean at every depth (last year of the run)
        # shades: ± SD; on the log axes the SD is taken of log10(value), so the band is a factor (never negative)
        if lg: sd_ = run.profile_sd(key, log=True); lo_, hi_ = m_ * 10 ** -sd_, m_ * 10 ** sd_
        else: sd_ = run.profile_sd(key); lo_, hi_ = m_ - sd_, m_ + sd_
        ax.fill_betweenx(Z, lo_, hi_, color="C3", alpha=0.3, lw=0, zorder=2.6, label="model ± SD (yearly)")
        ax.plot(m_, Z, color="C3", lw=2, zorder=3, label="model, yearly mean")
        FF = O.FIELD[O.FIELD[fc] > 0] if lg else O.FIELD
        fs = (np.log10(FF[fc]) if lg else FF[fc]).groupby(FF["depth_m"]).agg(["mean", "std", "count"]); fs = fs[fs["count"] >= 3]
        flo, fhi = (10 ** (fs["mean"] - fs["std"]), 10 ** (fs["mean"] + fs["std"])) if lg else (fs["mean"] - fs["std"], fs["mean"] + fs["std"])
        ax.fill_betweenx(fs.index, flo, fhi, color="dodgerblue", alpha=0.2, lw=0, zorder=1.5, label="field ± SD (per depth)")
        ax.scatter(O.FIELD[fc], O.FIELD["depth_m"], s=18, color="0.35", alpha=0.4, zorder=2, label="field (all samples)")
        ax.scatter(fm.values, fm.index, s=70, marker="D", facecolor="dodgerblue", edgecolor="k", zorder=4, label="field (depth mean)")
        if lab not in ("O2(aq)", "pH"): ax.set_xscale("log")
        ax.set_xlabel(lab if lab == "pH" else f"{lab}  [µM]", fontsize=9); ax.tick_params(labelsize=8)
    axes[0].set_ylim(0.35, 0); axes[0].set_ylabel("Depth [m]"); axes[0].legend(fontsize=7)
    fig.suptitle(title or f"{run.label}: depth profiles vs field, top 35 cm", fontsize=12, y=1.02); plt.tight_layout(); plt.show()

def plot_bars(run, title=None):
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 7, figsize=(18.2, 5))
    for ax, (lab, fc, key) in zip(axes, PANELS):
        fx = O.FIELD[fc].dropna(); m = run.profile(key)[:NTOP]
        ax.bar(0, m.mean(), width=0.6, color="C3", zorder=2); ax.bar(1, fx.mean(), width=0.6, color="dodgerblue", alpha=0.6, zorder=2)
        for x_, v_ in ((0, m), (1, fx)):                                     # mean ± SD (lower arm clipped at 0 for the log axes)
            mu, sd = v_.mean(), v_.std(); ax.errorbar(x_, mu, yerr=[[min(sd, mu * 0.999)], [sd]], color="k", capsize=5, lw=1.5, zorder=4)
        ax.scatter(1 + rng.uniform(-0.15, 0.15, len(fx)), fx, s=14, color="0.3", alpha=0.4, zorder=3)
        if lab not in ("O2(aq)", "pH"): ax.set_yscale("log")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["model\n(top 35 cm)", "field\n(top 35 cm)"], fontsize=7)
        ax.set_title(f"{lab}\nmodel/field = {m.mean() / fx.mean():.2f}" if lab != "pH" else f"pH\nmodel {m.mean():.2f}, field {fx.mean():.2f}", fontsize=9); ax.tick_params(labelsize=7)
    axes[0].set_ylabel("µM"); fig.suptitle(title or f"{run.label}: top-35 cm average ± SD vs field average ± SD (day {run.last})", fontsize=12, y=1.04); plt.tight_layout(); plt.show()

def plot_violins(run, title=None):
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 7, figsize=(18.2, 5))
    for ax, (lab, fc, key) in zip(axes, PANELS):
        mv = run.profile(key)[:NTOP]; fv = O.FIELD[fc].dropna().to_numpy()
        parts = ax.violinplot([mv, fv], positions=[0, 1], showmeans=True, showextrema=True, widths=0.7)
        for body, col in zip(parts["bodies"], ("C3", "dodgerblue")): body.set_facecolor(col); body.set_alpha(0.6)
        for k in ("cmeans", "cmaxes", "cmins", "cbars"): parts[k].set_color("0.2"); parts[k].set_linewidth(1)
        ax.scatter(1 + rng.uniform(-0.15, 0.15, len(fv)), fv, s=14, color="0.3", alpha=0.4, zorder=3)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["model cells\n(top 35 cm)", "field samples\n(top 35 cm)"], fontsize=7)
        ax.set_title(lab, fontsize=9); ax.tick_params(labelsize=7)
    axes[0].set_ylabel("µM"); fig.suptitle(title or f"{run.label}: per-cell model distribution vs field distribution, top 35 cm (day {run.last})", fontsize=12, y=1.04); plt.tight_layout(); plt.show()

def plot_raincloud(run, title=None):
    """Raincloud per species: density cloud (left), box (centre) and the raw values as dots (right), for the 35 model cells of the top 35 cm (red)
    and the field samples (blue). Log axis for every species except O2 and pH; the density is estimated in log10 on the log axes. White diamond = mean."""
    from scipy.stats import gaussian_kde
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 7, figsize=(18.2, 5.4))
    for ax, (lab, fc, key) in zip(axes, PANELS):
        logy = lab not in ("O2(aq)", "pH")
        for x, v, col in ((0, run.profile(key)[:NTOP], "C3"), (1, O.FIELD[fc].dropna().to_numpy(), "dodgerblue")):
            v = v[np.isfinite(v)]; v = v[v > 0] if logy else v; u = np.log10(v) if logy else v
            try:
                pad = 0.15 * max(np.ptp(u), 1e-6); ys = np.linspace(u.min() - pad, u.max() + pad, 200); ys = ys[ys >= 0] if lab == "O2(aq)" else ys; d = gaussian_kde(u)(ys); d = d / d.max() * 0.36
                yy = 10 ** ys if logy else ys; ax.fill_betweenx(yy, x - 0.06, x - 0.06 - d, color=col, alpha=0.55, lw=0)
            except Exception: pass                                   # constant data (e.g. DOC at the ceiling): no density, box and dots only
            bp = ax.boxplot([v], positions=[x + 0.06], widths=0.1, showfliers=False, patch_artist=True, manage_ticks=False)
            for b in bp["boxes"]: b.set_facecolor(col); b.set_alpha(0.7)
            for k_ in ("medians", "whiskers", "caps", "boxes"): [l.set_color("0.2") for l in bp[k_]]
            ax.scatter(x + 0.24 + rng.uniform(-0.07, 0.07, len(v)), v, s=11, color=col, edgecolor="0.25", lw=0.3, alpha=0.7, zorder=3)
            ax.scatter(x + 0.06, v.mean(), s=36, marker="D", color="w", edgecolor="k", zorder=5)
        if logy: ax.set_yscale("log")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["model cells\n(top 35 cm)", "field samples\n(top 35 cm)"], fontsize=7)
        ax.set_xlim(-0.55, 1.45); ax.set_title(lab, fontsize=9); ax.tick_params(labelsize=7); ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("µM"); fig.suptitle(title or f"{run.label}: raincloud, per-cell model distribution vs field distribution, top 35 cm (day {run.last})", fontsize=12, y=1.04); plt.tight_layout(); plt.show()


def plot_timeseries(run, compare=(), title=None, t0=0.0):   # t0 [years]: left edge of the time axis (e.g. 3 = last year of a 4-year run)
    PAN = [("DOC", "DOC", TOT + "DOM1"), ("CO$_2$(aq)", "CO2", TOT + "CO2(aq)"), ("CH$_4$(aq)", "CH4", TOT + "CH4(aq)"),
           ("Fe$^{2+}$", "FeII", TOT + "Fe++"), ("Fe(III)", "FeIII", TOT + "Fe+++"), ("pH", "pH", "pH.0")]
    greys = ["0.6", "0.78", "0.45", "0.3"]
    tmax = max(r.years for r in [run, *compare])
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.4), sharex=True)
    for ax, (lab, fc, key) in zip(axes.flat, PAN):
        fv = O.FIELD[fc].dropna(); fm, fs = fv.mean(), fv.std()
        ax.axhline(fm, color="#eb6834", lw=2.4, ls=(0, (4, 2)), label="field mean (top 35 cm)")
        ax.axhspan(fm - fs, fm + fs, color="#eb6834", alpha=0.15, lw=0, label="field ± SD")
        ys = []
        for g, r in zip(greys, compare):
            t_, y_ = r.series(key); ax.plot(t_, y_, color=g, lw=2.2, label=r.label); ys.append(y_)
        t_, y_ = run.series(key); _, sd_ = run.series_sd(key)
        ax.fill_between(t_, y_ - sd_, y_ + sd_, color="C3", alpha=0.2, lw=0, label="model ± SD (35 cells)")
        ax.plot(t_, y_, color="C3", lw=3, label=run.label); ys.append(y_)
        if lab == "pH": ax.set_ylim(1, 7)                                      # pH on the full 1-7 scale
        if lab == "DOC" and min(y.min() for y in ys) > 1200 and max(y.max() for y in ys) < 1500: ax.set_ylim(1200, 1500)
        ax.set_xlim(t0, tmax * 1.03); ax.set_title(lab + ("" if lab == "pH" else "  [µM]"), loc="left", fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[1]: ax.set_xlabel("years")
    h, l = axes[0, 0].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(title or f"{run.label}: top-35 cm column means", y=1.07); plt.tight_layout(); plt.show()

def plot_soil_gas(run, title=None, obs=True):   # obs=False: model only, no observations or ratios
    cols = ["#1f77b4", "#2ca02c", "#9467bd"]
    cy = run.days[1:]; t_ = np.array(cy) / 365.0
    ser = {g: np.full((len(cy), 3), np.nan) for g in GAS_OBS}
    for j, c in enumerate(cy):
        gm, pg = run.gas(c)
        if gm.sum() > 1:
            for g in GAS_OBS: ser[g][j] = np.interp(GAS_DEPTHS, Z[gm], pg[g][gm])
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), gridspec_kw={"width_ratios": [2.2, 1]})
    for r, g in enumerate(GAS_OBS):
        ax = axes[r, 0]
        for k in (0, 1):                                                         # 5 cm and 8 cm model cells
            ax.plot(t_, ser[g][:, k], color=cols[k], lw=2.4, label=f"model, {GAS_LABELS[k].split(' ')[0]} {GAS_LABELS[k].split(' ')[1]}")
        if obs:
            for k in range(3): ax.axhline(GAS_OBS[g][k], color=cols[k], lw=1.6, ls=(0, (4, 2)))
        ax.axhline(GAS_ATM[g], color="0.4", lw=1, ls=":"); ax.set_yscale("log"); ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(f"soil-air p{g[:3]} [µatm]  ({'dashed = observation, ' if obs else ''}dotted = atmosphere)", loc="left", fontweight="bold")
        ax.set_xlim(0, run.years * 1.03); ax.set_xlabel("years"); ax.legend(fontsize=9)
        ax = axes[r, 1]
        for i in range(3):
            m_ = ser[g][-1, i]
            if obs: ax.scatter(i - 0.12, GAS_OBS[g][i], s=150, marker="D", color="#eb6834", edgecolor="white", zorder=5, label="observation" if i == 0 else None)
            ax.scatter(i + (0.12 if obs else 0), m_, s=130, color="C3", edgecolor="white", zorder=4, label=f"model, day {run.last}" if i == 0 else None)
            ax.text(i + (0.12 if obs else 0), m_ * 1.4, f"x{m_ / GAS_OBS[g][i]:.2g}" if obs else f"{m_:.3g}", ha="center", fontsize=10)
        ax.axhline(GAS_ATM[g], color="0.4", lw=1, ls=":"); ax.set_yscale("log"); ax.set_xlim(-0.6, 2.6); ax.set_xticks([0, 1, 2], GAS_LABELS)
        lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi * 3); ax.legend(fontsize=9)
    fig.suptitle(title or f"{run.label}: soil-air gases at the sampled depths vs the 18 Aug 2026 samples" if obs else f"{run.label}: modelled soil-air gases at the sampled depths", fontsize=12); plt.tight_layout(); plt.show()

def plot_one_to_one(run, title=None):
    fig, axes = plt.subplots(2, 5, figsize=(21, 8.6)); axes = axes.ravel(); stats = {}
    for ax, (lab, fc, key) in zip(axes, PANELS):
        d = O.FIELD[["depth_m", fc]].dropna(); obs = d[fc].values; mod = np.interp(d["depth_m"].values, Z, run.profile(key))
        isph = fc == "pH"; ok = np.ones(len(obs), bool) if isph else (obs > 0) & (mod > 0)
        sc = ax.scatter(obs[ok], mod[ok], c=d["depth_m"].values[ok], cmap="viridis_r", s=36, alpha=0.8, edgecolor="none", vmin=0, vmax=0.35)
        lo = min(obs[ok].min(), mod[ok].min()); hi = max(obs[ok].max(), mod[ok].max())
        if isph: xx = np.array([lo - 0.1, hi + 0.1])
        else:
            xx = np.array([lo / 1.5, hi * 1.5]); ax.set_xscale("log"); ax.set_yscale("log"); ax.fill_between(xx, xx / 2, xx * 2, color="0.9", zorder=0)
        ax.plot(xx, xx, "k-", lw=1); ax.set_xlim(xx); ax.set_ylim(xx); ax.set_aspect("equal")
        ax.set_title(lab if isph else f"{lab} [µM]", loc="left", fontweight="bold"); ax.set_xlabel("observation"); ax.set_ylabel("model")
        stats[lab] = round(float(mod.mean() - obs.mean()), 2) if isph else round(float(np.mean(mod[ok]) / np.mean(obs[ok])), 2)
    gm, pg = run.gas()
    for ax, g in zip(axes[7:9], GAS_OBS):
        mod = np.interp(GAS_DEPTHS, Z[gm], pg[g][gm]); o = np.array(GAS_OBS[g]); ax.scatter(o, mod, s=90, color="C3", edgecolor="white", zorder=3)
        lo, hi = min(o.min(), mod.min()) / 2, max(o.max(), mod.max()) * 2; xx = np.array([lo, hi]); ax.fill_between(xx, xx / 2, xx * 2, color="0.9", zorder=0)
        ax.plot(xx, xx, "k-", lw=1); ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(xx); ax.set_ylim(xx); ax.set_aspect("equal")
        ax.set_title(f"soil-air p{g[:3]} [µatm]", loc="left", fontweight="bold"); ax.set_xlabel("observation"); ax.set_ylabel("model")
        stats["air " + g[:3]] = [round(float(m / x), 2) for m, x in zip(mod, o)]
    axes[9].axis("off"); axes[9].text(0, 0.5, "black line: 1:1\ngrey band: within a factor 2\ncolour (aqueous panels): sample depth, 0 to 0.35 m", fontsize=11, va="center")
    fig.suptitle(title or f"{run.label} (day {run.last}): model vs observation, one point per field sample in the top 35 cm", fontsize=13, y=0.98); plt.tight_layout(); plt.show()
    print("mean model/observation (pH: model - observation):", stats)

def scorecard(run):
    """Top-35 cm model mean / field mean at the last output, plus pH and the number of clamped cells."""
    out = {"run": run.label, "day": run.last}
    for lab, fc, key in PANELS:
        if lab == "O2(aq)": continue
        m = run.profile(key)[:NTOP].mean(); f = O.FIELD[fc].dropna().mean()
        out[lab if lab != "pH" else "pH (model)"] = round(float(m if lab == "pH" else m / f), 3)
    out["pH (field)"] = round(float(O.FIELD["pH"].dropna().mean()), 3); out["clamped cells"] = run.clamped_cells()
    return out

def scorecard_table(runs):
    return pd.DataFrame([scorecard(r) for r in runs]).set_index("run")

def plot_profiles_time(run, zmax=0.35, field=False, title=None):
    days = run.days[1:]                                     # the first output has no pH
    cmap = plt.cm.viridis; norm = plt.Normalize(0, run.years)
    fig, axes = plt.subplots(1, 7, figsize=(20, 6), sharey=True)
    for ax, (lab, fc, key) in zip(axes, PANELS):
        for d in days: ax.plot(run.profile(key, d), Z, color=cmap(norm(d / 365.0)), lw=1.4)
        if field:
            ax.scatter(O.FIELD[fc], O.FIELD["depth_m"], s=18, color="0.35", alpha=0.4, zorder=1)
        if lab not in ("O2(aq)", "pH", "DOC"):                # DOC spans <2x, so it stays linear
            ax.set_xscale("log"); ax.xaxis.set_major_locator(plt.LogLocator(numticks=4)); ax.xaxis.set_minor_formatter(plt.NullFormatter())
        if lab == "DOC": ax.set_xlim(0, None)                     # zero-based, so a flat DOC reads as flat
        ax.set_xlabel(lab if lab == "pH" else f"{lab}  [µM]", fontsize=9); ax.tick_params(labelsize=8)
    axes[0].set_ylim(zmax, 0); axes[0].set_ylabel("Depth [m]")
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, label="time [years]", fraction=0.02, pad=0.01)
    fig.suptitle(title or f"{run.label}: depth profiles at every output ({len(days)} outputs), coloured by time", fontsize=12, y=1.02); plt.show()


def plot_gas_profiles_time(run, title=None):
    gases = [("CO2(gas)", "pCO$_2$", 425.0), ("CH4(gas)", "pCH$_4$", 1.9), ("O2(gas)", "pO$_2$", 209500.0), ("H2(gas)", "pH$_2$", None)]
    days = run.days[1:]; cmap = plt.cm.viridis; norm = plt.Normalize(0, run.years)
    fig, axes = plt.subplots(1, 4, figsize=(15, 5.5), sharey=True); zdeep = 0.0
    with h5py.File(run.file, "r") as f:
        for d in days:
            c_ = str(d); gm = f["saturation_gas"][c_][:, 0] > 1e-6                       # cells that contain a gas phase
            p = f["pressure"][c_][:, 0] / 101325.0 * 1e6
            if gm.any(): zdeep = max(zdeep, Z[gm].max() + 0.005)
            for ax, (g, lab, atm) in zip(axes, gases):
                ax.plot((f["gas_mole_fraction." + g][c_][:, 0] * p)[gm], Z[gm], "-o", ms=3, lw=1.4, color=cmap(norm(d / 365.0)))
    for ax, (g, lab, atm) in zip(axes, gases):
        if atm: ax.axvline(atm, color="0.4", lw=1, ls=":")
        ax.set_xscale("log"); ax.xaxis.set_major_locator(plt.LogLocator(numticks=4)); ax.xaxis.set_minor_formatter(plt.NullFormatter()); ax.set_xlabel(f"soil-air {lab}  [µatm]" + ("   (dotted = atmosphere)" if atm else ""), fontsize=9)
    axes[0].set_ylim(zdeep, 0); axes[0].set_ylabel("Depth [m]  (gas-filled cells only)")
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, label="time [years]", fraction=0.025, pad=0.01)
    plt.show()
