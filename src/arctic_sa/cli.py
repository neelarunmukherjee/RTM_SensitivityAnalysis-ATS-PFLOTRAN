"""Command line: arctic-sa {check,design,run,analyse,verify,report} --config configs/arctic_c8.yaml.
Nothing launches an ATS run unless `run` / `verify` is given --yes; without it they print what they would do (dry run)."""
import argparse, json, os, sys
import pandas as pd
from .config import load_config, ConfigError

def _cfg(a): return load_config(a.config)

def cmd_check(a):
    from .runner import check
    res = check(_cfg(a))
    for ok, msg in res: print(("OK   " if ok else "FAIL ") + msg)
    return 0 if all(ok for ok, _ in res) else 1

def cmd_design(a):
    from . import designs
    cfg = _cfg(a)
    df = designs.morris_design(**cfg.morris) if a.method == "morris" else designs.lhs_design(cfg) if a.method == "lhs" else designs.control_design(cfg)
    f = cfg.design_file(a.method); os.makedirs(os.path.dirname(f), exist_ok=True); df.to_csv(f, index_label="member"); print(f"{len(df)} members -> {f}"); return 0

def _design_df(cfg, name, file=None):
    from .parameters import NAMES
    f = file or cfg.design_file(name)
    if not os.path.exists(f): raise ConfigError(f"design file {f} not found; run `arctic-sa design --method {name}` first")
    return pd.read_csv(f, index_col=0)[NAMES]

def cmd_run(a, name=None, years=None):
    from . import runner
    cfg = _cfg(a); name = name or a.design
    if a.workers: cfg = cfg.with_(workers=a.workers)
    if years or a.years: cfg = cfg.with_(years=years or a.years)
    df = _design_df(cfg, name, getattr(a, "file", None)); p = runner.plan(cfg, name, df)
    print(f"design {name}: {p['members']} members ({p['skipped']} already done, {p['to_run']} to run), {p['workers']} in parallel, {p['years']} simulated years each")
    print("command per member:", " ".join(p["command"]))
    if not a.yes: print("dry run: nothing started.  Add --yes to launch the runs."); return 0
    bad = [m for ok, m in runner.check(cfg) if not ok]
    if bad: print("pre-flight check failed:\n  " + "\n  ".join(bad)); return 1
    runner.run_design(cfg, name, df); return 0

def cmd_verify(a):
    cfg = _cfg(a); a.design = "verify"; a.file = a.candidates or cfg.design_file("candidates"); return cmd_run(a, "verify", years=a.years or 6)

def _prep(cfg):
    from . import observations; observations.load(cfg)

def cmd_analyse(a):
    from . import morris, outputs, match
    cfg = _cfg(a); _prep(cfg); res = os.path.join(cfg.output_dir, "results"); rep = os.path.join(cfg.output_dir, "analysis"); os.makedirs(rep, exist_ok=True)
    mo = outputs.load_outputs(os.path.join(res, "morris"), cfg) if os.path.isdir(os.path.join(res, "morris")) or os.path.exists(os.path.join(res, "morris_outputs.csv")) else pd.DataFrame()
    if len(mo) >= 48:
        m = morris.run_morris(_design_df(cfg, "morris"), mo, cfg); m["ranking"].to_csv(f"{rep}/morris_ranking.csv"); print(m["ranking"].round(3).to_string())
        print("kept for the Latin hypercube:", int(m["ranking"].keep_for_LHS.sum()))
    else: print("Morris: not enough finished members (need >= 48 with outputs); skipped")
    lo = outputs.load_outputs(os.path.join(res, "lhs"), cfg) if os.path.isdir(os.path.join(res, "lhs")) or os.path.exists(os.path.join(res, "lhs_outputs.csv")) else pd.DataFrame()
    if len(lo) >= 100:
        fit = match.fit_lhs(_design_df(cfg, "lhs"), lo, seed=cfg.seed); fit["q2"].round(3).to_csv(f"{rep}/emulator_q2.csv"); print(fit["q2"].round(2).to_string())
        sob = match.sobol_scores(fit, cfg.sobol_n, cfg.seed)
        pd.DataFrame({f"{c}_{k}": v for c, (s1, st, _, _) in sob.items() for k, v in (("S1", s1), ("ST", st))}, index=fit["names"]).to_csv(f"{rep}/sobol.csv")
        hm = match.history_match(fit, cfg); print(f"NROY {hm['NROY'].mean() * 100:.2f} %; Pareto front {len(hm['front'])} points")
        if len(hm["front"]) >= 6: c = match.pick_candidates(fit, hm, cfg); c.to_csv(f"{rep}/candidates.csv", index_label="member"); print(f"candidates -> {rep}/candidates.csv  (copy to designs/candidates.csv or pass --candidates to `verify`)")
    else: print("Latin hypercube: not enough finished members (need >= 100); skipped")
    return 0

def cmd_report(a):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from . import observations, plots
    from .run import Run
    cfg = _cfg(a); observations.load(cfg); out = os.path.join(cfg.output_dir, "report"); os.makedirs(out, exist_ok=True)
    runs = [Run(f, os.path.basename(f).replace("_ats_vis_data.h5", "")) for f in a.runs]
    for r in runs:
        for fn in (plots.plot_profiles, plots.plot_bars, plots.plot_timeseries):
            plt.show = lambda *x, **k: None; fn(r); plt.gcf().savefig(f"{out}/{r.label}_{fn.__name__}.png", dpi=130, bbox_inches="tight"); plt.close("all")
    print(f"figures -> {out}"); return 0

def main(argv=None):
    p = argparse.ArgumentParser(prog="arctic-sa", description=__doc__); sub = p.add_subparsers(dest="cmd", required=True)
    def add(name, fn, **kw):
        s = sub.add_parser(name, **kw); s.add_argument("--config", required=True, help="path to the YAML config"); s.set_defaults(fn=fn); return s
    add("check", cmd_check, help="verify that ATS, templates and the restart file are in place")
    s = add("design", cmd_design, help="write a design CSV"); s.add_argument("--method", choices=["morris", "lhs", "controls"], required=True)
    s = add("run", cmd_run, help="run a design (dry run unless --yes)"); s.add_argument("--design", required=True); s.add_argument("--workers", type=int); s.add_argument("--years", type=int); s.add_argument("--yes", action="store_true")
    s = add("verify", cmd_verify, help="run the candidates for 6 years (dry run unless --yes)"); s.add_argument("--candidates"); s.add_argument("--workers", type=int); s.add_argument("--years", type=int); s.add_argument("--yes", action="store_true")
    add("analyse", cmd_analyse, help="Morris ranking, emulator, Sobol, history matching from the finished outputs")
    s = add("report", cmd_report, help="save standard figures for finished runs (.h5)"); s.add_argument("runs", nargs="+")
    a = p.parse_args(argv)
    try: return a.fn(a)
    except ConfigError as e: print("config error:", e, file=sys.stderr); return 2

if __name__ == "__main__": sys.exit(main())
