"""Build and run ensemble members.  Every member restarts from the equilibrated baseline checkpoint (no spinup) in its own folder.
Runs are resumable: a member whose result file already exists is skipped.  Nothing here starts a run unless run_design() is called."""
import glob, json, os, platform, re, shutil, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from .parameters import PARAMS, NAMES
from .templates import apply_values, read_lines, TemplateError

RESTART_LOCAL = "../../run/restart.h5"        # path of the restart file inside a member (relative to run/run_rxn)

def manifest(cfg, design):
    """provenance record written next to every design / result set"""
    import numpy, pandas, scipy, sklearn
    try: git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=cfg.study_dir).stdout.strip()
    except Exception: git = ""
    from . import __version__
    return dict(design=design, created=time.strftime("%Y-%m-%d %H:%M:%S"), config_hash=cfg.config_hash, years=cfg.years, restart=cfg.restart, git_commit=git,
                versions=dict(arctic_sa=__version__, python=platform.python_version(), numpy=numpy.__version__, pandas=pandas.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__),
                seeds=dict(morris=cfg.morris["seed"], lhs=cfg.lhs["seed"]), parameters=[list(p) for p in PARAMS])

def check(cfg):
    """verify that everything a run needs is present; returns a list of (ok, message)"""
    out = []
    def add(ok, msg): out.append((bool(ok), msg))
    add(shutil.which(cfg.ats_executable) or os.path.isfile(cfg.ats_executable), f"ATS executable: {cfg.ats_executable}")
    for lab, p in (("restart file", cfg.restart), ("reaction database", cfg.database), ("data directory", cfg.data_dir)): add(os.path.exists(p), f"{lab}: {p}")
    for lab, pat in (("reaction .in", cfg.reaction_in_glob), ("ATS XML", cfg.xml_glob)): add(len(glob.glob(pat)) == 1, f"{lab}: {pat} (exactly 1 match)")
    try:
        from .templates import base_values; b = base_values(cfg, PARAMS); add(len(b) == len(PARAMS), f"baseline values read for all {len(PARAMS)} parameters")
    except Exception as e: add(False, f"reading baseline values failed: {e}")
    return out

def build_member(cfg, mdir, values):
    """create the member folder (data copy, restart copy, edited .in/.dat/XML); returns the XML file name"""
    shutil.rmtree(mdir, ignore_errors=True); os.makedirs(mdir + "/inputs"); os.makedirs(mdir + "/run/run_rxn")
    shutil.copytree(cfg.data_dir, mdir + "/data", ignore=shutil.ignore_patterns("*.alquimia")); shutil.copy(cfg.restart, mdir + "/run/restart.h5")
    inf = glob.glob(mdir + "/data/pflotran_dir/" + os.path.basename(cfg.reaction_in_glob))
    if len(inf) != 1: raise TemplateError(f"reaction .in not found in the copied data directory ({os.path.basename(cfg.reaction_in_glob)})")
    inf = inf[0]; dbf = mdir + "/data/" + os.path.relpath(cfg.database, cfg.data_dir)
    Lin, Ldb, Lx = read_lines(inf), read_lines(dbf), read_lines(cfg.xml); apply_values(Lin, Ldb, Lx, values, PARAMS)
    open(inf, "w").write("\n".join(Lin)); open(dbf, "w").write("\n".join(Ldb)); t = "\n".join(Lx)
    t, n = re.subn(r'(name="restart file" type="string" value=")[^"]*(")', rf'\g<1>{RESTART_LOCAL}\g<2>', t)
    if n == 0: raise TemplateError('no <Parameter name="restart file"> found in the ATS XML: cannot point the member at the baseline checkpoint')
    t = re.sub(r'(name="end time" type="double" value=")[^"]*"', rf'\g<1>{cfg.years}.0"', t, count=1)
    xn = os.path.basename(cfg.xml); open(mdir + "/inputs/" + xn, "w").write(t); return xn

def _run_member(args):
    cfg, design, mid, values = args; res = cfg.results_dir(design); os.makedirs(res, exist_ok=True)
    if os.path.exists(f"{res}/{mid}_ats_vis_data.h5"): return mid, "skipped (done)"
    mdir = os.path.join(cfg.output_dir, "members", design, mid); xn = build_member(cfg, mdir, values); t0 = time.time()
    cmd = [cfg.ats_executable, *cfg.ats_args, f"../../inputs/{xn}"]
    with open(mdir + "/run/run_rxn/out.log", "w") as log: r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=mdir + "/run/run_rxn")
    ok = r.returncode == 0 and os.path.exists(mdir + "/run/run_rxn/ats_vis_data.h5")
    if ok:
        shutil.copy(mdir + "/run/run_rxn/ats_vis_data.h5", f"{res}/{mid}_ats_vis_data.h5"); shutil.copy(glob.glob(mdir + "/data/pflotran_dir/*_Rxn_*.in")[0], f"{res}/{mid}.in")
        if not cfg.keep_members: shutil.rmtree(mdir, ignore_errors=True)
    else: open(f"{res}/{mid}.FAILED", "w").write(f"return code {r.returncode}")
    return mid, ("ok %.0f s" % (time.time() - t0)) if ok else "FAILED"

def plan(cfg, design, df):
    """what run_design would do: members still to run and the command line (no side effects)"""
    res = cfg.results_dir(design); todo = [m for m in df.index if not os.path.exists(f"{res}/{m}_ats_vis_data.h5")]
    return dict(members=len(df), to_run=len(todo), skipped=len(df) - len(todo), command=[cfg.ats_executable, *cfg.ats_args, "<member>/inputs/" + os.path.basename(cfg.xml)],
                workers=cfg.workers, years=cfg.years)

def run_design(cfg, design, df, workers=None):
    """run every member of `df` (index = member id, columns = parameter names).  Resumable.  Writes the design CSV, manifest.json, log and .done marker."""
    missing = [n for n in NAMES if n not in df.columns]
    if missing: raise ValueError(f"design lacks parameter columns: {missing}")
    os.makedirs(os.path.dirname(cfg.design_file(design)), exist_ok=True); df[NAMES].to_csv(cfg.design_file(design), index_label="member")
    res = cfg.results_dir(design); os.makedirs(res, exist_ok=True); mf = f"{res}/manifest.json"
    if os.path.exists(mf):
        old = json.load(open(mf))
        if old.get("config_hash") != cfg.config_hash or old.get("years") != cfg.years: raise RuntimeError(f"{res} holds results made with a different config/years; use a new output_dir or delete it")
    else: json.dump(manifest(cfg, design), open(mf, "w"), indent=1)
    jobs = [(cfg, design, mid, row.to_dict()) for mid, row in df.iterrows()]; log = os.path.join(cfg.output_dir, f"{design}.log")
    with ThreadPoolExecutor(workers or cfg.workers) as ex:
        for mid, msg in ex.map(_run_member, jobs):
            with open(log, "a") as f: f.write(f"{time.strftime('%H:%M:%S')} {mid} {msg}\n")
    open(os.path.join(cfg.output_dir, f"{design}.done"), "w").write("DONE")
