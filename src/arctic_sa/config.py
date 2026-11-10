"""YAML configuration -> Settings.  Relative paths are resolved against the config file, so a study can be moved or shared."""
import glob, hashlib, os
from dataclasses import dataclass, field
import yaml

class ConfigError(Exception): pass

@dataclass
class Settings:
    study_dir: str; reaction_in_glob: str; database: str; xml_glob: str; data_dir: str; restart: str
    ats_executable: str = "ats"; ats_args: tuple = ("--verbosity=low",)
    workers: int = 8; years: int = 2; keep_members: bool = False; output_dir: str = "sa"
    morris: dict = field(default_factory=lambda: dict(r=10, levels=4, candidates=300, seed=1))
    lhs: dict = field(default_factory=lambda: dict(n=200, seed=2))
    tolerance: float = 1.33; sigma_ph: float = 0.15; rate_factor: float = 2.0; top_cells: int = 35
    morris_keep_threshold: float = 0.1; sobol_n: int = 4096; seed: int = 0
    field_chem: str = ""; lit_rates: str = ""; config_hash: str = ""

    def _one(self, pattern):
        hits = sorted(glob.glob(pattern))
        if len(hits) != 1: raise ConfigError(f"expected exactly one file for {pattern!r}, found {len(hits)}")
        return hits[0]
    @property
    def reaction_in(self): return self._one(self.reaction_in_glob)
    @property
    def xml(self): return self._one(self.xml_glob)
    @property
    def out(self): return self.output_dir
    def results_dir(self, design): return os.path.join(self.output_dir, "results", design)
    def design_file(self, design): return os.path.join(self.output_dir, "designs", f"{design}.csv")
    def with_(self, **kw):
        import dataclasses; return dataclasses.replace(self, **kw)

def load_config(path):
    path = os.path.abspath(path)
    if not os.path.isfile(path): raise ConfigError(f"config file not found: {path}")
    raw = open(path, "rb").read(); c = yaml.safe_load(raw); base = os.path.dirname(path)
    for k in ("study_dir", "templates", "restart"):
        if k not in c: raise ConfigError(f"missing required key {k!r} in {path}")
    for k in ("reaction_in", "database", "xml", "data_dir"):
        if k not in c["templates"]: raise ConfigError(f"missing templates.{k}")
    sd = os.path.expandvars(str(c["study_dir"]))
    if "$" in sd: raise ConfigError(f"study_dir {c['study_dir']!r} uses an environment variable that is not set (export ARCTIC_SA_STUDY=/path/to/study)")
    study = os.path.normpath(os.path.join(base, sd)); P = lambda p: p if os.path.isabs(p) else os.path.join(study, p)
    t, ats, run, an, d = c["templates"], c.get("ats", {}), c.get("run", {}), c.get("analysis", {}), c.get("data", {})
    des = c.get("design", {})
    s = Settings(study_dir=study, reaction_in_glob=P(t["reaction_in"]), database=P(t["database"]), xml_glob=P(t["xml"]), data_dir=P(t["data_dir"]),
                 restart=P(c["restart"]), ats_executable=ats.get("executable", "ats"), ats_args=tuple(ats.get("args", ["--verbosity=low"])),
                 workers=int(run.get("workers", 8)), years=int(run.get("years", 2)), keep_members=bool(run.get("keep_members", False)),
                 output_dir=P(c.get("output_dir", "sa")), tolerance=float(an.get("tolerance", 1.33)), sigma_ph=float(an.get("sigma_ph", 0.15)),
                 rate_factor=float(an.get("rate_factor", 2.0)), top_cells=int(an.get("top_cells", 35)),
                 morris_keep_threshold=float(an.get("morris_keep_threshold", 0.1)), sobol_n=int(an.get("sobol_n", 4096)), seed=int(an.get("seed", 0)),
                 field_chem=P(d.get("field_chem", "data/field_chem.csv")), lit_rates=P(d.get("lit_rates", "data/lit_rates.csv")),
                 config_hash=hashlib.sha256(raw).hexdigest()[:12])
    if "morris" in des: s.morris = {**s.morris, **des["morris"]}
    if "lhs" in des: s.lhs = {**s.lhs, **des["lhs"]}
    if s.workers < 1 or s.years < 1: raise ConfigError("run.workers and run.years must be >= 1")
    return s
