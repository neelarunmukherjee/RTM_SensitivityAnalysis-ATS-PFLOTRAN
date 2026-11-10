# arctic-sa

Reproducible global sensitivity analysis and calibration support for the **Arctic soil-column ATS + PFLOTRAN (Alquimia) model** (config 8, Imnavait):

```
Morris screening (23 parameters, 240 runs)  ->  Latin hypercube on the 17 retained parameters (200 runs)
   ->  Gaussian-process emulator (cross-validated Q2)  ->  Sobol indices  ->  history matching (NROY, Pareto)
   ->  verification runs of the best candidates (6 years)
```

Every run restarts from the equilibrated baseline checkpoint (no spinups). Outputs are scored against the field concentrations
(CO2, CH4, DOC, Fe2+, Fe(III), pH) and against measured incubation rates.

## Install
```bash
conda env create -f environment.yml && conda activate arctic-sa      # or:  pip install -e .[dev]
pytest                                                                  # 13 tests, ~3 s; the 4 that compare with the study folder need ARCTIC_SA_STUDY set
```
**ATS (with PFLOTRAN/Alquimia) is not pip- or conda-installable**: build it separately and put `ats` on `PATH` (or set `ats.executable`
in the config). It is only needed to *run* members; the analysis works from the cached outputs.

## Quick start: reproduce the published analysis without ATS
```bash
arctic-sa analyse --config examples/example.yaml      # Morris ranking, emulator Q2, Sobol, history matching (about 3 min)
```
The tutorial `notebooks/tutorial.ipynb` walks through the same steps with figures.

## Full workflow (needs ATS and the study folder)
All settings live in one YAML file (`configs/arctic_c8.yaml`, paths relative to the file). Point it at your study folder (the one holding `data/`, `inputs/`, `run/`) with an environment variable:
```bash
export ARCTIC_SA_STUDY=/path/to/config8_sensitivity_analysis
arctic-sa check   --config configs/arctic_c8.yaml                    # ats, templates, restart file, baseline values
arctic-sa design  --method morris --config configs/arctic_c8.yaml    # also: lhs, controls  -> sa/designs/*.csv
arctic-sa run     --design morris --config configs/arctic_c8.yaml    # DRY RUN: prints members and command
arctic-sa run     --design morris --config configs/arctic_c8.yaml --yes     # launches the runs (resumable)
arctic-sa analyse --config configs/arctic_c8.yaml                    # ranking, emulator, Sobol, NROY, candidates -> sa/analysis/
arctic-sa verify  --candidates sa/analysis/candidates.csv --config configs/arctic_c8.yaml --yes   # 6-year runs
arctic-sa report  sa/results/verify/cand3_ats_vis_data.h5 --config configs/arctic_c8.yaml          # standard figures
```
`run` and `verify` never start anything without `--yes`. Re-running a command continues where it stopped (finished members are skipped).
Each result folder gets a `manifest.json` (config hash, git commit, package versions, seeds, parameter table); a folder made with another
config or simulation length is refused.

## Python API
```python
from arctic_sa import load_config, observations, designs, outputs, morris, match
cfg = load_config("configs/arctic_c8.yaml"); observations.load(cfg)
design = designs.morris_design(**cfg.morris)                     # same 240 members as the paper (seed 1)
out = outputs.load_outputs("sa/results/morris", cfg)             # score finished members (cached CSV)
rank = morris.run_morris(design, out, cfg)["ranking"]
fit = match.fit_lhs(lhs_design, lhs_out); hm = match.history_match(fit, cfg)
```

## Layout
| module | role |
|---|---|
| `parameters` | the 23 parameters, ranges, the 17 retained for the Latin hypercube |
| `templates` | read/write one value in the reaction `.in`, the database `.dat`, the ATS XML (fails loudly if the layout differs) |
| `designs` | Morris trajectories, Latin hypercube, controls |
| `runner` | build member folders, run `ats`, resumable, manifests, pre-flight `check` |
| `run`, `observations`, `rates`, `outputs` | ATS output access, field and incubation data, rate-law rebuild, scored outputs |
| `morris`, `emulator`, `match` | elementary effects, GP emulator + Sobol, history matching, Pareto, candidates |
| `plots` | figures (no gridlines) |

## Scope and limits
* Specific to this model: parameter names, template layout and the scoring are tied to the config-8 Arctic column. `parameters.py` is the one place to extend.
* Never varies `alpha_CO2` / `alpha_CH4`. Large model data (meshes, forcing, restart `.h5`) are referenced by path, not shipped.
* About half of the ensemble members have clamped H+ cells (ATS clips negative proton totals); the Morris ranking excludes steps with clamped members, as in the paper.
* The equilibration/spinup (`eq_continue.py`, `run_all.sh`) is a prerequisite and is not part of this package.
