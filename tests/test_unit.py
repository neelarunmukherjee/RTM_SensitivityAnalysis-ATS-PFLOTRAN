import numpy as np, pytest, yaml, os
from arctic_sa import config, templates, parameters as P
from arctic_sa.morris import elementary_effects, morris_table, rank_parameters

def test_unit_roundtrip():
    for n in P.NAMES:
        u = np.array([0.0, 0.3, 1.0]); np.testing.assert_allclose(P.to_unit(P.from_unit(u, n), n), u, atol=1e-12)
def test_bounds():
    for sp in P.PARAMS: assert P.from_unit(0.0, sp[0]) == pytest.approx(sp[4]) and P.from_unit(1.0, sp[0]) == pytest.approx(sp[5])
def test_cap_parameters_never_include_alphas(): assert not any("alpha" in n.lower() for n in P.NAMES)
def test_wrong_block_count_raises():
    with pytest.raises(templates.TemplateError): templates.in_get_set(["MICROBIAL_REACTION", "/"], P.PARAMS[0])
def test_config_errors(tmp_path):
    with pytest.raises(config.ConfigError): config.load_config(tmp_path / "missing.yaml")
    f = tmp_path / "c.yaml"; f.write_text(yaml.safe_dump({"study_dir": "."})); 
    with pytest.raises(config.ConfigError): config.load_config(f)
def test_config_paths_relative_to_file(cfg): assert os.path.isabs(cfg.restart) and cfg.restart.endswith("yr09_end.h5") and cfg.workers == 8 and cfg.years == 2
def test_morris_recovers_known_effects():
    """linear model y = 3 u0 + 0 u1: mu*(u0) = 3, mu*(u1) = 0"""
    import pandas as pd
    from arctic_sa.designs import morris_design
    d = morris_design(r=4, candidates=20, seed=1); u = __import__("arctic_sa.morris", fromlist=["unit"]).unit(d)
    out = pd.DataFrame({"y": 3 * u[P.NAMES[0]]}, index=d.index); EE, cols = elementary_effects(d, out); mu_s = morris_table(EE, cols, nboot=20)[0]
    assert mu_s[0, 0] == pytest.approx(3.0) and mu_s[1:, 0].max() == pytest.approx(0.0, abs=1e-9)

def test_example_morris_reproduces_published_ranking():
    """cached outputs shipped in examples/: R1 and R2 on top, 15 parameters flagged"""
    import pandas as pd
    from arctic_sa import load_config, observations
    from arctic_sa.morris import run_morris
    here = os.path.dirname(__file__); cfg = load_config(os.path.join(here, "..", "examples", "example.yaml")); observations.load(cfg)
    d = pd.read_csv(os.path.join(cfg.output_dir, "designs", "morris.csv"), index_col=0); out = pd.read_csv(os.path.join(cfg.output_dir, "results", "morris_outputs.csv"), index_col=0)
    r = run_morris(d, out, cfg)["ranking"]
    assert list(r.index[:2]) == ["R1_k_DOM_aerobic", "R2_k_fermentation"] and int(r.keep_for_LHS.sum()) == 15
