"""The package must reproduce the legacy sa/ code exactly (designs, template edits, member files)."""
import os, numpy as np, pandas as pd, pytest
from arctic_sa import designs, templates, runner
from arctic_sa.parameters import PARAMS, NAMES, LHS_PARAMETERS

def test_parameter_table_identical(old): assert [tuple(p) for p in PARAMS] == [tuple(p) for p in old.PARAMS]

def test_base_values_match(cfg, old):
    new = templates.base_values(cfg, PARAMS); o = old.base_values()
    assert set(new) == set(NAMES) and all(new[k] == pytest.approx(o[k], rel=0, abs=0) for k in NAMES)

def test_morris_design_matches_saved_design(cfg, old):
    d = designs.morris_design(r=10, seed=1); saved = pd.read_csv(os.path.join(cfg.output_dir, "designs", "morris.csv"), index_col=0)
    assert d.shape == (240, 23) and list(d.index) == list(saved.index); np.testing.assert_allclose(d[NAMES].to_numpy(), saved[NAMES].to_numpy(), rtol=1e-12)

def test_lhs_design_matches_saved_design(cfg):
    d = designs.lhs_design(cfg); saved = pd.read_csv(os.path.join(cfg.output_dir, "designs", "lhs.csv"), index_col=0)
    assert d.shape == (200, 23); np.testing.assert_allclose(d[NAMES].to_numpy(), saved[NAMES].to_numpy(), rtol=1e-12)

def test_member_files_identical_to_legacy(cfg, old, tmp_path):
    vals = pd.read_csv(os.path.join(cfg.output_dir, "designs", "lhs.csv"), index_col=0).iloc[3][NAMES].to_dict()
    new_dir, old_dir = str(tmp_path / "new"), str(tmp_path / "old"); xn = runner.build_member(cfg, new_dir, vals); old.build_member(old_dir, vals)
    import glob
    pairs = [("data/pflotran_dir/" + os.path.basename(cfg.reaction_in), None), (os.path.relpath(cfg.database, cfg.study_dir), None), ("inputs/" + xn, None)]
    for rel, _ in pairs:
        assert open(os.path.join(new_dir, rel)).read() == open(os.path.join(old_dir, rel)).read(), rel
