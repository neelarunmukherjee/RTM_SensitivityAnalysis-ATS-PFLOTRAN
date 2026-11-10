import os, sys, pytest
HERE = os.path.dirname(os.path.abspath(__file__)); CFG = os.path.join(HERE, "..", "configs", "arctic_c8.yaml")
STUDY = os.environ.get("ARCTIC_SA_STUDY", "")
@pytest.fixture(scope="session")
def cfg():
    from arctic_sa import load_config
    if not STUDY: pytest.skip("set ARCTIC_SA_STUDY to run the tests that need the study folder")
    c = load_config(CFG)
    if not os.path.isdir(c.data_dir): pytest.skip("study data not present")
    return c
@pytest.fixture(scope="session")
def old():
    """the original sa/sa_lib.py (parity reference)"""
    p = os.path.join(STUDY, "sa")
    if not os.path.exists(os.path.join(p, "sa_lib.py")): pytest.skip("legacy sa/ not present")
    sys.path.insert(0, p); import sa_lib; return sa_lib
