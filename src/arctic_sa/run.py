"""ATS output access: the Run class wraps one ats_vis_data.h5."""
import os
import h5py, numpy as np
from .observations import TOT, NTOP, GAS_OBS

class Run:
    """One finished ATS run. `path` = run folder (containing ats_vis_data.h5) or a saved .h5 file."""
    def __init__(self, path, label, expect_days=None):
        self.file = path if path.endswith(".h5") else os.path.join(path, "ats_vis_data.h5")
        self.label = label
        with h5py.File(self.file, "r") as f:
            self.days = sorted(int(c) for c in f["pH.0"])
        self.last = self.days[-1]                       # last output day
        self.years = self.last / 365.0
        if expect_days is not None and self.last < expect_days:
            raise RuntimeError(f"{label}: run stopped at day {self.last}, expected >= {expect_days}; not plotting an unfinished run")

    def profile(self, key, day=None):
        """Value in every cell at one output (default: the last), in uM (pH unscaled)."""
        with h5py.File(self.file, "r") as f:
            v = f[key][str(day or self.last)][:, 0]
        return v if key == "pH.0" else v * 1e6

    def profile_mean(self, key, window=365):
        """Mean over time, in every cell, of the outputs of the last `window` days (default: the last year; same units as `profile`)."""
        ds = [d for d in self.days if d > 0 and d > self.last - window]
        with h5py.File(self.file, "r") as f: v = np.array([f[key][str(d)][:, 0] for d in ds])
        return v.mean(axis=0) if key == "pH.0" else v.mean(axis=0) * 1e6

    def profile_sd(self, key, window=365, log=False):
        """SD over time, in every cell, of the outputs of the last `window` days (same units as `profile`; of log10 of the values if log=True)."""
        ds = [d for d in self.days if d > 0 and d > self.last - window]
        with h5py.File(self.file, "r") as f: v = np.array([f[key][str(d)][:, 0] for d in ds])
        v = v if key == "pH.0" else v * 1e6
        return np.log10(np.maximum(v, 1e-30)).std(axis=0) if log else v.std(axis=0)

    def series(self, key):
        """(years, mean over the top 35 cells) at every output after the first (the first has no pH)."""
        with h5py.File(self.file, "r") as f:
            cy = self.days[1:]
            y = np.array([f[key][str(c)][:NTOP, 0].mean() for c in cy])
        return np.array(cy) / 365.0, (y if key == "pH.0" else y * 1e6)

    def series_sd(self, key):
        """(years, SD over the top 35 cells) at every output after the first, same units as `series`."""
        with h5py.File(self.file, "r") as f:
            cy = self.days[1:]
            y = np.array([f[key][str(c)][:NTOP, 0].std() for c in cy])
        return np.array(cy) / 365.0, (y if key == "pH.0" else y * 1e6)

    def gas(self, day=None):
        """Soil-air partial pressures [uatm] in the cells that contain a gas phase, at one output."""
        with h5py.File(self.file, "r") as f:
            c = str(day or self.last); sg = f["saturation_gas"][c][:, 0]; p = f["pressure"][c][:, 0] / 101325.0 * 1e6
            pg = {g: f["gas_mole_fraction." + g][c][:, 0] * p for g in GAS_OBS}
        return sg > 1e-6, pg

    def clamped_cells(self):
        """Cells in the top 35 whose proton total sits at the ATS clip floor (pH invalid there)."""
        with h5py.File(self.file, "r") as f:
            return int((f[TOT + "H+"][str(self.last)][:NTOP, 0] <= 1e-30).sum())

    def __repr__(self): return f"Run({self.label!r}, last day {self.last})"
