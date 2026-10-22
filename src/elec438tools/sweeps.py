"""Parameter sweeps (e.g. neff vs width) and wavelength sweeps (neff, ng, D vs wavelength)."""

from __future__ import annotations

import warnings
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from .solver import ModeSolver, _overlap

C_UM_PER_S = 2.99792458e14


class SweepResult:
    """Mode data at each value of a swept parameter, split by polarization.

    Arrays are shaped (len(values), max modes) and padded with NaN; within each polarization
    the modes are ordered by neff, so column 0 of neff_te is TE0, column 1 is TE1, and so on.
    Modes that are not guided are NaN unless the sweep was run with guided_only=False.

    Attributes: values, wavelength, neff_te, neff_tm, ng_te, ng_tm (if group_index=True),
    n_clad (per value), modes (list of ModeSet per value, if keep_modes=True).
    """

    def __init__(self, values, wavelength, modesets, param_name, guided_only):
        self.values = np.asarray(values, float)
        self.wavelength = wavelength
        self.param_name = param_name
        self.n_clad = np.array([ms[0].n_clad if ms else np.nan for ms in modesets])
        self.modes = None
        for pol in ("TE", "TM"):
            rows_n, rows_g = [], []
            for ms in modesets:
                sel = [m for m in ms if m.polarization == pol and (m.guided or not guided_only)]
                rows_n.append([m.neff.real for m in sel])
                rows_g.append([m.ng if m.ng is not None else np.nan for m in sel])
            width = max([len(r) for r in rows_n] + [1])
            pad = lambda rows: np.array([r + [np.nan] * (width - len(r)) for r in rows])
            setattr(self, f"neff_{pol.lower()}", pad(rows_n))
            setattr(self, f"ng_{pol.lower()}", pad(rows_g))

    def num_guided(self, polarization: str | None = None) -> np.ndarray:
        """Number of guided modes at each value (TE, TM, or both if None)."""
        arrays = [self.neff_te, self.neff_tm] if polarization is None else [getattr(self, f"neff_{polarization.lower()}")]
        return sum(np.sum(np.isfinite(a), axis=1) for a in arrays)

    def plot(self, quantity: str = "neff", ax=None):
        """neff (or ng) vs the swept value; TE solid, TM dashed; dotted line = cladding index."""
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 4.2))
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for pol, ls in (("te", "-"), ("tm", "--")):
            A = getattr(self, f"{quantity}_{pol}")
            for k in range(A.shape[1]):
                if np.isfinite(A[:, k]).any():
                    ax.plot(self.values, A[:, k], ls, marker="o", ms=3, color=colors[k % len(colors)],
                            label=f"{pol.upper()}{k}")
        if quantity == "neff":
            ax.plot(self.values, self.n_clad, ":", color="gray", label="cladding index")
        ax.set_xlabel(self.param_name)
        ax.set_ylabel({"neff": "effective index", "ng": "group index"}.get(quantity, quantity))
        ax.set_title(f"{quantity} at {self.wavelength} um")
        ax.legend(fontsize=8, ncol=2)
        return ax

    def to_dataframe(self):
        import pandas as pd
        rows = []
        for pol in ("TE", "TM"):
            A, G = getattr(self, f"neff_{pol.lower()}"), getattr(self, f"ng_{pol.lower()}")
            for i, v in enumerate(self.values):
                for k in range(A.shape[1]):
                    if np.isfinite(A[i, k]):
                        rows.append({self.param_name: v, "mode": f"{pol}{k}", "neff": A[i, k], "ng": G[i, k]})
        return pd.DataFrame(rows)


def sweep(build: Callable[[float], ModeSolver], values: Sequence[float], wavelength: float = 1.55,
          num_modes: int = 6, param_name: str = "parameter", guided_only: bool = True,
          keep_modes: bool = False, **solve_kwargs) -> SweepResult:
    """Rebuild and solve the cross-section for each value.

    build: function value -> ModeSolver, e.g.  lambda w: strip_waveguide(width=w)
    **solve_kwargs: passed to ModeSolver.solve (n_guess, group_index, bend_radius).

    Example:  res = sweep(lambda w: strip_waveguide(width=w), np.arange(0.3, 1.01, 0.05),
                          param_name="width (um)");  res.plot()
    """
    modesets = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # boundary warnings would repeat at every point
        for v in values:
            modesets.append(build(v).solve(wavelength, num_modes, **solve_kwargs))
    res = SweepResult(values, wavelength, modesets, param_name, guided_only)
    if keep_modes:
        res.modes = modesets
    return res


class DispersionResult:
    """Modes followed across wavelength. Arrays shaped (len(wavelengths), num_modes).

    neff : effective index (real part)
    ng   : group index  ng = neff - wavelength * dneff/dwavelength
    D    : dispersion parameter in ps/(nm km); D = (1/c) dng/dwavelength = -(wavelength/c) d2neff/dwavelength2
    names: names of the followed modes at the first wavelength
    """

    def __init__(self, wavelengths, neff, names):
        self.wavelengths = np.asarray(wavelengths, float)
        self.neff = np.asarray(neff)
        self.names = names
        wl = self.wavelengths
        n = self.neff.real
        if len(wl) >= 3:
            self.ng = n - wl[:, None] * np.gradient(n, wl, axis=0, edge_order=2)
            dng = np.gradient(self.ng, wl, axis=0, edge_order=2)  # 1/um
            self.D = dng / C_UM_PER_S * 1e18  # s/um^2 -> ps/(nm km)
        else:
            self.ng = self.D = np.full_like(n, np.nan)

    def plot(self, quantity: str = "ng", ax=None):
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 4.2))
        A = {"neff": self.neff.real, "ng": self.ng, "D": self.D}[quantity]
        for k, name in enumerate(self.names):
            ax.plot(self.wavelengths, A[:, k], marker="o", ms=3, label=name or f"mode {k}")
        ax.set_xlabel("wavelength (um)")
        ax.set_ylabel({"neff": "effective index", "ng": "group index", "D": "D (ps/(nm km))"}[quantity])
        if quantity == "D":
            ax.axhline(0, color="gray", lw=0.8)
        ax.legend()
        return ax

    def to_dataframe(self):
        import pandas as pd
        rows = []
        for i, wl in enumerate(self.wavelengths):
            for k, name in enumerate(self.names):
                rows.append(dict(wavelength=wl, mode=name or k, neff=self.neff[i, k].real, ng=self.ng[i, k], D=self.D[i, k]))
        return pd.DataFrame(rows)


def wavelength_sweep(sim: ModeSolver, wavelengths, num_modes=2, n_guess=None) -> DispersionResult:
    wavelengths = np.sort(np.asarray(wavelengths, float))
    first = sim.solve(wavelengths[0], num_modes, n_guess)
    names = [m.name for m in first]
    prev = [m._yee for m in first]
    neff = [[m.neff for m in first]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for wl in wavelengths[1:]:
            cand = sim._raw(wl, num_modes + 2, n_guess or sim._check(wl), None)
            cost = np.array([[-_overlap(p, c[1]) for c in cand] for p in prev])
            rows, cols = linear_sum_assignment(cost)
            order = cols[np.argsort(rows)]
            neff.append([cand[j][0] for j in order])
            prev = [cand[j][1] for j in order]
    return DispersionResult(wavelengths, np.array(neff), names)
