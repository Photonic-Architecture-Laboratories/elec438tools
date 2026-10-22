"""Mode and ModeSet: solved modes, their properties, and plotting/saving."""

from __future__ import annotations

import re
import warnings

import numpy as np

from . import fde

DB_PER_NEPER_POWER = 10 / np.log(10)  # 4.343 dB per neper of power


class Mode:
    """One solved mode. Fields are on the cell centres, arrays indexed [ix, iy].

    Field normalisation: max |E| = 1, with H in the matching physical units (|H| ~ n/377 ~ 3e-3).

    Attributes
    ----------
    neff : complex effective index (Im > 0 means loss).
    name : "TE0", "TM0", "TE1", ... for guided modes; "" for modes that are not guided.
    polarization : "TE" if most transverse E is along x (te_fraction >= 0.5), else "TM".
    te_fraction : integral |Ex|^2 / integral (|Ex|^2 + |Ey|^2)  (the "TE polarization fraction").
    guided : True if Re(neff) is above the highest index at the region edge (n_clad).
    loss_db_per_cm : propagation loss from Im(neff) (0 for lossless guided modes with metal walls).
    effective_area : (integral |E|^2)^2 / integral |E|^4  in um^2 (in um for a 1D problem).
    ng : group index, if solve(..., group_index=True) or a sweep computed it; else None.
    x, y : coordinates (um) of the field arrays.
    """

    def __init__(self, index, neff, yee, wavelength, sim, n_clad, bend_radius=None):
        self.index = index
        self.wavelength = wavelength
        self.bend_radius = bend_radius
        self.n_clad = n_clad
        self._sim = sim
        self._axes = sim.axes
        ax, ay = self._axes
        C = fde.to_centers(yee, ax, ay)
        # normalise: max |E| = 1, dominant transverse component real and positive at its peak
        E2 = abs(C["Ex"]) ** 2 + abs(C["Ey"]) ** 2 + abs(C["Ez"]) ** 2
        scale = 1 / np.sqrt(E2.max())
        dom = "Ex" if np.sum(abs(C["Ex"]) ** 2) >= np.sum(abs(C["Ey"]) ** 2) else "Ey"
        peak = C[dom].flat[np.argmax(abs(C[dom]))]
        scale *= np.conj(peak) / abs(peak)
        self._yee = {k: v * scale for k, v in yee.items()}
        self.fields = {k: v * scale for k, v in C.items()}
        self.neff = complex(neff)
        self.x, self.y = ax.centers.copy(), ay.centers.copy()
        self._w = np.outer(ax.h, ay.h)
        self.power = fde.yee_power(self._yee, ax, ay)
        ex = np.sum(abs(self.fields["Ex"]) ** 2 * self._w)
        ey = np.sum(abs(self.fields["Ey"]) ** 2 * self._w)
        self.te_fraction = float(ex / (ex + ey))
        self.polarization = "TE" if self.te_fraction >= 0.5 else "TM"
        self.guided = bool(self.neff.real > n_clad + 1e-9)
        self.name = ""
        self.ng = None
        e2 = self.E2
        self.effective_area = float(np.sum(e2 * self._w) ** 2 / np.sum(e2**2 * self._w))
        self.boundary_fraction = self._boundary_fraction()

    # --- field access --------------------------------------------------------------
    @property
    def Ex(self): return self.fields["Ex"]
    @property
    def Ey(self): return self.fields["Ey"]
    @property
    def Ez(self): return self.fields["Ez"]
    @property
    def Hx(self): return self.fields["Hx"]
    @property
    def Hy(self): return self.fields["Hy"]
    @property
    def Hz(self): return self.fields["Hz"]

    @property
    def E2(self) -> np.ndarray:
        """Electric field intensity |Ex|^2 + |Ey|^2 + |Ez|^2."""
        return abs(self.Ex) ** 2 + abs(self.Ey) ** 2 + abs(self.Ez) ** 2

    @property
    def Sz(self) -> np.ndarray:
        """z component of the time-averaged Poynting vector, 0.5 Re(E x H*)_z."""
        return 0.5 * np.real(self.Ex * np.conj(self.Hy) - self.Ey * np.conj(self.Hx))

    @property
    def beta(self) -> complex:
        """Propagation constant in rad/um."""
        return 2 * np.pi / self.wavelength * self.neff

    @property
    def loss_db_per_cm(self) -> float:
        alpha = 2 * (2 * np.pi / self.wavelength) * self.neff.imag  # power attenuation, 1/um
        return float(DB_PER_NEPER_POWER * alpha * 1e4)

    def get(self, quantity: str = "E2", part: str | None = None):
        """Return (array, label) for a quantity name; see plot() for the accepted names."""
        return _resolve(self, quantity, part)

    def power_fraction(self, region) -> float:
        """Fraction of the mode's power (integral of Sz) inside a shape or a (x_min, x_max, y_min, y_max) box.
        With the core shape this is the confinement factor."""
        X, Y = np.meshgrid(self.x, self.y, indexing="ij")
        if hasattr(region, "contains"):
            inside = region.contains(X, Y)
        else:
            x0, x1, y0, y1 = region
            inside = (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)
        S = self.Sz * self._w
        return float(S[inside].sum() / S.sum())

    def overlap(self, other: "Mode") -> float:
        """Power coupling efficiency between two modes on the same mesh (the power overlap)."""
        if self.Ex.shape != other.Ex.shape or not np.allclose(self.x, other.x) or not np.allclose(self.y, other.y):
            raise ValueError("overlap() needs both modes solved on the same mesh (same region and dx, dy).")
        w = self._w
        c12 = np.sum((self.Ex * np.conj(other.Hy) - self.Ey * np.conj(other.Hx)) * w)
        c21 = np.sum((other.Ex * np.conj(self.Hy) - other.Ey * np.conj(self.Hx)) * w)
        p1 = np.real(np.sum((self.Ex * np.conj(self.Hy) - self.Ey * np.conj(self.Hx)) * w))
        p2 = np.real(np.sum((other.Ex * np.conj(other.Hy) - other.Ey * np.conj(other.Hx)) * w))
        return float(np.real(c12 * c21 / p1) / p2)

    def _boundary_fraction(self):
        """Share of |E|^2 within a thin band (5 % of the span, >= 3 cells) along the walls."""
        ax, ay = self._axes
        e2 = self.E2 * self._w
        near = np.zeros(e2.shape, bool)
        for axis, a in ((0, ax), (1, ay)):
            if a.uniform:
                continue
            band = max(0.05 * (a.hi - a.lo), 3 * a.h.max())
            c = a.centers
            m = (c < a.lo + band) | (c > a.hi - band)
            if axis == 0:
                near[m, :] = True
            else:
                near[:, m] = True
        return float(e2[near].sum() / e2.sum())

    # --- plotting / export -----------------------------------------------------------
    def plot(self, quantity: str = "E2", part: str | None = None, ax=None, cmap: str | None = None,
             show_structure: bool = True, xlim=None, ylim=None):
        """Plot one quantity over the cross-section (a line vs y for a 1D problem).

        quantity: "Ex", "Ey", "Ez", "Hx", "Hy", "Hz" (magnitude by default),
                  "E" (|E|), "E2" (|E|^2, intensity), "H" (|H|), "H2", "Sz" (Poynting, power flow).
                  Wrapped forms also work: "|Ex|", "Re(Ex)", "Im(Hy)", "phase(Ex)", "|E|^2".
        part: "abs" | "real" | "imag" | "phase" for single components (overrides the wrapper).
        xlim, ylim: (min, max) in um to zoom in, e.g. xlim=(-0.6, 0.6).
        """
        from .plotting import plot_mode
        return plot_mode(self, quantity, part=part, ax=ax, cmap=cmap, show_structure=show_structure,
                         xlim=xlim, ylim=ylim)

    def plot_fields(self, cmap: str = "turbo", xlim=None, ylim=None):
        """2 x 3 panel of |Ex| |Ey| |Ez| / |Hx| |Hy| |Hz| with shared colour scales per row."""
        from .plotting import plot_fields
        return plot_fields(self, cmap=cmap, xlim=xlim, ylim=ylim)

    def plot_slice(self, quantity: str = "E2", x: float | None = None, y: float | None = None, ax=None):
        """Line cut. Give y= for a horizontal cut (vs x), x= for a vertical cut (vs y);
        with neither, plot both cuts through the field maximum."""
        from .plotting import plot_slice
        return plot_slice(self, quantity, x=x, y=y, ax=ax)

    def save(self, path: str):
        """Save fields and properties to .npz (NumPy) or .mat (scipy.io format)."""
        data = dict(x=self.x, y=self.y, neff=self.neff, wavelength=self.wavelength, name=self.name,
                    te_fraction=self.te_fraction, n_clad=self.n_clad, **self.fields)
        if self.ng is not None:
            data["ng"] = self.ng
        if str(path).lower().endswith(".mat"):
            from scipy.io import savemat
            savemat(path, data)
        else:
            np.savez(path, **data)

    def __repr__(self):
        tag = self.name or ("guided" if self.guided else "not guided")
        ng = f", ng={self.ng:.4f}" if self.ng is not None else ""
        return f"<Mode {self.index} {tag}: neff={_fmt_n(self.neff)}{ng}, TE fraction={100*self.te_fraction:.1f}%>"


class ModeSet(list):
    """List of modes sorted by Re(neff), highest first. Index by number or by name: modes["TE0"]."""

    def __init__(self, modes, wavelength):
        super().__init__(modes)
        self.wavelength = wavelength
        counts = {"TE": 0, "TM": 0}
        for m in self:
            if m.guided:
                m.name = f"{m.polarization}{counts[m.polarization]}"
                counts[m.polarization] += 1

    def __getitem__(self, key):
        if isinstance(key, str):
            for m in self:
                if m.name.lower() == key.lower():
                    return m
            names = [m.name for m in self if m.name] or ["(no guided modes)"]
            raise KeyError(f"No mode named {key!r}. Guided modes here: {', '.join(names)}.")
        return super().__getitem__(key)

    @property
    def neff(self) -> np.ndarray:
        return np.array([m.neff for m in self])

    def guided(self) -> "ModeSet":
        """Only the guided modes."""
        out = ModeSet([], self.wavelength)
        out.extend(m for m in self if m.guided)
        return out

    def table(self) -> str:
        has_ng = any(m.ng is not None for m in self)
        head = f"{'#':>3} {'name':<5} {'neff':>22} {'loss(dB/cm)':>12} {'TE frac':>8}"
        head += f" {'ng':>8}" if has_ng else ""
        head += f" {'Aeff(um2)':>10} {'guided':>7}"
        lines = [f"wavelength {self.wavelength} um; guided means Re(neff) > {self[0].n_clad:.4f} "
                 "(highest index at the region edge)" if self else "no modes", head]
        for m in self:
            row = f"{m.index:>3} {m.name or '-':<5} {_fmt_n(m.neff):>22} {m.loss_db_per_cm:>12.4g} {100*m.te_fraction:>7.1f}%"
            row += f" {m.ng:>8.4f}" if has_ng and m.ng is not None else (" " * 9 if has_ng else "")
            row += f" {m.effective_area:>10.4g} {'yes' if m.guided else 'no':>7}"
            lines.append(row)
        return "\n".join(lines)

    def __repr__(self):
        return self.table()

    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame([dict(index=m.index, name=m.name, neff=m.neff.real, neff_imag=m.neff.imag,
                                  loss_db_per_cm=m.loss_db_per_cm, te_fraction=m.te_fraction, ng=m.ng,
                                  effective_area=m.effective_area, guided=m.guided) for m in self])

    def plot(self, quantity: str = "E2", max_cols: int = 4, cmap: str = "turbo", xlim=None, ylim=None):
        """Grid of all modes, one panel each, titled with name and neff."""
        from .plotting import plot_modeset
        return plot_modeset(self, quantity, max_cols=max_cols, cmap=cmap, xlim=xlim, ylim=ylim)

    def _warn_boundary(self):
        for m in self:
            if m.guided and m.boundary_fraction > 1e-2:
                warnings.warn(f"{m.name}: {100*m.boundary_fraction:.1f}% of its field is next to the metal "
                              "boundary, so its neff is affected by the walls. Enlarge x_span / y_span.",
                              stacklevel=3)


def _fmt_n(n: complex) -> str:
    if abs(n.imag) < 1e-12:
        return f"{n.real:.6f}"
    return f"{n.real:.6f}{n.imag:+.2e}j"


_ALIASES = {
    "e": "E", "|e|": "E", "e2": "E2", "|e|^2": "E2", "|e|2": "E2", "e^2": "E2", "intensity": "E2",
    "eintensity": "E2", "h": "H", "|h|": "H", "h2": "H2", "|h|^2": "H2", "|h|2": "H2",
    "sz": "Sz", "poynting": "Sz", "power": "Sz",
}
_PARTS = {"re": "real", "real": "real", "im": "imag", "imag": "imag", "abs": "abs",
          "phase": "phase", "angle": "phase"}


def _resolve(mode: Mode, quantity: str, part: str | None):
    q = quantity.replace(" ", "")
    ql = q.lower()
    if ql in _ALIASES:
        key = _ALIASES[ql]
        if key == "E":
            return np.sqrt(mode.E2), "|E|"
        if key == "E2":
            return mode.E2, "|E|$^2$"
        H2 = abs(mode.Hx) ** 2 + abs(mode.Hy) ** 2 + abs(mode.Hz) ** 2
        if key == "H":
            return np.sqrt(H2), "|H|"
        if key == "H2":
            return H2, "|H|$^2$"
        return mode.Sz, "Sz"
    wrap = None
    m = re.fullmatch(r"(\w+)\((\w+)\)", ql)
    if m and m.group(1) in _PARTS:
        wrap, ql = _PARTS[m.group(1)], m.group(2)
    m = re.fullmatch(r"\|(\w+)\|(\^?2)?", ql)
    if m:
        ql = m.group(1)
        wrap = "abs2" if m.group(2) else "abs"
    comp = {"ex": "Ex", "ey": "Ey", "ez": "Ez", "hx": "Hx", "hy": "Hy", "hz": "Hz"}.get(ql)
    if comp is None:
        raise ValueError(f"Unknown quantity {quantity!r}. Use Ex, Ey, Ez, Hx, Hy, Hz, E, E2, H, H2, Sz, "
                         "or wrapped forms like '|Ex|', 'Re(Ex)', 'phase(Hy)', '|E|^2'.")
    p = _PARTS.get(part, part) if part else (wrap or "abs")
    A = mode.fields[comp]
    if p == "abs":
        return abs(A), f"|{comp}|"
    if p == "abs2":
        return abs(A) ** 2, f"|{comp}|$^2$"
    if p == "real":
        return A.real, f"Re({comp})"
    if p == "imag":
        return A.imag, f"Im({comp})"
    if p == "phase":
        return np.angle(A), f"phase({comp}) (rad)"
    raise ValueError(f"Unknown part {part!r}: use 'abs', 'real', 'imag' or 'phase'.")
