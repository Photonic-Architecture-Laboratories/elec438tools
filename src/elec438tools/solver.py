"""ModeSolver: set up a waveguide cross-section and find its modes (finite-difference eigenmode)."""

from __future__ import annotations

import warnings

import numpy as np

from . import fde
from .geometry import Polygon, Rect, Shape
from .materials import get_material
from .results import Mode, ModeSet


class ModeSolver:
    """A waveguide cross-section and its solver region.

    Coordinates in um: x horizontal, y vertical, light propagates along z.

    Parameters
    ----------
    region_width, region_height : size of the solver region (um), along x and y.
        region_width=None makes a 1D problem: the structure is invariant along x (a slab),
        and only y is meshed.
    dx, dy : mesh cell size (um). dy defaults to dx. Refine locally with refine_mesh().
    center : (x, y) of the centre of the region (um).
    background : material filling everything no structure covers (the cladding).
    boundary : "metal" (perfect electric conductor on all four sides). Fields must decay
        before reaching it, so place it >= ~1 um from any high-index core.
    subpixel : sub-samples per cell edge used to average the permittivity at interfaces.

    Example
    -------
    >>> sim = ModeSolver(region_width=3, region_height=3, dx=0.02, background="SiO2")
    >>> sim.add(Rect(width=0.5, height=0.22, material="Si"))
    >>> modes = sim.solve(wavelength=1.55, num_modes=4)
    >>> print(modes); modes[0].plot("E2")
    """

    def __init__(self, region_width: float | None, region_height: float, dx: float = 0.02,
                 dy: float | None = None, center: tuple[float, float] = (0.0, 0.0), background="air",
                 boundary: str = "metal", subpixel: int = 4):
        if boundary != "metal":
            raise ValueError("Only boundary='metal' is available for now.")
        if region_width is not None and region_width <= 0 or region_height <= 0:
            raise ValueError("region_width and region_height must be positive (um).")
        self.region_width, self.region_height = region_width, region_height
        self.dx = float(dx)
        self.dy = float(dy if dy is not None else dx)
        self.center = (float(center[0]), float(center[1]))
        self.background = get_material(background)
        self.boundary = boundary
        self.subpixel = int(subpixel)
        self.structures: list[Shape] = []
        self.refinements: list[tuple] = []
        self._bc = {"xmin": "pec", "xmax": "pec", "ymin": "pec", "ymax": "pec"}
        self._n_pml = 0
        self._axes = None

    # --- setup ---------------------------------------------------------------------
    def add(self, *shapes: Shape):
        """Add one or more structures. Later structures override earlier ones where they overlap."""
        for s in shapes:
            if not isinstance(s, Shape):
                raise TypeError(f"add() takes Rect/Polygon/Circle/Layer shapes, got {type(s).__name__}.")
            self.structures.append(s)
        self._axes = None
        return shapes[0] if len(shapes) == 1 else shapes

    def refine_mesh(self, x_min: float, x_max: float, y_min: float, y_max: float,
                          dx: float, dy: float | None = None):
        """Use a finer mesh (dx, dy) inside a box, graded smoothly into the default mesh outside."""
        self.refinements.append((x_min, x_max, y_min, y_max, float(dx), float(dy if dy is not None else dx)))
        self._axes = None

    @property
    def axes(self):
        if self._axes is None:
            ox = [(a, b, d) for a, b, _, _, d, _ in self.refinements]
            oy = [(c, e, d) for _, _, c, e, _, d in self.refinements]
            bx, by = self._breakpoints()
            ax = fde.make_axis(self.region_width, self.center[0], self.dx, ox, self._bc["xmin"], self._bc["xmax"],
                               self._n_pml, bx)
            ay = fde.make_axis(self.region_height, self.center[1], self.dy, oy, self._bc["ymin"], self._bc["ymax"],
                               self._n_pml, by)
            self._axes = (ax, ay)
        return self._axes

    def _breakpoints(self):
        """Interface coordinates of flat faces (rectangle sides, polygon vertices) for mesh alignment."""
        bx, by = [], []
        for s in self.structures:
            if isinstance(s, Polygon):
                bx += list(s.vertices[:, 0])
                by += list(s.vertices[:, 1])
            elif isinstance(s, Rect):
                x0, x1, y0, y1 = s.bounds()
                bx += [v for v in (x0, x1) if np.isfinite(v)]
                by += [v for v in (y0, y1) if np.isfinite(v)]
        return bx, by

    @property
    def x(self) -> np.ndarray:
        """x coordinates (um) of the cell centres, where fields are reported."""
        return self.axes[0].centers

    @property
    def y(self) -> np.ndarray:
        """y coordinates (um) of the cell centres, where fields are reported."""
        return self.axes[1].centers

    def mesh_info(self) -> str:
        ax, ay = self.axes
        nx = "1 (invariant in x)" if ax.uniform else f"{ax.N}  (dx {ax.h.min()*1e3:.1f}-{ax.h.max()*1e3:.1f} nm)"
        return f"mesh: Nx = {nx},  Ny = {ay.N}  (dy {ay.h.min()*1e3:.1f}-{ay.h.max()*1e3:.1f} nm),  {2*ax.N*ay.N} unknowns"

    def index(self, wavelength: float = 1.55) -> np.ndarray:
        """Refractive index at the cell centres, shape (len(x), len(y)), indexed [ix, iy]."""
        X, Y = np.meshgrid(self.x, self.y, indexing="ij")
        return np.sqrt(fde._paint(self.structures, self.background, wavelength, X, Y))

    def plot_index(self, wavelength: float = 1.55, ax=None, show_mesh: bool = False, cmap: str = "turbo",
                   xlim=None, ylim=None):
        """Plot the real refractive index of the cross-section (check this before solving).
        show_mesh=True draws the mesh lines; xlim/ylim=(min, max) zoom in."""
        from .plotting import plot_index
        return plot_index(self, wavelength, ax=ax, show_mesh=show_mesh, cmap=cmap, xlim=xlim, ylim=ylim)

    # --- solving -------------------------------------------------------------------
    def _check(self, wavelength):
        if not 0.2 <= wavelength <= 20:
            raise ValueError(f"wavelength={wavelength}: wavelengths are in micrometres (1.55, not 1550).")
        n_max = max([self.background.n(wavelength).real] + [s.material.n(wavelength).real for s in self.structures])
        ax, ay = self.axes
        h = max(ay.h.max(), 0 if ax.uniform else ax.h.max())
        if h > wavelength / (8 * n_max):
            warnings.warn(f"Mesh cells up to {h*1e3:.0f} nm are coarse for n = {n_max:.2f} at {wavelength} um "
                          f"(aim for <= {wavelength/(10*n_max)*1e3:.0f} nm, or use refine_mesh around the core).",
                          stacklevel=3)
        if 2 * ax.N * ay.N > 400_000:
            warnings.warn(f"{2*ax.N*ay.N} unknowns: this solve will be slow and memory-hungry. "
                          "Use a coarser default mesh plus refine_mesh() around the core.", stacklevel=3)
        return n_max

    def _raw(self, wavelength, num_modes, n_guess, bend_radius):
        ax, ay = self.axes
        eps = fde.averaged_eps(self.structures, self.background, wavelength, ax, ay, self.subpixel,
                               bend_radius=bend_radius, bend_x=self.center[0])
        return fde.solve(ax, ay, eps, wavelength, num_modes, n_guess)

    def solve(self, wavelength: float = 1.55, num_modes: int = 6, n_guess: float | None = None,
              bend_radius: float | None = None, group_index: bool = False) -> ModeSet:
        """Find the modes with effective index nearest n_guess (default: the highest index present).

        Parameters
        ----------
        wavelength : vacuum wavelength in um.
        num_modes : number of modes to return.
        n_guess : the solver returns the modes whose effective index is closest to this value.
        bend_radius : bend radius in um (conformal transformation); the centre of curvature is
            at x = center[0] - bend_radius, so the mode shifts towards +x. neff is referred to
            the radius at center[0]. Radiation loss of bends is not computed with metal boundaries.
        group_index : also compute ng = neff - wavelength * dneff/dwavelength for each mode
            (two extra solves at wavelength * (1 +- 1e-3)).

        Returns a ModeSet (a list of Mode objects sorted by Re(neff), highest first).
        """
        n_max = self._check(wavelength)
        user_guess = n_guess is not None
        if n_guess is None:
            n_guess = n_max
        raw = self._raw(wavelength, num_modes, n_guess, bend_radius)
        n_clad = self._edge_index(wavelength, bend_radius)
        modes = [Mode(i, neff, F, wavelength, self, n_clad, bend_radius) for i, (neff, F) in enumerate(raw)]
        if group_index:
            d = 1e-3 * wavelength
            # a shift just above the top mode finds the same set of modes, faster
            guess = n_guess if user_guess else max(m.neff.real for m in modes) + 0.01
            plus = self._raw(wavelength + d, num_modes + 2, guess, bend_radius)
            minus = self._raw(wavelength - d, num_modes + 2, guess, bend_radius)
            for m in modes:
                n_p = _match(m._yee, plus)
                n_m = _match(m._yee, minus)
                m.ng = (m.neff - wavelength * (n_p - n_m) / (2 * d)).real
        ms = ModeSet(modes, wavelength=wavelength)
        ms._warn_boundary()
        return ms

    def _edge_index(self, wavelength, bend_radius=None):
        """The effective index a mode must exceed to be guided.

        Top and bottom edges: the highest material index there. Left and right edges: the
        effective index of the fundamental slab mode of the layer stack at that edge (a
        uniform cladding gives its own index), since a mode below it leaks sideways into
        that slab, as in a rib waveguide. In a bend the transformed material index is used,
        which grows towards +x (the outside of the bend)."""
        n = self.index(wavelength).real
        ax, ay = self.axes
        if bend_radius:
            n = n * (1 + (ax.centers[:, None] - self.center[0]) / bend_radius)
            edges = [n[0, :], n[-1, :], n[:, 0], n[:, -1]]
            return float(max(e.max() for e in edges))
        levels = [n[:, 0].max(), n[:, -1].max()]
        if not ax.uniform:
            levels += [self._slab_index(wavelength, n[0, :], ax.centers[0]),
                       self._slab_index(wavelength, n[-1, :], ax.centers[-1])]
        return float(max(levels))

    def _slab_index(self, wavelength, column, x):
        """Fundamental effective index of the vertical layer stack at position x."""
        if column.max() - column.min() < 1e-9:
            return float(column.max())
        ax1 = fde.make_axis(None, x, 1.0, [], "pec", "pec", 0)
        ay = self.axes[1]
        eps = fde.averaged_eps(self.structures, self.background, wavelength, ax1, ay, self.subpixel)
        res = fde.solve(ax1, ay, eps, wavelength, 2, column.max())
        return float(max(neff.real for neff, _ in res))

    def wavelength_sweep(self, wavelengths, num_modes: int = 2, n_guess: float | None = None):
        """Solve at each wavelength and follow the first `num_modes` modes by field overlap.

        Returns a DispersionResult with neff, group index ng and dispersion D vs wavelength.
        Use >= 5 wavelengths spaced <= 20 nm apart for a smooth ng and D.
        """
        from .sweeps import wavelength_sweep
        return wavelength_sweep(self, wavelengths, num_modes, n_guess)

    def __repr__(self):
        xs = "invariant" if self.region_width is None else f"{self.region_width} um"
        return (f"ModeSolver(region_width={xs}, region_height={self.region_height} um, background={self.background.name}, "
                f"{len(self.structures)} structures; {self.mesh_info()})")


def _overlap(Fa, Fb):
    a = np.concatenate([Fa["Ex"].ravel(), Fa["Ey"].ravel()])
    b = np.concatenate([Fb["Ex"].ravel(), Fb["Ey"].ravel()])
    return abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300)


def _match(F, candidates):
    """neff of the candidate (neff, fields) whose transverse E best overlaps F."""
    best = max(candidates, key=lambda c: _overlap(F, c[1]))
    return best[0]
