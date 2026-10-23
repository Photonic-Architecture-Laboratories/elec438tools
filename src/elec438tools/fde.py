"""Finite-difference eigenmode (FDE) core: Yee mesh, permittivity averaging, eigen-solve.

Method: full-vectorial finite differences on a 2D Yee cell (Zhu & Brown, Opt. Express 10, 853,
2002). The unknowns are the transverse
electric fields (Ex, Ey); the eigenvalue is neff^2 = (beta / k0)^2. Fields vary as
exp(i(beta*z - omega*t)), so a lossy or leaky mode has Im(neff) > 0.

Yee positions on a cell with edges e_i and centres c_i (same layout in y):
    Ex (c_i, e_j)   Ey (e_i, c_j)   Ez (e_i, e_j)
    Hx (e_i, c_j)   Hy (c_i, e_j)   Hz (c_i, c_j)
Node e_0 lies on the min wall and the implicit node e_N on the max wall, so tangential E
vanishes on a metal (PEC) wall exactly where the mesh puts it.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, eigs, splu

Z0 = 376.730313668  # free-space impedance (ohm)
GRADING = 0.2  # mesh refinement grading: spacing grows by ~20 % per cell away from a refined box
PML_ORDER = 3
PML_REFLECTION = 1e-8


class Axis:
    """Mesh along one axis. `lo`/`hi` bound the user region; PML cells sit outside it."""

    def __init__(self, edges, lo, hi, bc_min, bc_max, n_pml_min=0, n_pml_max=0, uniform=False):
        self.edges = np.asarray(edges, float)
        self.lo, self.hi = lo, hi
        self.bc_min, self.bc_max = bc_min, bc_max
        self.uniform = uniform  # True for a 1D problem: structure invariant along this axis
        self.N = len(self.edges) - 1
        self.h = np.diff(self.edges)  # primal cell widths
        self.nodes = self.edges[:-1]  # integer positions e_i
        self.centers = 0.5 * (self.edges[:-1] + self.edges[1:])  # half positions c_i
        # spacing between consecutive centres (c_{-1} mirrored about the min wall)
        self.delta = np.concatenate([[self.h[0]], 0.5 * (self.h[:-1] + self.h[1:])])
        # integration widths for quantities at integer nodes (node 0 sits on the wall)
        self.w_int = np.concatenate([[0.5 * self.h[0]], 0.5 * (self.h[:-1] + self.h[1:])])
        if uniform:
            self.h = self.delta = self.w_int = np.ones(1)
        # dual cells (for averaging at integer nodes)
        c_prev = np.concatenate([[self.edges[0] - 0.5 * self.h[0]], self.centers[:-1]])
        self.dual_lo, self.dual_hi = c_prev, self.centers
        # PML depth (0 at the region boundary, 1 at the outer wall)
        self.L_min = lo - self.edges[0] if n_pml_min else 0.0
        self.L_max = self.edges[-1] - hi if n_pml_max else 0.0

    def depth(self, pos):
        d = np.zeros_like(pos)
        if self.L_min > 0:
            m = pos < self.lo
            d[m] = (self.lo - pos[m]) / self.L_min
        if self.L_max > 0:
            m = pos > self.hi
            d[m] = (pos[m] - self.hi) / self.L_max
        return np.clip(d, 0, 1)

    def stretch(self, pos, k0):
        """Complex coordinate stretching s = 1 + i*sigma (dimensionless, k0-normalised)."""
        s = np.ones(len(pos), complex)
        d = self.depth(pos)
        for L, mask in ((self.L_min, pos < self.lo), (self.L_max, pos > self.hi)):
            if L > 0:
                sigma0 = (PML_ORDER + 1) * np.log(1 / PML_REFLECTION) / (2 * k0 * L)
                s[mask] = 1 + 1j * sigma0 * d[mask] ** PML_ORDER
        return s

    def interior(self, pos):
        return (pos >= self.lo - 1e-12) & (pos <= self.hi + 1e-12)


def _graded_edges(lo, hi, d, overrides, breakpoints=()):
    """Cell edges from lo to hi with default spacing d, refined inside the refinement intervals.

    Every breakpoint (a material interface) becomes a cell edge: the
    tangential E nodes then sit on the interface and the averaging is exact for flat faces.
    """
    ov = [(max(a, lo), min(b, hi), df) for a, b, df in overrides if b > lo and a < hi]

    def spacing(x):
        v = d
        for a, b, df in ov:
            dist = max(a - x, 0.0, x - b)
            v = min(v, df + GRADING * dist)
        return v

    pts = [lo, hi] + [p for p in breakpoints if lo < p < hi] + [p for a, b, _ in ov for p in (a, b) if lo < p < hi]
    pts = np.unique(np.round(pts, 9))
    edges = [lo]
    for a, b in zip(pts[:-1], pts[1:]):
        seg = [a]
        x = a
        while x < b - 1e-12:
            x = x + spacing(x + 0.5 * spacing(x))
            seg.append(x)
        seg = np.array(seg)
        if len(seg) > 2 and (seg[-1] - b) > 0.5 * (seg[-1] - seg[-2]):
            seg = seg[:-1]  # overshoot by more than half a cell: one cell fewer, stretched
        seg = a + (seg - a) * (b - a) / (seg[-1] - a)
        edges.extend(seg[1:])
    return np.array(edges)


def make_axis(span, center, d, overrides, bc_min, bc_max, n_pml, breakpoints=()):
    if span is None:  # 1D problem: invariant along this axis
        return Axis([center - 0.5, center + 0.5], center - 0.5, center + 0.5, "none", "none", uniform=True)
    lo, hi = center - span / 2, center + span / 2
    edges = _graded_edges(lo, hi, d, overrides, breakpoints)
    n_min = n_pml if bc_min == "pml" else 0
    n_max = n_pml if bc_max == "pml" else 0
    if n_min:
        h0 = edges[1] - edges[0]
        edges = np.concatenate([lo - h0 * np.arange(n_min, 0, -1), edges])
    if n_max:
        h1 = edges[-1] - edges[-2]
        edges = np.concatenate([edges, hi + h1 * np.arange(1, n_max + 1)])
    return Axis(edges, lo, hi, bc_min, bc_max, n_min, n_max)


# --- permittivity ----------------------------------------------------------------------

def _samples(lo, hi, S):
    t = (np.arange(S) + 0.5) / S
    return (lo[:, None] + (hi - lo)[:, None] * t).ravel()


def _paint(structures, background, wl, X, Y):
    eps = np.full(X.shape, background.eps(wl), complex)
    for s in structures:
        eps[s.contains(X, Y)] = s.material.eps(wl)
    return eps


def averaged_eps(structures, background, wl, ax, ay, subpixel=4, bend_radius=None, bend_x=0.0):
    """Permittivity at the Ex, Ey, Ez positions, averaged over each Yee cell.

    Across an interface normal to x, eps_x takes the harmonic mean and eps_y, eps_z the
    arithmetic mean (and likewise for y). This removes staircasing for axis-aligned
    interfaces and makes neff vary smoothly when a width is swept.
    """
    Sx = 1 if ax.uniform else subpixel
    Sy = 1 if ay.uniform else subpixel
    xh = _samples(ax.edges[:-1], ax.edges[1:], Sx) if not ax.uniform else ax.centers
    xi = _samples(ax.dual_lo, ax.dual_hi, Sx) if not ax.uniform else ax.centers
    yh = _samples(ay.edges[:-1], ay.edges[1:], Sy) if not ay.uniform else ay.centers
    yi = _samples(ay.dual_lo, ay.dual_hi, Sy) if not ay.uniform else ay.centers
    Nx, Ny = ax.N, ay.N

    def grid(xs, ys):
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        return _paint(structures, background, wl, X, Y).reshape(Nx, Sx, Ny, Sy)

    E = grid(xh, yi)
    eps_x = (1 / (1 / E).mean(axis=1)).mean(axis=-1)
    E = grid(xi, yh)
    eps_y = (1 / (1 / E).mean(axis=3)).mean(axis=1)
    E = grid(xi, yi)
    eps_z = E.mean(axis=(1, 3))

    if bend_radius:
        # conformal transformation: the bend becomes a straight guide with a tilted index
        fx_half = (1 + (ax.centers - bend_x) / bend_radius) ** 2
        fx_int = (1 + (ax.nodes - bend_x) / bend_radius) ** 2
        eps_x = eps_x * fx_half[:, None]
        eps_y = eps_y * fx_int[:, None]
        eps_z = eps_z * fx_int[:, None]
    return eps_x, eps_y, eps_z


# --- operators and eigen-solve ---------------------------------------------------------

def _derivatives(ax: Axis, k0: float):
    """Forward (integer -> half) and backward (half -> integer) derivatives, divided by k0."""
    N = ax.N
    if ax.uniform or N == 1:
        Z = sp.csr_matrix((N, N))
        return Z, Z
    h, delta = ax.h, ax.delta
    f_main = -1 / h
    b_main = 1 / delta
    if ax.bc_min in ("pec", "pml"):
        f_main = f_main.copy(); f_main[0] = 0.0  # E_tangential = 0 on the wall
        b_main = b_main.copy(); b_main[0] = 0.0  # H_tangential even about the wall
    elif ax.bc_min == "pmc":
        b_main = b_main.copy(); b_main[0] = 2 / delta[0]  # H_tangential odd about the wall
    Df = sp.diags([f_main, 1 / h[:-1]], [0, 1], shape=(N, N), format="csr")
    Db = sp.diags([b_main, -1 / delta[1:]], [0, -1], shape=(N, N), format="csr")
    s_half = ax.stretch(ax.centers, k0)
    s_int = ax.stretch(ax.nodes, k0)
    Df = sp.diags(1 / s_half) @ Df
    Db = sp.diags(1 / s_int) @ Db
    return Df / k0, Db / k0


def build_operators(ax, ay, eps_x, eps_y, eps_z, k0):
    Nx, Ny = ax.N, ay.N
    Dxf, Dxb = _derivatives(ax, k0)
    Dyf, Dyb = _derivatives(ay, k0)
    Ix, Iy = sp.identity(Nx, format="csr"), sp.identity(Ny, format="csr")
    Ux, Vx = sp.kron(Dxf, Iy, "csr"), sp.kron(Dxb, Iy, "csr")
    Uy, Vy = sp.kron(Ix, Dyf, "csr"), sp.kron(Ix, Dyb, "csr")
    ex = sp.diags(eps_x.ravel())
    ey = sp.diags(eps_y.ravel())
    iez = sp.diags(1 / eps_z.ravel())
    I = sp.identity(Nx * Ny, format="csr")
    # beta [Hx; Hy] = Q [Ex; Ey]   and   beta [Ex; Ey] = R [Hx; Hy]
    Q = sp.bmat([[Vx @ Uy, -ey - Vx @ Ux], [ex + Vy @ Uy, -Vy @ Ux]], format="csr")
    R = sp.bmat([[-Ux @ iez @ Vy, I + Ux @ iez @ Vx], [-I - Uy @ iez @ Vy, Uy @ iez @ Vx]], format="csr")
    return dict(Q=Q, R=R, Ux=Ux, Uy=Uy, Vx=Vx, Vy=Vy, iez=iez)


def _kept_dofs(ax, ay):
    """Drop tangential-E unknowns that sit on a min-side metal wall (they are zero)."""
    Nx, Ny = ax.N, ay.N
    keep = np.ones(2 * Nx * Ny, bool)
    if not ax.uniform and Nx > 1 and ax.bc_min in ("pec", "pml"):
        keep[Nx * Ny + np.arange(Ny)] = False  # Ey at x = e_0
    if not ay.uniform and Ny > 1 and ay.bc_min in ("pec", "pml"):
        keep[np.arange(Nx) * Ny] = False  # Ex at y = e_0
    return keep


def solve(ax, ay, eps, wavelength, num_modes, n_guess):
    """Solve the eigenproblem. Returns neff (complex, sorted by Re desc) and raw Yee fields."""
    k0 = 2 * np.pi / wavelength
    ops = build_operators(ax, ay, *eps, k0)
    P = (ops["R"] @ ops["Q"]).tocsc()
    keep = _kept_dofs(ax, ay)
    idx = np.flatnonzero(keep)
    Pr = P[idx][:, idx].tocsc()
    n = Pr.shape[0]
    k = int(min(num_modes, n - 2))
    if k < 1:
        raise ValueError("Mesh too small to solve: refine the mesh or enlarge the region.")
    sigma = complex(n_guess) ** 2
    if not np.any(Pr.data.imag) and sigma.imag == 0:  # lossless: real arithmetic is ~10x faster
        Pr, sigma = Pr.real.tocsc(), sigma.real
    # shift-invert: the eigenvalues nearest n_guess^2 become the largest of (P - sigma)^-1
    lu = splu((Pr - sigma * sp.identity(n, format="csc")).tocsc(), permc_spec="COLAMD")
    op = LinearOperator(Pr.shape, matvec=lu.solve, dtype=Pr.dtype)
    v0 = np.random.default_rng(0).standard_normal(n).astype(Pr.dtype)
    mu, vecs = eigs(op, k=k, which="LM", v0=v0, tol=1e-8, ncv=min(n - 1, max(2 * k + 1, 20)))
    vals = sigma + 1 / mu
    neff = np.sqrt(vals.astype(complex))
    neff = np.where(neff.real < 0, -neff, neff)
    order = np.argsort(-neff.real)
    Nx, Ny = ax.N, ay.N
    out = []
    for m in order:
        e = np.zeros(2 * Nx * Ny, complex)
        e[idx] = vecs[:, m]
        out.append((neff[m], fields_from_e(e, neff[m], ops, Nx, Ny)))
    return out


def fields_from_e(e, neff, ops, Nx, Ny):
    """All six components on the Yee mesh from [Ex; Ey]. H is physical (E / Z0 units)."""
    N = Nx * Ny
    Ex, Ey = e[:N], e[N:]
    h = ops["Q"] @ e / neff
    Hx, Hy = h[:N], h[N:]
    Hz = -1j * (ops["Ux"] @ Ey - ops["Uy"] @ Ex)
    Ez = 1j * (ops["iez"] @ (ops["Vx"] @ Hy - ops["Vy"] @ Hx))
    shape = (Nx, Ny)
    return {
        "Ex": Ex.reshape(shape), "Ey": Ey.reshape(shape), "Ez": Ez.reshape(shape),
        "Hx": (Hx / Z0).reshape(shape), "Hy": (Hy / Z0).reshape(shape), "Hz": (Hz / Z0).reshape(shape),
    }


def to_centers(F, ax, ay):
    """Interpolate raw Yee components to the cell centres (c_i, c_j) for display/integrals."""

    def avg_x(A):
        if ax.uniform or ax.N == 1:
            return A
        return 0.5 * (A + np.vstack([A[1:], np.zeros((1, A.shape[1]), A.dtype)]))

    def avg_y(A):
        if ay.uniform or ay.N == 1:
            return A
        return 0.5 * (A + np.hstack([A[:, 1:], np.zeros((A.shape[0], 1), A.dtype)]))

    return {
        "Ex": avg_y(F["Ex"]), "Ey": avg_x(F["Ey"]), "Ez": avg_x(avg_y(F["Ez"])),
        "Hx": avg_x(F["Hx"]), "Hy": avg_y(F["Hy"]), "Hz": F["Hz"],
    }


def yee_power(F, ax, ay):
    """0.5 * Re integral of (E x H*) . z using co-located Yee pairs (Ex-Hy, Ey-Hx)."""
    w1 = np.outer(ax.h, ay.w_int)  # Ex, Hy at (c_i, e_j)
    w2 = np.outer(ax.w_int, ay.h)  # Ey, Hx at (e_i, c_j)
    return 0.5 * np.real(np.sum(F["Ex"] * np.conj(F["Hy"]) * w1) - np.sum(F["Ey"] * np.conj(F["Hx"]) * w2))
