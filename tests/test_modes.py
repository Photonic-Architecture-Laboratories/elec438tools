"""Checks against exact slab solutions, the S02 lecture slides' reference values, and femwell (FEM)."""

import numpy as np
import pytest
from scipy.optimize import brentq

import elec438tools as et


def slab_neff(wl, d, n1, n2, n3, pol="TE", m=0):
    """Exact three-layer slab: core n1 (thickness d) between substrate n2 and cover n3."""
    k0 = 2 * np.pi / wl

    def f(ne):
        kappa = k0 * np.sqrt(n1**2 - ne**2)
        g2 = k0 * np.sqrt(ne**2 - n2**2)
        g3 = k0 * np.sqrt(ne**2 - n3**2)
        r2, r3 = (1, 1) if pol == "TE" else ((n1 / n2) ** 2, (n1 / n3) ** 2)
        return kappa * d - m * np.pi - np.arctan(r2 * g2 / kappa) - np.arctan(r3 * g3 / kappa)

    lo = max(n2, n3) + 1e-12
    return brentq(f, lo, n1 - 1e-12)


@pytest.mark.parametrize("pol", ["TE", "TM"])
def test_1d_slab_matches_exact_solution(pol):
    sim = et.slab_waveguide(thickness=0.22, dy=0.002)
    modes = sim.solve(1.55, num_modes=2)
    nSi, nOx = et.Si.n(1.55).real, et.SiO2.n(1.55).real
    exact = slab_neff(1.55, 0.22, nSi, nOx, nOx, pol)
    assert abs(modes[f"{pol}0"].neff.real - exact) < 2e-4


def test_symmetric_slab_s02():
    """S02: 1 um slab of n = 1.444 in air; first two TE modes (walls far away, as the exact solution assumes)."""
    sim = et.slab_waveguide(thickness=1.0, core=1.444, substrate="air", cladding="air", region_height=12.0, dy=0.005)
    modes = sim.solve(1.55, num_modes=4)
    for m in (0, 1):
        exact = slab_neff(1.55, 1.0, 1.444, 1.0, 1.0, "TE", m)
        assert abs(modes[f"TE{m}"].neff.real - exact) < 2e-4


def test_rectangle_matches_lecture_s02():
    """S02 slides: 2 x 1 um core of n = 1.444 in air, 3 x 4 um region, metal walls, 1.55 um (values on the slides)."""
    sim = et.ModeSolver(region_width=3.0, region_height=4.0, dx=0.02, background="air")
    sim.add(et.Rect(width=2.0, height=1.0, material=1.444))
    with pytest.warns(UserWarning):  # the slide's region is tight: TE3 touches the walls
        modes = sim.solve(1.55, num_modes=5)
    slides = {"TE0": 1.304020, "TM0": 1.271321, "TE1": 1.176642, "TM1": 1.155666, "TE2": 1.029682}
    for name, n in slides.items():
        assert abs(modes[name].neff.real - n) < 5e-4, name
    assert modes["TE0"].te_fraction > 0.99 and modes["TM0"].te_fraction < 0.01


def test_group_index_matches_exact_slab():
    sim = et.slab_waveguide(thickness=0.22, dy=0.002)
    te0 = sim.solve(1.55, num_modes=2, group_index=True)["TE0"]
    d = 1e-3

    def n_at(wl):
        return slab_neff(wl, 0.22, et.Si.n(wl).real, et.SiO2.n(wl).real, et.SiO2.n(wl).real)

    ng_exact = n_at(1.55) - 1.55 * (n_at(1.55 + d) - n_at(1.55 - d)) / (2 * d)
    assert abs(te0.ng - ng_exact) < 2e-3


def test_mesh_lands_on_interfaces():
    sim = et.strip_waveguide(width=0.5, height=0.22)
    ax, ay = sim.axes
    for v in (-0.25, 0.25):
        assert np.min(abs(ax.edges - v)) < 1e-9
    for v in (0.0, 0.22):
        assert np.min(abs(ay.edges - v)) < 1e-9


def test_overlap_and_power_fraction():
    sim = et.strip_waveguide(width=0.5, height=0.22, dx=0.04, fine_dx=0.02)
    modes = sim.solve(1.55, num_modes=2)
    te0, tm0 = modes["TE0"], modes["TM0"]
    assert abs(te0.overlap(te0) - 1) < 1e-9
    assert te0.overlap(tm0) < 1e-3
    gamma = te0.power_fraction(sim.structures[1])
    assert 0.5 < gamma < 1.0


def test_width_sweep_and_dispersion():
    res = et.sweep(lambda w: et.strip_waveguide(width=w, dx=0.04, fine_dx=0.02), [0.4, 0.5, 0.6],
                   param_name="width (um)", num_modes=3)
    assert np.all(np.diff(res.neff_te[:, 0]) > 0)  # neff grows with width
    disp = et.slab_waveguide(0.22, dy=0.004).wavelength_sweep(np.linspace(1.50, 1.60, 6), num_modes=1)
    ng_direct = et.slab_waveguide(0.22, dy=0.004).solve(1.55, 1, group_index=True)[0].ng
    assert abs(np.interp(1.55, disp.wavelengths, disp.ng[:, 0]) - ng_direct) < 5e-3


def test_strip_matches_femwell():
    pytest.importorskip("femwell")
    from collections import OrderedDict

    from femwell.maxwell.waveguide import compute_modes
    from femwell.mesh import mesh_from_OrderedDict
    from shapely.geometry import box
    from skfem import Basis, ElementTriP0
    from skfem.io.meshio import from_meshio

    nSi, nOx = et.Si.n(1.55).real, et.SiO2.n(1.55).real
    polys = OrderedDict(core=box(-0.25, 0, 0.25, 0.22), box=box(-2, -1.5, 2, 0), clad=box(-2, 0, 2, 1.72))
    mesh = from_meshio(mesh_from_OrderedDict(polys, {"core": {"resolution": 0.03, "distance": 0.5}},
                                             default_resolution_max=0.15))
    basis = Basis(mesh, ElementTriP0())
    eps = basis.zeros()
    for k, n in {"core": nSi, "box": nOx, "clad": nOx}.items():
        eps[basis.get_dofs(elements=k)] = n**2
    fem = compute_modes(basis, eps, wavelength=1.55, num_modes=2, order=2)
    ours = et.strip_waveguide(0.5, 0.22).solve(1.55, num_modes=2)
    assert abs(ours["TE0"].neff.real - fem[0].n_eff.real) < 1e-3
    assert abs(ours["TM0"].neff.real - fem[1].n_eff.real) < 3e-3


def test_rib_guided_threshold_is_slab_mode_index():
    """A rib's slab runs into the side walls: modes are guided only above the slab mode's neff."""
    slab = et.slab_waveguide(0.09).solve(1.55, num_modes=1)["TE0"].neff.real
    modes = et.rib_waveguide(width=0.5, height=0.22, slab_height=0.09).solve(1.55, num_modes=4)
    assert abs(modes[0].n_clad - slab) < 5e-3
    assert modes["TE0"].guided and modes["TE0"].neff.real > slab
