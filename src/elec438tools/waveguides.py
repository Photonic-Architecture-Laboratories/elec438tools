"""Ready-made cross-sections. Each returns a ModeSolver you can solve, plot, or add to.

Conventions for all builders: x = 0 is the waveguide centre and y = 0 is the top of the
substrate (the bottom of the core). The substrate fills y < 0; the cladding is the
background and fills everything else. Default margins put the metal walls 1.5 um from the core.
"""

from __future__ import annotations

import numpy as np

from .geometry import Layer, Rect, Trapezoid
from .solver import ModeSolver

MARGIN = 1.5


def _solver(x_span, y_span, height, dx, background):
    return ModeSolver(x_span=x_span, y_span=y_span, dx=dx, x_center=0.0, y_center=height / 2,
                      background=background)


def _fine(sim, x0, x1, y0, y1, fine_dx):
    if fine_dx:
        pad = max(0.1, 2 * fine_dx)  # refine the near field around the core, not just the core
        sim.add_mesh_override(x0 - pad, x1 + pad, y0 - pad, y1 + pad, fine_dx)


def strip_waveguide(width: float = 0.5, height: float = 0.22, core="Si", substrate="SiO2", cladding="SiO2",
                    sidewall_angle: float = 90.0, x_span: float | None = None, y_span: float | None = None,
                    dx: float = 0.02, fine_dx: float | None = 0.01) -> ModeSolver:
    """Rectangular (or trapezoidal) strip/channel waveguide on a substrate.

    fine_dx: mesh size inside a box around the core (None to use dx everywhere).
    sidewall_angle: degrees from horizontal; 90 = vertical sidewalls.
    """
    sim = _solver(x_span or width + 2 * MARGIN, y_span or height + 2 * MARGIN, height, dx, cladding)
    sim.add(Layer(-np.inf, 0.0, substrate, name="substrate"))
    sim.add(Trapezoid(width, height, sidewall_angle, x=0.0, y_bottom=0.0, material=core, name="core"))
    _fine(sim, -width / 2, width / 2, 0.0, height, fine_dx)
    return sim


def rib_waveguide(width: float = 0.5, height: float = 0.22, slab_height: float = 0.09, core="Si",
                  substrate="SiO2", cladding="SiO2", x_span: float | None = None, y_span: float | None = None,
                  dx: float = 0.02, fine_dx: float | None = 0.01) -> ModeSolver:
    """Rib waveguide: a full-height ridge of `width` on a partially etched slab of `slab_height`.
    The slab runs into the side walls, so make x_span wide enough that the mode has decayed."""
    if not 0 < slab_height < height:
        raise ValueError("Need 0 < slab_height < height.")
    sim = _solver(x_span or width + 2 * MARGIN + 1.0, y_span or height + 2 * MARGIN, height, dx, cladding)
    sim.add(Layer(-np.inf, 0.0, substrate, name="substrate"))
    sim.add(Layer(0.0, slab_height, core, name="slab"))
    sim.add(Rect(width, height, x=0.0, y=height / 2, material=core, name="rib"))
    _fine(sim, -width / 2, width / 2, 0.0, height, fine_dx)
    return sim


def slot_waveguide(rail_width: float = 0.175, gap: float = 0.15, height: float = 0.22, core="Si",
                   substrate="SiO2", cladding="SiO2", x_span: float | None = None, y_span: float | None = None,
                   dx: float = 0.02, fine_dx: float | None = 0.005) -> ModeSolver:
    """Two rails of `rail_width` separated by a `gap` (the slot), centred on x = 0.
    The slot field is sharp: keep fine_dx <= gap / 10."""
    total = 2 * rail_width + gap
    sim = _solver(x_span or total + 2 * MARGIN, y_span or height + 2 * MARGIN, height, dx, cladding)
    sim.add(Layer(-np.inf, 0.0, substrate, name="substrate"))
    xc = gap / 2 + rail_width / 2
    sim.add(Rect(rail_width, height, x=-xc, y=height / 2, material=core, name="left rail"))
    sim.add(Rect(rail_width, height, x=+xc, y=height / 2, material=core, name="right rail"))
    _fine(sim, -total / 2, total / 2, 0.0, height, fine_dx)
    return sim


def coupled_waveguides(width: float = 0.5, gap: float = 0.2, height: float = 0.22, core="Si", substrate="SiO2",
                       cladding="SiO2", x_span: float | None = None, y_span: float | None = None,
                       dx: float = 0.02, fine_dx: float | None = 0.01) -> ModeSolver:
    """Two identical strips separated by `gap` (a directional-coupler cross-section).
    Its two lowest TE modes are the even and odd supermodes; L_cross = wavelength / (2 * delta_neff)."""
    total = 2 * width + gap
    sim = _solver(x_span or total + 2 * MARGIN, y_span or height + 2 * MARGIN, height, dx, cladding)
    sim.add(Layer(-np.inf, 0.0, substrate, name="substrate"))
    xc = gap / 2 + width / 2
    sim.add(Rect(width, height, x=-xc, y=height / 2, material=core, name="left guide"))
    sim.add(Rect(width, height, x=+xc, y=height / 2, material=core, name="right guide"))
    _fine(sim, -total / 2, total / 2, 0.0, height, fine_dx)
    return sim


def slab_waveguide(thickness: float = 0.22, core="Si", substrate="SiO2", cladding="SiO2",
                   y_span: float | None = None, dy: float = 0.005) -> ModeSolver:
    """1D slab: a core layer of `thickness` between substrate and cladding, invariant in x.
    Modes are pure TE (Ex, Hy, Hz) or pure TM (Ey, Hx, Ez)."""
    sim = ModeSolver(x_span=None, y_span=y_span or thickness + 2 * MARGIN, dx=dy, y_center=thickness / 2,
                     background=cladding)
    sim.add(Layer(-np.inf, 0.0, substrate, name="substrate"))
    sim.add(Layer(0.0, thickness, core, name="core"))
    return sim
