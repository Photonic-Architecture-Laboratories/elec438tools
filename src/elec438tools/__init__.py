"""elec438tools: photonic waveguide tools for ELEC 438 (Integrated Photonics Design, Koc University).

Mode solver: finite-difference eigenmode (FDE) on a Yee mesh.
Units: um for lengths and wavelengths. Axes: x horizontal, y vertical, z propagation.

    import elec438tools as et
    sim = et.strip_waveguide(width=0.5, height=0.22)
    modes = sim.solve(wavelength=1.55, num_modes=4)
    print(modes)
    modes["TE0"].plot("E2")

See AGENTS.md in the repository for the full reference.
"""

from .geometry import Circle, Layer, Polygon, Rect, Trapezoid
from .materials import Air, Material, Si, SiO2, get_material
from .results import Mode, ModeSet
from .solver import ModeSolver
from .sweeps import DispersionResult, SweepResult, sweep
from .waveguides import coupled_waveguides, rib_waveguide, slab_waveguide, slot_waveguide, strip_waveguide

__version__ = "0.1.0"

__all__ = [
    "ModeSolver", "Mode", "ModeSet", "sweep", "SweepResult", "DispersionResult",
    "Rect", "Layer", "Polygon", "Trapezoid", "Circle",
    "Material", "Si", "SiO2", "Air", "get_material",
    "strip_waveguide", "rib_waveguide", "slot_waveguide", "coupled_waveguides", "slab_waveguide",
]
