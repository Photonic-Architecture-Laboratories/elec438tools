"""Optical materials. Wavelength in micrometres (um) everywhere.

Anywhere a material is expected you may pass:
    - a library name (case-insensitive): "Si", "SiO2", "air"
    - a number for a constant index:     1.444   (complex for loss: 3.48 + 0.001j, k > 0 = absorption)
    - a Material object
"""

from __future__ import annotations

import warnings
from typing import Callable, Sequence, Union

import numpy as np


class Material:
    """A material with a constant or wavelength-dependent refractive index.

    >>> Material(1.444)                                   # constant index
    >>> Material(lambda wl: 1.45 + 0.004 / wl**2, name="Cauchy glass")
    """

    def __init__(self, n: Union[float, complex, Callable[[float], complex]], name: str | None = None,
                 valid_range: tuple[float, float] | None = None):
        if callable(n):
            self._f = n
            self.name = name or "custom"
        else:
            value = complex(n)
            self._f = lambda wl: value
            self.name = name or f"n={value.real:g}"
        self.valid_range = valid_range
        self._warned = False

    @classmethod
    def sellmeier(cls, B: Sequence[float], C: Sequence[float], name: str, valid_range=None) -> "Material":
        """n^2 = 1 + sum_i B_i * wl^2 / (wl^2 - C_i), with C_i in um^2."""
        B, C = np.asarray(B, float), np.asarray(C, float)

        def n(wl):
            l2 = wl * wl
            return np.sqrt(1.0 + np.sum(B * l2 / (l2 - C)))

        return cls(n, name=name, valid_range=valid_range)

    def n(self, wavelength: float) -> complex:
        """Complex refractive index at `wavelength` (um)."""
        if self.valid_range and not self._warned:
            lo, hi = self.valid_range
            if not lo <= wavelength <= hi:
                warnings.warn(f"{self.name}: wavelength {wavelength} um is outside the model's range "
                              f"{lo}-{hi} um.", stacklevel=3)
                self._warned = True
        return complex(self._f(wavelength))

    def eps(self, wavelength: float) -> complex:
        """Relative permittivity n^2."""
        return self.n(wavelength) ** 2

    def __repr__(self):
        return f"Material({self.name!r})"


# Silicon: Salzberg & Villa, JOSA 47, 244 (1957); fit range 1.36-11 um. n(1.55) = 3.4777.
Si = Material.sellmeier(
    B=[10.6684293, 0.0030434748, 1.54133408],
    C=[0.301516485**2, 1.13475115**2, 1104.0**2],
    name="Si", valid_range=(1.2, 11.0),
)
# Fused silica: Malitson, JOSA 55, 1205 (1965); fit range 0.21-6.7 um. n(1.55) = 1.4440.
SiO2 = Material.sellmeier(
    B=[0.6961663, 0.4079426, 0.8974794],
    C=[0.0684043**2, 0.1162414**2, 9.896161**2],
    name="SiO2", valid_range=(0.21, 6.7),
)
Air = Material(1.0, name="air")

LIBRARY = {"si": Si, "silicon": Si, "sio2": SiO2, "oxide": SiO2, "silica": SiO2, "air": Air}


def get_material(m) -> Material:
    """Turn a Material, a library name, or a number into a Material."""
    if isinstance(m, Material):
        return m
    if isinstance(m, str):
        key = m.strip().lower()
        if key not in LIBRARY:
            raise ValueError(f"Unknown material {m!r}. Library: 'Si', 'SiO2', 'air'. "
                             "For anything else pass its refractive index as a number, e.g. 2.0.")
        return LIBRARY[key]
    if isinstance(m, (int, float, complex, np.number)):
        return Material(complex(m))
    raise TypeError(f"Cannot interpret {m!r} as a material.")
