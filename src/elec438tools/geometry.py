"""Cross-section shapes. Coordinates in um; x horizontal, y vertical, z = propagation.

Structures are painted in the order they are added to the solver: a later shape overrides
an earlier one where they overlap (so add the substrate first and the core last).
"""

from __future__ import annotations

import numpy as np
from matplotlib.path import Path

from .materials import get_material


class Shape:
    material = None
    name = ""

    def contains(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def outline(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Closed outline (xs, ys) for plotting, or None for unbounded shapes."""
        return None

    def bounds(self) -> tuple[float, float, float, float]:
        """(x_min, x_max, y_min, y_max); may be infinite."""
        raise NotImplementedError


class Rect(Shape):
    """Axis-aligned rectangle centred at (x, y). Use width=np.inf for a layer spanning all x."""

    def __init__(self, width: float, height: float, x: float = 0.0, y: float = 0.0, material="Si", name: str = ""):
        if width <= 0 or height <= 0:
            raise ValueError(f"Rect needs width > 0 and height > 0 (got {width}, {height}).")
        self.width, self.height, self.x, self.y = float(width), float(height), float(x), float(y)
        self.material = get_material(material)
        self.name = name or f"Rect({self.material.name})"

    @classmethod
    def from_bounds(cls, x_min, x_max, y_min, y_max, material="Si", name=""):
        x = 0.0 if np.isinf(x_min) and np.isinf(x_max) else 0.5 * (x_min + x_max)
        y = 0.0 if np.isinf(y_min) and np.isinf(y_max) else 0.5 * (y_min + y_max)
        r = cls.__new__(cls)
        r.width, r.height = x_max - x_min, y_max - y_min
        r.x, r.y = x, y
        r._b = (x_min, x_max, y_min, y_max)
        r.material = get_material(material)
        r.name = name or f"Rect({r.material.name})"
        return r

    def bounds(self):
        if hasattr(self, "_b"):
            return self._b
        return (self.x - self.width / 2, self.x + self.width / 2, self.y - self.height / 2, self.y + self.height / 2)

    def contains(self, X, Y):
        x0, x1, y0, y1 = self.bounds()
        return (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)

    def outline(self):
        x0, x1, y0, y1 = self.bounds()
        if not all(np.isfinite([x0, x1, y0, y1])):
            return None
        return np.array([x0, x1, x1, x0, x0]), np.array([y0, y0, y1, y1, y0])


def Layer(y_min: float, y_max: float, material="SiO2", name: str = "") -> Rect:
    """A horizontal layer spanning all x between y_min and y_max (use -np.inf / np.inf freely)."""
    return Rect.from_bounds(-np.inf, np.inf, y_min, y_max, material, name or f"Layer({get_material(material).name})")


class Polygon(Shape):
    """Polygon from a list of (x, y) vertices (either orientation)."""

    def __init__(self, vertices, material="Si", name: str = ""):
        v = np.asarray(vertices, float)
        if v.ndim != 2 or v.shape[1] != 2 or len(v) < 3:
            raise ValueError("Polygon needs at least 3 (x, y) vertices.")
        self.vertices = v
        self._path = Path(np.vstack([v, v[:1]]), closed=True)
        self.material = get_material(material)
        self.name = name or f"Polygon({self.material.name})"

    def bounds(self):
        v = self.vertices
        return (v[:, 0].min(), v[:, 0].max(), v[:, 1].min(), v[:, 1].max())

    def contains(self, X, Y):
        x0, x1, y0, y1 = self.bounds()
        X, Y = np.broadcast_arrays(X, Y)
        out = np.zeros(X.shape, bool)
        box = (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)
        if box.any():
            pts = np.column_stack([X[box], Y[box]])
            # small positive radius so points exactly on the edge count as inside
            out[box] = self._path.contains_points(pts, radius=1e-12) | self._path.contains_points(pts, radius=-1e-12)
        return out

    def outline(self):
        v = np.vstack([self.vertices, self.vertices[:1]])
        return v[:, 0], v[:, 1]


def Trapezoid(bottom_width: float, height: float, sidewall_angle: float = 90.0, x: float = 0.0,
              y_bottom: float = 0.0, material="Si", name: str = "") -> Polygon | Rect:
    """Etched ridge with slanted sidewalls. sidewall_angle in degrees from horizontal (90 = vertical)."""
    if sidewall_angle == 90:
        return Rect(bottom_width, height, x=x, y=y_bottom + height / 2, material=material, name=name)
    inset = height / np.tan(np.radians(sidewall_angle))
    top = bottom_width - 2 * inset
    if top <= 0:
        raise ValueError("Sidewall angle too shallow: the top width would be <= 0.")
    v = [(x - bottom_width / 2, y_bottom), (x + bottom_width / 2, y_bottom),
         (x + top / 2, y_bottom + height), (x - top / 2, y_bottom + height)]
    return Polygon(v, material, name)


class Circle(Shape):
    """Circle (e.g. a fibre core) centred at (x, y)."""

    def __init__(self, radius: float, x: float = 0.0, y: float = 0.0, material="SiO2", name: str = ""):
        self.radius, self.x, self.y = float(radius), float(x), float(y)
        self.material = get_material(material)
        self.name = name or f"Circle({self.material.name})"

    def bounds(self):
        r = self.radius
        return (self.x - r, self.x + r, self.y - r, self.y + r)

    def contains(self, X, Y):
        return (X - self.x) ** 2 + (Y - self.y) ** 2 <= self.radius**2

    def outline(self):
        t = np.linspace(0, 2 * np.pi, 181)
        return self.x + self.radius * np.cos(t), self.y + self.radius * np.sin(t)
