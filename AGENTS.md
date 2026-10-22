# AGENTS.md: reference for AI coding assistants

This repository is `elec438tools`, the Python toolkit for ELEC 438 (Integrated Photonics Design, Koç University). Its first tool is a **finite-difference eigenmode (FDE) waveguide mode solver** on a Yee mesh, in plain NumPy/SciPy. Users are students writing analysis scripts and notebooks; write code they can read.

Install: `pip install git+https://github.com/Photonic-Architecture-Laboratories/elec438tools.git` (Colab: same line with `!` or `%pip`). Import as `import elec438tools as et`.

## Conventions (get these right first)

- **Units: micrometres** for every length and wavelength. `wavelength=1.55`, never `1550`. Widths like `0.5` (= 500 nm).
- **Axes:** x horizontal, y vertical, z propagation. Fields vary as exp(i(βz − ωt)); loss means Im(neff) > 0.
- **Field arrays are indexed `[ix, iy]`** (shape `(len(mode.x), len(mode.y))`). For `imshow`/`pcolormesh` pass `A.T`, or just use the built-in plot methods.
- **Builders' origin:** x = 0 at the waveguide centre, y = 0 at the top of the substrate (bottom of the core). The substrate fills y < 0; the cladding is the solver background.
- **Boundaries:** metal (perfect electric conductor) on all four sides; the only option in this version. There are no PML/absorbing boundaries, so bend *loss* and leakage loss cannot be computed.
- **Materials:** `"Si"`, `"SiO2"`, `"air"` (Si and SiO2 are dispersive Sellmeier models), or any number for a constant index (`1.444`, `3.17`, `2.0 + 0.001j`). There is no other material library; do not invent names like `"Si3N4"` (pass `2.0` instead).
- Structures are painted in the order added; a later one overrides an earlier one where they overlap. Add substrate layers first, cores last.

## Minimal examples

```python
import numpy as np
import matplotlib.pyplot as plt
import elec438tools as et

# 1. Standard SOI strip (Si 500 x 220 nm on SiO2, SiO2 cladding)
sim = et.strip_waveguide(width=0.5, height=0.22)
modes = sim.solve(wavelength=1.55, num_modes=4)
print(modes)                         # table: name, neff, loss, TE fraction, Aeff, guided
te0 = modes["TE0"]                   # access by name ("TE0", "TM0", "TE1", ...) or index modes[0]
te0.plot("E2", xlim=(-0.8, 0.8), ylim=(-0.5, 0.7))
te0.plot_fields()                    # 2x3 panel |Ex| |Ey| |Ez| / |Hx| |Hy| |Hz|
te0.plot_slice("Ex")                 # line cuts through the field maximum

# 2. Any cross-section by hand
sim = et.ModeSolver(x_span=3.0, y_span=3.0, y_center=0.11, dx=0.02, background="SiO2")
sim.add(et.Layer(-np.inf, 0.0, "SiO2"))                            # substrate
core = sim.add(et.Rect(width=0.5, height=0.22, x=0, y=0.11, material="Si"))
sim.add_mesh_override(x_min=-0.35, x_max=0.35, y_min=-0.1, y_max=0.32, dx=0.01)
sim.plot_index()                     # ALWAYS check geometry before solving
modes = sim.solve(1.55, num_modes=4, group_index=True)

# 3. Width sweep (cutoffs, single-mode condition)
res = et.sweep(lambda w: et.strip_waveguide(width=w), np.arange(0.3, 1.01, 0.05),
               wavelength=1.55, num_modes=6, param_name="width (um)")
res.plot()                           # TE solid, TM dashed, dotted line = cladding index
res.neff_te[:, 0]                    # TE0 at every width (NaN where not guided)
res.num_guided("TE")                 # number of guided TE modes per width

# 4. Dispersion vs wavelength
disp = et.strip_waveguide(0.5, 0.22).wavelength_sweep(np.linspace(1.5, 1.6, 11), num_modes=2)
disp.plot("ng"); disp.plot("D")      # also disp.neff, disp.ng, disp.D arrays, shape (n_wl, n_modes)
```

## API

### Geometry (`et.`)
| call | meaning |
|---|---|
| `Rect(width, height, x=0, y=0, material="Si", name="")` | rectangle centred at (x, y); `width=np.inf` spans all x |
| `Rect.from_bounds(x_min, x_max, y_min, y_max, material)` | rectangle from its edges (±np.inf allowed) |
| `Layer(y_min, y_max, material)` | horizontal layer spanning all x (substrates, slabs, claddings) |
| `Polygon(vertices, material)` | polygon from `[(x, y), ...]` |
| `Trapezoid(bottom_width, height, sidewall_angle=90, x=0, y_bottom=0, material)` | ridge with slanted sidewalls (degrees from horizontal) |
| `Circle(radius, x=0, y=0, material)` | circle |
| `Material(n)` / `Material(lambda wl: ..., name=...)` | constant or wavelength-dependent index; `et.Si.n(1.55)` returns the index |

### Builders (return a ready `ModeSolver`; walls 1.5 um from the core by default)
- `strip_waveguide(width=0.5, height=0.22, core="Si", substrate="SiO2", cladding="SiO2", sidewall_angle=90, x_span=None, y_span=None, dx=0.02, fine_dx=0.01)`
- `rib_waveguide(width=0.5, height=0.22, slab_height=0.09, ...)`: ridge on a partially etched slab
- `slot_waveguide(rail_width=0.175, gap=0.15, height=0.22, ..., fine_dx=0.005)`
- `coupled_waveguides(width=0.5, gap=0.2, height=0.22, ...)`: directional-coupler cross-section; even/odd supermodes are `modes["TE0"]`, `modes["TE1"]`; coupling length `wl / (2 * (n_even - n_odd))`
- `slab_waveguide(thickness=0.22, core="Si", substrate="SiO2", cladding="SiO2", y_span=None, dy=0.005)`: 1D, invariant in x

`fine_dx` is the mesh inside a box 0.1 um around the core; `dx` elsewhere; `fine_dx=None` gives a uniform mesh.

### `ModeSolver(x_span, y_span, dx=0.02, dy=None, x_center=0, y_center=0, background="air", boundary="metal", subpixel=4)`
- `x_span=None` → 1D problem (invariant in x).
- `.add(*shapes)`, `.add_mesh_override(x_min, x_max, y_min, y_max, dx, dy=None)`
- `.plot_index(wavelength=1.55, show_mesh=False, xlim=None, ylim=None)`, `.index(wavelength)` → n array `[ix, iy]`, `.x`, `.y`, `.mesh_info()`
- `.solve(wavelength=1.55, num_modes=6, n_guess=None, bend_radius=None, group_index=False)` → `ModeSet`
  - `n_guess`: search near this neff (default: highest index present → fundamental first).
  - `bend_radius` (um): conformal transformation; centre of curvature at x = x_center − R; the mode shifts to +x. Gives neff and shape only (no bend loss with metal walls). Make `x_span` large enough.
  - `group_index=True`: fills `mode.ng` (2 extra solves).
- `.wavelength_sweep(wavelengths, num_modes=2, n_guess=None)` → `DispersionResult`

### `ModeSet` (list of `Mode`, sorted by Re(neff), highest first)
- `modes["TE0"]`, `modes[0]`, `len(modes)`, `modes.neff` (array), `modes.guided()`, `print(modes)` / `modes.table()`, `modes.to_dataframe()` (needs pandas), `modes.plot(quantity="E2", max_cols=4, xlim, ylim)`

### `Mode`
- Scalars: `neff` (complex), `ng` (None unless computed), `beta` (rad/um), `loss_db_per_cm`, `te_fraction` (∫|Ex|²/∫(|Ex|²+|Ey|²)), `polarization` ("TE"/"TM"), `name`, `guided`, `n_clad` (guided threshold), `effective_area` (um²), `boundary_fraction`, `wavelength`, `power`.
- Fields (complex, on cell centres, `[ix, iy]`, normalised to max|E| = 1; H in matching physical units ≈ n/377): `Ex Ey Ez Hx Hy Hz`, `E2` (|E|²), `Sz` (Poynting z), `x`, `y`, `fields` dict.
- `power_fraction(shape_or_(x0, x1, y0, y1))` → fraction of the mode's power inside: integral of Sz = ½Re(E×H*)·z over the shape divided by the integral over the whole cross-section (power confinement factor). Not an |E|² integral.
- `overlap(other_mode)` → power coupling between two modes on the same mesh.
- `plot(quantity="E2", part=None, ax=None, cmap=None, xlim=None, ylim=None)`: quantities `Ex Ey Ez Hx Hy Hz` (magnitude by default), `E`, `E2`, `H`, `H2`, `Sz`, and wrapped forms `"|Ex|"`, `"Re(Ex)"`, `"Im(Hy)"`, `"phase(Ex)"`, `"|E|^2"`.
- `plot_fields(xlim, ylim)`, `plot_slice(quantity="E2", x=None, y=None)`: `y=` horizontal cut, `x=` vertical cut, neither: both through the maximum.
- `save("file.mat")` (scipy.io `.mat` format) or `save("file.npz")` (NumPy).

### Sweeps
- `et.sweep(build, values, wavelength=1.55, num_modes=6, param_name="parameter", guided_only=True, keep_modes=False, **solve_kwargs)` → `SweepResult` with `values`, `neff_te`, `neff_tm`, `ng_te`, `ng_tm` (arrays `(n_values, n_modes)`, NaN-padded, ordered by neff within each polarization), `n_clad`, `num_guided(pol)`, `plot(quantity="neff"|"ng")`, `to_dataframe()`. `build` is a function value → ModeSolver. Pass `group_index=True` to get ng.
- `DispersionResult`: `wavelengths`, `neff`, `ng`, `D` (ps/(nm·km)), `names`, `plot("neff"|"ng"|"D")`, `to_dataframe()`. Use ≥ 5 wavelengths, ≤ 20 nm apart.

## Checks to run before trusting a number (tell the student when you skip one)

1. **Plot the index** (`sim.plot_index()`) and confirm the geometry.
2. **Guided?** Use only modes with `mode.guided == True`; modes below `n_clad` are modes of the metal box, not of the waveguide.
3. **Walls far enough?** The solver warns when a guided mode has > 1 % of its field next to a wall; enlarge `x_span`/`y_span`. Weakly guided modes near cutoff need the most room.
4. **Mesh converged?** Halve `fine_dx` (or `dx`) and confirm neff changes less than the precision you report. Typical: 10 nm in the core gives neff to ~1e-3 for SOI; 5 nm to ~3e-4.
5. **Polarization labels** come from `te_fraction`; near a TE/TM crossing in a width sweep the modes hybridise and labels can swap. Look at the fields.

## Performance

A default SOI strip solve takes ~2–5 s on a laptop (~50–130k unknowns). Cost grows quickly with mesh size: prefer a coarse `dx` with a fine `fine_dx`/mesh override over a uniformly fine mesh; use `dx=0.04, fine_dx=0.02` for quick exploration and refine for final numbers. Sweeps solve once per point (three times with `group_index=True`).

## Repository layout

- `src/elec438tools/`: `solver.py` (ModeSolver), `results.py` (Mode, ModeSet), `fde.py` (mesh, permittivity averaging, eigen-solve), `geometry.py`, `materials.py`, `waveguides.py` (builders), `sweeps.py`, `plotting.py`
- `tutorials/`: notebooks numbered in the order students use them (`01_mode_solver_basics.ipynb` is the HW1 tutorial, `02_guided_modes.ipynb` the S02 lecture examples); built by `dev/build_notebooks.py`
- `homework/`: assignment notebooks (`hw1.ipynb`, ...). Students answer in their own copy: code in the code cells, typed answers in the text cells starting with the question label, submitted as one PDF. Follow each question's requested quantities, precision and figure requirements exactly.
- `tests/`: `python -m pytest`; checks against exact slab solutions, the reference values on the S02 lecture slides, and femwell (FEM, optional)

When changing solver code, run the tests. Keep the public API above stable: notebooks and student code depend on it.
