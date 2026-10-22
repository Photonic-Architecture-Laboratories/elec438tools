# elec438tools

Photonic waveguide tools for **ELEC 438 Integrated Photonics Design** (Koç University). The first tool is a waveguide **mode solver**: you define a cross-section (geometry, materials, region, mesh) and it finds the guided modes, their effective and group indices, polarization, loss, effective area, and fields, and plots them.

It uses the finite-difference eigenmode (FDE) method on a Yee mesh, written in plain Python (NumPy, SciPy, Matplotlib).

## Start here

| | |
|---|---|
| 01 Finding waveguide modes (the HW1 tutorial) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Photonic-Architecture-Laboratories/elec438tools/blob/main/tutorials/01_mode_solver_basics.ipynb) |
| 02 Guided modes: the S02 lecture examples | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Photonic-Architecture-Laboratories/elec438tools/blob/main/tutorials/02_guided_modes.ipynb) |
| Homework 1 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Photonic-Architecture-Laboratories/elec438tools/blob/main/homework/hw1.ipynb) |

How to work with a notebook in Colab:

1. Click its **Open in Colab** button. You can run and edit the course copy, but nothing is kept: your changes disappear when you close the tab.
2. To keep your work, make your own copy first: **File → Save a copy in Drive**. Work in that copy (it is in your Google Drive under *Colab Notebooks*).
3. Run the cells in order (**Shift + Enter**, or **Runtime → Run all**). The first cell installs the package, about 20 s, once per session.
4. Keep the default CPU runtime; the solver does not use a GPU.

## Install on your own computer

```bash
pip install git+https://github.com/Photonic-Architecture-Laboratories/elec438tools.git
```

Python 3.9 or newer. For working on the code itself: `git clone`, then `pip install -e ".[dev]"` and `python -m pytest`.

## Example

```python
import elec438tools as et

sim = et.strip_waveguide(width=0.5, height=0.22)      # Si strip on SiO2, SiO2 cladding
sim.plot_index()
modes = sim.solve(wavelength=1.55, num_modes=4, group_index=True)
print(modes)
modes["TE0"].plot_fields()
```

```
wavelength 1.55 um; guided means Re(neff) > 1.4440 (highest index at the region edge)
  # name                    neff  loss(dB/cm)  TE frac       ng  Aeff(um2)  guided
  0 TE0                 2.446...            0    98.3%   4.19...     0.19     yes
  1 TM0                 1.773...            0     4.4%   3.75...     0.34     yes
  ...
```

Units are micrometres throughout; x is horizontal, y vertical, z the propagation direction.

## Using an AI assistant

You are welcome to use AI coding assistants. Point yours at [AGENTS.md](AGENTS.md): it is a compact reference of the whole package, written for them. Whatever the assistant writes, check the geometry with `plot_index()`, use only modes marked guided, and confirm the answer does not change when you refine the mesh or enlarge the region.

## Accuracy

The solver is tested against exact slab solutions, the reference values on the S02 lecture slides (agreement to ~1e-4 in neff), and an independent finite-element solver (femwell; TE0 of the 500 x 220 nm strip agrees to ~5e-4 at the default mesh). See `tests/`.

## Current scope

Metal (PEC) boundaries; materials Si, SiO2, air, and any constant index. Bends give neff and mode shape; bend loss needs absorbing boundaries, which are not in this version.
