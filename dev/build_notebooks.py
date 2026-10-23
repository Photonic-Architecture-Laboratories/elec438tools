"""Build the notebooks from the cell lists below.

Edit the cells here, then run:  python dev/build_notebooks.py
Tutorials are written to tutorials/ with their outputs, so they read well on GitHub.
"""

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
REPO = "Photonic-Architecture-Laboratories/elec438tools"

SETUP = '''# Setup: installs elec438tools if it is missing (on Colab this takes ~20 s)
import importlib.util, subprocess, sys
if importlib.util.find_spec("elec438tools") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           "git+https://github.com/%s.git"])

import numpy as np
import matplotlib.pyplot as plt
import elec438tools as et''' % REPO


def howto(name, folder="tutorials"):
    url = f"https://colab.research.google.com/github/{REPO}/blob/main/{folder}/{name}"
    return f"""## How to use this notebook

**In Google Colab (recommended: nothing to install)**

1. Open the course copy: {url}
   You can run and even edit this copy, but nothing you do is kept: Colab cannot save it back to GitHub, so your changes disappear when you close the tab.
2. To keep your work, make your own copy first: **File → Save a copy in Drive**. Colab opens the copy ("Copy of {name}"), which is saved in your Google Drive under *Colab Notebooks*. Work in that copy from now on.
3. Run the cells in order with **Shift + Enter**, or all at once with **Runtime → Run all**. The first code cell installs `elec438tools` (about 20 s). Colab forgets installed packages when its runtime restarts, so that cell runs again in each new session.
4. Keep the default **CPU** runtime. The solver does not use a GPU, so a GPU runtime will not make it faster.

**On your own computer:** run `pip install git+https://github.com/Photonic-Architecture-Laboratories/elec438tools.git` once, then open the notebook in Jupyter or VS Code.

**Updates:** if the course copy changes during the semester, open the link again and save a new copy; your old copy keeps your own work."""


def badge(name, folder="tutorials"):
    return (f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
            f"(https://colab.research.google.com/github/{REPO}/blob/main/{folder}/{name})")


TUTORIAL_01 = [
    ("md", f"""# 01 Finding waveguide modes

{badge("01_mode_solver_basics.ipynb")}

This tutorial introduces `elec438tools`, the course's Python mode solver. By the end you can set up a waveguide cross-section, find its modes, read their properties, plot their fields, check that a result has converged, and compute group index and dispersion.

Run the cells from top to bottom. The eight numbered **Exercises** ask you to change the code, run it, and report what you find. Write each answer in new cells directly below its exercise: your code in a code cell, and your answer typed in a text cell that starts with the exercise number."""),
    ("md", howto("01_mode_solver_basics.ipynb")),
    ("code", SETUP),
    ("md", r"""## 1. What a mode solver computes

A waveguide that does not change along its length $z$ supports *modes*: field patterns that keep their shape as they propagate,

$$\mathbf{E}(x, y, z, t) = \mathbf{E}(x, y)\, e^{i(\beta z - \omega t)}, \qquad n_\mathrm{eff} = \frac{\beta}{k_0} = \frac{\beta \lambda}{2\pi}.$$

The solver takes the refractive index of the cross-section $n(x, y)$ and a wavelength, and returns the allowed pairs of $n_\mathrm{eff}$ and $\mathbf{E}(x, y), \mathbf{H}(x, y)$.

It uses the **finite-difference eigenmode (FDE)** method: the cross-section is divided into a rectangular (Yee) mesh, Maxwell's equations become a large sparse matrix, and its eigenvalues are $n_\mathrm{eff}^2$.

Conventions used everywhere in `elec438tools`:

* lengths and wavelengths in **micrometres** (1.55, not 1550);
* $x$ horizontal, $y$ vertical, $z$ the propagation direction;
* field arrays are indexed `[ix, iy]`."""),
    ("md", """## 2. A first waveguide

`strip_waveguide` builds the standard silicon strip: a Si core on SiO2 with SiO2 cladding. $x = 0$ is the centre of the core and $y = 0$ the top of the substrate. Printing the solver shows the region and the mesh."""),
    ("code", '''sim = et.strip_waveguide(width=0.5, height=0.22)
print(sim)
sim.plot_index(show_mesh=True, xlim=(-0.8, 0.8), ylim=(-0.4, 0.6));'''),
    ("md", """Always look at the index profile before solving: it is the quickest way to catch a misplaced structure.

The solver region is surrounded by **metal walls** (perfect electric conductors). The fields of a guided mode must have decayed before they reach the walls, otherwise the walls change the answer; section 6 shows how to check this.

The mesh is finer (10 nm) in a box around the core and coarser (20 nm) elsewhere: a *mesh override region*. Mesh lines always land on material interfaces."""),
    ("md", """## 3. Building a cross-section yourself

The builder above is a shortcut. Here is the same waveguide set up by hand, which is how you would build any other geometry. Structures are painted in the order they are added: a later one overrides an earlier one where they overlap."""),
    ("code", '''sim = et.ModeSolver(x_span=3.5, y_span=3.2, y_center=0.11, dx=0.02, background="SiO2")
sim.add(et.Layer(-np.inf, 0.0, "SiO2", name="substrate"))          # everything below y = 0
core = sim.add(et.Rect(width=0.5, height=0.22, x=0, y=0.11, material="Si", name="core"))
sim.add_mesh_override(x_min=-0.35, x_max=0.35, y_min=-0.1, y_max=0.32, dx=0.01)
print(sim)'''),
    ("md", """Materials can be `"Si"`, `"SiO2"`, `"air"`, or any number (a constant refractive index, complex for loss). Si and SiO2 include their wavelength dependence (Sellmeier models), which matters for group index:"""),
    ("code", '''for wl in (1.31, 1.55):
    print(f"{wl} um:  n_Si = {et.Si.n(wl).real:.4f},  n_SiO2 = {et.SiO2.n(wl).real:.4f}")'''),
    ("md", """## 4. Finding modes

`solve` returns the modes whose effective index is closest to `n_guess`; by default that is the highest index in the structure, so the fundamental mode comes first."""),
    ("code", '''modes = sim.solve(wavelength=1.55, num_modes=4)
modes'''),
    ("md", """Reading the table:

* **neff**: effective index. Modes are sorted from highest to lowest.
* **loss**: propagation loss from the imaginary part of $n_\\mathrm{eff}$ (zero here: no absorbing materials).
* **TE frac**: $\\int |E_x|^2 / \\int (|E_x|^2 + |E_y|^2)$, the *TE polarization fraction*. Above 50 % the mode is called TE-like, below it TM-like.
* **Aeff**: effective area $(\\int |E|^2)^2 / \\int |E|^4$.
* **guided**: whether $n_\\mathrm{eff}$ is above the highest index at the edge of the region (here SiO2, 1.444). A mode below it is not confined by the core; it is a mode of the metal box.

Guided modes are named in order within each polarization: TE0, TE1, ... and TM0, TM1, .... Each mode can be fetched by name:"""),
    ("code", '''te0 = modes["TE0"]
print(te0)
print("beta =", te0.beta, "rad/um")
print("fraction of power in the core:", round(te0.power_fraction(core), 3))'''),
    ("md", r"""The fraction of power in the core (the *confinement factor*) is the integral of the Poynting vector $S_z = \tfrac{1}{2}\mathrm{Re}(\mathbf{E}\times\mathbf{H}^*)\cdot\hat{z}$ over the core, divided by its integral over the whole cross-section.

**Exercise 1.** Solve the same waveguide with a width of 450 nm instead of 500 nm. List its guided modes with their $n_\mathrm{eff}$ (4 decimals). For the guided mode with the lowest $n_\mathrm{eff}$, give $n_\mathrm{eff} - n_\mathrm{SiO_2}$ (4 decimals), and explain in 1-2 sentences why such a mode is called weakly guided."""),
    ("md", """## 5. Plotting the fields

`plot_fields` shows all six field components, normalised so that the largest $|E|$ is 1. The H field is in matching physical units, so it is about $n/377$ times smaller."""),
    ("code", '''te0.plot_fields(xlim=(-0.8, 0.8), ylim=(-0.5, 0.7));'''),
    ("code", '''fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
modes["TE0"].plot("E2", ax=axs[0], xlim=(-0.8, 0.8), ylim=(-0.5, 0.7))   # |E|^2, "E intensity"
modes["TM0"].plot("E2", ax=axs[1], xlim=(-0.8, 0.8), ylim=(-0.5, 0.7))
plt.tight_layout()'''),
    ("md", """Any single component can be plotted as a magnitude, real part, imaginary part or phase: `"Ex"`, `"|Ex|"`, `"Re(Ex)"`, `"Im(Hy)"`, `"phase(Ex)"`; and the totals `"E"`, `"E2"`, `"H"`, `"Sz"` (power flow). Line cuts through the field maximum:"""),
    ("code", '''te0.plot_slice("Ex");'''),
    ("md", r"""**Exercise 2.** In the horizontal cut, $|E_x|$ jumps at the sidewalls ($x = \pm 0.25$ um). Name the boundary condition that causes the jump, and use it to predict the ratio of $|E_x|$ just outside to just inside the silicon from $n_\mathrm{Si}$ and $n_\mathrm{SiO_2}$ at 1.55 um (2 decimals). Then plot $|E_y|$ of TM0 along a vertical line through the core (`modes["TM0"].plot_slice("Ey", x=0)`), and say at which interfaces it jumps and why."""),
    ("md", """## 6. Convergence testing

A mode solver's answer depends on the mesh and on the size of the region. Before trusting a number, change each of them and check that the number stops changing. This is the most important habit in simulation.

**Mesh.** Refine the mesh and watch $n_\\mathrm{eff}$ settle:"""),
    ("code", '''dxs = [0.04, 0.02, 0.01, 0.005]
neff_te, neff_tm = [], []
for dx in dxs:
    # default mesh 20 nm (or dx if coarser), refined to dx around the core
    m = et.strip_waveguide(0.5, 0.22, dx=max(dx, 0.02), fine_dx=dx).solve(1.55, num_modes=2)
    neff_te.append(m["TE0"].neff.real)
    neff_tm.append(m["TM0"].neff.real)
    print(f"fine mesh {dx*1e3:4.0f} nm:  TE0 {neff_te[-1]:.5f}   TM0 {neff_tm[-1]:.5f}")

fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
axs[0].semilogx(dxs, neff_te, "o-"); axs[0].set_title("TE0")
axs[1].semilogx(dxs, neff_tm, "o-"); axs[1].set_title("TM0")
for a in axs:
    a.set_xlabel("mesh size in the core (um)"); a.set_ylabel("neff")
plt.tight_layout()'''),
    ("md", """**Region size.** Move the metal walls closer and see when they start to matter. Weakly guided modes, whose fields spread far into the cladding, need the largest region. The solver warns when a guided mode has more than 1 % of its field next to a wall."""),
    ("code", '''import warnings
margins = [0.3, 0.5, 0.8, 1.2, 1.5, 2.0]
for margin in margins:
    sim_m = et.strip_waveguide(0.5, 0.22, x_span=0.5 + 2 * margin, y_span=0.22 + 2 * margin)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m = sim_m.solve(1.55, num_modes=3)
    names = "  ".join(f"{mode.name or '-'} {mode.neff.real:.4f}" for mode in m)
    print(f"walls {margin} um from the core:  {names}   ({len(w)} warnings)")'''),
    ("md", r"""**Exercise 3.** Extend the region test to margins of 2.5, 3 and 4 um. How far from the core must the walls be for the $n_\mathrm{eff}$ of TE0, and of TE1, to stop changing in the 4th decimal? Explain the difference using the decay of each mode's field outside the core, $e^{-\gamma d}$ with $\gamma = k_0 \sqrt{n_\mathrm{eff}^2 - n_\mathrm{SiO_2}^2}$: give $\gamma$ for both modes (1/um, 2 decimals)."""),
    ("md", r"""## 7. Wavelength dependence: group index and dispersion

The **group index** sets how fast a pulse travels and how far apart the resonances of a ring are:

$$n_g = n_\mathrm{eff} - \lambda \frac{dn_\mathrm{eff}}{d\lambda}.$$

`group_index=True` computes it for every mode (with two extra solves at nearby wavelengths, including the materials' own dispersion):"""),
    ("code", '''modes = et.strip_waveguide(0.5, 0.22).solve(1.55, num_modes=2, group_index=True)
modes'''),
    ("md", r"""To see the whole curve, sweep the wavelength. The modes are followed from one wavelength to the next by the overlap of their fields. The result also gives the dispersion parameter $D = -\frac{\lambda}{c}\frac{d^2 n_\mathrm{eff}}{d\lambda^2}$ in ps/(nm km)."""),
    ("code", '''disp = et.strip_waveguide(0.5, 0.22).wavelength_sweep(np.linspace(1.45, 1.65, 11), num_modes=2)
fig, axs = plt.subplots(1, 3, figsize=(15, 3.6))
disp.plot("neff", ax=axs[0]); disp.plot("ng", ax=axs[1]); disp.plot("D", ax=axs[2])
plt.tight_layout()'''),
    ("md", r"""**Exercise 4.** From `disp.wavelengths` and `disp.neff`, compute $dn_\mathrm{eff}/d\lambda$ of TE0 at 1.55 um (`np.gradient`; 1/um, 3 decimals), and from it $n_g = n_\mathrm{eff} - \lambda\, dn_\mathrm{eff}/d\lambda$ (3 decimals). Compare it with `disp.ng` and with the group index from `solve(..., group_index=True)` above. Explain in 2-3 sentences why $n_g$ is larger than $n_\mathrm{eff}$."""),
    ("md", """## 8. Sweeping a dimension

`et.sweep` rebuilds and solves the waveguide for each value of a parameter. Here is the effective index of every guided mode against the core width. The dotted line is the cladding index: a mode is guided only above it, so the width where a curve meets the line is that mode's cutoff."""),
    ("code", '''widths = np.arange(0.25, 1.01, 0.05)
res = et.sweep(lambda w: et.strip_waveguide(width=w, height=0.22), widths,
               wavelength=1.55, num_modes=6, param_name="width (um)")
res.plot()
print("guided TE modes:", res.num_guided("TE"))
print("guided TM modes:", res.num_guided("TM"))'''),
    ("md", """`res.neff_te[:, 0]` is TE0 at every width, `res.neff_te[:, 1]` is TE1, and so on (NaN where the mode is not guided). `res.to_dataframe()` gives a table you can save to CSV.

**Exercise 5.** From the sweep, give the range of widths over which the waveguide has exactly one guided TE mode at 1.55 um (to the nearest 50 nm). Repeat the sweep at 1.31 um and give the range there. Explain in 1-2 sentences why the range changes with wavelength."""),
    ("md", """## 9. Other cross-sections

The same tools work for any geometry. A few that come up in this course:"""),
    ("code", '''rib = et.rib_waveguide(width=0.5, height=0.22, slab_height=0.09)
slot = et.slot_waveguide(rail_width=0.175, gap=0.15, height=0.22)
pair = et.coupled_waveguides(width=0.5, gap=0.2, height=0.22)

fig, axs = plt.subplots(1, 3, figsize=(15, 3.4))
for a, s, title in zip(axs, (rib, slot, pair), ("rib", "slot", "two coupled strips")):
    s.solve(1.55, num_modes=2)[0].plot("E2", ax=a, xlim=(-1.0, 1.0), ylim=(-0.5, 0.7))
    a.set_title(title)
plt.tight_layout()'''),
    ("md", r"""Two coupled waveguides have an even and an odd **supermode**. Light launched into one guide moves fully to the other after the coupling length $L_c = \lambda / (2\,\Delta n_\mathrm{eff})$:"""),
    ("code", '''sm = pair.solve(1.55, num_modes=4)
even, odd = sm["TE0"], sm["TE1"]
fig, axs = plt.subplots(1, 2, figsize=(11, 3.2))
even.plot("Re(Ex)", ax=axs[0], xlim=(-1.2, 1.2), ylim=(-0.5, 0.7))
odd.plot("Re(Ex)", ax=axs[1], xlim=(-1.2, 1.2), ylim=(-0.5, 0.7))
plt.tight_layout()
print(f"coupling length = {1.55 / (2 * (even.neff.real - odd.neff.real)):.2f} um")'''),
    ("md", r"""**Exercise 6.** Compute the coupling length for gaps of 0.1, 0.2, 0.3 and 0.4 um (2 decimals), and plot it against the gap with a logarithmic vertical axis. Explain the trend in 1-2 sentences in terms of how far the field of one guide reaches into the other."""),
    ("md", """A **1D slab** is invariant in $x$; give `x_span=None` (or use `slab_waveguide`). Its modes are pure TE ($E_x, H_y, H_z$) or pure TM ($E_y, H_x, E_z$):"""),
    ("code", '''slab = et.slab_waveguide(thickness=0.22)
sm = slab.solve(1.55, num_modes=2, group_index=True)
print(sm)
sm["TE0"].plot("Ex");'''),
    ("md", r"""**Exercise 7.** Compare the $n_\mathrm{eff}$ of the slab's TE0 with that of TE0 in 220 nm thick strips of width 0.5, 1, 2 and 4 um (4 decimals). Explain the trend in 1-2 sentences."""),
    ("md", """A **bend** of radius $R$ can be modelled with `solve(..., bend_radius=R)`. The mode moves towards the outside of the bend. (Bend *loss* needs absorbing boundaries, which this version does not have yet, so only $n_\\mathrm{eff}$ and the mode shape are meaningful.)"""),
    ("code", '''wide = et.strip_waveguide(0.5, 0.22, x_span=4.0)
fig, axs = plt.subplots(1, 2, figsize=(11, 3.2))
for a, R in zip(axs, (None, 5.0)):
    m = wide.solve(1.55, num_modes=1, bend_radius=R)[0]
    m.plot("E2", ax=a, xlim=(-1.0, 1.0), ylim=(-0.5, 0.7))
    a.set_title(f"R = {R} um, neff = {m.neff.real:.4f}" if R else f"straight, neff = {m.neff.real:.4f}")
plt.tight_layout()'''),
    ("md", r"""**Exercise 8.** Replace the SiO2 cladding of the 500 nm strip with air (`et.strip_waveguide(0.5, 0.22, cladding="air")`; the substrate stays SiO2). Give $n_\mathrm{eff}$ of TE0 and TM0 with each cladding (4 decimals). Which mode changes more, and why?"""),
    ("md", """## 10. Saving results, and working with an AI assistant

`mode.save("te0.mat")` writes the fields and properties to a `.mat` file, `mode.save("te0.npz")` to a NumPy file; `modes.to_dataframe()` and `res.to_dataframe()` give tables.

If you use an AI coding assistant, point it at `AGENTS.md` in the repository: it is a compact reference of everything in this package, written for exactly that purpose. Whatever the assistant writes, the checks in this tutorial still apply: look at the index profile, check convergence, and check that the walls are far enough away."""),
]

S02 = [
    ("md", f"""# 02 Guided modes: the S02 lecture examples

{badge("02_guided_modes.ipynb")}

The two examples from the *S02 Guided Modes* slides, solved with `elec438tools`: a 1 um thick slab and a 2 x 1 um rectangular core, both of index 1.444 in air, at 1.55 um, in a 3 x 4 um region with metal walls. The values on the slides are printed alongside for comparison."""),
    ("md", howto("02_guided_modes.ipynb")),
    ("code", SETUP),
    ("md", """## Slab: 1 um thick, n = 1.444, uniform in x

On the slides the slab fills the region in $x$, so the metal walls at $x = \\pm 1.5$ um also act on it: modes 2 to 6 have the same vertical profile as mode 1 but vary between the side walls."""),
    ("code", '''sim = et.ModeSolver(x_span=3.0, y_span=4.0, dx=0.02, background="air")
sim.add(et.Layer(-0.5, 0.5, 1.444))
slab_modes = sim.solve(1.55, num_modes=8)
slides = [1.347850, 1.322865, 1.283040, 1.244908, 1.202506, 1.102848, 1.072984]
for m, ref in zip(slab_modes, slides):
    print(f"mode {m.index + 1}:  neff {m.neff.real:.6f}   slides {ref:.6f}   TE fraction {100*m.te_fraction:.0f} %")'''),
    ("code", '''for k in (0, 6):           # mode A and mode B of the slides
    slab_modes[k].plot_fields()
    slab_modes[k].plot_slice("Ex")'''),
    ("md", """## Rectangle: 2 x 1 um core, n = 1.444"""),
    ("code", '''sim = et.ModeSolver(x_span=3.0, y_span=4.0, dx=0.02, background="air")
sim.add(et.Rect(width=2.0, height=1.0, material=1.444))
sim.plot_index()
modes = sim.solve(1.55, num_modes=8)
modes'''),
    ("code", '''slides = {"TE0": 1.304020, "TM0": 1.271321, "TE1": 1.176642, "TM1": 1.155666, "TE2": 1.029682}
for name, ref in slides.items():
    print(f"{name}:  neff {modes[name].neff.real:.6f}   slides {ref:.6f}")'''),
    ("md", """The warnings say the region of the slides is tight for the higher-order modes: they reach the walls. Enlarging the region shifts their $n_\\mathrm{eff}$ in the 3rd or 4th decimal."""),
    ("code", '''modes.plot("E", max_cols=5);'''),
    ("code", '''for name in ("TE0", "TM0"):
    modes[name].plot_fields()
    modes[name].plot_slice("Ex" if name == "TE0" else "Ey")'''),
]


def build(cells, name, execute=True, folder="tutorials"):
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["colab"] = {"provenance": []}
    for kind, src in cells:
        nb.cells.append(nbformat.v4.new_markdown_cell(src) if kind == "md" else nbformat.v4.new_code_cell(src))
    out = ROOT / folder / name
    out.parent.mkdir(exist_ok=True)
    if execute:
        NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
    nbformat.write(nb, out)
    print("wrote", out)


if __name__ == "__main__":
    execute = "--no-exec" not in sys.argv
    build(TUTORIAL_01, "01_mode_solver_basics.ipynb", execute)
    build(S02, "02_guided_modes.ipynb", execute)
