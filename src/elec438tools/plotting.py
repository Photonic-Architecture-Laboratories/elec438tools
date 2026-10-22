"""Matplotlib plots of index profiles, mode fields, slices and mode sets."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

SIGNED_CMAP = "RdBu_r"


def _region_limits(ax_plot, sim, xlim=None, ylim=None):
    axx, ayy = sim.axes
    if not axx.uniform:
        ax_plot.set_xlim(xlim or (axx.lo, axx.hi))
    ax_plot.set_ylim(ylim or (ayy.lo, ayy.hi))


def draw_structures(ax_plot, sim, color="white", lw=0.8):
    """Outline every structure; infinite layers become horizontal lines at their faces."""
    axx, ayy = sim.axes
    for s in sim.structures:
        out = s.outline()
        if out is not None:
            ax_plot.plot(*out, color=color, lw=lw, alpha=0.8)
            continue
        x0, x1, y0, y1 = s.bounds()
        for yy in (y0, y1):
            if np.isfinite(yy) and ayy.lo < yy < ayy.hi:
                xa = max(x0, axx.lo) if np.isfinite(x0) else axx.lo
                xb = min(x1, axx.hi) if np.isfinite(x1) else axx.hi
                ax_plot.plot([xa, xb], [yy, yy], color=color, lw=lw, alpha=0.8)


def _new_ax(ax):
    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 4))
    return ax


def plot_index(sim, wavelength, ax=None, show_mesh=False, cmap="turbo", xlim=None, ylim=None):
    axx, ayy = sim.axes
    n = sim.index(wavelength).real
    ax = _new_ax(ax)
    if axx.uniform:
        ax.plot(n[0], sim.y)
        ax.set_xlabel("refractive index")
        ax.set_ylabel("y (um)")
        ax.set_title(f"index profile at {wavelength} um (invariant in x)")
        return ax
    im = ax.pcolormesh(sim.x, sim.y, n.T, shading="nearest", cmap=cmap)
    plt.colorbar(im, ax=ax, label="refractive index")
    if show_mesh:
        for e in axx.edges:
            ax.axvline(e, color="k", lw=0.2, alpha=0.4)
        for e in ayy.edges:
            ax.axhline(e, color="k", lw=0.2, alpha=0.4)
    ax.set_aspect("equal")
    _region_limits(ax, sim, xlim, ylim)
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_title(f"index at {wavelength} um")
    return ax


def _title(mode):
    tag = mode.name or ("not guided" if not mode.guided else "")
    return f"{tag}  neff = {mode.neff.real:.4f}".strip()


def plot_mode(mode, quantity="E2", part=None, ax=None, cmap=None, show_structure=True, colorbar=True,
              vmax=None, title=True, xlim=None, ylim=None):
    A, label = mode.get(quantity, part)
    signed = label.startswith(("Re(", "Im(", "Sz"))
    if cmap is None:
        cmap = SIGNED_CMAP if signed else "turbo"
    ax = _new_ax(ax)
    sim = mode._sim
    if sim.axes[0].uniform:
        ax.plot(A[0], mode.y)
        ax.set_xlabel(label)
        ax.set_ylabel("y (um)")
        if title:
            ax.set_title(_title(mode))
        return ax
    if signed:
        lim = vmax or np.abs(A).max()
        vmin, vmax_ = -lim, lim
    else:
        vmin, vmax_ = (np.nanmin(A) if label.startswith("phase") else 0), (vmax or np.nanmax(A))
    im = ax.pcolormesh(mode.x, mode.y, A.T, shading="nearest", cmap=cmap, vmin=vmin, vmax=vmax_)
    if colorbar:
        plt.colorbar(im, ax=ax, label=label)
    if show_structure:
        draw_structures(ax, sim)
    ax.set_aspect("equal")
    _region_limits(ax, sim, xlim, ylim)
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    if title:
        ax.set_title(f"{_title(mode)}" if colorbar else f"{label}  {_title(mode)}")
    return ax


def plot_fields(mode, cmap="turbo", xlim=None, ylim=None):
    comps = ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]
    if mode._sim.axes[0].uniform:
        fig, axs = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
        for c in comps[:3]:
            axs[0].plot(abs(mode.fields[c][0]), mode.y, label=f"|{c}|")
        for c in comps[3:]:
            axs[1].plot(abs(mode.fields[c][0]), mode.y, label=f"|{c}|")
        for a in axs:
            a.legend()
        axs[0].set_ylabel("y (um)")
        fig.suptitle(_title(mode))
        return fig
    fig, axs = plt.subplots(2, 3, figsize=(12, 7))
    e_max = max(abs(mode.fields[c]).max() for c in comps[:3])
    h_max = max(abs(mode.fields[c]).max() for c in comps[3:])
    for a, c in zip(axs.flat, comps):
        A = abs(mode.fields[c])
        vm = e_max if c[0] == "E" else h_max
        im = a.pcolormesh(mode.x, mode.y, A.T, shading="nearest", cmap=cmap, vmin=0, vmax=vm)
        fig.colorbar(im, ax=a)
        draw_structures(a, mode._sim)
        a.set_aspect("equal")
        _region_limits(a, mode._sim, xlim, ylim)
        a.set_title(f"|{c}|")
        a.set_xlabel("x (um)")
        a.set_ylabel("y (um)")
    fig.suptitle(_title(mode))
    fig.tight_layout()
    return fig


def plot_slice(mode, quantity="E2", x=None, y=None, ax=None):
    A, label = mode.get(quantity)
    if mode._sim.axes[0].uniform:
        ax = _new_ax(ax)
        ax.plot(mode.y, A[0])
        ax.set_xlabel("y (um)")
        ax.set_ylabel(label)
        ax.set_title(_title(mode))
        return ax
    if x is None and y is None:
        i, j = np.unravel_index(np.argmax(np.abs(A)), A.shape)
        fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
        axs[0].plot(mode.y, A[i, :])
        axs[0].set_xlabel("y (um)")
        axs[0].set_title(f"vertical cut at x = {mode.x[i]:.3f} um")
        axs[1].plot(mode.x, A[:, j])
        axs[1].set_xlabel("x (um)")
        axs[1].set_title(f"horizontal cut at y = {mode.y[j]:.3f} um")
        for a in axs:
            a.set_ylabel(label)
        fig.suptitle(_title(mode))
        fig.tight_layout()
        return axs
    ax = _new_ax(ax)
    if y is not None:
        j = int(np.argmin(abs(mode.y - y)))
        ax.plot(mode.x, A[:, j])
        ax.set_xlabel("x (um)")
        ax.set_title(f"{_title(mode)}, cut at y = {mode.y[j]:.3f} um")
    else:
        i = int(np.argmin(abs(mode.x - x)))
        ax.plot(mode.y, A[i, :])
        ax.set_xlabel("y (um)")
        ax.set_title(f"{_title(mode)}, cut at x = {mode.x[i]:.3f} um")
    ax.set_ylabel(label)
    return ax


def plot_modeset(modes, quantity="E2", max_cols=4, cmap="turbo", xlim=None, ylim=None):
    n = len(modes)
    cols = min(max_cols, n)
    rows = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.2 * rows), squeeze=False)
    for a, m in zip(axs.flat, modes):
        plot_mode(m, quantity, ax=a, cmap=cmap, colorbar=False, xlim=xlim, ylim=ylim)
    for a in axs.flat[n:]:
        a.axis("off")
    fig.tight_layout()
    return fig
