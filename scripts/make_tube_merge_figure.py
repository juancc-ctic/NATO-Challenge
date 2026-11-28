"""Genera la figura sintética de `tube_merge` para el README.

Dos paneles: (a) tubo 3D alrededor de un track y (b) vista cenital con la
decisión de conservar/eliminar cada track.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "tube_merge.png"

A = "#1F4E79"
TUBE = "#4C78A8"
B = "#2CA02C"
C = "#E07B39"
GRID = "#D9E1E8"
MUTED = "#5B6770"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.edgecolor": GRID,
    "axes.linewidth": 0.8,
    "axes.titleweight": "bold",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


def track_a(n: int = 120) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 1.0, n)
    x = 8.0 + 84.0 * t
    y = 26.0 + 13.0 * np.sin(2.0 * np.pi * 0.90 * t) + 4.0 * np.sin(2.0 * np.pi * 2.1 * t)
    z = 24.0 + 5.0 * np.sin(2.0 * np.pi * 1.3 * t + 0.6)
    return x, y, z


def offset_track(x: np.ndarray, y: np.ndarray, z: np.ndarray, dx: float, dy: float, dz: float):
    return x + dx, y + dy, z + dz


def tube_surface(x, y, z, radius: float, n_theta: int = 22):
    tangent = np.gradient(np.column_stack([x, y, z]), axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True) + 1e-9
    ref = np.array([0.0, 0.0, 1.0])
    normal = np.cross(tangent, ref)
    normal /= np.linalg.norm(normal, axis=1, keepdims=True) + 1e-9
    binormal = np.cross(tangent, normal)
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta)
    surface = np.empty((len(x), n_theta, 3))
    for i in range(len(x)):
        surface[i] = (
            np.array([x[i], y[i], z[i]])[None, :]
            + radius * (np.cos(theta)[:, None] * normal[i][None, :]
                        + np.sin(theta)[:, None] * binormal[i][None, :])
        )
    return surface


def style_2d(ax):
    ax.set_aspect("equal")
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.7)
    ax.set_xlabel("Este (m)")
    ax.set_ylabel("Norte (m)")
    for spine in ax.spines.values():
        spine.set_color(GRID)


def main() -> None:
    x, y, z = track_a()
    xb, yb, zb = offset_track(x, y, z, 0.6, 8.5, 0.0)
    xc, yc, zc = offset_track(x, y, z, 0.0, 52.0, 0.0)
    radius = 10.5

    fig = plt.figure(figsize=(14.0, 6.4))
    gs = fig.add_gridspec(1, 2, left=0.04, right=0.98, bottom=0.08, top=0.78,
                          wspace=0.10, width_ratios=[1.02, 1.0])

    # (a) Tubo 3D
    ax3 = fig.add_subplot(gs[0, 0], projection="3d")
    surface = tube_surface(x, y, z, radius)
    ax3.plot_surface(surface[:, :, 0], surface[:, :, 1], surface[:, :, 2],
                     color=TUBE, alpha=0.08, linewidth=0, antialiased=True, shade=False,
                     rstride=1, cstride=1)
    n_theta = surface.shape[1]
    for k in range(0, n_theta, max(1, n_theta // 10)):
        ax3.plot(surface[:, k, 0], surface[:, k, 1], surface[:, k, 2],
                 color=TUBE, lw=0.65, alpha=0.38)
    for i in range(0, len(x), max(1, len(x) // 8)):
        ax3.plot(surface[i, :, 0], surface[i, :, 1], surface[i, :, 2],
                 color=TUBE, lw=0.85, alpha=0.48)
    ax3.plot(x, y, z, color=A, lw=2.8, label="Track A (se conserva)")
    ax3.scatter(xb, yb, zb, color=B, s=16, depthshade=False, label="Track B (se elimina)")
    ax3.scatter(xc, yc, zc, color=C, s=16, depthshade=False, label="Track C (se conserva)")
    ax3.set_xlabel("Este (m)", labelpad=4)
    ax3.set_ylabel("Norte (m)", labelpad=4)
    ax3.set_zlabel("Altura (m)", labelpad=4)
    ax3.set_box_aspect((1.5, 1.0, 0.55))
    ax3.view_init(elev=23, azim=-58)
    ax3.set_title("(a) Tubo 3D alrededor del track A", loc="left", pad=10)
    ax3.legend(loc="upper left", frameon=False, fontsize=8, handletextpad=0.4)
    ax3.tick_params(labelsize=7, colors=MUTED)
    for axis in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
    ax3.grid(False)

    # (b) Vista cenital: tubo, duplicado y track distinto
    ax2 = fig.add_subplot(gs[0, 1])
    for xi, yi in zip(x, y):
        ax2.add_patch(Circle((xi, yi), radius, facecolor=TUBE, edgecolor="none", alpha=0.05))
    ax2.plot(x, y, color=A, lw=2.6, label="A")
    ax2.scatter(xb, yb, color=B, s=26, zorder=5, edgecolor="white", linewidth=0.5, label="B")
    ax2.scatter(xc, yc, color=C, s=26, zorder=5, edgecolor="white", linewidth=0.5, label="C")
    i = len(x) // 2
    ax2.add_patch(FancyArrowPatch((x[i], y[i]), (x[i] + radius, y[i]),
                                 arrowstyle="<->", mutation_scale=10, color=MUTED, lw=1.1))
    ax2.text(x[i] + radius / 2, y[i] + 2.5, "r", ha="center", color=MUTED, fontsize=9)

    ax2.annotate("A: se conserva\n(keep = longest | hits)",
                 xy=(x[0], y[0]), xytext=(x[0] - 4, y[0] - 18),
                 fontsize=8.5, color=A, ha="left", va="top",
                 arrowprops=dict(arrowstyle="-", color=A, lw=0.9, alpha=0.6),
                 bbox=dict(boxstyle="round,pad=0.35", facecolor="#F5F8FB", edgecolor=GRID))
    ax2.annotate("B ⊂ tubo(A)\nf = 1.00 → se elimina",
                 xy=(xb[-1], yb[-1]), xytext=(x[-1] + 8, yb[-1] + 10),
                 fontsize=8.5, color=B, ha="left", va="bottom",
                 arrowprops=dict(arrowstyle="->", color=B, lw=1.0),
                 bbox=dict(boxstyle="round,pad=0.35", facecolor="#F2FAF2", edgecolor=B, alpha=0.9))
    ax2.annotate("C ∉ tubo(A)\nf = 0.00 → se conserva",
                 xy=(xc[-1], yc[-1]), xytext=(x[-1] + 8, yc[-1] + 8),
                 fontsize=8.5, color=C, ha="left", va="bottom",
                 arrowprops=dict(arrowstyle="->", color=C, lw=1.0),
                 bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFF7F0", edgecolor=C, alpha=0.9))
    ax2.set_xlim(x.min() - 18, x.max() + 58)
    ax2.set_ylim(min(y.min(), yc.min()) - 18, max(y.max(), yc.max()) + 18)
    style_2d(ax2)
    ax2.set_title("(b) Decisión: qué track se conserva", loc="left")
    fig.text(0.04, 0.965, "tube_merge: fusión de tracks duplicados por tubo 3D",
             fontsize=18, fontweight="bold", color="#111111", ha="left", va="top")
    fig.text(0.04, 0.905,
             "El tubo de A son esferas de radio r en cada punto. B cae dentro (f = 1.00 ≥ 0.80) "
             "y se elimina; C queda fuera (f = 0.00) y se conserva.",
             fontsize=10, color=MUTED, ha="left", va="top")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
