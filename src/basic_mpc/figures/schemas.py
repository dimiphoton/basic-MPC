"""Schémas RC et Kalman pour publication (matplotlib, pas de visio).

Trois circuits thermiques, même langage visuel : nœuds, R en zigzag,
C vers la masse, flèches d'entrée, capteur distinct de l'état.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

from basic_mpc.config import REPO_ROOT
from basic_mpc.figures.style import ACCENT, ACCENT_2, BG, INK, MUTED
from basic_mpc.figures.style import apply_publication_rc, save_figure

NODE_R = 0.38


def _zigzag(
    ax: plt.Axes,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    n_teeth: int = 6,
    amplitude: float = 0.11,
) -> None:
    """Résistance : zigzag perpendiculaire au segment."""
    vec = np.array([x1 - x0, y1 - y0], dtype=float)
    length = float(np.linalg.norm(vec))
    if length < 1e-9:
        return
    direction = vec / length
    normal = np.array([-direction[1], direction[0]])
    # Ligne droite au début et à la fin (patte de connexion)
    pad = 0.12
    p0 = np.array([x0, y0]) + direction * pad
    p1 = np.array([x1, y1]) - direction * pad
    ax.plot([x0, p0[0]], [y0, p0[1]], color=INK, lw=1.4, solid_capstyle="round")
    ax.plot([p1[0], x1], [p1[1], y1], color=INK, lw=1.4, solid_capstyle="round")
    xs = [p0[0]]
    ys = [p0[1]]
    span = p1 - p0
    for i in range(n_teeth):
        t_a = (i + 0.25) / n_teeth
        sign = 1.0 if i % 2 == 0 else -1.0
        pt = p0 + t_a * span + sign * amplitude * normal
        xs.append(float(pt[0]))
        ys.append(float(pt[1]))
    xs.append(float(p1[0]))
    ys.append(float(p1[1]))
    ax.plot(xs, ys, color=INK, lw=1.4, solid_capstyle="round")


def _capacitor_down(ax: plt.Axes, x: float, y_top: float, y_gnd: float) -> None:
    """Capacité vers la référence (deux armatures + terre)."""
    gap = 0.09
    width = 0.34
    y_mid = (y_top + y_gnd) * 0.52
    ax.plot([x, x], [y_top, y_mid + gap], color=INK, lw=1.4)
    ax.plot([x - width, x + width], [y_mid + gap, y_mid + gap], color=INK, lw=2.0)
    ax.plot([x - width, x + width], [y_mid - gap, y_mid - gap], color=INK, lw=2.0)
    ax.plot([x, x], [y_mid - gap, y_gnd + 0.12], color=INK, lw=1.4)
    # Symbole terre
    ax.plot([x - 0.18, x + 0.18], [y_gnd + 0.12, y_gnd + 0.12], color=INK, lw=1.6)
    ax.plot([x - 0.12, x + 0.12], [y_gnd + 0.05, y_gnd + 0.05], color=INK, lw=1.4)
    ax.plot([x - 0.06, x + 0.06], [y_gnd - 0.02, y_gnd - 0.02], color=INK, lw=1.3)


def _node(ax: plt.Axes, x: float, y: float, label: str, fill: str = "#FBF7F0") -> None:
    """Nœud de température."""
    circle = Circle((x, y), NODE_R, facecolor=fill, edgecolor=INK, lw=1.6, zorder=3)
    ax.add_patch(circle)
    ax.text(x, y, label, ha="center", va="center", color=INK, fontsize=10, zorder=4)


def _ext_box(ax: plt.Axes, x: float, y: float, label: str) -> None:
    """Source / condition aux limites."""
    box = FancyBboxPatch(
        (x - 0.55, y - 0.32),
        1.10,
        0.64,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        facecolor="#E7E0D4",
        edgecolor=INK,
        lw=1.4,
        zorder=3,
    )
    ax.add_patch(box)
    ax.text(x, y, label, ha="center", va="center", color=INK, fontsize=10, zorder=4)


def _arrow(
    ax: plt.Axes,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    color: str,
    label: str | None = None,
    label_off: tuple[float, float] = (0.0, 0.14),
) -> None:
    """Flèche d'apport ou d'observation."""
    patch = FancyArrowPatch(
        (x0, y0),
        (x1, y1),
        arrowstyle="-|>",
        mutation_scale=12,
        lw=1.5,
        color=color,
        zorder=2,
    )
    ax.add_patch(patch)
    if label:
        ax.text(
            (x0 + x1) / 2 + label_off[0],
            (y0 + y1) / 2 + label_off[1],
            label,
            ha="center",
            va="bottom",
            color=color,
            fontsize=10,
        )


def _caption(ax: plt.Axes, title: str, subtitle: str, y: float = 2.55) -> None:
    ax.text(0.0, y, title, ha="left", va="top", color=INK, fontsize=13, fontweight="medium")
    ax.text(0.0, y - 0.28, subtitle, ha="left", va="top", color=MUTED, fontsize=9.5)


def _blank_ax(xlim: tuple[float, float], ylim: tuple[float, float]) -> tuple[plt.Figure, plt.Axes]:
    apply_publication_rc()
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def draw_r1c1(ax: plt.Axes, origin: tuple[float, float] = (0.0, 0.0)) -> None:
    """Un nœud air, une R vers l'extérieur, C, apports S et P, capteur y."""
    ox, oy = origin
    x_ext, x_air = ox + 1.1, ox + 3.6
    y = oy + 1.35
    y_gnd = oy + 0.05
    _ext_box(ax, x_ext, y, r"$T_{\mathrm{ext}}$")
    _zigzag(ax, x_ext + 0.55, y, x_air - NODE_R, y)
    ax.text((x_ext + x_air) / 2, y + 0.28, r"$R$", ha="center", color=INK, fontsize=11)
    _node(ax, x_air, y, r"$T_{\mathrm{air}}$")
    _capacitor_down(ax, x_air, y - NODE_R, y_gnd)
    ax.text(x_air + 0.48, (y + y_gnd) / 2 - 0.05, r"$C$", color=INK, fontsize=11)
    _arrow(ax, x_air, y + 1.15, x_air, y + NODE_R + 0.04, ACCENT, r"$S,\;P$", (0.55, 0.0))
    _arrow(ax, x_air + NODE_R + 0.02, y, x_air + 1.35, y, ACCENT_2, r"$y=T_{\mathrm{air}}+v$", (0.15, 0.16))


def draw_r2c2(
    ax: plt.Axes,
    origin: tuple[float, float] = (0.0, 0.0),
    solar_on_mass: bool = False,
) -> None:
    """Deux nœuds. Solaire sur l'air ; option plant : aussi sur la masse."""
    ox, oy = origin
    x_ext, x_air, x_m = ox + 0.95, ox + 3.25, ox + 5.55
    y = oy + 1.35
    y_gnd = oy + 0.05
    _ext_box(ax, x_ext, y, r"$T_{\mathrm{ext}}$")
    _zigzag(ax, x_ext + 0.55, y, x_air - NODE_R, y)
    ax.text((x_ext + 0.55 + x_air) / 2, y + 0.28, r"$R_{ae}$", ha="center", color=INK, fontsize=11)
    _node(ax, x_air, y, r"$T_{\mathrm{air}}$")
    _zigzag(ax, x_air + NODE_R, y, x_m - NODE_R, y)
    ax.text((x_air + x_m) / 2, y + 0.28, r"$R_{am}$", ha="center", color=INK, fontsize=11)
    _node(ax, x_m, y, r"$T_{\mathrm{masse}}$", fill="#EFE6DC")
    _capacitor_down(ax, x_air, y - NODE_R, y_gnd)
    ax.text(x_air + 0.48, (y + y_gnd) / 2 - 0.05, r"$C_a$", color=INK, fontsize=11)
    _capacitor_down(ax, x_m, y - NODE_R, y_gnd)
    ax.text(x_m + 0.50, (y + y_gnd) / 2 - 0.05, r"$C_m$", color=INK, fontsize=11)
    _arrow(ax, x_air, y + 1.15, x_air, y + NODE_R + 0.04, ACCENT, r"$S,\;P$", (0.50, 0.0))
    if solar_on_mass:
        _arrow(
            ax,
            x_m,
            y + 1.15,
            x_m,
            y + NODE_R + 0.04,
            ACCENT_2,
            r"$\alpha_{s,\mathrm{mass}}\,S$",
            (0.15, 0.0),
        )
    _arrow(
        ax,
        x_air + NODE_R * 0.5,
        y - NODE_R * 0.85,
        x_air + 1.55,
        y - 0.55,
        ACCENT_2,
        r"$y=T_{\mathrm{air}}+v$",
        (0.35, -0.22),
    )


def draw_kalman_blocks(ax: plt.Axes) -> None:
    """Boucle prédiction → innovation → mise à jour (pas de matrice A)."""
    def bloc(x: float, y: float, w: float, h: float, title: str, body: str) -> None:
        box = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.05,rounding_size=0.08",
            facecolor="#FBF7F0",
            edgecolor=INK,
            lw=1.4,
        )
        ax.add_patch(box)
        ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="top", color=INK, fontsize=11)
        ax.text(x + w / 2, y + 0.18, body, ha="center", va="bottom", color=MUTED, fontsize=9)

    bloc(0.2, 0.85, 1.9, 1.15, "Prédiction", r"$\hat x^-_{k},\;P^-_{k}$")
    bloc(2.5, 0.85, 2.1, 1.15, "Innovation", r"$e_k = y_k - C\hat x^-_{k}$")
    bloc(5.0, 0.85, 2.0, 1.15, "Mise à jour", r"$\hat x^+_{k},\;P^+_{k}$")
    _arrow(ax, 2.1, 1.42, 2.5, 1.42, INK)
    _arrow(ax, 4.6, 1.42, 5.0, 1.42, INK)
    ax.text(0.2, 2.35, r"$u_{k-1}$", color=ACCENT, fontsize=10)
    _arrow(ax, 0.55, 2.25, 1.0, 2.00, ACCENT)
    ax.text(3.1, 2.35, r"$y_k$  (capteur)", color=ACCENT_2, fontsize=10)
    _arrow(ax, 3.55, 2.22, 3.55, 2.00, ACCENT_2)
    ax.text(6.0, 0.45, r"état pour le MPC", color=MUTED, fontsize=9.5)
    _arrow(ax, 6.0, 0.85, 6.0, 0.58, INK)


def figure_r1c1() -> plt.Figure:
    """Schéma R1C1 seul."""
    fig, ax = _blank_ax((-0.2, 6.5), (-0.25, 2.7))
    _caption(ax, "R1C1 — baseline", "Un état, une échelle de temps. Le capteur n'est pas l'état.")
    draw_r1c1(ax, origin=(0.15, 0.15))
    return fig


def figure_r2c2() -> plt.Figure:
    """Schéma R2C2 d'identification (solaire sur l'air)."""
    fig, ax = _blank_ax((-0.2, 7.4), (-0.35, 2.7))
    _caption(ax, "R2C2 identifié", "Masse cachée. Solaire et chauffage sur l'air seulement.")
    draw_r2c2(ax, origin=(0.0, 0.05), solar_on_mass=False)
    return fig


def figure_plant() -> plt.Figure:
    """Plant littérature : une flèche solaire de plus sur la masse."""
    fig, ax = _blank_ax((-0.2, 7.4), (-0.35, 2.7))
    _caption(
        ax,
        "Plant (littérature)",
        "Même structure, mais le soleil chauffe aussi la masse — pas le modèle du MPC.",
    )
    draw_r2c2(ax, origin=(0.0, 0.05), solar_on_mass=True)
    return fig


def figure_kalman() -> plt.Figure:
    """Schéma du filtre, sans matrices."""
    fig, ax = _blank_ax((-0.15, 7.4), (0.15, 2.7))
    _caption(ax, "Filtre de Kalman", "Prédiction du RC, correction par y, état pour le contrôle.")
    draw_kalman_blocks(ax)
    return fig


def figure_famille() -> plt.Figure:
    """Les trois circuits l'un sous l'autre (planche unique)."""
    apply_publication_rc()
    fig, axes = plt.subplots(3, 1, figsize=(8.4, 10.2))
    fig.patch.set_facecolor(BG)
    titres = (
        ("R1C1 — baseline", "Une capacité. Suffit si une seule constante de temps."),
        ("R2C2 identifié", "Air + masse. Solaire sur l'air (brief)."),
        ("Plant", "Solaire aussi sur la masse : le MPC ne se valide pas sur lui-même."),
    )
    drawers = (
        lambda a: draw_r1c1(a, origin=(0.3, 0.15)),
        lambda a: draw_r2c2(a, origin=(0.05, 0.1), solar_on_mass=False),
        lambda a: draw_r2c2(a, origin=(0.05, 0.1), solar_on_mass=True),
    )
    for ax, (title, sub), draw in zip(axes, titres, drawers, strict=True):
        ax.set_facecolor(BG)
        ax.set_xlim(-0.2, 7.4)
        ax.set_ylim(-0.3, 2.55)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.text(0.0, 2.42, title, color=INK, fontsize=13, fontweight="medium")
        ax.text(0.0, 2.18, sub, color=MUTED, fontsize=9.5)
        draw(ax)
    fig.tight_layout(h_pad=0.4)
    return fig


def run_draw_schemas(pictures_dir: Path | None = None) -> dict[str, str]:
    """Écrit les schémas dans ``pictures/experiments/``.

    Parameters
    ----------
    pictures_dir : Path, optional
        Dossier images. Défaut : ``pictures/experiments``.

    Returns
    -------
    dict
        Stem → chemin PNG relatif.
    """
    out = pictures_dir or (REPO_ROOT / "pictures" / "experiments")
    out.mkdir(parents=True, exist_ok=True)
    jobs = {
        "schema-r1c1": figure_r1c1,
        "schema-r2c2": figure_r2c2,
        "schema-plant": figure_plant,
        "schema-kalman": figure_kalman,
        "schema-famille-rc": figure_famille,
    }
    written: dict[str, str] = {}
    for stem, factory in jobs.items():
        paths = save_figure(factory(), out / f"{stem}.png")
        written[stem] = paths[0].as_posix()
    return written
