"""Schémas RC et Kalman pour publication (matplotlib, pas de visio).

Même langage qu'un cours : nœud = température, zigzag = résistance,
deux traits = capacité, flèche = apport, pointillé = mesure.
Les fils ne se croisent pas.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

from basic_mpc.config import REPO_ROOT
from basic_mpc.figures.style import ACCENT, ACCENT_2, BG, INK, MUTED
from basic_mpc.figures.style import apply_publication_rc, save_figure

NODE_R = 0.36


def _zigzag(ax: plt.Axes, x0: float, y0: float, x1: float, y1: float, n_teeth: int = 7) -> None:
    """Résistance : dents égales, pattes droites aux extrémités."""
    vec = np.array([x1 - x0, y1 - y0], dtype=float)
    length = float(np.linalg.norm(vec))
    if length < 1e-9:
        return
    direction = vec / length
    normal = np.array([-direction[1], direction[0]])
    lead = min(0.22, 0.14 * length)
    p0 = np.array([x0, y0]) + direction * lead
    p1 = np.array([x1, y1]) - direction * lead
    ax.plot([x0, p0[0]], [y0, p0[1]], color=INK, lw=1.35, solid_capstyle="butt", zorder=2)
    ax.plot([p1[0], x1], [p1[1], y1], color=INK, lw=1.35, solid_capstyle="butt", zorder=2)
    span = p1 - p0
    amp = 0.12
    xs = [float(p0[0])]
    ys = [float(p0[1])]
    for i in range(n_teeth):
        t = (i + 0.5) / n_teeth
        sign = 1.0 if i % 2 == 0 else -1.0
        pt = p0 + t * span + sign * amp * normal
        xs.append(float(pt[0]))
        ys.append(float(pt[1]))
    xs.append(float(p1[0]))
    ys.append(float(p1[1]))
    ax.plot(xs, ys, color=INK, lw=1.35, solid_capstyle="butt", solid_joinstyle="miter", zorder=2)


def _ground(ax: plt.Axes, x: float, y: float) -> None:
    """Symbole de référence, trois traits."""
    ax.plot([x - 0.22, x + 0.22], [y, y], color=INK, lw=1.5, zorder=2)
    ax.plot([x - 0.14, x + 0.14], [y - 0.08, y - 0.08], color=INK, lw=1.35, zorder=2)
    ax.plot([x - 0.07, x + 0.07], [y - 0.16, y - 0.16], color=INK, lw=1.2, zorder=2)


def _capacitor(ax: plt.Axes, x: float, y_node: float, y_gnd: float) -> None:
    """Capacité sous le nœud, jusqu'à la référence."""
    y_top = y_node - NODE_R
    gap = 0.07
    half = 0.22
    y_plate = y_gnd + 0.72
    ax.plot([x, x], [y_top, y_plate + gap], color=INK, lw=1.35, zorder=2)
    ax.plot([x - half, x + half], [y_plate + gap, y_plate + gap], color=INK, lw=2.2, zorder=2)
    ax.plot([x - half, x + half], [y_plate - gap, y_plate - gap], color=INK, lw=2.2, zorder=2)
    ax.plot([x, x], [y_plate - gap, y_gnd + 0.16], color=INK, lw=1.35, zorder=2)
    _ground(ax, x, y_gnd)


def _node(ax: plt.Axes, x: float, y: float, label: str, fill: str = "#FBF7F0") -> None:
    """Température : une bulle, le nom dedans."""
    circle = Circle((x, y), NODE_R, facecolor=fill, edgecolor=INK, lw=1.5, zorder=3)
    ax.add_patch(circle)
    ax.text(x, y, label, ha="center", va="center", color=INK, fontsize=11, zorder=4)


def _box(ax: plt.Axes, x: float, y: float, label: str, w: float = 1.35, h: float = 0.62) -> None:
    """Condition aux limites ou bloc."""
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        facecolor="#E7E0D4",
        edgecolor=INK,
        lw=1.3,
        zorder=3,
    )
    ax.add_patch(patch)
    ax.text(x, y, label, ha="center", va="center", color=INK, fontsize=11, zorder=4)


def _arrow(ax: plt.Axes, x0: float, y0: float, x1: float, y1: float, color: str) -> None:
    """Flèche d'apport."""
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=11,
            lw=1.4,
            color=color,
            zorder=2,
            shrinkA=0,
            shrinkB=0,
        )
    )


def _label(ax: plt.Axes, x: float, y: float, text: str, color: str = INK) -> None:
    ax.text(x, y, text, ha="center", va="center", color=color, fontsize=11, zorder=4)


def _caption(ax: plt.Axes, title: str, subtitle: str) -> None:
    ax.text(0.15, 5.15, title, ha="left", va="top", color=INK, fontsize=13)
    ax.text(0.15, 4.78, subtitle, ha="left", va="top", color=MUTED, fontsize=10)


def _panel() -> tuple[plt.Figure, plt.Axes]:
    apply_publication_rc()
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0.0, 10.0)
    ax.set_ylim(0.0, 5.4)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def _sensor(ax: plt.Axes, x_node: float, y_gnd: float) -> None:
    """Mesure en pointillé, sur le fil de la capacité, vers la droite."""
    y_tap = y_gnd + 1.15
    x_s = x_node + 1.05
    ax.plot(
        [x_node, x_s - 0.2],
        [y_tap, y_tap],
        color=ACCENT_2,
        lw=1.15,
        ls=(0, (3, 2)),
        zorder=2,
    )
    circle = Circle((x_s, y_tap), 0.2, facecolor=BG, edgecolor=ACCENT_2, lw=1.3, zorder=3)
    ax.add_patch(circle)
    ax.text(x_s, y_tap, "y", ha="center", va="center", color=ACCENT_2, fontsize=10, zorder=4)
    ax.text(x_s + 0.32, y_tap, "mesure", ha="left", va="center", color=ACCENT_2, fontsize=9)


def _heat_arrow(ax: plt.Axes, x: float, y_node: float, name: str, color: str) -> None:
    """Apport vertical. Le nom est à droite de la flèche, sous le titre."""
    y_tip = y_node + NODE_R + 0.02
    y_tail = y_tip + 0.62
    _arrow(ax, x, y_tail, x, y_tip, color)
    ax.text(x + 0.16, (y_tip + y_tail) / 2, name, ha="left", va="center", color=color, fontsize=10)


def draw_r1c1(ax: plt.Axes, origin: tuple[float, float] = (0.0, 0.0)) -> None:
    """Un nœud air, R vers l'extérieur, C, apports, capteur."""
    ox, oy = origin
    y = oy + 2.15
    y_gnd = oy + 0.45
    x_ext, x_air = ox + 1.7, ox + 4.6
    _box(ax, x_ext, y, "dehors")
    _zigzag(ax, x_ext + 0.72, y, x_air - NODE_R, y)
    _label(ax, (x_ext + x_air) / 2, y + 0.38, r"$R$")
    _node(ax, x_air, y, "air")
    _capacitor(ax, x_air, y, y_gnd)
    _label(ax, x_air - 0.48, y_gnd + 0.85, r"$C$")
    _heat_arrow(ax, x_air, y, "soleil, chauffage", ACCENT)
    _sensor(ax, x_air, y_gnd)


def draw_r2c2(
    ax: plt.Axes,
    origin: tuple[float, float] = (0.0, 0.0),
    solar_on_mass: bool = False,
) -> None:
    """Deux nœuds. Solaire sur l'air ; le plant ajoute le soleil sur les murs."""
    ox, oy = origin
    y = oy + 2.15
    y_gnd = oy + 0.45
    x_ext, x_air, x_mass = ox + 1.35, ox + 4.15, ox + 7.15
    _box(ax, x_ext, y, "dehors")
    _zigzag(ax, x_ext + 0.72, y, x_air - NODE_R, y)
    _label(ax, (x_ext + 0.72 + x_air - NODE_R) / 2, y + 0.38, r"$R_{ae}$")
    _node(ax, x_air, y, "air")
    _zigzag(ax, x_air + NODE_R, y, x_mass - NODE_R, y)
    _label(ax, (x_air + x_mass) / 2, y + 0.38, r"$R_{am}$")
    _node(ax, x_mass, y, "murs", fill="#EFE6DC")
    _capacitor(ax, x_air, y, y_gnd)
    _label(ax, x_air - 0.48, y_gnd + 0.85, r"$C_a$")
    _capacitor(ax, x_mass, y, y_gnd)
    _label(ax, x_mass - 0.48, y_gnd + 0.85, r"$C_m$")
    _heat_arrow(ax, x_air, y, "soleil, chauffage", ACCENT)
    if solar_on_mass:
        _heat_arrow(ax, x_mass, y, "soleil", ACCENT_2)
    _sensor(ax, x_air, y_gnd)


def draw_kalman_blocks(ax: plt.Axes) -> None:
    """Prédiction, innovation, mise à jour. Pas de matrice."""
    def bloc(x: float, title: str, body: str) -> None:
        patch = FancyBboxPatch(
            (x, 1.55),
            2.15,
            1.25,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor="#FBF7F0",
            edgecolor=INK,
            lw=1.3,
            zorder=3,
        )
        ax.add_patch(patch)
        ax.text(x + 1.075, 2.45, title, ha="center", va="center", color=INK, fontsize=12)
        ax.text(x + 1.075, 1.95, body, ha="center", va="center", color=MUTED, fontsize=10)

    bloc(0.45, "Prédire", "on avance l'état")
    bloc(3.55, "Comparer", "mesure − prévision")
    bloc(6.65, "Corriger", "on met l'état à jour")
    _arrow(ax, 2.6, 2.18, 3.55, 2.18, INK)
    _arrow(ax, 5.7, 2.18, 6.65, 2.18, INK)
    _label(ax, 1.5, 3.55, "météo, chauffage", ACCENT)
    _arrow(ax, 1.5, 3.35, 1.5, 2.82, ACCENT)
    _label(ax, 4.6, 3.55, "capteur  y", ACCENT_2)
    _arrow(ax, 4.6, 3.35, 4.6, 2.82, ACCENT_2)
    _label(ax, 7.7, 1.15, "état pour le contrôleur", MUTED)
    _arrow(ax, 7.7, 1.55, 7.7, 1.32, INK)


def figure_r1c1() -> plt.Figure:
    """Schéma R1C1 seul."""
    fig, ax = _panel()
    _caption(ax, "Un état", "Une résistance vers l'extérieur, une capacité. Le capteur n'est pas l'état.")
    draw_r1c1(ax)
    return fig


def figure_r2c2() -> plt.Figure:
    """Schéma R2C2 d'identification (solaire sur l'air)."""
    fig, ax = _panel()
    _caption(ax, "Deux états", "L'air et les murs. Le soleil et le chauffage n'entrent que dans l'air.")
    draw_r2c2(ax, solar_on_mass=False)
    return fig


def figure_plant() -> plt.Figure:
    """Plant littérature : une flèche solaire de plus sur la masse."""
    fig, ax = _panel()
    _caption(ax, "Maison simulée", "Même circuit, plus une entrée : le soleil chauffe aussi les murs.")
    draw_r2c2(ax, solar_on_mass=True)
    return fig


def figure_kalman() -> plt.Figure:
    """Schéma du filtre, sans matrices."""
    fig, ax = _panel()
    _caption(ax, "Filtre de Kalman", "Le modèle avance. Le capteur corrige. On obtient l'état.")
    draw_kalman_blocks(ax)
    return fig


def figure_famille() -> plt.Figure:
    """Les trois circuits, l'un sous l'autre."""
    apply_publication_rc()
    fig, axes = plt.subplots(3, 1, figsize=(8.6, 12.6))
    fig.patch.set_facecolor(BG)
    panels = (
        ("Un état", "Une capacité. Une seule constante de temps.", False, False),
        ("Deux états", "Air et murs. Soleil et chauffage sur l'air.", True, False),
        ("Maison simulée", "Le soleil entre aussi dans les murs.", True, True),
    )
    for ax, (title, subtitle, two, solar_mass) in zip(axes, panels, strict=True):
        ax.set_facecolor(BG)
        ax.set_xlim(0.0, 10.0)
        ax.set_ylim(0.0, 5.4)
        ax.set_aspect("equal")
        ax.axis("off")
        _caption(ax, title, subtitle)
        if two:
            draw_r2c2(ax, solar_on_mass=solar_mass)
        else:
            draw_r1c1(ax)
    fig.subplots_adjust(hspace=0.05, left=0.02, right=0.98, top=0.98, bottom=0.02)
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
