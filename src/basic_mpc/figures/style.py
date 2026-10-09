"""Palette et export communs aux figures d'expérience et aux schémas."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Thème portfolio : crème, encre, accent (pas de cadre blanc)
BG = "#F4EFE6"
INK = "#2C2416"
ACCENT = "#3D6B6B"
ACCENT_2 = "#8C4A32"
MUTED = "#8A7E6E"

logging.getLogger("fontTools").setLevel(logging.WARNING)
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)


def apply_publication_rc() -> None:
    """Typo et PDF avec polices éditables (Type 42)."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def set_local_clock(ax: plt.Axes, hours: np.ndarray, start_hour: float = 18.0) -> None:
    """Gradue « heures depuis le début » en heure locale.

    Marques aux changements du scénario : 18 h, 22 h, 7 h, 12 h.
    Le scénario de contrôle part à 18 h (``ControlConfig.start_local``).

    Parameters
    ----------
    ax : Axes
        Axe dont l'abscisse est en heures depuis le départ.
    hours : ndarray
        Abscisse déjà tracée.
    start_hour : float
        Heure locale du premier point.
    """
    if len(hours) == 0:
        return
    end = float(hours[-1])
    marks = (7, 12, 18, 22)
    ticks: list[float] = []
    labels: list[str] = []
    t = 0.0
    while t <= end + 1e-6:
        clock = (start_hour + t) % 24.0
        if min(abs(clock - mark) for mark in marks) < 0.05:
            ticks.append(t)
            labels.append(f"{int(round(clock)) % 24} h")
        t += 1.0
    if not ticks:
        return
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_xlabel("heure locale")


def save_figure(fig: plt.Figure, path: Path, dpi: int = 200) -> list[Path]:
    """PNG (catalogue) + PDF (publication).

    Parameters
    ----------
    fig : Figure
        Figure matplotlib déjà dessinée.
    path : Path
        Chemin ``.png`` (le PDF est le même stem).
    dpi : int
        Résolution du raster.

    Returns
    -------
    list of Path
        Fichiers écrits.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    png = path.with_suffix(".png")
    pdf = path.with_suffix(".pdf")
    fig.savefig(png, dpi=dpi, facecolor=BG, bbox_inches="tight", pad_inches=0.18)
    fig.savefig(pdf, facecolor=BG, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    return [png, pdf]
