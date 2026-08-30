"""Palette et export communs aux figures d'expérience et aux schémas."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
