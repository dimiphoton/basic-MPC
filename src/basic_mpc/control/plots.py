"""Figures S4–S6 : MPC vs bang-bang."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from basic_mpc.figures.style import ACCENT, ACCENT_2, BG, INK, MUTED
from basic_mpc.figures.style import apply_publication_rc, save_figure


def _axes_ink(ax: plt.Axes) -> None:
    ax.set_facecolor(BG)
    ax.tick_params(colors=INK)
    for spine in ax.spines.values():
        spine.set_color(INK)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def plot_s4_tair(
    hours: np.ndarray,
    ta_mpc: np.ndarray,
    ta_bb: np.ndarray,
    t_min: float,
    t_max: float,
    path: Path,
) -> None:
    """S4 : air plant + bande de confort."""
    apply_publication_rc()
    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    fig.patch.set_facecolor(BG)
    ax.axhspan(t_min, t_max, color=ACCENT, alpha=0.12, zorder=0)
    ax.plot(hours, ta_bb, color=MUTED, lw=1.4, label="bang-bang")
    ax.plot(hours, ta_mpc, color=ACCENT, lw=1.6, label="MPC")
    ax.axhline(t_min, color=MUTED, lw=0.6, ls="--")
    ax.axhline(t_max, color=MUTED, lw=0.6, ls="--")
    ax.set_xlabel("heures")
    ax.set_ylabel(r"$T_{\mathrm{air}}$ (°C)")
    ax.set_title("Après le départ froid, le MPC tient 20 °C ; le thermostat oscille")
    ax.legend(frameon=False)
    _axes_ink(ax)
    save_figure(fig, path)


def plot_s5_commande(
    hours: np.ndarray,
    p_mpc: np.ndarray,
    p_bb: np.ndarray,
    path: Path,
) -> None:
    """S5 : P dosé vs tout-ou-rien."""
    apply_publication_rc()
    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    fig.patch.set_facecolor(BG)
    ax.plot(hours, p_bb, color=MUTED, lw=1.2, label="bang-bang")
    ax.plot(hours, p_mpc, color=ACCENT_2, lw=1.5, label="MPC")
    ax.set_xlabel("heures")
    ax.set_ylabel("P (proxy, pas des W)")
    ax.set_title("Tout-ou-rien contre une commande dosée")
    ax.legend(frameon=False)
    _axes_ink(ax)
    save_figure(fig, path)


def plot_s6_score(
    p_hours_mpc: float,
    p_hours_bb: float,
    hours_out_mpc: float,
    hours_out_bb: float,
    path: Path,
) -> None:
    """S6 : cumul de P et heures hors bande."""
    apply_publication_rc()
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6))
    fig.patch.set_facecolor(BG)
    labels = ["MPC", "bang-bang"]
    colors = [ACCENT, MUTED]
    axes[0].bar(labels, [p_hours_mpc, p_hours_bb], color=colors, width=0.55)
    axes[0].set_ylabel("P × heures (proxy)")
    axes[0].set_title("Consommation")
    axes[1].bar(labels, [hours_out_mpc, hours_out_bb], color=colors, width=0.55)
    axes[1].set_ylabel("heures hors bande")
    axes[1].set_title("Confort")
    for ax, vals in zip(
        axes,
        ([p_hours_mpc, p_hours_bb], [hours_out_mpc, hours_out_bb]),
        strict=True,
    ):
        for i, val in enumerate(vals):
            label = f" {val:.0f}" if val > 20 else f" {val:.1f}"
            ax.text(i, val, label, ha="center", va="bottom", color=INK, fontsize=10)
        ymax = max(vals) if max(vals) > 0 else 1.0
        ax.set_ylim(0.0, ymax * 1.18)
        _axes_ink(ax)
    fig.suptitle("Même météo, deux régulateurs", color=INK, fontsize=12)
    save_figure(fig, path)
