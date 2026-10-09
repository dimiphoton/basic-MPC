"""Figures S4–S6 : MPC vs bang-bang."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from basic_mpc.figures.style import ACCENT, ACCENT_2, BG, INK, MUTED
from basic_mpc.figures.style import apply_publication_rc, save_figure, set_local_clock


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
    ax.plot(hours, ta_bb, color=MUTED, lw=1.4, label="thermostat")
    ax.plot(hours, ta_mpc, color=ACCENT, lw=1.6, label="MPC")
    ax.axhline(t_min, color=MUTED, lw=0.6, ls="--")
    ax.axhline(t_max, color=MUTED, lw=0.6, ls="--")
    ax.set_xlabel("heures")
    ax.set_ylabel("air (°C)")
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
    ax.plot(hours, p_bb, color=MUTED, lw=1.2, label="thermostat")
    ax.plot(hours, p_mpc, color=ACCENT_2, lw=1.5, label="MPC")
    ax.set_xlabel("heures")
    ax.set_ylabel("chauffage (proxy)")
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
    labels = ["MPC", "thermostat"]
    colors = [ACCENT, MUTED]
    axes[0].bar(labels, [p_hours_mpc, p_hours_bb], color=colors, width=0.55)
    axes[0].set_ylabel("chauffage × heures (proxy)")
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


def plot_s4_confort(
    hours: np.ndarray,
    ta_mpc: np.ndarray,
    ta_bb: np.ndarray,
    t_conf: np.ndarray,
    pi: np.ndarray,
    pi_hc: float,
    path: Path,
) -> None:
    """S4 : air + programmation T_conf, bandes grises = heures creuses."""
    apply_publication_rc()
    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    fig.patch.set_facecolor(BG)
    hc = np.asarray(pi) <= pi_hc + 1e-12
    # Plages HP/HC pour lire le décalage de charge d'un coup d'œil.
    in_band = False
    start = 0.0
    labeled = False
    for i, flag in enumerate(hc):
        if flag and not in_band:
            in_band = True
            start = float(hours[i])
        elif not flag and in_band:
            ax.axvspan(
                start,
                float(hours[i]),
                color=MUTED,
                alpha=0.12,
                zorder=0,
                label="heures creuses" if not labeled else None,
            )
            labeled = True
            in_band = False
    if in_band:
        ax.axvspan(
            start,
            float(hours[-1]),
            color=MUTED,
            alpha=0.12,
            zorder=0,
            label="heures creuses" if not labeled else None,
        )
    ax.plot(hours, t_conf, color=INK, lw=1.1, ls="--", label="consigne de confort")
    ax.plot(hours, ta_bb, color=MUTED, lw=1.1, label="thermostat")
    ax.plot(hours, ta_mpc, color=ACCENT, lw=1.8, label="MPC")
    set_local_clock(ax, hours, start_hour=18.0)
    ax.set_ylabel("air (°C)")
    ax.set_title("48 h, maison simulée : le MPC monte avant 7 h")
    ax.legend(frameon=False, ncol=2, loc="upper right")
    _axes_ink(ax)
    save_figure(fig, path)


def plot_s5_consigne(
    hours: np.ndarray,
    tsp_mpc: np.ndarray,
    tsp_bb: np.ndarray,
    p_mpc: np.ndarray,
    p_bb: np.ndarray,
    path: Path,
) -> None:
    """S5 : consigne thermostat (décision) et P produit par la bande n."""
    apply_publication_rc()
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 5.2), sharex=True)
    fig.patch.set_facecolor(BG)
    axes[0].plot(hours, tsp_bb, color=MUTED, lw=1.3, label="thermostat (= confort)")
    axes[0].plot(hours, tsp_mpc, color=ACCENT, lw=1.5, label="MPC")
    axes[0].set_ylabel("consigne thermostat (°C)")
    axes[0].set_title("La décision est une consigne, pas des watts")
    axes[0].legend(frameon=False)
    # Moyenne sur le pas de décision (30 min) : le tout-ou-rien
    # à 5 min n'est pas lisible.
    axes[1].plot(hours, _mean_30min(p_bb), color=MUTED, lw=1.6, label="thermostat")
    axes[1].plot(hours, _mean_30min(p_mpc), color=ACCENT_2, lw=1.6, label="MPC")
    set_local_clock(axes[1], hours, start_hour=18.0)
    axes[1].set_ylabel("chauffage proxy")
    axes[1].set_title("Moyenne sur 30 min : le thermostat s'allume par à-coups")
    axes[1].legend(frameon=False)
    for ax in axes:
        _axes_ink(ax)
    save_figure(fig, path)


def _mean_30min(values: np.ndarray, samples: int = 6) -> np.ndarray:
    """Moyenne glissante sur 30 min (6 pas de 5 min)."""
    v = np.asarray(values, dtype=float)
    if samples <= 1 or len(v) < samples:
        return v
    kernel = np.ones(samples) / samples
    return np.convolve(v, kernel, mode="same")


def plot_s6_euros(
    bill_mpc: float,
    bill_bb: float,
    hours_under_mpc: float,
    hours_under_bb: float,
    path: Path,
) -> None:
    """S6 : facture proxy € et heures sous T_conf."""
    apply_publication_rc()
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6))
    fig.patch.set_facecolor(BG)
    labels = ["MPC", "thermostat"]
    colors = [ACCENT, MUTED]
    ecart_h = hours_under_bb - hours_under_mpc
    axes[0].bar(labels, [bill_mpc, bill_bb], color=colors, width=0.55)
    axes[0].set_ylabel("euros proxy")
    axes[0].set_title("Facture (heures pleines / creuses)")
    axes[1].bar(labels, [hours_under_mpc, hours_under_bb], color=colors, width=0.55)
    axes[1].set_ylabel("heures")
    axes[1].set_title("Sous le confort de plus de 0,1 °C")
    for ax, vals in zip(
        axes,
        ([bill_mpc, bill_bb], [hours_under_mpc, hours_under_bb]),
        strict=True,
    ):
        for i, val in enumerate(vals):
            label = f" {val:.2f}" if val < 20 else f" {val:.0f}"
            ax.text(i, val, label, ha="center", va="bottom", color=INK, fontsize=10)
        ymax = max(vals) if max(vals) > 0 else 1.0
        ax.set_ylim(0.0, ymax * 1.28)
        _axes_ink(ax)
    fig.suptitle("48 h, maison simulée", color=INK, fontsize=13)
    fig.text(
        0.5,
        0.01,
        f"Écart : {ecart_h:.0f} h de froid en moins. "
        "Le thermostat oscille de ±0,5 °C autour de la consigne.",
        ha="center",
        va="bottom",
        color=INK,
        fontsize=9,
    )
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.90))
    save_figure(fig, path)
