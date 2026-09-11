"""Trois panneaux datetime du labo Streamlit."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.pyplot as plt

from basic_mpc.figures.style import ACCENT, ACCENT_2, BG, INK, MUTED, apply_publication_rc


def _hc_spans(ax: Axes, index: pd.DatetimeIndex, pi: np.ndarray, pi_hc: float) -> None:
    """Bandes grises = heures creuses."""
    hc = np.asarray(pi) <= pi_hc + 1e-12
    in_band = False
    start = index[0]
    labeled = False
    for i, flag in enumerate(hc):
        if flag and not in_band:
            in_band = True
            start = index[i]
        elif not flag and in_band:
            ax.axvspan(
                start,
                index[i],
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
            index[-1],
            color=MUTED,
            alpha=0.12,
            zorder=0,
            label="heures creuses" if not labeled else None,
        )


def fig_lab_panels(
    index: pd.DatetimeIndex,
    traj: pd.DataFrame,
    t_conf: np.ndarray,
    pi: np.ndarray,
    pi_hc: float,
    bill_cum: np.ndarray,
    overlay: dict | None = None,
) -> Figure:
    """Physique, commande, facture cumulée — même axe temps.

    Parameters
    ----------
    index : DatetimeIndex
        Horloge scorée.
    traj : DataFrame
        Colonnes ``ta_true``, ``tm_true``, ``t_ext``, ``t_sp``, ``P``.
    t_conf, pi : ndarray
        Confort et tarif.
    pi_hc : float
        Prix heures creuses (€/kWh).
    bill_cum : ndarray
        Facture cumulée (€).
    overlay : dict, optional
        Run hystérésis (clés ``traj``, ``metrics``).

    Returns
    -------
    Figure
    """
    apply_publication_rc()
    fig, axes = plt.subplots(3, 1, figsize=(9.2, 8.4), sharex=True)
    fig.patch.set_facecolor(BG)
    ov = overlay["traj"] if overlay is not None else None

    ax = axes[0]
    _hc_spans(ax, index, pi, pi_hc)
    ax.plot(index, traj["t_ext"], color=INK, lw=1.1, label=r"$T_{\mathrm{ext}}$")
    ax.plot(index, t_conf, color=INK, lw=1.0, ls="--", label=r"$T_{\mathrm{conf}}$")
    if ov is not None:
        ax.plot(
            index,
            ov["ta_true"],
            color=MUTED,
            lw=1.2,
            label="air (hystérésis)",
        )
    ax.plot(index, traj["ta_true"], color=ACCENT, lw=1.7, label=r"$T_{\mathrm{air}}$")
    ax.plot(
        index,
        traj["tm_true"],
        color=ACCENT_2,
        lw=1.3,
        ls=":",
        label=r"$T_{\mathrm{masse}}$",
    )
    ax.set_ylabel("°C")
    ax.set_title("La masse retarde ; le capteur ne voit que l'air")
    ax.legend(frameon=False, ncol=3, fontsize=9)
    ax.set_facecolor(BG)

    ax = axes[1]
    _hc_spans(ax, index, pi, pi_hc)
    if ov is not None:
        ax.plot(index, ov["t_sp"], color=MUTED, lw=1.1, label=r"$T_{\mathrm{sp}}$ hyst.")
        ax.plot(index, ov["P"], color=MUTED, lw=0.9, alpha=0.7)
    ax.plot(index, traj["t_sp"], color=ACCENT, lw=1.5, label=r"$T_{\mathrm{sp}}$")
    ax.set_ylabel(r"$T_{\mathrm{sp}}$ (°C)")
    ax_p = ax.twinx()
    ax_p.plot(index, traj["P"], color=ACCENT_2, lw=1.2, label="P")
    ax_p.set_ylabel("P (proxy)")
    ax.set_title("La décision est une consigne ; P suit la bande n")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.set_facecolor(BG)
    ax_p.set_facecolor(BG)

    ax = axes[2]
    _hc_spans(ax, index, pi, pi_hc)
    if overlay is not None:
        ax.plot(
            index,
            overlay["metrics"]["bill_cum"],
            color=MUTED,
            lw=1.2,
            label="hystérésis",
        )
    ax.plot(index, bill_cum, color=ACCENT, lw=1.6, label="stratégie")
    ax.set_ylabel("€ cumulés")
    ax.set_title("Facture HP/HC au fil des jours")
    ax.legend(frameon=False, fontsize=9)
    ax.set_facecolor(BG)
    ax.set_xlabel("heure locale (Bruxelles)")

    for spine_ax in list(axes) + [ax_p]:
        spine_ax.tick_params(colors=INK)
        for spine in spine_ax.spines.values():
            spine.set_color(INK)
        spine_ax.xaxis.label.set_color(INK)
        spine_ax.yaxis.label.set_color(INK)
    fig.tight_layout()
    return fig
