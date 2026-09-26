"""Figures de devoir : données, dynamique, Kalman, sensibilités MPC."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.stats import norm

from basic_mpc.control.actuator import proportional_band_p
from basic_mpc.figures.style import ACCENT, ACCENT_2, BG, INK, MUTED
from basic_mpc.figures.style import apply_publication_rc, save_figure
from matplotlib import pyplot as plt


def _axes_ink(ax: plt.Axes) -> None:
    ax.set_facecolor(BG)
    ax.tick_params(colors=INK)
    for spine in ax.spines.values():
        spine.set_color(INK)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def plot_loi_bande_n(path: Path) -> None:
    """H1 : P / P_max vs écart consigne − mesure, trois n."""
    apply_publication_rc()
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    fig.patch.set_facecolor(BG)
    gap = np.linspace(-0.5, 2.5, 200)
    t_sp = 20.0
    t_meas = t_sp - gap
    for n_band, color in ((0.5, MUTED), (1.0, ACCENT), (2.0, ACCENT_2)):
        p = [proportional_band_p(t_sp, t, n_band, 1.0) for t in t_meas]
        ax.plot(gap, p, color=color, lw=1.8, label=f"n = {n_band:g} °C")
    ax.axvline(0.0, color=INK, lw=0.6, ls=":")
    ax.set_xlabel(r"$T_{\mathrm{sp}} - y$ (°C)")
    ax.set_ylabel(r"$P / P_{\max}$")
    ax.set_title("La commande n'est pas des watts : une bande de n degrés")
    ax.legend(frameon=False)
    _axes_ink(ax)
    save_figure(fig, path)


def plot_reponse_indicielle(
    hours: np.ndarray,
    ta: np.ndarray,
    tm: np.ndarray,
    path: Path,
) -> None:
    """H2 : échelon de P, air vs masse (plant)."""
    apply_publication_rc()
    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    fig.patch.set_facecolor(BG)
    ax.plot(hours, ta, color=ACCENT, lw=1.6, label=r"$T_{\mathrm{air}}$")
    ax.plot(hours, tm, color=ACCENT_2, lw=1.6, label=r"$T_{\mathrm{masse}}$")
    ax.set_xlabel("heures")
    ax.set_ylabel("°C")
    ax.set_title("Échelon de chauffage : les murs retardent l'air")
    ax.legend(frameon=False)
    _axes_ink(ax)
    save_figure(fig, path)


def plot_bode_module(
    periods: np.ndarray,
    mag_plant: np.ndarray,
    mag_r1: np.ndarray | None,
    mag_r2: np.ndarray | None,
    path: Path,
) -> None:
    """H3 : |Z/Z(0)| vs période."""
    apply_publication_rc()
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    fig.patch.set_facecolor(BG)
    ax.semilogx(periods, mag_plant, color=INK, lw=1.8, label="plant")
    if mag_r1 is not None:
        ax.semilogx(periods, mag_r1, color=MUTED, lw=1.5, label="R1C1 fit")
    if mag_r2 is not None:
        ax.semilogx(periods, mag_r2, color=ACCENT, lw=1.5, label="R2C2 fit")
    ax.axvline(24.0, color=INK, lw=0.7, ls="--")
    ax.set_xlabel("période (h)")
    ax.set_ylabel(r"$|Z/Z(0)|$")
    ax.set_title("Bode module : le plant filtre autrement que le fit salon")
    ax.legend(frameon=False)
    _axes_ink(ax)
    save_figure(fig, path)


def plot_qq_innovations(innov: np.ndarray, path: Path) -> None:
    """H4 : QQ-plot gaussien des innovations Kalman (plant)."""
    apply_publication_rc()
    e = np.asarray(innov, dtype=float)
    e = e[np.isfinite(e)]
    e = np.sort(e)
    n = e.size
    p = (np.arange(n) + 0.5) / n
    theo = norm.ppf(p, loc=float(e.mean()), scale=float(e.std(ddof=1)))
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    fig.patch.set_facecolor(BG)
    ax.plot(theo, e, "o", color=ACCENT, ms=3, alpha=0.7)
    lim = [float(np.min(theo)), float(np.max(theo))]
    ax.plot(lim, lim, color=INK, lw=1.0)
    ax.set_xlabel("quantiles N(μ, σ)")
    ax.set_ylabel("innovations (°C)")
    ax.set_title("Kalman : les innovations sont-elles gaussiennes ?")
    ax.set_aspect("equal", adjustable="box")
    _axes_ink(ax)
    save_figure(fig, path)


def plot_kalman_trou(
    hours: np.ndarray,
    tm_true: np.ndarray,
    tm_hat: np.ndarray,
    t0: float,
    t1: float,
    path: Path,
) -> None:
    """H5 : trou de mesure — le filtre prédit la masse sans y."""
    apply_publication_rc()
    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    fig.patch.set_facecolor(BG)
    ax.axvspan(t0, t1, color=MUTED, alpha=0.18, label="y manquant")
    ax.plot(hours, tm_true, color=INK, lw=1.5, label=r"$T_{\mathrm{masse}}$ vraie")
    ax.plot(hours, tm_hat, color=ACCENT, lw=1.4, label=r"$\hat T_{\mathrm{masse}}$")
    ax.set_xlabel("heures")
    ax.set_ylabel("°C")
    ax.set_title("Sans mesure, le Kalman continue de prédire (pas d'interpolation)")
    ax.legend(frameon=False)
    _axes_ink(ax)
    save_figure(fig, path)


def plot_split_chronologique(
    train_start: str,
    pem_start: str,
    train_end: str,
    test_end: str,
    path: Path,
) -> None:
    """H6 : protocole PEM — une trajectoire, pas des semaines mélangées."""
    apply_publication_rc()
    fig, ax = plt.subplots(figsize=(8.2, 2.6))
    fig.patch.set_facecolor(BG)
    ax.broken_barh([(0, 7), (7, 3)], (0.6, 0.8), facecolors=(MUTED, ACCENT))
    ax.broken_barh([(10, 4)], (0.6, 0.8), facecolors=ACCENT_2)
    ax.set_yticks([1.0])
    ax.set_yticklabels(["année"])
    ax.set_xticks([0, 7, 10, 14])
    ax.set_xticklabels(
        [train_start[:10], pem_start[:10], train_end[:10], test_end[:10]],
        rotation=15,
        ha="right",
    )
    ax.set_xlim(0, 14)
    ax.set_ylim(0.2, 2.2)
    ax.text(3.5, 1.55, "train (hors PEM)", ha="center", color=INK, fontsize=9)
    ax.text(8.5, 1.55, "fenêtre PEM", ha="center", color=INK, fontsize=9)
    ax.text(12.0, 1.55, "test", ha="center", color=INK, fontsize=9)
    ax.set_title("Split chronologique : R et C ne changent pas le samedi")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _axes_ink(ax)
    save_figure(fig, path)


def plot_mae_horizons(
    hours: list[int],
    mae_r1: list[float],
    mae_r2: list[float],
    path: Path,
) -> None:
    """H7 : MAE vs horizon (complète le RMSE I1)."""
    apply_publication_rc()
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    fig.patch.set_facecolor(BG)
    ax.plot(hours, mae_r1, "o-", color=MUTED, label="R1C1", lw=1.8)
    ax.plot(hours, mae_r2, "s-", color=ACCENT, label="R2C2", lw=1.8)
    ax.set_xlabel("horizon (h)")
    ax.set_ylabel("MAE (°C)")
    ax.set_title("Même test : la MAE dit la même chose que le RMSE")
    ax.legend(frameon=False)
    _axes_ink(ax)
    save_figure(fig, path)


def plot_pareto_lambda(
    lambdas: np.ndarray,
    bill_mpc: np.ndarray,
    hours_mpc: np.ndarray,
    bill_bb: float,
    hours_bb: float,
    path: Path,
) -> None:
    """H8 : front facture vs inconfort quand on tourne λ."""
    apply_publication_rc()
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    fig.patch.set_facecolor(BG)
    ax.plot(hours_mpc, bill_mpc, "o-", color=ACCENT, lw=1.8, label="MPC (λ variable)")
    for lam, x, y in zip(lambdas, hours_mpc, bill_mpc, strict=True):
        ax.annotate(
            f"λ={lam:g}",
            (x, y),
            textcoords="offset points",
            xytext=(6, 6),
            color=INK,
            fontsize=9,
        )
    ax.scatter([hours_bb], [bill_bb], color=MUTED, s=70, zorder=3, label="hystérésis")
    ax.set_xlabel("heures sous T_conf")
    ax.set_ylabel("facture proxy (€)")
    ax.set_title("λ pèse l'inconfort : on ne minimise pas « un peu de tout »")
    ax.legend(frameon=False)
    _axes_ink(ax)
    save_figure(fig, path)


def plot_oracle_vs_retard(
    hours: np.ndarray,
    ta_oracle: np.ndarray,
    ta_delay: np.ndarray,
    t_conf: np.ndarray,
    path: Path,
) -> None:
    """H9 : prévision T_ext retardée de 6 h vs oracle."""
    apply_publication_rc()
    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    fig.patch.set_facecolor(BG)
    ax.plot(hours, t_conf, color=INK, lw=1.0, ls="--", label=r"$T_{\mathrm{conf}}$")
    ax.plot(hours, ta_oracle, color=ACCENT, lw=1.6, label="MPC oracle")
    ax.plot(hours, ta_delay, color=ACCENT_2, lw=1.5, label="MPC, T_ext retardé 6 h")
    ax.set_xlabel("heures")
    ax.set_ylabel(r"$T_{\mathrm{air}}$ (°C)")
    ax.set_title("Limite dite : l'oracle n'est pas un prévisionniste")
    ax.legend(frameon=False)
    _axes_ink(ax)
    save_figure(fig, path)


def plot_quantification_plant(
    hours: np.ndarray,
    ta: np.ndarray,
    y: np.ndarray,
    path: Path,
) -> None:
    """H10 : une journée, pas 0,1 °C du capteur vs air vrai."""
    apply_publication_rc()
    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    fig.patch.set_facecolor(BG)
    ax.plot(hours, ta, color=INK, lw=1.5, label="T_air vraie")
    ax.plot(hours, y, color=ACCENT, lw=1.0, drawstyle="steps-post", label="y (0,1 °C)")
    ax.set_xlabel("heures")
    ax.set_ylabel("°C")
    ax.set_title("Le capteur n'est pas l'état : quantification 0,1 °C")
    ax.legend(frameon=False)
    _axes_ink(ax)
    save_figure(fig, path)
