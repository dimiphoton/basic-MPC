"""Figures de comparaison R1C1 / R2C2 (catalogue I1, I2, Z, S2, S3, I5)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from basic_mpc.figures.style import ACCENT, ACCENT_2, BG, INK, MUTED
from basic_mpc.figures.style import apply_publication_rc, save_figure
from basic_mpc.models.impedance import nyquist_omegas, omega_period_hours, z_r1c1, z_r2c2
from basic_mpc.models.r1c1 import R1C1Params
from basic_mpc.models.r2c2 import R2C2Params


def _axes_ink(ax: plt.Axes) -> None:
    ax.set_facecolor(BG)
    ax.tick_params(colors=INK)
    for spine in ax.spines.values():
        spine.set_color(INK)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def plot_rmse_horizons(
    scores_r1: dict,
    scores_r2: dict,
    path: Path,
) -> None:
    """I1 : RMSE vs horizon, les deux modèles."""
    apply_publication_rc()
    keys = ["1h", "3h", "6h", "12h", "24h"]
    hours = [1, 3, 6, 12, 24]
    r1 = [scores_r1[k]["rmse"] for k in keys]
    r2 = [scores_r2[k]["rmse"] for k in keys]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    fig.patch.set_facecolor(BG)
    ax.plot(hours, r1, "o-", color=MUTED, label="R1C1", lw=1.8)
    ax.plot(hours, r2, "s-", color=ACCENT, label="R2C2", lw=1.8)
    ax.set_xlabel("horizon (h)")
    ax.set_ylabel("RMSE (°C)")
    ax.set_title("Le second état se juge à 6–24 h, pas à un pas")
    ax.legend(frameon=False)
    _axes_ink(ax)
    save_figure(fig, path)


def plot_48h_overlay(
    hours: np.ndarray,
    y: np.ndarray,
    yhat_r1: np.ndarray,
    yhat_r2: np.ndarray,
    path: Path,
    title: str,
) -> None:
    """I2 : une fenêtre 48 h."""
    apply_publication_rc()
    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    fig.patch.set_facecolor(BG)
    ax.plot(hours, y, color=INK, lw=1.5, label="y")
    ax.plot(hours, yhat_r1, color=MUTED, lw=1.2, label="ŷ R1C1")
    ax.plot(hours, yhat_r2, color=ACCENT, lw=1.2, label="ŷ R2C2")
    ax.set_xlabel("heures")
    ax.set_ylabel("°C")
    ax.set_title(title)
    ax.legend(frameon=False, ncol=3)
    _axes_ink(ax)
    save_figure(fig, path)


def plot_z1_vectors(
    params_r1: R1C1Params,
    params_r2: R2C2Params,
    path: Path,
) -> None:
    """Z1 : Z(jω_24h) / Z(0), une flèche par modèle."""
    apply_publication_rc()
    w = omega_period_hours(24.0)
    z1 = z_r1c1(params_r1, np.array([w]))
    z2 = z_r2c2(params_r2, np.array([w]))
    z1n = np.ravel(z1 / z_r1c1(params_r1, np.array([1e-12])))[0]
    z2n = np.ravel(z2 / z_r2c2(params_r2, np.array([1e-12])))[0]
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    fig.patch.set_facecolor(BG)
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.axvline(0, color=MUTED, lw=0.6)
    ax.arrow(
        0,
        0,
        float(np.real(z1n)),
        float(np.imag(z1n)),
        color=MUTED,
        width=0.004,
        head_width=0.03,
        length_includes_head=True,
        label="R1C1",
    )
    ax.arrow(
        0,
        0,
        float(np.real(z2n)),
        float(np.imag(z2n)),
        color=ACCENT,
        width=0.004,
        head_width=0.03,
        length_includes_head=True,
        label="R2C2",
    )
    ax.set_xlabel(r"$\mathrm{Re}\,Z(j\omega_{24h})/Z(0)$")
    ax.set_ylabel(r"$\mathrm{Im}\,Z(j\omega_{24h})/Z(0)$")
    ax.set_title("À 24 h, le R2C2 n'est pas un R1C1 renommé")
    ax.set_aspect("equal")
    ax.legend(frameon=False)
    _axes_ink(ax)
    save_figure(fig, path)


def plot_nyquist(
    params_r1: R1C1Params,
    params_r2: R2C2Params,
    path: Path,
) -> None:
    """Z2 : lieu de Nyquist, Z/Z(0)."""
    apply_publication_rc()
    w = nyquist_omegas()
    z1 = z_r1c1(params_r1, w) / z_r1c1(params_r1, np.array([1e-12]))
    z2 = z_r2c2(params_r2, w) / z_r2c2(params_r2, np.array([1e-12]))
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    fig.patch.set_facecolor(BG)
    ax.plot(z1.real, z1.imag, color=MUTED, lw=1.6, label="R1C1")
    ax.plot(z2.real, z2.imag, color=ACCENT, lw=1.6, label="R2C2")
    ax.axhline(0, color=MUTED, lw=0.5)
    ax.axvline(0, color=MUTED, lw=0.5)
    ax.set_xlabel(r"$\mathrm{Re}\,Z/Z(0)$")
    ax.set_ylabel(r"$\mathrm{Im}\,Z/Z(0)$")
    ax.set_title("Nyquist : 5 min → 7 jours")
    ax.set_aspect("equal")
    ax.legend(frameon=False)
    _axes_ink(ax)
    save_figure(fig, path)


def plot_phase_portrait(
    ta: np.ndarray,
    tm: np.ndarray,
    path: Path,
) -> None:
    """I5 : air filtré vs masse estimée."""
    apply_publication_rc()
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    fig.patch.set_facecolor(BG)
    ax.plot(ta, tm, color=ACCENT, lw=0.6, alpha=0.85)
    ax.set_xlabel(r"$\hat T_{\mathrm{air}}$ (°C)")
    ax.set_ylabel(r"$\hat T_{\mathrm{masse}}$ (°C)")
    ax.set_title("La masse n'est pas une copie de l'air")
    _axes_ink(ax)
    save_figure(fig, path)


def plot_mass_on_plant(
    hours: np.ndarray,
    tm_true: np.ndarray,
    tm_hat: np.ndarray,
    path: Path,
) -> None:
    """S2 : Kalman vs vérité plant."""
    apply_publication_rc()
    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    fig.patch.set_facecolor(BG)
    ax.plot(hours, tm_true, color=INK, lw=1.5, label=r"$T_{\mathrm{masse}}$ plant")
    ax.plot(hours, tm_hat, color=ACCENT, lw=1.2, label=r"$\hat T_{\mathrm{masse}}$ Kalman")
    ax.set_xlabel("heures")
    ax.set_ylabel("°C")
    ax.set_title("État caché : le filtre suit la masse du plant")
    ax.legend(frameon=False)
    _axes_ink(ax)
    save_figure(fig, path)


def plot_plant_vs_identified(
    hours: np.ndarray,
    y_plant: np.ndarray,
    y_model: np.ndarray,
    path: Path,
) -> None:
    """S3 : même u, plant ≠ R2C2 identifié."""
    apply_publication_rc()
    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    fig.patch.set_facecolor(BG)
    ax.plot(hours, y_plant, color=INK, lw=1.5, label="y plant")
    ax.plot(hours, y_model, color=ACCENT_2, lw=1.2, label="y R2C2 identifié")
    ax.set_xlabel("heures")
    ax.set_ylabel("°C")
    ax.set_title("Même météo, deux physiques : pas de validation circulaire")
    ax.legend(frameon=False)
    _axes_ink(ax)
    save_figure(fig, path)


def plot_params_r2c2(params: R2C2Params, path: Path) -> None:
    """I4 R2C2 : constantes de temps et gains."""
    apply_publication_rc()
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    fig.patch.set_facecolor(BG)
    ax.axis("off")
    ax.set_title("R2C2 identifié (C_a = 1, unités proxy)")
    lignes = [
        f"τ_air  = {params.tau_air_hours:.2f} h    (R_ae C_a)",
        f"τ_masse = {params.tau_mass_hours:.2f} h    (R_am C_m)",
        f"C_m / C_a = {params.cm:.2f}",
        f"g_S = {params.g_solar:.2e} / pas    g_P = {params.g_heating:.2e} / pas",
    ]
    for i, ligne in enumerate(lignes):
        ax.text(0.08, 0.72 - 0.16 * i, ligne, fontsize=12, color=INK, family="monospace")
    save_figure(fig, path)
