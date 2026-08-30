"""Figures du catalogue pour l'étape R1C1 (I3, I4, S1)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from basic_mpc.models.r1c1 import R1C1Params

# Fond crème du thème portfolio, pas un cadre blanc
_BG = "#F4EFE6"
_INK = "#2C2416"
_ACCENT = "#3D6B6B"


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(_BG)
    ax.figure.set_facecolor(_BG)
    ax.tick_params(colors=_INK)
    for spine in ax.spines.values():
        spine.set_color(_INK)


def _acf(values: np.ndarray, nlags: int) -> np.ndarray:
    """Autocorrélation empirique (biaisée), lag 0…nlags."""
    x = np.asarray(values, dtype=float)
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom <= 0.0:
        return np.zeros(nlags + 1)
    out = np.empty(nlags + 1)
    out[0] = 1.0
    for lag in range(1, nlags + 1):
        out[lag] = float(np.dot(x[:-lag], x[lag:])) / denom
    return out


def plot_innovations(
    innov: np.ndarray,
    path: Path,
    nlags: int = 48,
) -> None:
    """I3 : histogramme des innovations + ACF (lags en pas de 5 min)."""
    e = np.asarray(innov, dtype=float)
    e = e[np.isfinite(e)]
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))
    fig.patch.set_facecolor(_BG)
    axes[0].hist(e, bins=40, color=_ACCENT, edgecolor=_INK, linewidth=0.4)
    axes[0].set_title("Innovations Kalman (°C)")
    axes[0].set_xlabel("y − ŷ")
    _style(axes[0])
    acf = _acf(e, nlags)
    lags_h = np.arange(nlags + 1) * 5.0 / 60.0
    axes[1].stem(lags_h, acf, linefmt=_ACCENT, markerfmt="o", basefmt=_INK)
    axes[1].axhline(0.0, color=_INK, linewidth=0.6)
    axes[1].set_title("ACF des innovations")
    axes[1].set_xlabel("lag (h)")
    _style(axes[1])
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, facecolor=_BG)
    plt.close(fig)


def plot_params(
    params: R1C1Params,
    stderr: dict[str, float] | None,
    path: Path,
) -> None:
    """I4 : τ, g_S, g_P en clair (unités différentes, pas un même axe log)."""
    se = stderr or {}

    def _ligne(nom: str, valeur: float, cle_se: str, fmt: str) -> str:
        texte = f"{nom} = {valeur:{fmt}}"
        if cle_se in se and se[cle_se] > 0.0:
            texte += f"  ± {se[cle_se]:{fmt}}"
        return texte

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)
    ax.axis("off")
    ax.set_title("R1C1 identifié (gains dans les unités proxy)")
    lignes = [
        _ligne("τ", params.tau_hours, "tau_hours", ".1f") + " h",
        _ligne("g_S", params.g_solar, "g_solar", ".2e") + " / pas",
        _ligne("g_P", params.g_heating, "g_heating", ".2e") + " / pas",
        f"σ_w = {params.process_noise_std:.3f} °C  (σ_v fixé à {params.sensor_noise_std:.2f})",
    ]
    for i, ligne in enumerate(lignes):
        ax.text(0.08, 0.72 - 0.18 * i, ligne, fontsize=13, color=_INK, family="monospace")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, facecolor=_BG)
    plt.close(fig)


def plot_tair_vs_y(
    ta_true: np.ndarray,
    y: np.ndarray,
    path: Path,
    n_show: int = 288,
) -> None:
    """S1 : état air vrai vs mesure quantifiée (une journée, 5 min)."""
    n = min(n_show, len(ta_true), len(y))
    hours = np.arange(n) * 5.0 / 60.0
    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    fig.patch.set_facecolor(_BG)
    ax.plot(hours, ta_true[:n], color=_INK, label="T_air vraie", linewidth=1.6)
    ax.plot(
        hours,
        y[:n],
        color=_ACCENT,
        label="y (0,1 °C)",
        linewidth=1.0,
        drawstyle="steps-post",
    )
    ax.set_xlabel("heures")
    ax.set_ylabel("°C")
    ax.set_title("Le capteur n'est pas l'état : T_air vs y")
    ax.legend(frameon=False)
    _style(ax)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, facecolor=_BG)
    plt.close(fig)
