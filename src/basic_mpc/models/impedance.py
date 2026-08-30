"""Impédance thermique Z(jω) = T_air / Q_air (T_ext = 0)."""

from __future__ import annotations

import numpy as np

from basic_mpc.models.r1c1 import R1C1Params
from basic_mpc.models.r2c2 import R2C2Params


def omega_period_hours(hours: float) -> float:
    """Pulsation d'une période en heures."""
    return float(2.0 * np.pi / (hours * 3600.0))


def z_r1c1(params: R1C1Params, omega: np.ndarray) -> np.ndarray:
    """Z = τ / (1 + jωτ)  (C normalisée à 1, R = τ).

    Parameters
    ----------
    params : R1C1Params
        Pour ``tau_hours``.
    omega : ndarray
        Pulsations (rad/s).

    Returns
    -------
    ndarray
        Impédance complexe.
    """
    tau = params.tau_hours * 3600.0
    w = np.asarray(omega, dtype=float)
    return tau / (1.0 + 1j * w * tau)


def z_r2c2(params: R2C2Params, omega: np.ndarray) -> np.ndarray:
    """Z(jω) vue du nœud air, T_ext = 0 (réseau RC, pas le gain proxy)."""
    w = np.asarray(omega, dtype=float)
    ya = 1.0 / params.ram + 1.0 / params.rae + 1j * w * params.ca
    ym = 1.0 / params.ram + 1j * w * params.cm
    return ym / (ya * ym - (1.0 / params.ram) ** 2)


def nyquist_omegas(dt_seconds: float = 300.0, n: int = 80) -> np.ndarray:
    """De plusieurs jours jusqu'à la maille 5 min, log-espacé."""
    w_slow = omega_period_hours(7.0 * 24.0)
    w_fast = 2.0 * np.pi / dt_seconds
    return np.logspace(np.log10(w_slow), np.log10(w_fast), n)
