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


def z_snapshot(z: complex | np.ndarray, period_hours: float) -> dict:
    """Partie réelle / imaginaire, phase et retard d'une impédance.

    Parameters
    ----------
    z : complex
        Z(jω) à la période donnée.
    period_hours : float
        Période (h), souvent 24.

    Returns
    -------
    dict
        ``re``, ``im``, ``mag``, ``phase_deg``, ``delay_hours``.
        Un ``delay_hours`` positif = l'air **retarde** sur l'apport.
    """
    zc = complex(np.ravel(z)[0])
    phase_rad = float(np.angle(zc))
    return {
        "re": float(np.real(zc)),
        "im": float(np.imag(zc)),
        "mag": float(np.abs(zc)),
        "phase_deg": float(np.degrees(phase_rad)),
        "delay_hours": float(-phase_rad / (2.0 * np.pi) * period_hours),
    }


def z_normalized(z: np.ndarray, z0: np.ndarray) -> np.ndarray:
    """Z / Z(0) pour comparer des échelles (P n'est pas en watts)."""
    z_dc = complex(np.ravel(z0)[0])
    if abs(z_dc) < 1e-18:
        return np.asarray(z, dtype=complex)
    return np.asarray(z, dtype=complex) / z_dc
