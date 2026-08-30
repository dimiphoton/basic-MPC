"""Modèle R2C2 d'identification : solaire sur l'air seulement.

``C_a`` est fixé à 1 (échelle absorbée par les gains et les R).
Le plant littérature a en plus ``alpha_s_mass`` — ne pas confondre.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm


@dataclass(frozen=True)
class R2C2Params:
    """Paramètres continus, ``C_a = 1``.

    ``R_{ae}`` et ``R_{am}`` sont des temps caractéristiques (s) une fois
    ``C_a`` normalisée. ``C_m`` est un ratio de capacités.

    Parameters
    ----------
    rae, ram : float
        Résistances avec ``C_a = 1``.
    cm : float
        Capacité de masse relative.
    g_solar, g_heating : float
        ``alpha / C_a`` (unités proxy de S et P).
    dt_seconds : float
        Pas.
    process_noise_std, process_noise_std_mass : float
        Bruits de process air / masse (°C).
    sensor_noise_std : float
        Bruit de mesure.
    """

    rae: float
    ram: float
    cm: float
    g_solar: float
    g_heating: float
    dt_seconds: float = 300.0
    process_noise_std: float = 0.05
    process_noise_std_mass: float = 0.05
    sensor_noise_std: float = 0.05
    ca: float = 1.0

    @property
    def tau_air_hours(self) -> float:
        """Échelle air : ``R_ae C_a``."""
        return float(self.rae * self.ca / 3600.0)

    @property
    def tau_mass_hours(self) -> float:
        """Échelle masse : ``R_am C_m``."""
        return float(self.ram * self.cm / 3600.0)


def continuous_matrices(params: R2C2Params) -> tuple[np.ndarray, np.ndarray]:
    """A (2×2) et B (2×3), u = [T_ext, S, P]. Solaire : ligne air seulement."""
    inv_ram_ca = 1.0 / (params.ram * params.ca)
    inv_rae_ca = 1.0 / (params.rae * params.ca)
    inv_ram_cm = 1.0 / (params.ram * params.cm)
    mat_a = np.array(
        [
            [-(inv_ram_ca + inv_rae_ca), inv_ram_ca],
            [inv_ram_cm, -inv_ram_cm],
        ],
        dtype=float,
    )
    mat_b = np.array(
        [
            [inv_rae_ca, params.g_solar, params.g_heating],
            [0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    return mat_a, mat_b


def discretize(params: R2C2Params) -> tuple[np.ndarray, np.ndarray]:
    """Discrétisation exacte ``Ad, Bd`` (comme le plant).

    Returns
    -------
    Ad, Bd : ndarray
        ``x_{k+1} = Ad x_k + Bd u_k``.
    """
    mat_a, mat_b = continuous_matrices(params)
    if not np.all(np.isfinite(mat_a)) or np.linalg.cond(mat_a) > 1e10:
        msg = "matrice A continue mal conditionnée"
        raise np.linalg.LinAlgError(msg)
    dt = params.dt_seconds
    mat_ad = expm(mat_a * dt)
    mat_bd = np.linalg.solve(mat_a, (mat_ad - np.eye(2)) @ mat_b)
    return mat_ad, mat_bd


def simulate_r2c2(
    params: R2C2Params,
    t_ext: np.ndarray,
    solar: np.ndarray,
    heating: np.ndarray,
    x0: np.ndarray | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """États vrais ``(n, 2)`` et mesure de l'air.

    Parameters
    ----------
    params : R2C2Params
        Dynamique.
    t_ext, solar, heating : ndarray
        Entrées.
    x0 : ndarray, optional
        ``[T_air, T_masse]``. Défaut 20 °C / 20 °C.
    seed : int
        Bruit.

    Returns
    -------
    x_true, y : ndarray
        États et observations.
    """
    ad, bd = discretize(params)
    n = len(t_ext)
    rng = np.random.default_rng(seed)
    x_true = np.empty((n, 2))
    y = np.empty(n)
    state = np.array([20.0, 20.0], dtype=float) if x0 is None else np.asarray(x0, dtype=float).reshape(2).copy()
    qa = params.process_noise_std
    qm = params.process_noise_std_mass
    r = params.sensor_noise_std
    for k in range(n):
        x_true[k] = state
        y[k] = state[0] + rng.normal(0.0, r)
        if k + 1 < n:
            u = np.array([t_ext[k], solar[k], heating[k]], dtype=float)
            noise = np.array([rng.normal(0.0, qa), rng.normal(0.0, qm)])
            state = ad @ state + bd @ u + noise
    return x_true, y
