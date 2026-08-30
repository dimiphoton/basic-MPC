"""Modèle R1C1 : un état (air), une résistance vers l'extérieur.

Forme continue : ``dT/dt = (T_ext - T)/tau + g_s S + g_h P``.
On identifie la forme discrète équivalente (grey-box : le gain sur
``T_ext`` vaut ``1 - a``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class R1C1Params:
    """Paramètres discrets du R1C1.

    ``a`` est le facteur d'oubli par pas : ``tau = -dt / log(a)``.
    ``g_solar`` et ``g_heating`` sont des gains par pas, dans les unités
    proxy de ``S`` et ``P`` (pas des SI).

    Parameters
    ----------
    a : float
        ``Ad`` scalaire, dans (0, 1).
    g_solar, g_heating : float
        Colonnes solaire et chauffage de ``Bd``.
    dt_seconds : float
        Pas d'échantillonnage.
    process_noise_std, sensor_noise_std : float
        Écarts-types de ``w`` et ``v`` (°C).
    """

    a: float
    g_solar: float
    g_heating: float
    dt_seconds: float = 300.0
    process_noise_std: float = 0.05
    sensor_noise_std: float = 0.05

    @property
    def tau_hours(self) -> float:
        """Constante de temps équivalente, en heures."""
        if self.a <= 0.0 or self.a >= 1.0:
            return float("inf") if self.a >= 1.0 else 0.0
        return float(-self.dt_seconds / np.log(self.a) / 3600.0)


def discretize(params: R1C1Params) -> tuple[np.ndarray, np.ndarray]:
    """Matrices discrètes ``x_{k+1} = Ad x_k + Bd u_k``.

    ``u = [T_ext, S, P]``. Le gain extérieur est ``1 - a`` (R1C1).

    Parameters
    ----------
    params : R1C1Params
        Paramètres discrets.

    Returns
    -------
    Ad, Bd : ndarray
        ``Ad`` (1×1) et ``Bd`` (1×3).
    """
    mat_ad = np.array([[params.a]], dtype=float)
    mat_bd = np.array(
        [[1.0 - params.a, params.g_solar, params.g_heating]],
        dtype=float,
    )
    return mat_ad, mat_bd


def simulate_r1c1(
    params: R1C1Params,
    t_ext: np.ndarray,
    solar: np.ndarray,
    heating: np.ndarray,
    x0: float = 20.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Trajectoire d'état vrai et d'observation (sans quantification).

    ``y_k`` mesure ``x_k`` ; ``u_k`` agit ensuite vers ``x_{k+1}``.

    Parameters
    ----------
    params : R1C1Params
        Dynamique et bruits.
    t_ext, solar, heating : ndarray
        Entrées alignées.
    x0 : float
        État initial.
    seed : int
        Bruit reproductible.

    Returns
    -------
    x_true, y : ndarray
        Température d'air vraie et mesure (même longueur).
    """
    ad, bd = discretize(params)
    n = len(t_ext)
    rng = np.random.default_rng(seed)
    x_true = np.empty(n)
    y = np.empty(n)
    state = np.array([x0], dtype=float)
    q = params.process_noise_std
    r = params.sensor_noise_std
    for k in range(n):
        x_true[k] = state[0]
        y[k] = state[0] + rng.normal(0.0, r)
        if k + 1 < n:
            u = np.array([t_ext[k], solar[k], heating[k]], dtype=float)
            state = ad @ state + bd @ u + rng.normal(0.0, q, size=1)
    return x_true, y
