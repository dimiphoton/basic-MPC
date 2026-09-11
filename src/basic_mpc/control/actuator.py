"""Bande proportionnelle n : consigne thermostat → P, et l'inverse."""

from __future__ import annotations

import numpy as np


def proportional_band_p(
    t_sp: float,
    t_meas: float,
    n_band: float,
    p_max: float,
) -> float:
    """Loi locale : à la consigne = arrêt, n °C en dessous = à fond.

    Parameters
    ----------
    t_sp : float
        Consigne thermostat (°C).
    t_meas : float
        Mesure (y), pas l'état vrai.
    n_band : float
        Largeur de bande (°C), défaut 1.
    p_max : float
        Commande max (unités proxy du plant).

    Returns
    -------
    float
        P dans ``[0, p_max]``.
    """
    if p_max <= 0.0:
        return 0.0
    if n_band <= 0.0:
        return float(p_max) if t_meas < t_sp else 0.0
    frac = (t_sp - t_meas) / n_band
    return float(p_max * np.clip(frac, 0.0, 1.0))


def t_sp_from_p(
    t_air: float,
    heating: float,
    n_band: float,
    p_max: float,
    t_sp_min: float,
    t_sp_max: float,
) -> float:
    """Consigne équivalente à un P, bornée dans ``[t_sp_min, t_sp_max]``.

    Parameters
    ----------
    t_air : float
        Air estimé au moment de la décision.
    heating : float
        P planifié.
    n_band, p_max : float
        Bande et plafond.
    t_sp_min, t_sp_max : float
        Bornes thermostat.

    Returns
    -------
    float
        T_sp (°C).
    """
    if p_max <= 0.0 or n_band <= 0.0:
        return float(np.clip(t_air, t_sp_min, t_sp_max))
    raw = t_air + n_band * heating / p_max
    return float(np.clip(raw, t_sp_min, t_sp_max))
