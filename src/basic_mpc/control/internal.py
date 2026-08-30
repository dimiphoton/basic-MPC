"""Modèle interne du MPC : R2C2 d'identification calé sur le plant.

On ne réutilise pas le R2C2 identifié sur la maison réelle : τ ~ 140 h
et l'échelle de P n'ont rien à voir avec le plant littérature (~3 h /
~11 h). La misspecification volontaire reste ``alpha_s_mass`` : le
contrôleur n'a le solaire que sur l'air.
"""

from __future__ import annotations

from basic_mpc.models.plant import PlantParams
from basic_mpc.models.r2c2 import R2C2Params


def r2c2_internal_from_plant(plant: PlantParams) -> R2C2Params:
    """Copie R, C, α_h, α_s,air du plant ; C_a → 1 ; pas de solaire masse.

    Parameters
    ----------
    plant : PlantParams
        Paramètres du simulateur.

    Returns
    -------
    R2C2Params
        Dynamique interne (structure d'identification).
    """
    ca = plant.ca
    return R2C2Params(
        rae=plant.rae * ca,
        ram=plant.ram * ca,
        cm=plant.cm / ca,
        g_solar=plant.alpha_s_air / ca,
        g_heating=plant.alpha_h / ca,
        dt_seconds=plant.dt_seconds,
        process_noise_std=0.02,
        process_noise_std_mass=0.02,
        sensor_noise_std=0.05,
    )


def p_to_hold(plant: PlantParams, t_set: float, t_ext: float) -> float:
    """P stationnaire pour tenir ``t_set`` à ``t_ext``, S = 0, T_m = T_a.

    Parameters
    ----------
    plant : PlantParams
        Plant (α_h et R_ae).
    t_set, t_ext : float
        Consigne et extérieur (°C).

    Returns
    -------
    float
        P ≥ 0 dans les unités proxy du plant.
    """
    denom = plant.rae * plant.alpha_h
    if denom <= 0.0:
        return 0.0
    return max(0.0, (t_set - t_ext) / denom)
