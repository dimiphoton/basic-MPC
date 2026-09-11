"""Thermostat hystérésis : tout-ou-rien sur la mesure, sans modèle."""


def bangbang_step(
    y: float,
    heating_on: bool,
    t_low: float,
    t_high: float,
    p_max: float,
) -> tuple[float, bool]:
    """Un pas de thermostat.

    Allume sous ``t_low``, éteint au-dessus de ``t_high``. Entre les deux,
    on garde l'état (hystérésis).

    Parameters
    ----------
    y : float
        Température mesurée.
    heating_on : bool
        État précédent.
    t_low, t_high : float
        Seuils (°C).
    p_max : float
        Commande ON (unités proxy).

    Returns
    -------
    p, heating_on : float, bool
        Commande et nouvel état.
    """
    if heating_on:
        if y >= t_high:
            heating_on = False
    elif y <= t_low:
        heating_on = True
    return (p_max if heating_on else 0.0), heating_on


def hysteresis_comfort_step(
    y: float,
    heating_on: bool,
    t_conf: float,
    n_band: float,
    p_max: float,
) -> tuple[float, bool]:
    """Baseline v1.1 : hystérésis autour de ``T_conf(t)``, sans modèle.

    Allume sous ``T_conf - n/2``, éteint au-dessus de ``T_conf + n/2``.
    """
    half = 0.5 * n_band
    return bangbang_step(y, heating_on, t_conf - half, t_conf + half, p_max)
