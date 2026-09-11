"""Calendriers confort et tarif bi-horaire (horloge Europe/Brussels)."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from basic_mpc.config import ControlConfig

TZ_BRUSSELS = ZoneInfo("Europe/Brussels")
JOULES_PER_KWH = 3.6e6


def parse_start_local(cfg: ControlConfig) -> datetime:
    """Début du scénario, heure locale Bruxelles."""
    naive = datetime.fromisoformat(cfg.start_local)
    if naive.tzinfo is None:
        return naive.replace(tzinfo=TZ_BRUSSELS)
    return naive.astimezone(TZ_BRUSSELS)


def scenario_index(
    start_local: datetime,
    n_steps: int,
    dt_seconds: float,
) -> pd.DatetimeIndex:
    """Index régulier à ``dt_seconds``, fuseau Bruxelles."""
    start = start_local
    if start.tzinfo is None:
        start = start.replace(tzinfo=TZ_BRUSSELS)
    else:
        start = start.astimezone(TZ_BRUSSELS)
    return pd.date_range(
        start=start,
        periods=n_steps,
        freq=pd.Timedelta(seconds=dt_seconds),
        tz=TZ_BRUSSELS,
    )


def hour_of_day(index: pd.DatetimeIndex) -> np.ndarray:
    """Heure locale décimale (7.5 = 07:30)."""
    local = index.tz_convert(TZ_BRUSSELS) if index.tz is not None else index
    return local.hour.to_numpy(dtype=float) + local.minute.to_numpy(dtype=float) / 60.0


def is_occupied(index: pd.DatetimeIndex, cfg: ControlConfig | None = None) -> np.ndarray:
    """True en période occupée : ``[occupied_start, occupied_end)``."""
    cfg = cfg or ControlConfig()
    hod = hour_of_day(index)
    return (hod >= cfg.occupied_start_hour) & (hod < cfg.occupied_end_hour)


def comfort_setpoint(index: pd.DatetimeIndex, cfg: ControlConfig | None = None) -> np.ndarray:
    """T_conf(t) : 20 °C le jour, 17 °C la nuit (défauts du brief)."""
    cfg = cfg or ControlConfig()
    occ = is_occupied(index, cfg)
    return np.where(occ, cfg.t_conf_occupied, cfg.t_conf_setback).astype(float)


def energy_price(index: pd.DatetimeIndex, cfg: ControlConfig | None = None) -> np.ndarray:
    """π(t) en €/kWh : heures pleines le jour, creuses la nuit (même horloge)."""
    cfg = cfg or ControlConfig()
    occ = is_occupied(index, cfg)
    return np.where(occ, cfg.pi_hp, cfg.pi_hc).astype(float)


def beta_kwh_per_p_second(alpha_h: float) -> float:
    """Conversion proxy P × secondes → kWh (chaleur qui entre dans l'air).

    Parameters
    ----------
    alpha_h : float
        Gain chauffage du plant (W par unité de P si P était SI).

    Returns
    -------
    float
        ``β = α_h / 3,6e6``.
    """
    return float(alpha_h) / JOULES_PER_KWH


def winter_weather(index: pd.DatetimeIndex, seed: int = 1) -> pd.DataFrame:
    """Extérieur et solaire calés sur l'horloge locale (pas sur t=0)."""
    rng = np.random.default_rng(seed)
    hod = hour_of_day(index)
    n = len(index)
    # Min vers 3 h, max vers 15 h — hiver belge, chauffage nécessaire.
    t_ext = 2.0 + 5.0 * np.sin(2.0 * np.pi * (hod - 9.0) / 24.0)
    t_ext = t_ext + rng.normal(0.0, 0.3, n)
    angle = 2.0 * np.pi * (hod - 6.0) / 24.0
    solar = np.clip(np.sin(angle), 0.0, None) * 2000.0
    return pd.DataFrame({"t_ext": t_ext, "S": solar}, index=index)
