"""Labo dynamique : météo tirée, burn-in habitation, une stratégie scorée."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import numpy as np
import pandas as pd

from basic_mpc.config import ControlConfig
from basic_mpc.control.cout import (
    _simulate_hysteresis,
    _simulate_mpc_setpoint,
    euro_metrics,
)
from basic_mpc.control.internal import p_to_hold, r2c2_internal_from_plant
from basic_mpc.control.tariffs import (
    TZ_BRUSSELS,
    beta_kwh_per_p_second,
    comfort_setpoint,
    energy_price,
    hour_of_day,
    parse_start_local,
    scenario_index,
)
from basic_mpc.models.plant import ThermalPlant, literature_plant_params
from basic_mpc.models.r2c2 import discretize as discretize_r2

BURN_IN_HOURS = 24.0
LAB_START_LOCAL = "2021-01-18T00:00:00"
STRATEGIES = ("hysteresis", "preheat", "mpc")
PREHEAT_HOURS = 2.0


def draw_weather(index: pd.DatetimeIndex, seed: int) -> pd.DataFrame:
    """Hiver belge, jours non identiques (marche aléatoire + nuages).

    Parameters
    ----------
    index : DatetimeIndex
        Horloge locale (Europe/Brussels).
    seed : int
        Graine du tirage (reproductible).

    Returns
    -------
    DataFrame
        Colonnes ``t_ext`` (°C) et ``S`` (proxy solaire).
    """
    rng = np.random.default_rng(seed)
    local = index.tz_convert(TZ_BRUSSELS) if index.tz is not None else index
    day_codes, _uniques = pd.factorize(local.normalize(), sort=False)
    n_days = int(day_codes.max()) + 1
    walk = np.cumsum(rng.normal(0.0, 1.2, n_days))
    means = np.clip(2.0 + walk, -5.0, 8.0)
    amps = rng.uniform(3.0, 7.0, n_days)
    clouds = rng.beta(4.0, 2.0, n_days)
    hod = hour_of_day(index)
    t_ext = means[day_codes] + amps[day_codes] * np.sin(
        2.0 * np.pi * (hod - 9.0) / 24.0
    )
    t_ext = t_ext + rng.normal(0.0, 0.4, len(index))
    angle = 2.0 * np.pi * (hod - 6.0) / 24.0
    solar = np.clip(np.sin(angle), 0.0, None) * 2000.0 * clouds[day_codes]
    return pd.DataFrame({"t_ext": t_ext, "S": solar}, index=index)


def lag_hours_xcorr(
    outdoor: np.ndarray,
    indoor: np.ndarray,
    dt_hours: float,
    discard_hours: float = 24.0,
    max_lag_hours: float = 18.0,
) -> float:
    """Retard indoor vs outdoor par corrélation croisée (après transitoire).

    Parameters
    ----------
    outdoor, indoor : ndarray
        Séries alignées.
    dt_hours : float
        Pas de temps (h).
    discard_hours : float
        Heures jetées en tête (burn-in / CI).
    max_lag_hours : float
        Retard max cherché (l'air ne précède pas l'extérieur).

    Returns
    -------
    float
        Retard en heures (positif = l'air retarde).
    """
    skip = int(round(discard_hours / dt_hours))
    x = np.asarray(outdoor[skip:], dtype=float)
    y = np.asarray(indoor[skip:], dtype=float)
    if x.size < 8 or y.size < 8:
        return 0.0
    x = x - float(x.mean())
    y = y - float(y.mean())
    max_lag = int(round(max_lag_hours / dt_hours))
    n = len(x)
    best_lag = 0
    best_c = -np.inf
    min_overlap = max(8, int(round(12.0 / dt_hours)))
    for lag in range(0, max_lag + 1):
        if n - lag < min_overlap:
            break
        a = x[: n - lag]
        b = y[lag:]
        denom = float(np.sqrt(np.dot(a, a) * np.dot(b, b))) + 1e-12
        corr = float(np.dot(a, b) / denom)
        if corr > best_c:
            best_c = corr
            best_lag = lag
    return best_lag * dt_hours


def lab_config(
    n_days: float = 4.0,
    t_conf_occupied: float = 20.0,
    t_conf_setback: float = 17.0,
    plant_seed: int = 0,
    start_local: str = LAB_START_LOCAL,
) -> ControlConfig:
    """ControlConfig du labo (lundi 00:00, durée en jours)."""
    return replace(
        ControlConfig(),
        n_hours=float(n_days) * 24.0,
        t_conf_occupied=float(t_conf_occupied),
        t_conf_setback=float(t_conf_setback),
        seed=int(plant_seed),
        start_local=start_local,
    )


def _score_slice(
    ta: np.ndarray,
    heating: np.ndarray,
    t_conf: np.ndarray,
    pi: np.ndarray,
    beta: float,
    dt_seconds: float,
    cfg: ControlConfig,
) -> dict:
    """Métriques euros + kWh HP/HC (fenêtre déjà scorée)."""
    metrics = euro_metrics(
        ta, heating, t_conf, pi, beta, dt_seconds, cfg.lambda_comfort
    )
    energy = beta * np.asarray(heating, dtype=float) * dt_seconds
    # Heures pleines = tarif haut (horloge occupation).
    is_hp = np.asarray(pi) >= 0.5 * (cfg.pi_hp + cfg.pi_hc)
    metrics["kwh_hp"] = float(np.sum(energy[is_hp]))
    metrics["kwh_hc"] = float(np.sum(energy[~is_hp]))
    metrics["bill_cum"] = np.cumsum(np.asarray(pi, dtype=float) * energy)
    return metrics


def build_scenario(
    n_days: float = 4.0,
    weather_seed: int = 1,
    plant_seed: int = 0,
    t_conf_occupied: float = 20.0,
    t_conf_setback: float = 17.0,
    cfg: ControlConfig | None = None,
) -> dict:
    """Météo + burn-in 24 h d'hystérésis. ``x0`` commun ensuite.

    Parameters
    ----------
    n_days : float
        Durée scorée (hors burn-in).
    weather_seed, plant_seed : int
        Graines météo et bruit du plant (période scorée).
    t_conf_occupied, t_conf_setback : float
        Programmation confort (°C).
    cfg : ControlConfig, optional
        Si fourni, écrase n_days / consignes / seed plant.

    Returns
    -------
    dict
        Index scoré, météo, T_conf, π, P_max, x0, matrices MPC.
    """
    cfg = cfg or lab_config(
        n_days=n_days,
        t_conf_occupied=t_conf_occupied,
        t_conf_setback=t_conf_setback,
        plant_seed=plant_seed,
    )
    plant_p = literature_plant_params()
    dt = plant_p.dt_seconds
    n_scored = int(round(cfg.n_hours * 3600.0 / dt))
    n_burn = int(round(BURN_IN_HOURS * 3600.0 / dt))
    start_scored = parse_start_local(cfg)
    start_burn = start_scored - timedelta(hours=BURN_IN_HOURS)
    index_full = scenario_index(start_burn, n_burn + n_scored, dt)
    weather = draw_weather(index_full, seed=weather_seed)
    t_ext = weather["t_ext"].to_numpy()
    solar = weather["S"].to_numpy()
    t_conf = comfort_setpoint(index_full, cfg)
    pi = energy_price(index_full, cfg)
    p_max = cfg.p_max_margin * p_to_hold(
        plant_p, cfg.t_conf_occupied, float(np.min(t_ext))
    )
    x_burn = np.array([float(t_conf[0]), float(t_conf[0])])
    plant_burn = ThermalPlant(
        params=plant_p,
        x0=x_burn,
        seed=cfg.seed + 10_003,
    )
    _simulate_hysteresis(
        plant_burn,
        t_ext[:n_burn],
        solar[:n_burn],
        t_conf[:n_burn],
        cfg.n_band,
        p_max,
    )
    x0 = plant_burn.x.copy()
    internal = r2c2_internal_from_plant(plant_p)
    ad, bd = discretize_r2(internal)
    q = np.diag(
        [internal.process_noise_std**2, internal.process_noise_std_mass**2]
    )
    r = np.array([[internal.sensor_noise_std**2]])
    return {
        "cfg": cfg,
        "plant_p": plant_p,
        "index": index_full[n_burn:],
        "t_ext": t_ext[n_burn:],
        "solar": solar[n_burn:],
        "t_conf": t_conf[n_burn:],
        "pi": pi[n_burn:],
        "p_max": p_max,
        "x0": x0,
        "weather_seed": int(weather_seed),
        "ad": ad,
        "bd": bd,
        "q": q,
        "r": r,
        "beta": beta_kwh_per_p_second(plant_p.alpha_h),
    }


def _run_named(name: str, scenario: dict) -> pd.DataFrame:
    cfg = scenario["cfg"]
    plant_p = scenario["plant_p"]
    plant = ThermalPlant(params=plant_p, x0=scenario["x0"], seed=cfg.seed)
    if name == "hysteresis":
        return _simulate_hysteresis(
            plant,
            scenario["t_ext"],
            scenario["solar"],
            scenario["t_conf"],
            cfg.n_band,
            scenario["p_max"],
        )
    if name == "preheat":
        early = replace(
            cfg,
            occupied_start_hour=cfg.occupied_start_hour - PREHEAT_HOURS,
        )
        t_sp_prog = comfort_setpoint(scenario["index"], early)
        return _simulate_hysteresis(
            plant,
            scenario["t_ext"],
            scenario["solar"],
            t_sp_prog,
            cfg.n_band,
            scenario["p_max"],
        )
    if name == "mpc":
        return _simulate_mpc_setpoint(
            plant,
            scenario["t_ext"],
            scenario["solar"],
            scenario["t_conf"],
            scenario["pi"],
            scenario["ad"],
            scenario["bd"],
            scenario["q"],
            scenario["r"],
            scenario["p_max"],
            cfg,
            plant_p.dt_seconds,
            scenario["beta"],
        )
    raise ValueError(f"stratégie inconnue: {name}")


def _pack_run(name: str, traj: pd.DataFrame, scenario: dict) -> dict:
    cfg = scenario["cfg"]
    metrics = _score_slice(
        traj["ta_true"].to_numpy(),
        traj["P"].to_numpy(),
        scenario["t_conf"],
        scenario["pi"],
        scenario["beta"],
        scenario["plant_p"].dt_seconds,
        cfg,
    )
    return {
        "strategy": name,
        "index": scenario["index"],
        "traj": traj,
        "t_conf": scenario["t_conf"],
        "pi": scenario["pi"],
        "metrics": metrics,
        "x0": np.asarray(scenario["x0"], dtype=float).copy(),
        "weather_seed": scenario["weather_seed"],
        "p_max": scenario["p_max"],
        "cfg": cfg,
    }


def run_strategy(
    name: str,
    scenario: dict,
    overlay_hysteresis: bool = False,
) -> dict:
    """Une loi sur le scénario (météo et x0 déjà figés).

    Parameters
    ----------
    name : {hysteresis, preheat, mpc}
    scenario : dict
        Sortie de ``build_scenario``.
    overlay_hysteresis : bool
        Second run hystérésis, même météo, même x0.

    Returns
    -------
    dict
        Trajectoire scorée + métriques ; ``overlay`` si demandé.
    """
    if name not in STRATEGIES:
        raise ValueError(f"stratégie inconnue: {name}")
    packed = _pack_run(name, _run_named(name, scenario), scenario)
    if overlay_hysteresis and name != "hysteresis":
        packed["overlay"] = _pack_run(
            "hysteresis",
            _run_named("hysteresis", scenario),
            scenario,
        )
    else:
        packed["overlay"] = None
    return packed
