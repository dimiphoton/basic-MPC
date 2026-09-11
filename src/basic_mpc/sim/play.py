"""Trois lois de chauffe sur le même plant (arène figée, CI magiques).

Le labo visiteur est ``basic_mpc.sim.dynamic`` (météo tirée, burn-in).
Ce module reste pour les exports historiques, pas pour le Streamlit.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from basic_mpc.config import ControlConfig
from basic_mpc.control.bangbang import hysteresis_comfort_step
from basic_mpc.control.cout import (
    _simulate_hysteresis,
    _simulate_mpc_setpoint,
    euro_metrics,
)
from basic_mpc.control.internal import p_to_hold, r2c2_internal_from_plant
from basic_mpc.control.tariffs import (
    beta_kwh_per_p_second,
    comfort_setpoint,
    energy_price,
    parse_start_local,
    scenario_index,
    winter_weather,
)
from basic_mpc.models.plant import ThermalPlant, literature_plant_params
from basic_mpc.models.r2c2 import discretize as discretize_r2


def _p_max(cfg: ControlConfig, t_ext: np.ndarray) -> float:
    plant_p = literature_plant_params()
    return cfg.p_max_margin * p_to_hold(plant_p, cfg.t_conf_occupied, float(np.min(t_ext)))


def scenario_bundle(cfg: ControlConfig | None = None) -> dict:
    """Météo, calendriers et P_max d'un scénario."""
    cfg = cfg or ControlConfig()
    plant_p = literature_plant_params()
    n_steps = int(cfg.n_hours * 3600.0 / plant_p.dt_seconds)
    index = scenario_index(parse_start_local(cfg), n_steps, plant_p.dt_seconds)
    weather = winter_weather(index, seed=cfg.seed + 1)
    t_ext = weather["t_ext"].to_numpy()
    solar = weather["S"].to_numpy()
    t_conf = comfort_setpoint(index, cfg)
    pi = energy_price(index, cfg)
    p_max = _p_max(cfg, t_ext)
    dt_hours = plant_p.dt_seconds / 3600.0
    hours = np.arange(n_steps) * dt_hours
    return {
        "cfg": cfg,
        "plant_p": plant_p,
        "index": index,
        "t_ext": t_ext,
        "solar": solar,
        "t_conf": t_conf,
        "pi": pi,
        "p_max": p_max,
        "hours": hours,
        "beta": beta_kwh_per_p_second(plant_p.alpha_h),
    }


def _metrics_of(traj: pd.DataFrame, bundle: dict) -> dict:
    return euro_metrics(
        traj["ta_true"].to_numpy(),
        traj["P"].to_numpy(),
        bundle["t_conf"],
        bundle["pi"],
        bundle["beta"],
        bundle["plant_p"].dt_seconds,
        bundle["cfg"].lambda_comfort,
    )


def run_hysteresis(bundle: dict, seed: int | None = None) -> pd.DataFrame:
    """Thermostat : T_sp = T_conf, hystérésis, pas d'anticipation."""
    plant_p = bundle["plant_p"]
    x0 = np.array([18.0, 16.0])
    plant = ThermalPlant(params=plant_p, x0=x0, seed=bundle["cfg"].seed if seed is None else seed)
    return _simulate_hysteresis(
        plant,
        bundle["t_ext"],
        bundle["solar"],
        bundle["t_conf"],
        bundle["cfg"].n_band,
        bundle["p_max"],
    )


def run_preheat(bundle: dict, hours_ahead: float = 2.0, seed: int | None = None) -> pd.DataFrame:
    """Hystérésis qui avance l'occupation de ``hours_ahead`` — règle simple."""
    cfg = bundle["cfg"]
    early = replace(cfg, occupied_start_hour=cfg.occupied_start_hour - hours_ahead)
    t_sp_prog = comfort_setpoint(bundle["index"], early)
    plant_p = bundle["plant_p"]
    x0 = np.array([18.0, 16.0])
    plant = ThermalPlant(
        params=plant_p,
        x0=x0,
        seed=bundle["cfg"].seed if seed is None else seed,
    )
    return _simulate_hysteresis(
        plant,
        bundle["t_ext"],
        bundle["solar"],
        t_sp_prog,
        cfg.n_band,
        bundle["p_max"],
    )


def run_mpc(bundle: dict, seed: int | None = None) -> pd.DataFrame:
    """MPC v1.1 (consigne, J euros) — plus lent, pour le scoreboard."""
    plant_p = bundle["plant_p"]
    internal = r2c2_internal_from_plant(plant_p)
    ad, bd = discretize_r2(internal)
    q = np.diag([internal.process_noise_std**2, internal.process_noise_std_mass**2])
    r = np.array([[internal.sensor_noise_std**2]])
    x0 = np.array([18.0, 16.0])
    plant = ThermalPlant(
        params=plant_p,
        x0=x0,
        seed=bundle["cfg"].seed if seed is None else seed,
    )
    return _simulate_mpc_setpoint(
        plant,
        bundle["t_ext"],
        bundle["solar"],
        bundle["t_conf"],
        bundle["pi"],
        ad,
        bd,
        q,
        r,
        bundle["p_max"],
        bundle["cfg"],
        plant_p.dt_seconds,
        bundle["beta"],
    )


def summarize_traj(name: str, traj: pd.DataFrame, bundle: dict) -> dict:
    """Métriques + séries sous-échantillonnées pour le JSON Pages."""
    metrics = _metrics_of(traj, bundle)
    # 10 min : assez pour Plotly, JSON raisonnable.
    stride = 2
    hours = bundle["hours"][::stride]
    return {
        "name": name,
        "metrics": metrics,
        "hours": hours.tolist(),
        "ta": traj["ta_true"].to_numpy()[::stride].tolist(),
        "tm": traj["tm_true"].to_numpy()[::stride].tolist(),
        "p": traj["P"].to_numpy()[::stride].tolist(),
        "t_sp": traj["t_sp"].to_numpy()[::stride].tolist(),
    }


def run_arena(
    cfg: ControlConfig | None = None,
    include_mpc: bool = True,
) -> dict:
    """Hystérésis, préchauffage 2 h, et MPC si demandé.

    Parameters
    ----------
    cfg : ControlConfig, optional
        Durée, bande n, calendriers.
    include_mpc : bool
        Le QP 48 h prend ~15 s ; les tests le coupent.

    Returns
    -------
    dict
        ``bundle`` allégé + ``strategies``.
    """
    bundle = scenario_bundle(cfg)
    strategies = {
        "hysteresis": summarize_traj("hysteresis", run_hysteresis(bundle), bundle),
        "preheat": summarize_traj("preheat", run_preheat(bundle, hours_ahead=2.0), bundle),
    }
    if include_mpc:
        strategies["mpc"] = summarize_traj("mpc", run_mpc(bundle), bundle)
    t_conf = bundle["t_conf"]
    pi = bundle["pi"]
    stride = 2
    return {
        "hours": bundle["hours"][::stride].tolist(),
        "t_conf": t_conf[::stride].tolist(),
        "pi": pi[::stride].tolist(),
        "t_ext": bundle["t_ext"][::stride].tolist(),
        "p_max": bundle["p_max"],
        "n_band": bundle["cfg"].n_band,
        "strategies": strategies,
    }
