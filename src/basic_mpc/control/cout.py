"""Boucle fermée v1.1 : consigne thermostat, bande n, J en euros."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from basic_mpc.config import REPO_ROOT, ControlConfig, DataConfig
from basic_mpc.control.actuator import proportional_band_p, t_sp_from_p
from basic_mpc.control.bangbang import hysteresis_comfort_step
from basic_mpc.control.internal import p_to_hold, r2c2_internal_from_plant
from basic_mpc.control.mpc import mpc_euro_first_move
from basic_mpc.control.plots import plot_s4_confort, plot_s5_consigne, plot_s6_euros
from basic_mpc.control.tariffs import (
    beta_kwh_per_p_second,
    comfort_setpoint,
    energy_price,
    parse_start_local,
    scenario_index,
    winter_weather,
)
from basic_mpc.identification.pem import P0_SCALE
from basic_mpc.models.kalman import KalmanTracker
from basic_mpc.models.plant import ThermalPlant, literature_plant_params
from basic_mpc.models.r2c2 import discretize as discretize_r2

logger = logging.getLogger(__name__)


def euro_metrics(
    ta: np.ndarray,
    heating: np.ndarray,
    t_conf: np.ndarray,
    pi: np.ndarray,
    beta: float,
    dt_seconds: float,
    lambda_comfort: float,
) -> dict:
    """Facture proxy, kWh, heures sous T_conf, inconfort en euros.

    Les heures d'inconfort comptent un écart **supérieur à 0,1 °C**
    (résolution du capteur) : 0,05 °C sous T_conf n'est pas du froid.
    """
    dt_h = dt_seconds / 3600.0
    energy = beta * np.asarray(heating, dtype=float) * dt_seconds
    under = np.maximum(np.asarray(t_conf) - np.asarray(ta), 0.0)
    skip = int(round(2.0 / dt_h))
    # 0,1 °C : en dessous, le capteur ne distingue pas de T_conf.
    noticed = under > 0.1
    return {
        "bill_eur": float(np.dot(pi, energy)),
        "energy_kwh": float(np.sum(energy)),
        "hours_under_conf": float(np.sum(noticed) * dt_h),
        "hours_under_after_2h": float(np.sum(noticed[skip:]) * dt_h),
        "discomfort_eur": float(lambda_comfort * dt_h * np.dot(under, under)),
        "ta_mean": float(np.mean(ta)),
        "ta_min": float(np.min(ta)),
        "ta_max": float(np.max(ta)),
    }


def _simulate_hysteresis(
    plant: ThermalPlant,
    t_ext: np.ndarray,
    solar: np.ndarray,
    t_conf: np.ndarray,
    n_band: float,
    p_max: float,
) -> pd.DataFrame:
    """T_sp = T_conf(t), hystérésis sur y, pas d'anticipation."""
    n = len(t_ext)
    ta = np.empty(n)
    tm = np.empty(n)
    y = np.empty(n)
    heating = np.empty(n)
    t_sp = np.empty(n)
    y0 = plant.observe()
    on = y0 < t_conf[0]
    p, on = hysteresis_comfort_step(y0, on, float(t_conf[0]), n_band, p_max)
    for k in range(n):
        t_sp[k] = float(t_conf[k])
        y[k] = plant.step(float(t_ext[k]), float(solar[k]), p)
        ta[k] = plant.x[0]
        tm[k] = plant.x[1]
        heating[k] = p
        p, on = hysteresis_comfort_step(float(y[k]), on, float(t_conf[k]), n_band, p_max)
    return pd.DataFrame(
        {
            "t_ext": t_ext,
            "S": solar,
            "P": heating,
            "t_sp": t_sp,
            "ta_true": ta,
            "tm_true": tm,
            "y": y,
        }
    )


def _simulate_mpc_setpoint(
    plant: ThermalPlant,
    t_ext: np.ndarray,
    solar: np.ndarray,
    t_conf: np.ndarray,
    pi: np.ndarray,
    ad: np.ndarray,
    bd: np.ndarray,
    q: np.ndarray,
    r: np.ndarray,
    p_max: float,
    cfg: ControlConfig,
    dt_seconds: float,
    beta: float,
) -> pd.DataFrame:
    """Horizon glissant : Kalman + QP euros, T_sp tenu par bloc, bande n."""
    n = len(t_ext)
    n_pred = max(1, int(round(cfg.horizon_hours * 3600.0 / dt_seconds)))
    block_len = max(1, int(round(cfg.block_minutes * 60.0 / dt_seconds)))
    y0 = plant.observe()
    tracker = KalmanTracker(
        ad,
        bd,
        q,
        r,
        x0=np.array([y0, y0]),
        p0=np.diag([P0_SCALE, 4.0 * P0_SCALE]),
    )
    x_hat = tracker.step(y0, u_prev=None)
    ta = np.empty(n)
    tm = np.empty(n)
    y = np.empty(n)
    heating = np.empty(n)
    t_sp_log = np.empty(n)
    p_moves: np.ndarray | None = None
    u_prev: np.ndarray | None = None
    t_sp_hold = float(t_conf[0])
    p_hold = 0.0
    steps_left = 0

    for k in range(n):
        if steps_left <= 0:
            end = min(n, k + n_pred)
            t_fc = t_ext[k:end]
            s_fc = solar[k:end]
            conf_fc = t_conf[k:end]
            pi_fc = pi[k:end]
            if len(t_fc) < n_pred:
                pad = n_pred - len(t_fc)
                t_fc = np.pad(t_fc, (0, pad), mode="edge")
                s_fc = np.pad(s_fc, (0, pad), mode="edge")
                conf_fc = np.pad(conf_fc, (0, pad), mode="edge")
                pi_fc = np.pad(pi_fc, (0, pad), mode="edge")
            p_hold, p_moves = mpc_euro_first_move(
                x_hat,
                t_fc,
                s_fc,
                ad,
                bd,
                p_max,
                conf_fc,
                pi_fc,
                block_len,
                cfg.n_band,
                cfg.t_sp_min,
                cfg.t_sp_max,
                beta,
                cfg.lambda_comfort,
                dt_seconds,
                p_guess=p_moves,
            )
            steps_left = block_len
        # T_sp mis à jour chaque pas pour réaliser le P du bloc (sinon
        # l'air monte, la bande n coupe, et on n'atteint jamais T_conf).
        t_sp_now = t_sp_from_p(
            float(x_hat[0]),
            float(p_hold),
            cfg.n_band,
            p_max,
            cfg.t_sp_min,
            cfg.t_sp_max,
        )
        y_now = y0 if k == 0 else float(y[k - 1])
        p = proportional_band_p(t_sp_now, y_now, cfg.n_band, p_max)
        y[k] = plant.step(float(t_ext[k]), float(solar[k]), p)
        ta[k] = plant.x[0]
        tm[k] = plant.x[1]
        heating[k] = p
        t_sp_log[k] = t_sp_now
        u_prev = np.array([t_ext[k], solar[k], p], dtype=float)
        x_hat = tracker.step(float(y[k]), u_prev)
        steps_left -= 1
        if p_moves is not None and steps_left == 0 and p_moves.size > 1:
            p_moves = np.concatenate([p_moves[1:], p_moves[-1:]])
    return pd.DataFrame(
        {
            "t_ext": t_ext,
            "S": solar,
            "P": heating,
            "t_sp": t_sp_log,
            "ta_true": ta,
            "tm_true": tm,
            "y": y,
        }
    )


def run_mpc_cout_consigne(
    cfg: ControlConfig | None = None,
    pictures_dir: Path | None = None,
    experiments_dir: Path | None = None,
    processed_dir: Path | None = None,
) -> dict:
    """Compare MPC (T_sp, J €) et hystérésis sur 48 h, écrit S4–S6.

    Parameters
    ----------
    cfg : ControlConfig, optional
        Bande n, calendriers, λ.
    pictures_dir, experiments_dir, processed_dir : Path, optional
        Figures, ``runs.jsonl``, JSON de rapport.

    Returns
    -------
    dict
        Métriques euros et chemins.
    """
    cfg = cfg or ControlConfig()
    pictures_dir = pictures_dir or (REPO_ROOT / "pictures" / "experiments")
    experiments_dir = experiments_dir or (REPO_ROOT / "experiments")
    processed_dir = processed_dir or DataConfig().processed_dir
    pictures_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    plant_p = literature_plant_params()
    internal = r2c2_internal_from_plant(plant_p)
    ad, bd = discretize_r2(internal)
    q = np.diag([internal.process_noise_std**2, internal.process_noise_std_mass**2])
    r = np.array([[internal.sensor_noise_std**2]])
    beta = beta_kwh_per_p_second(plant_p.alpha_h)

    n_steps = int(cfg.n_hours * 3600.0 / plant_p.dt_seconds)
    index = scenario_index(parse_start_local(cfg), n_steps, plant_p.dt_seconds)
    weather = winter_weather(index, seed=cfg.seed + 1)
    t_ext = weather["t_ext"].to_numpy()
    solar = weather["S"].to_numpy()
    t_conf = comfort_setpoint(index, cfg)
    pi = energy_price(index, cfg)
    t_ext_min = float(np.min(t_ext))
    p_hold = p_to_hold(plant_p, cfg.t_conf_occupied, t_ext_min)
    p_max = cfg.p_max_margin * p_hold
    x0 = np.array([18.0, 16.0])

    plant_bb = ThermalPlant(params=plant_p, x0=x0, seed=cfg.seed)
    traj_bb = _simulate_hysteresis(
        plant_bb, t_ext, solar, t_conf, cfg.n_band, p_max
    )
    plant_mpc = ThermalPlant(params=plant_p, x0=x0, seed=cfg.seed)
    traj_mpc = _simulate_mpc_setpoint(
        plant_mpc,
        t_ext,
        solar,
        t_conf,
        pi,
        ad,
        bd,
        q,
        r,
        p_max,
        cfg,
        plant_p.dt_seconds,
        beta,
    )

    dt_hours = plant_p.dt_seconds / 3600.0
    hours = np.arange(n_steps) * dt_hours
    m_bb = euro_metrics(
        traj_bb["ta_true"].to_numpy(),
        traj_bb["P"].to_numpy(),
        t_conf,
        pi,
        beta,
        plant_p.dt_seconds,
        cfg.lambda_comfort,
    )
    m_mpc = euro_metrics(
        traj_mpc["ta_true"].to_numpy(),
        traj_mpc["P"].to_numpy(),
        t_conf,
        pi,
        beta,
        plant_p.dt_seconds,
        cfg.lambda_comfort,
    )

    plot_s4_confort(
        hours,
        traj_mpc["ta_true"].to_numpy(),
        traj_bb["ta_true"].to_numpy(),
        t_conf,
        pi,
        cfg.pi_hc,
        pictures_dir / "s4-mpc-vs-bang-bang.png",
    )
    plot_s5_consigne(
        hours,
        traj_mpc["t_sp"].to_numpy(),
        traj_bb["t_sp"].to_numpy(),
        traj_mpc["P"].to_numpy(),
        traj_bb["P"].to_numpy(),
        pictures_dir / "s5-commande-p.png",
    )
    plot_s6_euros(
        m_mpc["bill_eur"],
        m_bb["bill_eur"],
        m_mpc["hours_under_conf"],
        m_bb["hours_under_conf"],
        pictures_dir / "s6-confort-conso.png",
    )

    report = {
        "model": "mpc_cout_consigne",
        "internal": "r2c2 from plant without alpha_s_mass",
        "plant_alpha_s_mass": plant_p.alpha_s_mass,
        "p_max": p_max,
        "p_hold_at_t_ext_min": p_hold,
        "t_ext_min": t_ext_min,
        "horizon_hours": cfg.horizon_hours,
        "block_minutes": cfg.block_minutes,
        "n_band": cfg.n_band,
        "beta": beta,
        "lambda_comfort": cfg.lambda_comfort,
        "start_local": cfg.start_local,
        "comfort": {
            "occupied": cfg.t_conf_occupied,
            "setback": cfg.t_conf_setback,
        },
        "tariff": {"hp": cfg.pi_hp, "hc": cfg.pi_hc},
        "mpc": m_mpc,
        "hysteresis": m_bb,
        "notes": (
            "Commande = T_sp, bande n, J = facture HP/HC + inconfort. "
            "Plant littérature (α_s,mass). Prévisions oracle. "
            "β convertit le proxy P en kWh, pas un compteur."
        ),
    }
    json_path = processed_dir / "mpc_cout_consigne_report.json"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("écrit %s", json_path)
    experiments_dir.mkdir(parents=True, exist_ok=True)
    line = {
        "time": datetime.now(timezone.utc).isoformat(),
        "model": "mpc_cout_consigne",
        "mpc_bill_eur": m_mpc["bill_eur"],
        "bb_bill_eur": m_bb["bill_eur"],
        "mpc_hours_under": m_mpc["hours_under_conf"],
        "bb_hours_under": m_bb["hours_under_conf"],
    }
    with (experiments_dir / "runs.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(line, ensure_ascii=False) + "\n")
    report["report_path"] = str(json_path)
    return report
