"""Boucle fermée : même plant, même météo, MPC vs bang-bang."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from basic_mpc.config import REPO_ROOT, ControlConfig, DataConfig
from basic_mpc.control.bangbang import bangbang_step
from basic_mpc.control.internal import p_to_hold, r2c2_internal_from_plant
from basic_mpc.control.mpc import mpc_first_move
from basic_mpc.control.plots import plot_s4_tair, plot_s5_commande, plot_s6_score
from basic_mpc.identification.pem import P0_SCALE
from basic_mpc.models.kalman import KalmanTracker
from basic_mpc.models.plant import ThermalPlant, literature_plant_params, synthetic_weather
from basic_mpc.models.r2c2 import discretize as discretize_r2

logger = logging.getLogger(__name__)


def _metrics(
    ta: np.ndarray,
    heating: np.ndarray,
    t_min: float,
    t_max: float,
    dt_hours: float,
) -> dict:
    """Heures hors bande et P cumulé (proxy conso)."""
    hors = (ta < t_min) | (ta > t_max)
    skip = int(round(2.0 / dt_hours))
    return {
        "hours_outside_band": float(hors.sum() * dt_hours),
        "hours_outside_after_2h": float(hors[skip:].sum() * dt_hours),
        "P_hours": float(np.sum(heating) * dt_hours),
        "ta_mean": float(np.mean(ta)),
        "ta_min": float(np.min(ta)),
        "ta_max": float(np.max(ta)),
    }


def _simulate_bangbang(
    plant: ThermalPlant,
    t_ext: np.ndarray,
    solar: np.ndarray,
    p_max: float,
    t_min: float,
    t_max: float,
    t_set: float,
) -> pd.DataFrame:
    """Thermostat : décide sur y, applique 0 ou P_max."""
    n = len(t_ext)
    ta = np.empty(n)
    tm = np.empty(n)
    y = np.empty(n)
    heating = np.empty(n)
    y0 = plant.observe()
    on = y0 < t_set
    p, on = bangbang_step(y0, on, t_min, t_max, p_max)
    for k in range(n):
        y[k] = plant.step(float(t_ext[k]), float(solar[k]), p)
        ta[k] = plant.x[0]
        tm[k] = plant.x[1]
        heating[k] = p
        p, on = bangbang_step(float(y[k]), on, t_min, t_max, p_max)
    return pd.DataFrame(
        {"t_ext": t_ext, "S": solar, "P": heating, "ta_true": ta, "tm_true": tm, "y": y}
    )


def _simulate_mpc(
    plant: ThermalPlant,
    t_ext: np.ndarray,
    solar: np.ndarray,
    ad: np.ndarray,
    bd: np.ndarray,
    q: np.ndarray,
    r: np.ndarray,
    p_max: float,
    cfg: ControlConfig,
    dt_seconds: float,
) -> pd.DataFrame:
    """Horizon glissant : Kalman + QP, météo future connue (oracle)."""
    n = len(t_ext)
    n_pred = max(1, int(round(cfg.horizon_hours * 3600.0 / dt_seconds)))
    block_len = max(1, int(round(cfg.block_minutes * 60.0 / dt_seconds)))
    r_u = cfg.r_rel / (p_max * p_max) if p_max > 0.0 else cfg.r_rel
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
    p_moves: np.ndarray | None = None
    u_prev: np.ndarray | None = None
    p_hold = 0.0
    steps_left = 0

    for k in range(n):
        if steps_left <= 0:
            end = min(n, k + n_pred)
            t_fc = t_ext[k:end]
            s_fc = solar[k:end]
            if len(t_fc) < n_pred:
                t_fc = np.pad(t_fc, (0, n_pred - len(t_fc)), mode="edge")
                s_fc = np.pad(s_fc, (0, n_pred - len(s_fc)), mode="edge")
            p_hold, p_moves = mpc_first_move(
                x_hat,
                t_fc,
                s_fc,
                ad,
                bd,
                p_max,
                cfg.t_set,
                cfg.t_min,
                cfg.t_max,
                block_len,
                cfg.q_track,
                cfg.q_band,
                r_u,
                p_guess=p_moves,
            )
            steps_left = block_len
        p = float(np.clip(p_hold, 0.0, p_max))
        y[k] = plant.step(float(t_ext[k]), float(solar[k]), p)
        ta[k] = plant.x[0]
        tm[k] = plant.x[1]
        heating[k] = p
        u_prev = np.array([t_ext[k], solar[k], p], dtype=float)
        x_hat = tracker.step(float(y[k]), u_prev)
        steps_left -= 1
        if p_moves is not None and steps_left == 0 and p_moves.size > 1:
            # Warm start : décale les mouvements d'un bloc
            p_moves = np.concatenate([p_moves[1:], p_moves[-1:]])
    return pd.DataFrame(
        {"t_ext": t_ext, "S": solar, "P": heating, "ta_true": ta, "tm_true": tm, "y": y}
    )


def run_mpc_vs_bangbang(
    cfg: ControlConfig | None = None,
    pictures_dir: Path | None = None,
    experiments_dir: Path | None = None,
    processed_dir: Path | None = None,
) -> dict:
    """Compare les deux contrôleurs sur 48 h de plant, écrit S4–S6.

    Parameters
    ----------
    cfg : ControlConfig, optional
        Bande, horizon, poids.
    pictures_dir, experiments_dir, processed_dir : Path, optional
        Figures, ``runs.jsonl``, JSON de rapport.

    Returns
    -------
    dict
        Métriques MPC / bang-bang et chemins.
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

    n_steps = int(cfg.n_hours * 3600.0 / plant_p.dt_seconds)
    weather = synthetic_weather(n_steps, plant_p.dt_seconds, seed=cfg.seed + 1)
    t_ext = weather["t_ext"].to_numpy()
    solar = weather["S"].to_numpy()
    t_ext_min = float(np.min(t_ext))
    p_hold = p_to_hold(plant_p, cfg.t_set, t_ext_min)
    p_max = cfg.p_max_margin * p_hold
    x0 = np.array([18.0, 16.0])

    plant_bb = ThermalPlant(params=plant_p, x0=x0, seed=cfg.seed)
    traj_bb = _simulate_bangbang(
        plant_bb, t_ext, solar, p_max, cfg.t_min, cfg.t_max, cfg.t_set
    )
    plant_mpc = ThermalPlant(params=plant_p, x0=x0, seed=cfg.seed)
    traj_mpc = _simulate_mpc(
        plant_mpc, t_ext, solar, ad, bd, q, r, p_max, cfg, plant_p.dt_seconds
    )

    dt_hours = plant_p.dt_seconds / 3600.0
    hours = np.arange(n_steps) * dt_hours
    m_bb = _metrics(traj_bb["ta_true"].to_numpy(), traj_bb["P"].to_numpy(), cfg.t_min, cfg.t_max, dt_hours)
    m_mpc = _metrics(
        traj_mpc["ta_true"].to_numpy(), traj_mpc["P"].to_numpy(), cfg.t_min, cfg.t_max, dt_hours
    )

    plot_s4_tair(
        hours,
        traj_mpc["ta_true"].to_numpy(),
        traj_bb["ta_true"].to_numpy(),
        cfg.t_min,
        cfg.t_max,
        pictures_dir / "s4-mpc-vs-bang-bang.png",
    )
    plot_s5_commande(
        hours,
        traj_mpc["P"].to_numpy(),
        traj_bb["P"].to_numpy(),
        pictures_dir / "s5-commande-p.png",
    )
    plot_s6_score(
        m_mpc["P_hours"],
        m_bb["P_hours"],
        m_mpc["hours_outside_band"],
        m_bb["hours_outside_band"],
        pictures_dir / "s6-confort-conso.png",
    )

    report = {
        "model": "mpc_vs_bangbang",
        "internal": "r2c2 from plant without alpha_s_mass",
        "plant_alpha_s_mass": plant_p.alpha_s_mass,
        "p_max": p_max,
        "p_hold_at_t_ext_min": p_hold,
        "t_ext_min": t_ext_min,
        "horizon_hours": cfg.horizon_hours,
        "block_minutes": cfg.block_minutes,
        "comfort": {"t_set": cfg.t_set, "t_min": cfg.t_min, "t_max": cfg.t_max},
        "mpc": m_mpc,
        "bangbang": m_bb,
        "notes": (
            "Boucle fermée sur le plant littérature (solaire aussi sur la "
            "masse). Modèle interne R2C2 sans alpha_s_mass. Prévisions "
            "météo parfaites (oracle). P n'est pas des watts."
        ),
    }
    json_path = processed_dir / "mpc_vs_bangbang_report.json"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("écrit %s", json_path)
    experiments_dir.mkdir(parents=True, exist_ok=True)
    line = {
        "time": datetime.now(timezone.utc).isoformat(),
        "model": "mpc_vs_bangbang",
        "mpc_hours_outside": m_mpc["hours_outside_band"],
        "bb_hours_outside": m_bb["hours_outside_band"],
        "mpc_P_hours": m_mpc["P_hours"],
        "bb_P_hours": m_bb["P_hours"],
    }
    with (experiments_dir / "runs.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(line, ensure_ascii=False) + "\n")
    report["report_path"] = str(json_path)
    return report
