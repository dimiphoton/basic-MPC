"""Génère les PNG de devoir (H1–H10) dans pictures/experiments/."""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from basic_mpc.config import REPO_ROOT, ControlConfig, DataConfig
from basic_mpc.control.cout import euro_closed_loop_pair
from basic_mpc.control.internal import r2c2_internal_from_plant
from basic_mpc.homework.plots import (
    plot_bode_module,
    plot_kalman_trou,
    plot_loi_bande_n,
    plot_mae_horizons,
    plot_oracle_vs_retard,
    plot_pareto_lambda,
    plot_qq_innovations,
    plot_quantification_plant,
    plot_reponse_indicielle,
    plot_split_chronologique,
)
from basic_mpc.identification.pem import P0_SCALE
from basic_mpc.models.impedance import omega_period_hours, z_normalized, z_r1c1, z_rc_network
from basic_mpc.models.kalman import run_kalman
from basic_mpc.models.plant import ThermalPlant, literature_plant_params
from basic_mpc.models.r2c2 import discretize as discretize_r2
from basic_mpc.sim.payload import default_fitted_if_missing

logger = logging.getLogger(__name__)

DELAY_HOURS = 6.0
LAMBDA_SWEEP = (0.25, 1.0, 4.0)
SWEEP_HOURS = 16.0


def _delay_series(values: np.ndarray, n_steps: int) -> np.ndarray:
    """Retard causal : début répété, pas de wrap-around du futur."""
    n = values.size
    lag = min(max(0, n_steps), n)
    if lag == 0:
        return values.copy()
    out = np.empty_like(values)
    out[:lag] = values[0]
    out[lag:] = values[:-lag]
    return out


def _step_response() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Échelon déterministe : extérieur 5 °C, départ à l'équilibre, P constant."""
    plant_p = replace(
        literature_plant_params(),
        process_noise_std=0.0,
        sensor_noise_std=0.0,
    )
    n = int(48.0 * 3600.0 / plant_p.dt_seconds)
    t_ext = np.full(n, 5.0)
    solar = np.zeros(n)
    heating = np.full(n, 80.0)
    plant = ThermalPlant(params=plant_p, x0=np.array([5.0, 5.0]), seed=0)
    traj = plant.simulate(t_ext, solar, heating)
    # Le simulateur enregistre l'état après le pas : on remet le départ à t = 0.
    hours = np.arange(n + 1) * plant_p.dt_seconds / 3600.0
    ta = np.concatenate([[5.0], traj["ta_true"].to_numpy()])
    tm = np.concatenate([[5.0], traj["tm_true"].to_numpy()])
    return hours, ta, tm


def _bode_curves() -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    plant = literature_plant_params()
    periods = np.logspace(np.log10(1.0), np.log10(7.0 * 24.0), 70)
    omegas = np.array([omega_period_hours(float(p)) for p in periods])
    z0 = z_rc_network(plant.rae, plant.ram, plant.ca, plant.cm, np.array([1e-12]))
    mag_p = np.abs(z_normalized(z_rc_network(plant.rae, plant.ram, plant.ca, plant.cm, omegas), z0))
    mag_r1 = mag_r2 = None
    try:
        params_r1, params_r2 = default_fitted_if_missing()
        mag_r1 = np.abs(z_normalized(z_r1c1(params_r1, omegas), z_r1c1(params_r1, np.array([1e-12]))))
        mag_r2 = np.abs(
            z_normalized(
                z_rc_network(params_r2.rae, params_r2.ram, params_r2.ca, params_r2.cm, omegas),
                z_rc_network(
                    params_r2.rae,
                    params_r2.ram,
                    params_r2.ca,
                    params_r2.cm,
                    np.array([1e-12]),
                ),
            )
        )
    except (OSError, KeyError, ValueError):
        logger.info("fits absents : Bode plant seulement")
    return periods, mag_p, mag_r1, mag_r2


def _kalman_on_plant() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Innovations + trou de 4 h au milieu."""
    plant_p = literature_plant_params()
    ad, bd = discretize_r2(r2c2_internal_from_plant(plant_p))
    # Filtre calé sur le modèle interne (sans α_s,mass) — misspec volontaire.
    n = int(18.0 * 3600.0 / plant_p.dt_seconds)
    t_ext = np.full(n, 4.0)
    solar = np.zeros(n)
    heating = np.full(n, 40.0)
    plant = ThermalPlant(params=plant_p, x0=np.array([17.0, 16.0]), seed=2)
    traj = plant.simulate(t_ext, solar, heating)
    y = traj["y"].to_numpy(dtype=float).copy()
    u = np.column_stack([t_ext, solar, heating])
    gap0 = int(8.0 * 3600.0 / plant_p.dt_seconds)
    gap1 = int(12.0 * 3600.0 / plant_p.dt_seconds)
    y[gap0:gap1] = np.nan
    q = np.diag([0.02**2, 0.02**2])
    r = np.array([[0.05**2]])
    y0 = float(traj["y"].iloc[0])
    result = run_kalman(
        y,
        u,
        ad,
        bd,
        np.array([[1.0, 0.0]]),
        q,
        r,
        x0=np.array([y0, y0]),
        p0=np.diag([P0_SCALE, 4.0 * P0_SCALE]),
    )
    hours = np.arange(n) * plant_p.dt_seconds / 3600.0
    return (
        hours,
        traj["tm_true"].to_numpy(),
        result.x_filt[:, 1],
        result.innov,
        8.0,
        12.0,
    )


def _mae_from_report(processed: Path) -> tuple[list[int], list[float], list[float]] | None:
    path = processed / "r1c1_r2c2_report.json"
    if not path.is_file():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    keys = ["1h", "3h", "6h", "12h", "24h"]
    hours = [1, 3, 6, 12, 24]
    r1 = [report["horizons_r1c1"][k]["mae"] for k in keys]
    r2 = [report["horizons_r2c2"][k]["mae"] for k in keys]
    return hours, r1, r2


def _split_from_report(processed: Path) -> tuple[str, str, str, str] | None:
    path = processed / "r1c1_report.json"
    if not path.is_file():
        return None
    split = json.loads(path.read_text(encoding="utf-8"))["split"]
    # Fenêtre PEM = ~50 derniers jours du train (JOURNAL).
    pem_start = "2020-12-16"
    return (
        str(split["train_start"]),
        pem_start,
        str(split["train_end"]),
        str(split["test_end"]),
    )


def run_homework_pngs(
    pictures_dir: Path | None = None,
    processed_dir: Path | None = None,
    experiments_dir: Path | None = None,
    include_sweeps: bool = True,
    n_hours_sweep: float = SWEEP_HOURS,
) -> dict:
    """Écrit H1–H10 (sweeps optionnels : Pareto λ et météo retardée).

    Parameters
    ----------
    pictures_dir, processed_dir, experiments_dir : Path, optional
        Sorties figures / JSON / ``runs.jsonl``.
    include_sweeps : bool
        False dans les tests unitaires (QP trop long).
    n_hours_sweep : float
        Durée des boucles fermées de sensibilité.

    Returns
    -------
    dict
        Chemins écrits et métriques des sweeps.
    """
    pictures_dir = pictures_dir or (REPO_ROOT / "pictures" / "experiments")
    processed_dir = processed_dir or DataConfig().processed_dir
    experiments_dir = experiments_dir or (REPO_ROOT / "experiments")
    pictures_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    plot_loi_bande_n(pictures_dir / "h1-loi-bande-n.png")
    written.append("h1-loi-bande-n.png")

    hours_s, ta, tm = _step_response()
    plot_reponse_indicielle(hours_s, ta, tm, pictures_dir / "h2-reponse-indicielle.png")
    written.append("h2-reponse-indicielle.png")

    periods, mag_p, mag_r1, mag_r2 = _bode_curves()
    plot_bode_module(periods, mag_p, mag_r1, mag_r2, pictures_dir / "h3-bode-module.png")
    written.append("h3-bode-module.png")

    h_k, tm_true, tm_hat, innov, g0, g1 = _kalman_on_plant()
    plot_qq_innovations(innov, pictures_dir / "h4-qq-innovations.png")
    plot_kalman_trou(h_k, tm_true, tm_hat, g0, g1, pictures_dir / "h5-kalman-trou.png")
    written.extend(["h4-qq-innovations.png", "h5-kalman-trou.png"])

    split = _split_from_report(processed_dir)
    if split is not None:
        plot_split_chronologique(*split, pictures_dir / "h6-split-chronologique.png")
        written.append("h6-split-chronologique.png")

    mae = _mae_from_report(processed_dir)
    if mae is not None:
        plot_mae_horizons(*mae, pictures_dir / "h7-mae-horizons.png")
        written.append("h7-mae-horizons.png")

    plant_p = literature_plant_params()
    n_q = int(24.0 * 3600.0 / plant_p.dt_seconds)
    weather_q = ThermalPlant(params=plant_p, x0=np.array([18.0, 17.0]), seed=3)
    w = np.full(n_q, 6.0)
    s = np.zeros(n_q)
    p = np.zeros(n_q)
    traj_q = weather_q.simulate(w, s, p)
    hq = np.arange(n_q) * plant_p.dt_seconds / 3600.0
    plot_quantification_plant(
        hq[:288],
        traj_q["ta_true"].to_numpy()[:288],
        traj_q["y"].to_numpy()[:288],
        pictures_dir / "h10-quantification.png",
    )
    written.append("h10-quantification.png")

    sweep_report: dict = {}
    if include_sweeps:
        lambdas = np.array(LAMBDA_SWEEP, dtype=float)
        bills = []
        hours_u = []
        bill_bb = hours_bb = 0.0
        for lam in lambdas:
            cfg = replace(ControlConfig(), n_hours=n_hours_sweep, lambda_comfort=float(lam))
            pair = euro_closed_loop_pair(cfg)
            bills.append(pair["mpc"]["bill_eur"])
            hours_u.append(pair["mpc"]["hours_under_conf"])
            bill_bb = pair["hysteresis"]["bill_eur"]
            hours_bb = pair["hysteresis"]["hours_under_conf"]
        plot_pareto_lambda(
            lambdas,
            np.array(bills),
            np.array(hours_u),
            bill_bb,
            hours_bb,
            pictures_dir / "h8-pareto-lambda.png",
        )
        written.append("h8-pareto-lambda.png")
        cfg_d = replace(ControlConfig(), n_hours=n_hours_sweep)
        pair_o = euro_closed_loop_pair(cfg_d)
        lag = int(round(DELAY_HOURS * 3600.0 / pair_o["dt_seconds"]))
        delayed = _delay_series(pair_o["t_ext"], lag)
        pair_d = euro_closed_loop_pair(cfg_d, t_ext_forecast=delayed)
        plot_oracle_vs_retard(
            pair_o["hours"],
            pair_o["traj_mpc"]["ta_true"].to_numpy(),
            pair_d["traj_mpc"]["ta_true"].to_numpy(),
            pair_o["t_conf"],
            pictures_dir / "h9-oracle-vs-retard.png",
        )
        written.append("h9-oracle-vs-retard.png")
        sweep_report = {
            "lambdas": lambdas.tolist(),
            "mpc_bill": bills,
            "mpc_hours_under": hours_u,
            "hysteresis_bill": bill_bb,
            "hysteresis_hours_under": hours_bb,
            "delay_hours": DELAY_HOURS,
            "oracle_hours_under": pair_o["mpc"]["hours_under_conf"],
            "delayed_hours_under": pair_d["mpc"]["hours_under_conf"],
        }

    report = {"figures": written, "sweeps": sweep_report}
    experiments_dir.mkdir(parents=True, exist_ok=True)
    line = {
        "time": datetime.now(timezone.utc).isoformat(),
        "model": "homework_pngs",
        "n_figures": len(written),
    }
    with (experiments_dir / "runs.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(line, ensure_ascii=False) + "\n")
    logger.info("écrit %s figures", len(written))
    return report
