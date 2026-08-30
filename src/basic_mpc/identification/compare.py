"""Comparaison R1C1 vs R2C2 : même split, mêmes horizons, schémas."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from basic_mpc.config import REPO_ROOT, DataConfig
from basic_mpc.features.inputs import run_build_inputs
from basic_mpc.figures.schemas import run_draw_schemas
from basic_mpc.identification.figures import plot_innovations
from basic_mpc.identification.metrics import multi_horizon_scores
from basic_mpc.identification.pem import filter_r1c1, fit_r1c1
from basic_mpc.identification.pem_r2c2 import filter_r2c2, fit_r2c2
from basic_mpc.identification.plots_compare import (
    plot_48h_overlay,
    plot_mass_on_plant,
    plot_nyquist,
    plot_params_r2c2,
    plot_phase_portrait,
    plot_plant_vs_identified,
    plot_rmse_horizons,
    plot_z1_vectors,
)
from basic_mpc.identification.run import (
    HORIZON_STRIDE,
    PEM_MAX_STEPS,
    _arrays_from_table,
    params_from_report,
    temporal_split,
)
from basic_mpc.models.plant import ThermalPlant, literature_plant_params, synthetic_weather
from basic_mpc.models.r1c1 import discretize as discretize_r1
from basic_mpc.models.r2c2 import R2C2Params, discretize as discretize_r2, simulate_r2c2

logger = logging.getLogger(__name__)

STEPS_48H = 576  # 48 h × 12 pas/h


def _one_step_yhat(y: np.ndarray, innov: np.ndarray) -> np.ndarray:
    """ŷ_{k|k-1} = y - innovation."""
    return np.asarray(y, dtype=float) - np.asarray(innov, dtype=float)


def _slice_48h(y: np.ndarray, y1: np.ndarray, y2: np.ndarray, start: int) -> tuple:
    end = min(start + STEPS_48H, len(y))
    hours = np.arange(end - start) * 5.0 / 60.0
    return hours, y[start:end], y1[start:end], y2[start:end]


def run_compare_r1c1_r2c2(
    config: DataConfig | None = None,
    pictures_dir: Path | None = None,
    experiments_dir: Path | None = None,
) -> dict:
    """Fit R2C2, compare au R1C1, écrit schémas + figures catalogue.

    Parameters
    ----------
    config : DataConfig, optional
        Chemins.
    pictures_dir, experiments_dir : Path, optional
        Images et ``runs.jsonl``.

    Returns
    -------
    dict
        Rapport de comparaison.
    """
    config = config or DataConfig()
    pictures_dir = pictures_dir or (REPO_ROOT / "pictures" / "experiments")
    experiments_dir = experiments_dir or (REPO_ROOT / "experiments")
    pictures_dir.mkdir(parents=True, exist_ok=True)

    schemas = run_draw_schemas(pictures_dir)
    logger.info("schémas : %s", list(schemas))

    ident_path = config.processed_dir / "identification_5min.csv"
    if not ident_path.is_file():
        run_build_inputs(config)
    table = pd.read_csv(ident_path, index_col="time", parse_dates=True)
    train, test = temporal_split(table)
    fit_slice = train.iloc[-min(PEM_MAX_STEPS, len(train)) :]
    y_fit, u_fit = _arrays_from_table(fit_slice)
    y_test, u_test = _arrays_from_table(test)

    r1_path = config.processed_dir / "r1c1_report.json"
    if r1_path.is_file():
        r1_report = json.loads(r1_path.read_text(encoding="utf-8"))
        params_r1 = params_from_report(r1_report)
        fit_r1 = {"loaded": True, "nll": r1_report.get("fit", {}).get("nll")}
        logger.info("R1C1 relu depuis %s (τ=%.1f h)", r1_path, params_r1.tau_hours)
    else:
        params_r1, fit_r1 = fit_r1c1(y_fit, u_fit)
        fit_r1["loaded"] = False

    params_r2, fit_r2 = fit_r2c2(y_fit, u_fit)
    logger.info(
        "R2C2 : τ_air=%.2f h τ_masse=%.2f h Cm=%.2f nll=%.1f",
        params_r2.tau_air_hours,
        params_r2.tau_mass_hours,
        params_r2.cm,
        fit_r2["nll"],
    )

    filt1 = filter_r1c1(y_test, u_test, params_r1)
    filt2 = filter_r2c2(y_test, u_test, params_r2)
    ad1, bd1 = discretize_r1(params_r1)
    ad2, bd2 = discretize_r2(params_r2)
    scores1 = multi_horizon_scores(
        y_test, u_test, filt1.x_filt, ad1, bd1, np.array([[1.0]]), 300.0, stride=HORIZON_STRIDE
    )
    scores2 = multi_horizon_scores(
        y_test,
        u_test,
        filt2.x_filt,
        ad2,
        bd2,
        np.array([[1.0, 0.0]]),
        300.0,
        stride=HORIZON_STRIDE,
    )

    yhat1 = _one_step_yhat(y_test, filt1.innov)
    yhat2 = _one_step_yhat(y_test, filt2.innov)

    plot_rmse_horizons(scores1, scores2, pictures_dir / "i1-rmse-horizons.png")
    h, yw, a, b = _slice_48h(y_test, yhat1, yhat2, 0)
    plot_48h_overlay(
        h, yw, a, b, pictures_dir / "i2-48h-hiver.png", "48 h en début de test (février)"
    )
    solar = test["S"].to_numpy(dtype=float)
    if np.isfinite(solar).any() and len(solar) > STEPS_48H:
        start_sun = int(np.nanargmax(solar))
        start_sun = max(0, min(start_sun - STEPS_48H // 4, len(y_test) - STEPS_48H))
    else:
        start_sun = max(0, len(y_test) // 2)
    h, yw, a, b = _slice_48h(y_test, yhat1, yhat2, start_sun)
    plot_48h_overlay(
        h, yw, a, b, pictures_dir / "i2-48h-soleil.png", "48 h autour d'un pic PV (test)"
    )
    plot_innovations(filt2.innov, pictures_dir / "i3-innovations-r2c2.png")
    plot_params_r2c2(params_r2, pictures_dir / "i4-params-r2c2.png")
    mask = np.isfinite(filt2.x_filt).all(axis=1)
    plot_phase_portrait(
        filt2.x_filt[mask, 0][::12],
        filt2.x_filt[mask, 1][::12],
        pictures_dir / "i5-phase-air-masse.png",
    )
    plot_z1_vectors(params_r1, params_r2, pictures_dir / "z1-impedance-24h.png")
    plot_nyquist(params_r1, params_r2, pictures_dir / "z2-nyquist.png")

    plant_p = literature_plant_params()
    n_day = int(48 * 3600 / plant_p.dt_seconds)
    weather = synthetic_weather(n_day, plant_p.dt_seconds, seed=1)
    plant = ThermalPlant(params=plant_p, x0=np.array([18.0, 16.0]), seed=0)
    traj = plant.simulate(
        weather["t_ext"].to_numpy(),
        weather["S"].to_numpy(),
        weather["P"].to_numpy(),
    )
    u_p = np.column_stack(
        [traj["t_ext"].to_numpy(), traj["S"].to_numpy(), traj["P"].to_numpy()]
    )
    filt_plant = filter_r2c2(traj["y"].to_numpy(), u_p, params_r2)
    hours_p = np.arange(len(traj)) * 5.0 / 60.0
    plot_mass_on_plant(
        hours_p,
        traj["tm_true"].to_numpy(),
        filt_plant.x_filt[:, 1],
        pictures_dir / "s2-masse-kalman-plant.png",
    )
    silent = R2C2Params(
        rae=params_r2.rae,
        ram=params_r2.ram,
        cm=params_r2.cm,
        g_solar=params_r2.g_solar,
        g_heating=params_r2.g_heating,
        dt_seconds=params_r2.dt_seconds,
        process_noise_std=0.0,
        process_noise_std_mass=0.0,
        sensor_noise_std=0.0,
    )
    x_open, y_open = simulate_r2c2(
        silent,
        traj["t_ext"].to_numpy(),
        traj["S"].to_numpy(),
        traj["P"].to_numpy(),
        x0=np.array([18.0, 16.0]),
        seed=0,
    )
    plot_plant_vs_identified(
        hours_p,
        traj["y"].to_numpy(),
        y_open,
        pictures_dir / "s3-plant-vs-r2c2.png",
    )

    report = {
        "model": "r1c1_vs_r2c2",
        "params_r1c1": {
            "a": params_r1.a,
            "tau_hours": params_r1.tau_hours,
            "g_solar": params_r1.g_solar,
            "g_heating": params_r1.g_heating,
        },
        "params_r2c2": {
            "rae": params_r2.rae,
            "ram": params_r2.ram,
            "cm": params_r2.cm,
            "tau_air_hours": params_r2.tau_air_hours,
            "tau_mass_hours": params_r2.tau_mass_hours,
            "g_solar": params_r2.g_solar,
            "g_heating": params_r2.g_heating,
            "process_noise_std": params_r2.process_noise_std,
        },
        "fit_r1c1": fit_r1,
        "fit_r2c2": fit_r2,
        "horizons_r1c1": scores1,
        "horizons_r2c2": scores2,
        "schemas": {k: f"pictures/experiments/{k}.png" for k in schemas},
        "notes": (
            "Même fenêtre PEM et même test que le R1C1. "
            "C_a = 1. Solaire identifié sur l'air seulement. "
            "Plant : alpha_s_mass > 0 (S3)."
        ),
    }
    json_path = config.processed_dir / "r1c1_r2c2_report.json"
    config.processed_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    logger.info("écrit %s", json_path)
    experiments_dir.mkdir(parents=True, exist_ok=True)
    line = {
        "time": datetime.now(timezone.utc).isoformat(),
        "model": "r2c2",
        "tau_air_hours": params_r2.tau_air_hours,
        "tau_mass_hours": params_r2.tau_mass_hours,
        "nll_train": fit_r2["nll"],
        "rmse_1h": scores2.get("1h", {}).get("rmse"),
        "rmse_24h": scores2.get("24h", {}).get("rmse"),
        "rmse_24h_r1c1": scores1.get("24h", {}).get("rmse"),
    }
    with (experiments_dir / "runs.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(line, ensure_ascii=False) + "\n")
    report["report_path"] = str(json_path)
    return report
