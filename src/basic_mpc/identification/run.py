"""Pipeline d'identification R1C1 sur les séries 5 min."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from basic_mpc.config import REPO_ROOT, DataConfig
from basic_mpc.features.inputs import run_build_inputs
from basic_mpc.identification.figures import (
    plot_innovations,
    plot_params,
    plot_tair_vs_y,
)
from basic_mpc.identification.metrics import multi_horizon_scores
from basic_mpc.identification.pem import SENSOR_NOISE_STD, filter_r1c1, fit_r1c1
from basic_mpc.models.plant import ThermalPlant, literature_plant_params, synthetic_weather
from basic_mpc.models.r1c1 import R1C1Params, discretize

logger = logging.getLogger(__name__)

TRAIN_FRACTION = 0.70
HORIZON_STRIDE = 12
# Derniers ~50 jours du train (chauffage) : assez pour le PEM, Python reste raisonnable.
PEM_MAX_STEPS = 14400


def _arrays_from_table(table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """y salon et u = [T_ext, S, P]."""
    y = table["livingroom_y"].to_numpy(dtype=float)
    u = np.column_stack(
        [
            table["outdoor_y"].to_numpy(dtype=float),
            table["S"].to_numpy(dtype=float),
            table["P"].to_numpy(dtype=float),
        ]
    )
    return y, u


def temporal_split(
    table: pd.DataFrame,
    train_fraction: float = TRAIN_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Coupe chronologique sur la maille (pas de mélange)."""
    n = len(table)
    cut = max(1, min(n - 1, int(n * train_fraction)))
    return table.iloc[:cut], table.iloc[cut:]


def run_identify_r1c1(
    config: DataConfig | None = None,
    pictures_dir: Path | None = None,
    experiments_dir: Path | None = None,
) -> dict:
    """Identifie le R1C1, écrit le JSON, les figures et une ligne d'expérience.

    Parameters
    ----------
    config : DataConfig, optional
        Chemins données.
    pictures_dir : Path, optional
        PNG du catalogue. Défaut ``pictures/experiments``.
    experiments_dir : Path, optional
        ``runs.jsonl``. Défaut ``experiments/``.

    Returns
    -------
    dict
        Rapport (params, RMSE multi-horizon, chemins).
    """
    config = config or DataConfig()
    pictures_dir = pictures_dir or (REPO_ROOT / "pictures" / "experiments")
    experiments_dir = experiments_dir or (REPO_ROOT / "experiments")

    ident_path = config.processed_dir / "identification_5min.csv"
    if not ident_path.is_file():
        run_build_inputs(config)

    table = pd.read_csv(ident_path, index_col="time", parse_dates=True)
    train, test = temporal_split(table)
    fit_slice = train.iloc[-min(PEM_MAX_STEPS, len(train)) :]
    logger.info(
        "PEM sur %s → %s (%s pas) ; test %s → %s",
        fit_slice.index[0],
        fit_slice.index[-1],
        len(fit_slice),
        test.index[0],
        test.index[-1],
    )
    y_fit, u_fit = _arrays_from_table(fit_slice)
    y_test, u_test = _arrays_from_table(test)

    params, fit_info = fit_r1c1(y_fit, u_fit)
    logger.info(
        "R1C1 : a=%.4f tau=%.2f h g_S=%.3g g_P=%.3g nll=%.1f",
        params.a,
        params.tau_hours,
        params.g_solar,
        params.g_heating,
        fit_info["nll"],
    )

    filtered_test = filter_r1c1(y_test, u_test, params)
    ad, bd = discretize(params)
    c = np.array([[1.0]])
    horizons = multi_horizon_scores(
        y_test,
        u_test,
        filtered_test.x_filt,
        ad,
        bd,
        c,
        params.dt_seconds,
        stride=HORIZON_STRIDE,
    )

    pictures_dir.mkdir(parents=True, exist_ok=True)
    i3_path = pictures_dir / "i3-innovations-r1c1.png"
    i4_path = pictures_dir / "i4-params-r1c1.png"
    s1_path = pictures_dir / "s1-tair-vs-y.png"
    plot_innovations(filtered_test.innov, i3_path)
    plot_params(params, fit_info.get("stderr"), i4_path)

    plant_params = literature_plant_params()
    n_day = int(24 * 3600 / plant_params.dt_seconds)
    weather = synthetic_weather(n_day, plant_params.dt_seconds, seed=1)
    plant = ThermalPlant(
        params=plant_params,
        x0=np.array([18.0, 16.0]),
        seed=0,
    )
    traj = plant.simulate(
        weather["t_ext"].to_numpy(),
        weather["S"].to_numpy(),
        weather["P"].to_numpy(),
    )
    plot_tair_vs_y(traj["ta_true"].to_numpy(), traj["y"].to_numpy(), s1_path)

    t_start = table.index[0]
    t_cut = train.index[-1]
    t_end = table.index[-1]
    report = {
        "model": "r1c1",
        "split": {
            "train_fraction": TRAIN_FRACTION,
            "train_start": str(t_start),
            "train_end": str(t_cut),
            "test_start": str(test.index[0]),
            "test_end": str(t_end),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "n_pem": int(len(fit_slice)),
            "pem_start": str(fit_slice.index[0]),
            "pem_end": str(fit_slice.index[-1]),
        },
        "params": {
            "a": params.a,
            "tau_hours": params.tau_hours,
            "g_solar": params.g_solar,
            "g_heating": params.g_heating,
            "process_noise_std": params.process_noise_std,
            "sensor_noise_std": params.sensor_noise_std,
            "dt_seconds": params.dt_seconds,
        },
        "fit": fit_info,
        "horizons_test": horizons,
        "innovations_test": {
            "n": filtered_test.n_obs,
            "mean": float(np.nanmean(filtered_test.innov)),
            "std": float(np.nanstd(filtered_test.innov)),
        },
        "figures": {
            "i3": "pictures/experiments/i3-innovations-r1c1.png",
            "i4": "pictures/experiments/i4-params-r1c1.png",
            "s1": "pictures/experiments/s1-tair-vs-y.png",
        },
        "notes": (
            "R1C1 baseline : un état air, T_ext avec gain 1-a. "
            "Bruit de capteur fixé à "
            f"{SENSOR_NOISE_STD} °C. "
            "PEM sur les derniers "
            f"{PEM_MAX_STEPS} pas du train (saison froide), "
            "métriques sur tout le test. "
            "S2 (masse) n'a pas de sens ici — pas d'état caché."
        ),
    }
    # hess_inv n'est pas JSON-natif si on a laissé un objet ; stderr déjà dict
    json_path = config.processed_dir / "r1c1_report.json"
    config.processed_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    logger.info("écrit %s", json_path)

    experiments_dir.mkdir(parents=True, exist_ok=True)
    run_line = {
        "time": datetime.now(timezone.utc).isoformat(),
        "model": "r1c1",
        "tau_hours": params.tau_hours,
        "nll_train": fit_info["nll"],
        "rmse_1h": horizons.get("1h", {}).get("rmse"),
        "rmse_24h": horizons.get("24h", {}).get("rmse"),
    }
    with (experiments_dir / "runs.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(run_line, ensure_ascii=False) + "\n")

    report["report_path"] = str(json_path)
    return report


def params_from_report(report: dict) -> R1C1Params:
    """Recharge les paramètres d'un JSON d'identification."""
    bloc = report["params"]
    return R1C1Params(
        a=float(bloc["a"]),
        g_solar=float(bloc["g_solar"]),
        g_heating=float(bloc["g_heating"]),
        dt_seconds=float(bloc["dt_seconds"]),
        process_noise_std=float(bloc["process_noise_std"]),
        sensor_noise_std=float(bloc["sensor_noise_std"]),
    )
