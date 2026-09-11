"""Impédance des fits et constantes du plant pour la page visiteur."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from basic_mpc.config import DataConfig, REPO_ROOT
from basic_mpc.models.impedance import (
    omega_period_hours,
    z_normalized,
    z_r1c1,
    z_r2c2,
    z_snapshot,
)
from basic_mpc.models.plant import discretize as discretize_plant
from basic_mpc.models.plant import literature_plant_params
from basic_mpc.models.r1c1 import R1C1Params
from basic_mpc.models.r2c2 import R2C2Params


def params_from_compare_report(report: dict) -> tuple[R1C1Params, R2C2Params]:
    """Reconstruit les params fittés depuis ``r1c1_r2c2_report.json``."""
    r1 = report["params_r1c1"]
    r2 = report["params_r2c2"]
    return (
        R1C1Params(a=r1["a"], g_solar=r1["g_solar"], g_heating=r1["g_heating"]),
        R2C2Params(
            rae=r2["rae"],
            ram=r2["ram"],
            cm=r2["cm"],
            g_solar=r2["g_solar"],
            g_heating=r2["g_heating"],
        ),
    )


def load_fitted_params(processed_dir: Path | None = None) -> tuple[R1C1Params, R2C2Params]:
    """Charge le rapport d'identification s'il existe."""
    processed_dir = processed_dir or DataConfig().processed_dir
    path = processed_dir / "r1c1_r2c2_report.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    return params_from_compare_report(report)


def vector_at_24h(params_r1: R1C1Params, params_r2: R2C2Params) -> dict:
    """Vecteurs Z(jω_24h)/Z(0) des deux fits."""
    w = omega_period_hours(24.0)
    w0 = np.array([1e-12])
    z1n = z_normalized(z_r1c1(params_r1, np.array([w])), z_r1c1(params_r1, w0))
    z2n = z_normalized(z_r2c2(params_r2, np.array([w])), z_r2c2(params_r2, w0))
    s1 = z_snapshot(z1n, 24.0)
    s2 = z_snapshot(z2n, 24.0)
    s1["tau_hours"] = params_r1.tau_hours
    s2["tau_air_hours"] = params_r2.tau_air_hours
    s2["tau_mass_hours"] = params_r2.tau_mass_hours
    return {"period_hours": 24.0, "r1c1": s1, "r2c2": s2}


def bode_curve(params_r1: R1C1Params, params_r2: R2C2Params, n: int = 40) -> dict:
    """Phase vs période (h) pour le graphe déphasage."""
    periods = np.logspace(np.log10(2.0), np.log10(7.0 * 24.0), n)
    omegas = np.array([omega_period_hours(float(p)) for p in periods])
    z1 = z_normalized(z_r1c1(params_r1, omegas), z_r1c1(params_r1, np.array([1e-12])))
    z2 = z_normalized(z_r2c2(params_r2, omegas), z_r2c2(params_r2, np.array([1e-12])))
    return {
        "period_hours": periods.tolist(),
        "phase_r1_deg": np.degrees(np.angle(z1)).tolist(),
        "phase_r2_deg": np.degrees(np.angle(z2)).tolist(),
    }


def plant_for_js() -> dict:
    """Matrices et RC du plant littérature, pour le labo JS."""
    plant = literature_plant_params()
    ad, bd = discretize_plant(plant)
    return {
        "dt_seconds": plant.dt_seconds,
        "ca": plant.ca,
        "cm": plant.cm,
        "ram": plant.ram,
        "rae": plant.rae,
        "alpha_h": plant.alpha_h,
        "alpha_s_air": plant.alpha_s_air,
        "alpha_s_mass": plant.alpha_s_mass,
        "tau_air_hours": plant.rae * plant.ca / 3600.0,
        "tau_mass_hours": plant.ram * plant.cm / 3600.0,
        "ad": ad.tolist(),
        "bd": bd.tolist(),
        "n_lab_days": 5.0,
        "discard_hours": 24.0,
    }


def default_fitted_if_missing() -> tuple[R1C1Params, R2C2Params]:
    """Valeurs du rapport salon si le JSON n'est pas là (CI minimale)."""
    fallback = REPO_ROOT / "data" / "processed" / "r1c1_r2c2_report.json"
    if fallback.is_file():
        return load_fitted_params(fallback.parent)
    return (
        R1C1Params(a=0.999, g_solar=0.0, g_heating=0.003),
        R2C2Params(rae=5e5, ram=1.25e5, cm=4.27, g_solar=0.0, g_heating=1e-5),
    )
