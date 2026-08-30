"""Tests du plant : distinct du modèle identifié, capteur ≠ état."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from basic_mpc.data.sensors import quantize_measurement
from basic_mpc.models.plant import (
    ThermalPlant,
    literature_plant_params,
    run_simulate_plant,
)


def test_literature_plant_a_du_solaire_sur_la_masse() -> None:
    """C'est ça qui distingue le plant du R2C2 d'identification."""
    params = literature_plant_params()
    assert params.alpha_s_mass > 0.0


def test_quantize_grille_dixieme() -> None:
    """Le capteur n'écrit pas 20.04 °C."""
    assert quantize_measurement(20.04, 0.1) == pytest.approx(20.0)
    assert quantize_measurement(20.06, 0.1) == pytest.approx(20.1)


def test_plant_refroidit_vers_lextérieur() -> None:
    """Sans chauffage ni soleil, l'air baisse si T_ext est plus froid."""
    params = replace(
        literature_plant_params(),
        process_noise_std=0.0,
        sensor_noise_std=0.0,
    )
    plant = ThermalPlant(params=params, x0=np.array([20.0, 20.0]), seed=0)
    t_ext = np.full(48, 5.0)  # 4 h
    traj = plant.simulate(t_ext, np.zeros(48), np.zeros(48))
    assert traj["ta_true"].iloc[-1] < 20.0
    assert traj["ta_true"].iloc[-1] > 5.0


def test_mesure_sur_la_grille_du_capteur() -> None:
    """y est un multiple de 0,1 °C."""
    plant = ThermalPlant(seed=0)
    y = plant.step(10.0, 0.0, 0.0)
    assert y == pytest.approx(round(y / 0.1) * 0.1)


def test_masse_pas_egale_a_la_mesure() -> None:
    """L'état caché n'est pas y."""
    params = replace(
        literature_plant_params(),
        process_noise_std=0.0,
        sensor_noise_std=0.0,
    )
    plant = ThermalPlant(params=params, x0=np.array([20.0, 15.0]), seed=0)
    plant.step(10.0, 0.0, 0.0)
    assert plant.x[1] != pytest.approx(plant.observe())


def test_meme_graine_meme_trajectoire() -> None:
    """Reproductibilité du bruit."""
    u = np.ones(10)
    t1 = ThermalPlant(seed=7).simulate(u, u, u)
    t2 = ThermalPlant(seed=7).simulate(u, u, u)
    np.testing.assert_allclose(t1["y"].to_numpy(), t2["y"].to_numpy())


def test_run_simulate_plant_ecrit_json(tmp_path: Path) -> None:
    """Le CLI métier écrit un rapport versionnable."""
    rapport = run_simulate_plant(n_hours=2.0, seed=0, processed_dir=tmp_path)
    assert (tmp_path / "plant_synthetic.csv").is_file()
    assert (tmp_path / "plant_report.json").is_file()
    assert rapport["alpha_s_mass"] > 0.0
