"""Pipeline de comparaison R1C1 / R2C2 sur un mini jeu synthétique."""

from pathlib import Path

import numpy as np
import pandas as pd

from basic_mpc.config import DataConfig
from basic_mpc.identification.compare import run_compare_r1c1_r2c2
from basic_mpc.identification.pem_r2c2 import fit_r2c2
from basic_mpc.models.r2c2 import R2C2Params, simulate_r2c2


def _entrees(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    t_ext = 10.0 + 8.0 * np.sin(2.0 * np.pi * t / 288.0) + rng.normal(0, 0.2, n)
    solar = np.clip(np.sin(2.0 * np.pi * (t - 72) / 288.0), 0.0, None) * 2500.0
    heating = ((t % 288) < 80).astype(float) * 25.0
    return t_ext, solar, heating


def test_pem_r2c2_cm_superieur_a_un() -> None:
    """Sur un vrai R2C2, C_m reste une masse, pas de l'air."""
    vrai = R2C2Params(
        rae=9e3,
        ram=2.5e3,
        cm=14.0,
        g_solar=2e-5,
        g_heating=4e-3,
        process_noise_std=0.03,
        process_noise_std_mass=0.03,
        sensor_noise_std=0.05,
    )
    n = 500
    t_ext, solar, heating = _entrees(n, seed=6)
    _x, y = simulate_r2c2(vrai, t_ext, solar, heating, x0=np.array([18.0, 17.0]), seed=6)
    u = np.column_stack([t_ext, solar, heating])
    estime, info = fit_r2c2(y, u)
    assert estime.cm > 1.2
    assert estime.tau_mass_hours > estime.tau_air_hours * 0.5
    assert np.isfinite(info["nll"])


def test_compare_ecrit_i1_et_schemas(tmp_path: Path) -> None:
    """JSON + I1 + au moins un schéma."""
    vrai = R2C2Params(
        rae=8e3,
        ram=2e3,
        cm=12.0,
        g_solar=2e-5,
        g_heating=4e-3,
        process_noise_std=0.03,
        process_noise_std_mass=0.03,
        sensor_noise_std=0.05,
    )
    n = 450
    t_ext, solar, heating = _entrees(n, seed=7)
    _x, y = simulate_r2c2(vrai, t_ext, solar, heating, x0=np.array([18.0, 17.0]), seed=7)
    index = pd.date_range("2020-01-01", periods=n, freq="5min", tz="UTC")
    table = pd.DataFrame(
        {
            "livingroom_y": y,
            "livingroom_setpoint": 21.0,
            "outdoor_y": t_ext,
            "P": heating,
            "S": solar,
        },
        index=index,
    )
    table.index.name = "time"
    processed = tmp_path / "processed"
    processed.mkdir()
    table.to_csv(processed / "identification_5min.csv")
    pictures = tmp_path / "pictures"
    rapport = run_compare_r1c1_r2c2(
        DataConfig(processed_dir=processed),
        pictures_dir=pictures,
        experiments_dir=tmp_path / "experiments",
    )
    assert (processed / "r1c1_r2c2_report.json").is_file()
    assert (pictures / "i1-rmse-horizons.png").is_file()
    assert (pictures / "schema-r1c1.png").is_file()
    assert (pictures / "schema-r1c1.pdf").is_file()
    assert "24h" in rapport["horizons_r2c2"]
    assert rapport["params_r2c2"]["cm"] > 1.0
