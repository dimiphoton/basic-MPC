"""Tests PEM R1C1 : récupération de paramètres et pipeline."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from basic_mpc.config import DataConfig
from basic_mpc.identification.metrics import multi_horizon_scores
from basic_mpc.identification.pem import filter_r1c1, fit_r1c1
from basic_mpc.identification.run import run_identify_r1c1, temporal_split
from basic_mpc.models.r1c1 import R1C1Params, discretize, simulate_r1c1


def _entrees_excitées(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    t_ext = 10.0 + 8.0 * np.sin(2.0 * np.pi * t / 288.0) + rng.normal(0, 0.2, n)
    solar = np.clip(np.sin(2.0 * np.pi * (t - 72) / 288.0), 0.0, None) * 2500.0
    heating = ((t % 288) < 80).astype(float) * 25.0
    return t_ext, solar, heating


def test_pem_retrouve_les_gains_sur_cas_synthétique() -> None:
    """État et params connus : l'identifiant ne s'égare pas trop."""
    vrai = R1C1Params(
        a=0.95,
        g_solar=3e-5,
        g_heating=5e-3,
        process_noise_std=0.04,
        sensor_noise_std=0.05,
    )
    n = 600
    t_ext, solar, heating = _entrees_excitées(n)
    _x, y = simulate_r1c1(vrai, t_ext, solar, heating, x0=18.0, seed=3)
    u = np.column_stack([t_ext, solar, heating])
    estime, info = fit_r1c1(y, u)
    assert estime.a == pytest.approx(vrai.a, rel=0.04)
    assert estime.g_heating == pytest.approx(vrai.g_heating, rel=0.35)
    assert estime.g_solar > 0.0
    assert np.isfinite(info["nll"])


def test_multi_horizon_a_les_cles_du_brief() -> None:
    """1, 3, 6, 12, 24 h."""
    params = R1C1Params(a=0.97, g_solar=1e-5, g_heating=1e-3, process_noise_std=0.02)
    n = 400
    t_ext, solar, heating = _entrees_excitées(n, seed=4)
    _x, y = simulate_r1c1(params, t_ext, solar, heating, x0=19.0, seed=4)
    u = np.column_stack([t_ext, solar, heating])
    filt = filter_r1c1(y, u, params)
    ad, bd = discretize(params)
    scores = multi_horizon_scores(
        y, u, filt.x_filt, ad, bd, np.array([[1.0]]), 300.0, stride=6
    )
    assert set(scores) == {"1h", "3h", "6h", "12h", "24h"}
    assert scores["1h"]["n"] > 0
    # Un pas plus loin : l'erreur 1 h ne doit pas exploser
    assert scores["1h"]["rmse"] < 1.5


def test_split_temporel_ne_melange_pas() -> None:
    """Train = début, test = fin."""
    index = pd.date_range("2020-01-01", periods=10, freq="5min")
    table = pd.DataFrame({"v": np.arange(10)}, index=index)
    train, test = temporal_split(table, train_fraction=0.7)
    assert train.index.max() < test.index.min()
    assert len(train) == 7


def test_run_identify_r1c1_ecrit_json_et_png(tmp_path: Path) -> None:
    """Pipeline complet sur un mini CSV synthétique."""
    vrai = R1C1Params(
        a=0.96,
        g_solar=2e-5,
        g_heating=4e-3,
        process_noise_std=0.03,
        sensor_noise_std=0.05,
    )
    n = 500
    t_ext, solar, heating = _entrees_excitées(n, seed=5)
    _x, y = simulate_r1c1(vrai, t_ext, solar, heating, x0=18.0, seed=5)
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
    experiments = tmp_path / "experiments"
    rapport = run_identify_r1c1(
        DataConfig(processed_dir=processed),
        pictures_dir=pictures,
        experiments_dir=experiments,
    )
    assert (processed / "r1c1_report.json").is_file()
    assert (pictures / "i3-innovations-r1c1.png").is_file()
    assert (pictures / "i4-params-r1c1.png").is_file()
    assert (pictures / "s1-tair-vs-y.png").is_file()
    assert (experiments / "runs.jsonl").is_file()
    assert rapport["params"]["tau_hours"] > 0.0
    assert "1h" in rapport["horizons_test"]
