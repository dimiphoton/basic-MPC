"""Moteur du simulateur : impédance fittée, préchauffage, export JSON."""

from dataclasses import replace
from pathlib import Path

import numpy as np

from basic_mpc.config import ControlConfig
from basic_mpc.models.impedance import omega_period_hours, z_r1c1, z_snapshot
from basic_mpc.models.r1c1 import R1C1Params
from basic_mpc.sim.export import run_export_simulator
from basic_mpc.sim.payload import vector_at_24h
from basic_mpc.sim.play import run_arena, run_hysteresis, run_preheat, scenario_bundle


def test_dephasage_r1c1_positif() -> None:
    """À 24 h, l'air retarde sur l'apport (Im Z < 0, delay > 0)."""
    params = R1C1Params(a=0.999, g_solar=0.0, g_heating=0.003)
    z = z_r1c1(params, np.array([omega_period_hours(24.0)]))
    snap = z_snapshot(z / z_r1c1(params, np.array([1e-12])), 24.0)
    assert snap["im"] < 0.0
    assert snap["delay_hours"] > 1.0
    assert snap["phase_deg"] < 0.0


def test_vecteur_fits_a_deux_fleches() -> None:
    """Les deux modèles fittés ont un vecteur 24 h utilisable."""
    r1 = R1C1Params(a=0.999, g_solar=0.0, g_heating=0.003)
    from basic_mpc.models.r2c2 import R2C2Params

    r2 = R2C2Params(rae=5e5, ram=1.25e5, cm=4.27, g_solar=0.0, g_heating=1e-5)
    vec = vector_at_24h(r1, r2)
    assert "re" in vec["r1c1"]
    assert vec["r2c2"]["delay_hours"] > 0.0


def test_prechauffage_avant_7h() -> None:
    """À 6 h, le préchauffage allume ; l'hystérésis attend T_conf=17."""
    cfg = replace(ControlConfig(), n_hours=14.0, horizon_hours=2.0)
    bundle = scenario_bundle(cfg)
    hyst = run_hysteresis(bundle)
    pre = run_preheat(bundle, hours_ahead=2.0)
    # 18 h → 6 h du matin = 12 h = index 144 à dt=5 min.
    i6 = int(12.0 * 3600.0 / bundle["plant_p"].dt_seconds)
    window = slice(i6 - 6, i6 + 6)
    assert float(np.mean(pre["P"].to_numpy()[window])) > float(
        np.mean(hyst["P"].to_numpy()[window])
    )


def test_export_json_et_z1(tmp_path: Path) -> None:
    """JSON Pages + flèches Z, sans MPC (trop long pour un test)."""
    cfg = replace(ControlConfig(), n_hours=6.0, horizon_hours=2.0, block_minutes=20.0)
    rapport = run_export_simulator(
        out_dir=tmp_path / "sim",
        pictures_dir=tmp_path / "pic",
        cfg=cfg,
        include_mpc=False,
    )
    json_path = Path(rapport["json_path"])
    assert json_path.is_file()
    assert (tmp_path / "pic" / "z1-impedance-24h.png").is_file()
    assert (tmp_path / "pic" / "z3-bode-phase.png").is_file()
    arena = run_arena(cfg, include_mpc=False)
    assert "hysteresis" in arena["strategies"]
    assert "preheat" in arena["strategies"]
    assert "mpc" not in arena["strategies"]
    assert arena["strategies"]["hysteresis"]["metrics"]["bill_eur"] >= 0.0
