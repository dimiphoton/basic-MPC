"""PNG de devoir : dynamique, Kalman, sensibilités — sans refit PEM."""

from dataclasses import replace
from pathlib import Path

import numpy as np

from basic_mpc.config import ControlConfig
from basic_mpc.control.actuator import proportional_band_p
from basic_mpc.control.cout import euro_closed_loop_pair
from basic_mpc.homework.run import _delay_series, _step_response, run_homework_pngs


def test_loi_bande_n_trois_pentes() -> None:
    """n plus large : même écart, moins de P."""
    assert proportional_band_p(20.0, 19.5, 0.5, 1.0) == 1.0
    assert proportional_band_p(20.0, 19.5, 1.0, 1.0) == 0.5
    assert proportional_band_p(20.0, 19.5, 2.0, 1.0) == 0.25


def test_delay_series_ne_voit_pas_le_futur() -> None:
    """Retard causal : les premières valeurs sont le passé immédiat."""
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = _delay_series(x, 2)
    assert y[0] == 1.0
    assert y[1] == 1.0
    assert y[2] == 1.0
    assert y[3] == 2.0


def test_echelon_masse_retarde_l_air() -> None:
    """Après un échelon P, l'air monte plus vite que la masse."""
    _hours, ta, tm = _step_response()
    i = 12  # 1 h
    assert (ta[i] - ta[0]) > (tm[i] - tm[0])
    assert ta[-1] > ta[0]
    assert tm[-1] > tm[0]


def test_homework_pngs_sans_sweep(tmp_path: Path) -> None:
    """H1–H5 et H10 s'écrivent sans lancer le QP."""
    rapport = run_homework_pngs(
        pictures_dir=tmp_path,
        processed_dir=tmp_path,
        experiments_dir=tmp_path,
        include_sweeps=False,
    )
    for nom in (
        "h1-loi-bande-n.png",
        "h2-reponse-indicielle.png",
        "h3-bode-module.png",
        "h4-qq-innovations.png",
        "h5-kalman-trou.png",
        "h10-quantification.png",
    ):
        assert (tmp_path / nom).is_file(), nom
    assert "h8-pareto-lambda.png" not in rapport["figures"]


def test_euro_pair_court() -> None:
    """Une boucle courte tourne (pour les sweeps du notebook)."""
    cfg = replace(ControlConfig(), n_hours=3.0, horizon_hours=2.0)
    pair = euro_closed_loop_pair(cfg)
    assert pair["mpc"]["bill_eur"] >= 0.0
    assert pair["hysteresis"]["bill_eur"] >= 0.0
    assert len(pair["hours"]) == int(3.0 * 3600.0 / pair["dt_seconds"])
