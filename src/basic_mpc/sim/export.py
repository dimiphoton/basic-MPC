"""Écrit ``docs/simulator/data.json`` et les figures Z1 / Z3 (leçon RC)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from basic_mpc.config import REPO_ROOT
from basic_mpc.identification.plots_compare import plot_bode_phase, plot_z1_vectors
from basic_mpc.sim.payload import (
    bode_curve,
    default_fitted_if_missing,
    plant_for_js,
    vector_at_24h,
)

logger = logging.getLogger(__name__)


def run_export_simulator(
    out_dir: Path | None = None,
    pictures_dir: Path | None = None,
) -> dict:
    """JSON Pages (plant + Z) sans arène précalculée.

    Parameters
    ----------
    out_dir : Path, optional
        Défaut ``docs/simulator``.
    pictures_dir : Path, optional
        Figures Z1 / Z3.

    Returns
    -------
    dict
        Chemins écrits.
    """
    out_dir = out_dir or (REPO_ROOT / "docs" / "simulator")
    pictures_dir = pictures_dir or (REPO_ROOT / "pictures" / "experiments")
    out_dir.mkdir(parents=True, exist_ok=True)
    pictures_dir.mkdir(parents=True, exist_ok=True)
    params_r1, params_r2 = default_fitted_if_missing()
    z24 = vector_at_24h(params_r1, params_r2)
    bode = bode_curve(params_r1, params_r2)
    payload = {
        "plant": plant_for_js(),
        "fitted_z_24h": z24,
        "bode": bode,
        "copy": {
            "rc": (
                "R isole (fuite vers l'extérieur). C stocke (murs, dalle). "
                "Le capteur ne voit que l'air."
            ),
            "phase": (
                "Un apport aujourd'hui n'arrive au confort que plus tard : "
                "c'est le déphasage. On chauffe avant 7 h, pas à 7 h."
            ),
            "lab": (
                "Le labo de stratégies (météo tirée, burn-in, MPC) tourne "
                "en local : streamlit run webapp/app.py"
            ),
        },
    }
    json_path = out_dir / "data.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")
    z1_path = pictures_dir / "z1-impedance-24h.png"
    z3_path = pictures_dir / "z3-bode-phase.png"
    plot_z1_vectors(params_r1, params_r2, z1_path)
    plot_bode_phase(params_r1, params_r2, z3_path)
    slides = REPO_ROOT / "pictures" / "presentations"
    if slides.is_dir():
        import shutil

        shutil.copy2(z1_path, slides / "z1-impedance-24h.png")
        shutil.copy2(z3_path, slides / "z3-bode-phase.png")
    logger.info("écrit %s", json_path)
    return {
        "json_path": str(json_path),
        "z1": str(z1_path),
        "z3": str(z3_path),
        "delay_r1_h": z24["r1c1"]["delay_hours"],
        "delay_r2_h": z24["r2c2"]["delay_hours"],
    }
