"""Démo locale Streamlit : RC, déphasage, vecteur Z, arène de stratégies."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from basic_mpc.config import REPO_ROOT
from basic_mpc.identification.plots_compare import plot_bode_phase, plot_z1_vectors
from basic_mpc.sim.payload import default_fitted_if_missing, plant_for_js, vector_at_24h

DATA_PATH = REPO_ROOT / "docs" / "simulator" / "data.json"


def _load_data() -> dict:
    if DATA_PATH.is_file():
        return json.loads(DATA_PATH.read_text(encoding="utf-8"))
    return {}


def _lab_curves(cm_scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    plant = plant_for_js()
    dt = plant["dt_seconds"]
    n = int(48 * 3600 / dt)
    ca = plant["ca"]
    cm = plant["cm"] * cm_scale
    ram = plant["ram"]
    rae = plant["rae"]
    ta = 12.0
    tm = 12.0
    hours = np.arange(n) * dt / 3600.0
    t_ext = 5.0 + 6.0 * np.sin(2.0 * np.pi * ((hours % 24.0) - 9.0) / 24.0)
    air = np.empty(n)
    for k in range(n):
        d_ta = (tm - ta) / (ram * ca) + (t_ext[k] - ta) / (rae * ca)
        d_tm = (ta - tm) / (ram * cm)
        ta += dt * d_ta
        tm += dt * d_tm
        air[k] = ta
    i_out = int(np.argmax(t_ext[: int(24 * 3600 / dt)]))
    i_in = int(np.argmax(air[: int(24 * 3600 / dt)]))
    lag = abs(hours[i_in] - hours[i_out])
    return hours, t_ext, air, lag


def main() -> None:
    """Page unique : leçon RC, labo de retard, arène."""
    st.set_page_config(page_title="Simulateur RC / MPC", layout="wide")
    st.title("Peux-tu chauffer avant que la maison ait froid ?")
    st.caption("Machine learning · Bâtiment · Python / NumPy / SciPy / Streamlit")
    data = _load_data()

    st.header("Le modèle RC")
    st.write(
        "R isole (fuite vers l'extérieur). C stocke (murs, dalle). "
        "Le capteur ne voit que l'air — la masse est cachée."
    )
    st.latex(r"C_a \dot T_a = \frac{T_m-T_a}{R_{am}} + \frac{T_e-T_a}{R_{ae}} + \alpha_h P")

    st.header("Déphasage")
    st.write(
        "Un apport aujourd'hui n'arrive au confort que plus tard. "
        "C'est pour ça qu'on préchauffe en heures creuses, pas à 7 h."
    )
    cm_scale = st.slider("Inertie des murs (× C)", 0.25, 4.0, 1.0, 0.05)
    hours, t_ext, air, lag = _lab_curves(cm_scale)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.plot(hours, t_ext, color="#2c2416", label="extérieur")
    ax.plot(hours, air, color="#3d6b6b", label="air")
    ax.set_xlabel("heures")
    ax.set_ylabel("°C")
    ax.legend(frameon=False)
    st.pyplot(fig)
    plt.close(fig)
    st.metric("Retard pic extérieur → air", f"{lag:.1f} h")

    st.header("Vecteur d'impédance des fits")
    params_r1, params_r2 = default_fitted_if_missing()
    z24 = vector_at_24h(params_r1, params_r2)
    col_a, col_b = st.columns(2)
    with col_a:
        st.write(
            f"R1C1 : retard **{z24['r1c1']['delay_hours']:.1f} h** à 24 h "
            f"(phase {z24['r1c1']['phase_deg']:.0f}°)."
        )
        st.write(
            f"R2C2 : retard **{z24['r2c2']['delay_hours']:.1f} h** "
            f"(phase {z24['r2c2']['phase_deg']:.0f}°)."
        )
    tmp = Path(tempfile.mkdtemp())
    z1 = tmp / "z1.png"
    z3 = tmp / "z3.png"
    plot_z1_vectors(params_r1, params_r2, z1)
    plot_bode_phase(params_r1, params_r2, z3)
    c1, c2 = st.columns(2)
    c1.image(str(z1.with_suffix(".png")))
    c2.image(str(z3.with_suffix(".png")))

    st.header("Arène")
    if not data.get("arena"):
        st.info("Lance `python -m basic_mpc export-simulator` pour précalculer MPC / hystérésis / préchauffage.")
        return
    arena = data["arena"]
    names = {
        "hysteresis": "Hystérésis",
        "preheat": "Préchauffage 2 h",
        "mpc": "MPC",
    }
    chosen = st.multiselect(
        "Stratégies à superposer",
        list(arena["strategies"].keys()),
        default=list(arena["strategies"].keys()),
        format_func=lambda k: names.get(k, k),
    )
    fig2, ax2 = plt.subplots(figsize=(8, 3.6))
    ax2.plot(arena["hours"], arena["t_conf"], ls="--", color="#2c2416", label="T_conf")
    colors = {"hysteresis": "#8a7e6e", "preheat": "#8c4a32", "mpc": "#3d6b6b"}
    for key in chosen:
        strat = arena["strategies"][key]
        ax2.plot(strat["hours"], strat["ta"], color=colors.get(key, "#3d6b6b"), label=names.get(key, key))
        m = strat["metrics"]
        st.write(
            f"**{names.get(key, key)}** — {m['bill_eur']:.2f} € · "
            f"{m['hours_under_conf']:.1f} h trop froid"
        )
    ax2.set_xlabel("heures depuis 18 h")
    ax2.set_ylabel("°C")
    ax2.legend(frameon=False)
    st.pyplot(fig2)
    plt.close(fig2)


if __name__ == "__main__":
    main()
