"""Labo Streamlit : météo tirée, une stratégie, graphes datetime."""

from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st

from basic_mpc.sim.charts import fig_lab_panels
from basic_mpc.sim.dynamic import STRATEGIES, build_scenario, run_strategy

LABELS = {
    "hysteresis": "Hystérésis (thermostat)",
    "preheat": "Préchauffage 2 h",
    "mpc": "MPC (consigne + facture €)",
}


def _init_state() -> None:
    if "weather_seed" not in st.session_state:
        st.session_state.weather_seed = 1
    if "plant_seed" not in st.session_state:
        st.session_state.plant_seed = 0


def main() -> None:
    """Scénario → run → trois panneaux (physique, commande, coût)."""
    st.set_page_config(page_title="Labo chauffage RC / MPC", layout="wide")
    _init_state()
    st.title("Chauffer avant que la maison ait froid")
    st.caption("Machine learning · Bâtiment · labo dynamique (pas une arène figée)")

    with st.sidebar:
        st.header("Scénario")
        n_days = st.slider("Durée scorée (jours)", 2, 7, 4)
        t_day = st.slider("T_conf jour (°C)", 18.0, 22.0, 20.0, 0.5)
        t_night = st.slider("T_conf nuit (°C)", 15.0, 19.0, 17.0, 0.5)
        strategy = st.selectbox(
            "Stratégie",
            list(STRATEGIES),
            format_func=lambda k: LABELS[k],
        )
        overlay = st.checkbox(
            "Superposer l'hystérésis",
            value=False,
            disabled=strategy == "hysteresis",
        )
        st.caption(f"Météo n° {st.session_state.weather_seed} (prévision parfaite).")
        if st.button("Nouvelle météo"):
            st.session_state.weather_seed += 1
            st.session_state.pop("lab_run", None)
        lancer = st.button("Lancer la simulation", type="primary")
        st.markdown(
            "Leçon RC (déphasage, vecteur Z) : "
            "[GitHub Pages](https://dimiphoton.github.io/basic-MPC/simulator/)."
        )

    if lancer:
        if strategy == "mpc":
            st.info(
                "Le QP tourne à chaque bloc de 30 min. "
                "Compter ~10 s par jour simulé."
            )
        with st.spinner("Burn-in 24 h puis stratégie…"):
            scenario = build_scenario(
                n_days=float(n_days),
                weather_seed=int(st.session_state.weather_seed),
                plant_seed=int(st.session_state.plant_seed),
                t_conf_occupied=float(t_day),
                t_conf_setback=float(t_night),
            )
            st.session_state.lab_run = run_strategy(
                strategy,
                scenario,
                overlay_hysteresis=bool(overlay) and strategy != "hysteresis",
            )

    run = st.session_state.get("lab_run")
    if run is None:
        st.write(
            "Règle le scénario, tire une météo, lance. "
            "La maison a déjà vécu 24 h (hystérésis) : on ne score pas un départ gelé."
        )
        return

    m = run["metrics"]
    x0 = run["x0"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Facture", f"{m['bill_eur']:.2f} €")
    c2.metric("Trop froid", f"{m['hours_under_conf']:.1f} h")
    c3.metric("kWh HP / HC", f"{m['kwh_hp']:.1f} / {m['kwh_hc']:.1f}")
    c4.metric("x0 air / murs", f"{x0[0]:.1f} / {x0[1]:.1f} °C")

    fig = fig_lab_panels(
        run["index"],
        run["traj"],
        run["t_conf"],
        run["pi"],
        run["cfg"].pi_hc,
        m["bill_cum"],
        overlay=run.get("overlay"),
    )
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)

    if run.get("overlay") is not None:
        om = run["overlay"]["metrics"]
        st.write(
            f"Hystérésis superposée : **{om['bill_eur']:.2f} €** · "
            f"**{om['hours_under_conf']:.1f} h** trop froid "
            f"(même météo, même x0)."
        )


if __name__ == "__main__":
    main()
