"""Point d'entrée en ligne de commande du projet."""

import argparse
import json
import logging

from basic_mpc.config import DataConfig
from basic_mpc.control.closed_loop import run_mpc_vs_bangbang
from basic_mpc.data.pipeline import run_preprocess
from basic_mpc.features.inputs import run_build_inputs
from basic_mpc.figures.schemas import run_draw_schemas
from basic_mpc.identification.compare import run_compare_r1c1_r2c2
from basic_mpc.identification.run import run_identify_r1c1
from basic_mpc.models.plant import run_simulate_plant


def main() -> None:
    """Point d'entrée principal du CLI."""
    parser = argparse.ArgumentParser(
        description="Grey-box RC, Kalman et MPC chauffage",
    )
    subparsers = parser.add_subparsers(dest="commande", required=True)

    subparsers.add_parser(
        "preprocess",
        help="Maille 5 min + modèle de capteur (salon, extérieur)",
    )
    subparsers.add_parser(
        "build-inputs",
        help="Construit P (chauffage) et S (PV), pas des watts",
    )
    subparsers.add_parser(
        "simulate-plant",
        help="Trajectoire du plant littérature (distinct de l'identification)",
    )
    subparsers.add_parser(
        "identify-r1c1",
        help="Identification PEM R1C1 + Kalman (split temporel)",
    )
    subparsers.add_parser(
        "draw-schemas",
        help="Schémas RC / Kalman (PNG + PDF, pictures/experiments)",
    )
    subparsers.add_parser(
        "compare-r1c1-r2c2",
        help="Identification R2C2 et comparaison multi-horizon",
    )
    subparsers.add_parser(
        "mpc-vs-bang-bang",
        help="MPC horizon glissant vs thermostat, plant littérature",
    )

    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if arguments.commande == "preprocess":
        rapport = run_preprocess(DataConfig())
        print(json.dumps(rapport["sensors"], indent=2, ensure_ascii=False))
    elif arguments.commande == "build-inputs":
        rapport = run_build_inputs(DataConfig())
        print(
            json.dumps(
                {
                    "heating_formula": rapport["heating_formula"],
                    "solar_formula": rapport["solar_formula"],
                    "fraction_heating_call": rapport["fraction_heating_call"],
                    "P_mean_when_positive": rapport["P_mean_when_positive"],
                    "S_max": rapport["S_max"],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    elif arguments.commande == "simulate-plant":
        rapport = run_simulate_plant()
        print(json.dumps(rapport, indent=2, ensure_ascii=False))
    elif arguments.commande == "identify-r1c1":
        rapport = run_identify_r1c1()
        print(
            json.dumps(
                {
                    "tau_hours": rapport["params"]["tau_hours"],
                    "a": rapport["params"]["a"],
                    "horizons_test": rapport["horizons_test"],
                    "nll": rapport["fit"]["nll"],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    elif arguments.commande == "draw-schemas":
        ecrits = run_draw_schemas()
        print(json.dumps(ecrits, indent=2, ensure_ascii=False))
    elif arguments.commande == "compare-r1c1-r2c2":
        rapport = run_compare_r1c1_r2c2()
        print(
            json.dumps(
                {
                    "tau_air_hours": rapport["params_r2c2"]["tau_air_hours"],
                    "tau_mass_hours": rapport["params_r2c2"]["tau_mass_hours"],
                    "horizons_r1c1": rapport["horizons_r1c1"],
                    "horizons_r2c2": rapport["horizons_r2c2"],
                    "nll_r2c2": rapport["fit_r2c2"]["nll"],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    elif arguments.commande == "mpc-vs-bang-bang":
        rapport = run_mpc_vs_bangbang()
        print(
            json.dumps(
                {
                    "p_max": rapport["p_max"],
                    "mpc": rapport["mpc"],
                    "bangbang": rapport["bangbang"],
                },
                indent=2,
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
