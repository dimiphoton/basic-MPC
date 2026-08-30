"""Point d'entrée en ligne de commande du projet."""

import argparse
import json
import logging

from basic_mpc.config import DataConfig
from basic_mpc.data.pipeline import run_preprocess
from basic_mpc.features.inputs import run_build_inputs
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


if __name__ == "__main__":
    main()
