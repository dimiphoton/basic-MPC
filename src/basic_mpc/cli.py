"""Point d'entrée en ligne de commande du projet."""

import argparse
import json
import logging

from basic_mpc.config import DataConfig
from basic_mpc.data.pipeline import run_preprocess


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

    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if arguments.commande == "preprocess":
        rapport = run_preprocess(DataConfig())
        print(json.dumps(rapport["sensors"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
