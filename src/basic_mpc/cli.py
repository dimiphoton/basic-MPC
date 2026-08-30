"""Point d'entrée en ligne de commande du projet."""

import argparse


def main() -> None:
    """Point d'entrée principal du CLI."""
    parser = argparse.ArgumentParser(
        description="Grey-box RC, Kalman et MPC chauffage",
    )
    parser.parse_args()


if __name__ == "__main__":
    main()
