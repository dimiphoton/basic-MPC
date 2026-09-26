"""Les expériences enregistrées (PNG) sont le livrable, pas l'app."""

import pytest

from notebooks._helpers import EXPERIMENTS, repo_root


def test_repo_root_a_le_package() -> None:
    root = repo_root()
    assert (root / "src" / "basic_mpc").is_dir()


@pytest.mark.parametrize(
    "nom",
    [
        "schema-r1c1.png",
        "z1-impedance-24h.png",
        "i1-rmse-horizons.png",
        "s4-mpc-vs-bang-bang.png",
        "h1-loi-bande-n.png",
        "h8-pareto-lambda.png",
    ],
)
def test_png_experiences_presents(nom: str) -> None:
    """Les figures clés sont déjà dans pictures/experiments/."""
    assert (EXPERIMENTS / nom).is_file(), nom
