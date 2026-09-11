"""Les notebooks de leçon sont du JSON Jupyter valide."""

import json
from pathlib import Path

from basic_mpc.config import REPO_ROOT

NOTEBOOKS = (
    "01-modeles-rc.ipynb",
    "02-modele-capteur.ipynb",
    "03-kalman-r1c1.ipynb",
    "04-kalman-masse-cachee.ipynb",
    "05-plant-vs-identifie.ipynb",
    "06-innovations-et-nll.ipynb",
    "07-impedance-dephasage.ipynb",
    "08-rapports-cli.ipynb",
)


def test_notebooks_lecon_valides() -> None:
    """Huit carnets, au moins une cellule markdown et une cellule code."""
    root = REPO_ROOT / "notebooks"
    for name in NOTEBOOKS:
        path = root / name
        assert path.is_file(), name
        nb = json.loads(path.read_text(encoding="utf-8"))
        kinds = {c.get("cell_type") for c in nb["cells"]}
        assert "markdown" in kinds
        assert "code" in kinds


def test_schemas_pages_copies() -> None:
    """La leçon HTML a les PNG de circuits."""
    img = REPO_ROOT / "docs" / "simulator" / "img"
    for stem in ("schema-r1c1", "schema-r2c2", "schema-plant", "schema-kalman"):
        png = img / f"{stem}.png"
        assert png.is_file()
        assert png.stat().st_size > 1000
