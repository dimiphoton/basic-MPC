"""Helpers des notebooks d'expérience : racine, PNG, journal.

Les notebooks n'inventent pas de logique : ils appellent ``src/`` et
affichent ce qui a déjà été écrit dans ``pictures/experiments/``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def repo_root() -> Path:
    """Racine du dépôt (``src/basic_mpc`` présent).

    Returns
    -------
    Path
        Dossier qui contient ``src/`` et ``pictures/``.

    Raises
    ------
    FileNotFoundError
        Si on n'est ni à la racine ni dans ``notebooks/``.
    """
    here = Path.cwd().resolve()
    for candidate in (here, here.parent):
        if (candidate / "src" / "basic_mpc").is_dir():
            return candidate
    msg = "ouvrir le notebook depuis notebooks/ ou la racine du repo"
    raise FileNotFoundError(msg)


def ensure_src_on_path() -> Path:
    """Ajoute ``src/`` au ``sys.path`` si le package n'est pas installé.

    Returns
    -------
    Path
        Racine du dépôt.
    """
    root = repo_root()
    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    return root


ROOT = ensure_src_on_path()
EXPERIMENTS = ROOT / "pictures" / "experiments"
RUNS_JSONL = ROOT / "experiments" / "runs.jsonl"


def show_pngs(*names: str) -> None:
    """Affiche des PNG déjà enregistrés sous ``pictures/experiments/``.

    Parameters
    ----------
    names : str
        Noms de fichiers (kebab-case, avec ``.png``).
    """
    from IPython.display import Image, Markdown, display

    for name in names:
        path = EXPERIMENTS / name
        if path.is_file():
            display(Markdown(f"**`{name}`**"))
            display(Image(filename=str(path)))
        else:
            display(
                Markdown(
                    f"*Manquant : `{name}` — passe `RECOMPUTE = True` "
                    "dans la cellule du dessus.*"
                )
            )


def last_run_lines(n: int = 8) -> str:
    """Dernières lignes de ``experiments/runs.jsonl``.

    Parameters
    ----------
    n : int
        Nombre de runs à afficher.

    Returns
    -------
    str
        JSON indenté, ou un message si le fichier n'existe pas.
    """
    if not RUNS_JSONL.is_file():
        return "(pas encore de experiments/runs.jsonl)"
    lines = [
        line.strip()
        for line in RUNS_JSONL.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    chunk = lines[-n:]
    pretty = []
    for raw in chunk:
        try:
            pretty.append(json.dumps(json.loads(raw), ensure_ascii=False, indent=2))
        except json.JSONDecodeError:
            pretty.append(raw)
    return "\n\n".join(pretty)
