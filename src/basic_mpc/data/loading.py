"""Lecture des CSV bruts — aucune écriture dans data/raw."""

from pathlib import Path

import pandas as pd


def load_raw_csv(path: Path) -> pd.DataFrame:
    """Charge un CSV capteur, timestamps UTC triés, sans doublon d'instant.

    Parameters
    ----------
    path : Path
        Fichier dans ``data/raw``.

    Returns
    -------
    DataFrame
        Colonne ``time`` en datetime UTC, index RangeIndex.

    Raises
    ------
    FileNotFoundError
        Si le fichier n'existe pas.
    ValueError
        Si la colonne ``time`` est absente.
    """
    if not path.is_file():
        msg = f"fichier brut introuvable : {path}"
        raise FileNotFoundError(msg)
    frame = pd.read_csv(path)
    # En-têtes PV du type 'L1 PV'
    frame.columns = [str(col).strip().strip("'\"") for col in frame.columns]
    if "time" not in frame.columns:
        msg = f"colonne time absente dans {path.name}"
        raise ValueError(msg)
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    frame = frame.sort_values("time")
    frame = frame.drop_duplicates(subset=["time"], keep="first")
    return frame.reset_index(drop=True)
