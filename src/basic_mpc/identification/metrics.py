"""Métriques multi-horizon : un MPC se juge à 1–24 h, pas à un pas."""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# Horizons du brief de cadrage (heures)
HORIZONS_HOURS = (1, 3, 6, 12, 24)


def rmse(errors: np.ndarray) -> float:
    """Racine de l'erreur quadratique moyenne.

    Parameters
    ----------
    errors : ndarray
        Résidus (prédiction − observation).

    Returns
    -------
    float
        RMSE, ou NaN si vide.
    """
    vals = np.asarray(errors, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(vals * vals)))


def mae(errors: np.ndarray) -> float:
    """Erreur absolue moyenne.

    Parameters
    ----------
    errors : ndarray
        Résidus.

    Returns
    -------
    float
        MAE, ou NaN si vide.
    """
    vals = np.asarray(errors, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    return float(np.mean(np.abs(vals)))


def multi_horizon_scores(
    y: np.ndarray,
    u: np.ndarray,
    x_filt: np.ndarray,
    ad: np.ndarray,
    bd: np.ndarray,
    c: np.ndarray,
    dt_seconds: float,
    horizons_hours: tuple[int, ...] = HORIZONS_HOURS,
    stride: int = 12,
) -> dict[str, dict[str, float]]:
    """Prédiction libre à partir de ``x_{t|t}``, entrées mesurées.

    On n'évalue pas chaque pas : ``stride`` (défaut 1 h sur maille 5 min)
    suffit pour des RMSE stables.

    Parameters
    ----------
    y : ndarray
        Observations ``(n,)``.
    u : ndarray
        Entrées ``(n, n_u)``.
    x_filt : ndarray
        États filtrés ``(n, n_x)``.
    ad, bd, c : ndarray
        Modèle discret.
    dt_seconds : float
        Durée d'un pas.
    horizons_hours : tuple of int
        Horizons à évaluer.
    stride : int
        Un point de départ tous les ``stride`` pas.

    Returns
    -------
    dict
        Clé ``\"1h\"``, etc. : ``rmse``, ``mae``, ``n``.
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    u = np.asarray(u, dtype=float)
    x_filt = np.asarray(x_filt, dtype=float)
    if u.ndim == 1:
        u = u.reshape(-1, 1)
    n = y.size
    scores: dict[str, dict[str, float]] = {}
    steps_per_hour = int(round(3600.0 / dt_seconds))
    ad = np.asarray(ad, dtype=float)
    bd = np.asarray(bd, dtype=float)
    c = np.asarray(c, dtype=float)

    for hours in horizons_hours:
        h = hours * steps_per_hour
        if h <= 0 or n <= h:
            scores[f"{hours}h"] = {"rmse": float("nan"), "mae": float("nan"), "n": 0.0}
            continue
        if ad.shape[0] == 1:
            arr = _horizon_errors_1d(y, u, x_filt, ad, bd, c, h, stride)
        else:
            arr = _horizon_errors_nd(y, u, x_filt, ad, bd, c, h, stride)
        scores[f"{hours}h"] = {
            "rmse": rmse(arr),
            "mae": mae(arr),
            "n": float(np.isfinite(arr).sum()),
        }
    return scores


def _horizon_errors_1d(
    y: np.ndarray,
    u: np.ndarray,
    x_filt: np.ndarray,
    ad: np.ndarray,
    bd: np.ndarray,
    c: np.ndarray,
    h: int,
    stride: int,
) -> np.ndarray:
    """Prédiction h pas : FIR sur les entrées (R1C1)."""
    a = float(ad.reshape(-1)[0])
    c_s = float(c.reshape(-1)[0])
    b = bd.reshape(-1)
    bu = u @ b
    n = y.size
    n_starts = n - h
    windows = sliding_window_view(bu, h)[:n_starts]
    weights = a ** np.arange(h - 1, -1, -1, dtype=float)
    yhat = c_s * ((a**h) * x_filt[:n_starts, 0] + windows @ weights)
    err = yhat - y[h : h + n_starts]
    idx = np.arange(0, n_starts, stride)
    ok = (
        np.isfinite(windows[idx]).all(axis=1)
        & np.isfinite(x_filt[idx, 0])
        & np.isfinite(y[idx + h])
    )
    return err[idx][ok]


def _horizon_errors_nd(
    y: np.ndarray,
    u: np.ndarray,
    x_filt: np.ndarray,
    ad: np.ndarray,
    bd: np.ndarray,
    c: np.ndarray,
    h: int,
    stride: int,
) -> np.ndarray:
    """Boucle d'état (R2C2)."""
    n = y.size
    errors: list[float] = []
    for t in range(0, n - h, stride):
        if not np.isfinite(x_filt[t]).all():
            continue
        if not np.isfinite(y[t + h]):
            continue
        if not np.isfinite(u[t : t + h]).all():
            continue
        x = x_filt[t].copy()
        for k in range(h):
            x = ad @ x + bd @ u[t + k]
        yhat = float((c @ x).reshape(-1)[0])
        errors.append(yhat - float(y[t + h]))
    return np.asarray(errors, dtype=float)
