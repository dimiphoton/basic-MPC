"""Filtre de Kalman linéaire, écrit à la main (pas de filterpy)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class KalmanResult:
    """Trajectoire filtrée et innovations.

    Parameters
    ----------
    x_filt : ndarray
        ``x_{k|k}``, shape ``(n, n_x)``. NaN si pas de mise à jour.
    innov : ndarray
        ``y_k - C x_{k|k-1}``, NaN si ``y`` manquant.
    innov_var : ndarray
        Variance d'innovation ``S_k`` (scalaire par pas).
    loglik : float
        Log-vraisemblance gaussienne des innovations finies.
    n_obs : int
        Nombre d'innovations utilisées.
    """

    x_filt: np.ndarray
    innov: np.ndarray
    innov_var: np.ndarray
    loglik: float
    n_obs: int


def run_kalman(
    y: np.ndarray,
    u: np.ndarray,
    ad: np.ndarray,
    bd: np.ndarray,
    c: np.ndarray,
    q: np.ndarray,
    r: np.ndarray,
    x0: np.ndarray,
    p0: np.ndarray,
) -> KalmanResult:
    """Filtre linéaire : prédiction puis mise à jour.

    Si ``y_k`` est NaN : prédiction seulement. Si ``u_{k-1}`` est NaN :
    on n'avance pas la dynamique (P inchangé) — les longs trous ne
    sont pas interpolés.

    Parameters
    ----------
    y : ndarray
        Observations, shape ``(n,)``.
    u : ndarray
        Entrées, shape ``(n, n_u)``. ``u_k`` agit après ``y_k``.
    ad, bd, c, q, r : ndarray
        Matrices discrètes. ``r`` est 1×1 (une mesure).
    x0, p0 : ndarray
        Prior avant ``y_0``.

    Returns
    -------
    KalmanResult
        États filtrés, innovations, log-vraisemblance.
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    u = np.asarray(u, dtype=float)
    if u.ndim == 1:
        u = u.reshape(-1, 1)
    n_x = int(np.asarray(ad).shape[0])
    if n_x == 1:
        return _kalman_1d(y, u, ad, bd, c, q, r, x0, p0)
    return _kalman_nd(y, u, ad, bd, c, q, r, x0, p0)


def _kalman_1d(
    y: np.ndarray,
    u: np.ndarray,
    ad: np.ndarray,
    bd: np.ndarray,
    c: np.ndarray,
    q: np.ndarray,
    r: np.ndarray,
    x0: np.ndarray,
    p0: np.ndarray,
) -> KalmanResult:
    """Chemin scalaire : ~10× plus rapide que les produits matriciels 1×1."""
    n = y.size
    a = float(np.asarray(ad).reshape(-1)[0])
    b = np.asarray(bd, dtype=float).reshape(-1)
    c_s = float(np.asarray(c).reshape(-1)[0])
    q_s = float(np.asarray(q).reshape(-1)[0])
    r_s = float(np.asarray(r).reshape(-1)[0])
    x = float(np.asarray(x0).reshape(-1)[0])
    p = float(np.asarray(p0).reshape(-1)[0])
    n_u = b.size
    u_f = np.ascontiguousarray(u[:, :n_u], dtype=float)
    b0 = float(b[0]) if n_u > 0 else 0.0
    b1 = float(b[1]) if n_u > 1 else 0.0
    b2 = float(b[2]) if n_u > 2 else 0.0

    x_filt = np.empty((n, 1))
    innov = np.full(n, np.nan)
    innov_var = np.full(n, np.nan)
    loglik = 0.0
    n_obs = 0
    log_2pi = np.log(2.0 * np.pi)

    for k in range(n):
        if k > 0:
            uk0 = u_f[k - 1, 0]
            uk1 = u_f[k - 1, 1] if n_u > 1 else 0.0
            uk2 = u_f[k - 1, 2] if n_u > 2 else 0.0
            # NaN ≠ NaN : un trou d'entrée → on n'avance pas
            if uk0 == uk0 and uk1 == uk1 and uk2 == uk2:
                x = a * x + b0 * uk0 + b1 * uk1 + b2 * uk2
                p = a * a * p + q_s

        yhat = c_s * x
        s_var = c_s * p * c_s + r_s
        if s_var <= 0.0:
            s_var = 1e-12
        yk = y[k]
        if yk == yk:
            e = yk - yhat
            innov[k] = e
            innov_var[k] = s_var
            gain = (p * c_s) / s_var
            x = x + gain * e
            p = (1.0 - gain * c_s) * p
            if p < 0.0:
                p = 0.0
            loglik += -0.5 * (log_2pi + np.log(s_var) + e * e / s_var)
            n_obs += 1
        x_filt[k, 0] = x

    return KalmanResult(
        x_filt=x_filt,
        innov=innov,
        innov_var=innov_var,
        loglik=float(loglik),
        n_obs=n_obs,
    )


def _kalman_nd(
    y: np.ndarray,
    u: np.ndarray,
    ad: np.ndarray,
    bd: np.ndarray,
    c: np.ndarray,
    q: np.ndarray,
    r: np.ndarray,
    x0: np.ndarray,
    p0: np.ndarray,
) -> KalmanResult:
    """Chemin matriciel (R2C2 et plus)."""
    n = y.size
    n_x = ad.shape[0]
    eye = np.eye(n_x)

    x_filt = np.full((n, n_x), np.nan)
    innov = np.full(n, np.nan)
    innov_var = np.full(n, np.nan)

    x = np.asarray(x0, dtype=float).reshape(n_x).copy()
    p = np.asarray(p0, dtype=float).reshape(n_x, n_x).copy()
    loglik = 0.0
    n_obs = 0

    for k in range(n):
        if k > 0 and np.isfinite(u[k - 1]).all():
            x = ad @ x + bd @ u[k - 1]
            p = ad @ p @ ad.T + q

        yhat = float((c @ x).reshape(-1)[0])
        s_var = float((c @ p @ c.T + r).reshape(-1)[0])
        if s_var <= 0.0:
            s_var = 1e-12

        if np.isfinite(y[k]):
            e = float(y[k]) - yhat
            innov[k] = e
            innov_var[k] = s_var
            gain = (p @ c.T) / s_var
            x = x + gain.reshape(n_x) * e
            # Joseph : un peu plus stable qu'un simple (I - K C) P
            ikc = eye - gain @ c
            r_mat = np.asarray(r, dtype=float).reshape(1, 1)
            p = ikc @ p @ ikc.T + gain @ r_mat @ gain.T
            loglik += -0.5 * (np.log(2.0 * np.pi * s_var) + e * e / s_var)
            n_obs += 1

        x_filt[k] = x

    return KalmanResult(
        x_filt=x_filt,
        innov=innov,
        innov_var=innov_var,
        loglik=float(loglik),
        n_obs=n_obs,
    )
