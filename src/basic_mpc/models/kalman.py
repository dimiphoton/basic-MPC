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
    if n_x == 2:
        return _kalman_2d(y, u, ad, bd, c, q, r, x0, p0)
    return _kalman_nd(y, u, ad, bd, c, q, r, x0, p0)


def _kalman_2d(
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
    """Chemin 2×2 déroulé (R2C2) : mesure = première composante si C ≈ [1, 0]."""
    c = np.asarray(c, dtype=float).reshape(-1)
    if abs(c[0] - 1.0) > 1e-9 or abs(c[1]) > 1e-9:
        return _kalman_nd(y, u, ad, bd, c.reshape(1, 2), q, r, x0, p0)

    n = y.size
    a11 = float(ad[0, 0])
    a12 = float(ad[0, 1])
    a21 = float(ad[1, 0])
    a22 = float(ad[1, 1])
    b00 = float(bd[0, 0])
    b01 = float(bd[0, 1])
    b02 = float(bd[0, 2])
    b10 = float(bd[1, 0])
    b11 = float(bd[1, 1])
    b12 = float(bd[1, 2])
    q11 = float(q[0, 0])
    q12 = float(q[0, 1])
    q21 = float(q[1, 0])
    q22 = float(q[1, 1])
    r_s = float(np.asarray(r).reshape(-1)[0])
    x_a = float(np.asarray(x0).reshape(-1)[0])
    x_m = float(np.asarray(x0).reshape(-1)[1])
    p11 = float(p0[0, 0])
    p12 = float(p0[0, 1])
    p21 = float(p0[1, 0])
    p22 = float(p0[1, 1])
    u0 = np.ascontiguousarray(u[:, 0], dtype=float)
    u1 = np.ascontiguousarray(u[:, 1], dtype=float)
    u2 = np.ascontiguousarray(u[:, 2], dtype=float)

    x_filt = np.empty((n, 2))
    innov = np.full(n, np.nan)
    innov_var = np.full(n, np.nan)
    loglik = 0.0
    n_obs = 0
    log_2pi = np.log(2.0 * np.pi)

    for k in range(n):
        if k > 0:
            uk0 = u0[k - 1]
            uk1 = u1[k - 1]
            uk2 = u2[k - 1]
            if uk0 == uk0 and uk1 == uk1 and uk2 == uk2:
                nx_a = a11 * x_a + a12 * x_m + b00 * uk0 + b01 * uk1 + b02 * uk2
                nx_m = a21 * x_a + a22 * x_m + b10 * uk0 + b11 * uk1 + b12 * uk2
                ap11 = a11 * p11 + a12 * p21
                ap12 = a11 * p12 + a12 * p22
                ap21 = a21 * p11 + a22 * p21
                ap22 = a21 * p12 + a22 * p22
                p11 = ap11 * a11 + ap12 * a12 + q11
                p12 = ap11 * a21 + ap12 * a22 + q12
                p21 = ap21 * a11 + ap22 * a12 + q21
                p22 = ap21 * a21 + ap22 * a22 + q22
                x_a, x_m = nx_a, nx_m

        s_var = p11 + r_s
        if s_var <= 0.0:
            s_var = 1e-12
        yk = y[k]
        if yk == yk:
            e = yk - x_a
            innov[k] = e
            innov_var[k] = s_var
            k0 = p11 / s_var
            k1 = p21 / s_var
            x_a = x_a + k0 * e
            x_m = x_m + k1 * e
            r0_11, r0_12 = p11, p12
            p11 = p11 - k0 * r0_11
            p12 = p12 - k0 * r0_12
            p21 = p21 - k1 * r0_11
            p22 = p22 - k1 * r0_12
            n_obs += 1
            loglik += -0.5 * (log_2pi + np.log(s_var) + e * e / s_var)
        x_filt[k, 0] = x_a
        x_filt[k, 1] = x_m

    return KalmanResult(
        x_filt=x_filt,
        innov=innov,
        innov_var=innov_var,
        loglik=float(loglik),
        n_obs=n_obs,
    )


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
