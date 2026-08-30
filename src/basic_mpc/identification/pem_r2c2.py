"""PEM R2C2 : mêmes innovations Kalman que le R1C1."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.optimize import minimize

from basic_mpc.identification.pem import GAIN_MIN, P0_SCALE, SENSOR_NOISE_STD
from basic_mpc.models.kalman import KalmanResult, run_kalman
from basic_mpc.models.r2c2 import R2C2Params, discretize


def filter_r2c2(y: np.ndarray, u: np.ndarray, params: R2C2Params) -> KalmanResult:
    """Kalman 2 états, mesure = air.

    Parameters
    ----------
    y : ndarray
        Capteur salon.
    u : ndarray
        ``[T_ext, S, P]``.
    params : R2C2Params
        Dynamique.

    Returns
    -------
    KalmanResult
        ``x_filt`` colonnes air, masse.
    """
    ad, bd = discretize(params)
    y = np.asarray(y, dtype=float).reshape(-1)
    finite = np.isfinite(y)
    y0 = float(y[finite][0]) if finite.any() else 20.0
    x0 = np.array([y0, y0])
    p0 = np.diag([P0_SCALE, 4.0 * P0_SCALE])
    q = np.diag(
        [params.process_noise_std**2, params.process_noise_std_mass**2]
    )
    r = np.array([[params.sensor_noise_std**2]])
    c = np.array([[1.0, 0.0]])
    return run_kalman(y, u, ad, bd, c, q, r, x0, p0)


def pack_theta_r2c2(params: R2C2Params) -> np.ndarray:
    """Logs des 6 paramètres libres."""
    return np.log(
        np.array(
            [
                max(params.rae, 60.0),
                max(params.ram, 60.0),
                max(params.cm, 1.1),
                max(params.g_solar, GAIN_MIN),
                max(params.g_heating, GAIN_MIN),
                max(params.process_noise_std, 1e-6),
            ],
            dtype=float,
        )
    )


def unpack_theta_r2c2(theta: np.ndarray, dt_seconds: float) -> R2C2Params:
    """Reconstruit un R2C2 ; q_masse = q_air (un seul bruit identifié)."""
    vals = np.exp(np.clip(np.asarray(theta, dtype=float), -20.0, 20.0))
    q = float(np.clip(vals[5], 1e-4, 2.0))
    return R2C2Params(
        rae=float(np.clip(vals[0], 200.0, 5.0e5)),
        ram=float(np.clip(vals[1], 200.0, 5.0e5)),
        cm=float(np.clip(vals[2], 1.2, 200.0)),
        g_solar=float(np.clip(vals[3], GAIN_MIN, 1.0)),
        g_heating=float(np.clip(vals[4], GAIN_MIN, 1.0)),
        dt_seconds=dt_seconds,
        process_noise_std=q,
        process_noise_std_mass=q,
        sensor_noise_std=SENSOR_NOISE_STD,
    )


def negative_loglik_r2c2(
    theta: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    dt: float,
) -> float:
    """NLL PEM."""
    params = unpack_theta_r2c2(theta, dt)
    if params.cm <= 1.0:
        return 1e12
    try:
        result = filter_r2c2(y, u, params)
    except np.linalg.LinAlgError:
        return 1e12
    if result.n_obs < 10:
        return 1e12
    nll = float(-result.loglik)
    if not np.isfinite(nll):
        return 1e12
    return nll


def fit_r2c2(
    y: np.ndarray,
    u: np.ndarray,
    dt_seconds: float = 300.0,
    initial: R2C2Params | None = None,
) -> tuple[R2C2Params, dict]:
    """Estime le R2C2 par PEM.

    Parameters
    ----------
    y, u : ndarray
        Mesures et entrées.
    dt_seconds : float
        Pas.
    initial : R2C2Params, optional
        Départ.

    Returns
    -------
    params, info
        Fit et NLL.
    """
    if initial is None:
        initial = R2C2Params(
            rae=4.0 * 3600.0,
            ram=(8.0 * 3600.0) / 15.0,
            cm=15.0,
            g_solar=1e-8,
            g_heating=1e-5,
            dt_seconds=dt_seconds,
            process_noise_std=0.08,
            process_noise_std_mass=0.08,
            sensor_noise_std=SENSOR_NOISE_STD,
        )
    else:
        initial = replace(initial, dt_seconds=dt_seconds)

    theta0 = pack_theta_r2c2(initial)
    opt = minimize(
        negative_loglik_r2c2,
        theta0,
        args=(y, u, dt_seconds),
        method="Nelder-Mead",
        options={"maxiter": 120, "xatol": 1e-3, "fatol": 2.0, "disp": False},
    )
    params = unpack_theta_r2c2(opt.x, dt_seconds)
    info = {
        "success": bool(opt.success),
        "message": str(opt.message),
        "nll": float(opt.fun),
        "n_iter": int(opt.nit),
        "sensor_noise_std_fixed": SENSOR_NOISE_STD,
    }
    if not opt.success and not np.isfinite(opt.fun):
        msg = f"identification R2C2 impossible : {opt.message}"
        raise RuntimeError(msg)
    return params, info
