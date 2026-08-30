"""Erreur de prédiction (PEM) : vraisemblance des innovations Kalman."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.optimize import minimize

from basic_mpc.models.kalman import KalmanResult, run_kalman
from basic_mpc.models.r1c1 import R1C1Params, discretize

# Prior large : le premier y du segment ancre l'état
P0_SCALE = 4.0
# Bruit de capteur fixé : un peu au-dessus de la quantification 0,1 °C
SENSOR_NOISE_STD = 0.05
A_BOUNDS = (0.80, 0.9998)
GAIN_MIN = 1e-12


def _sigmoid(z: float) -> float:
    """Ramène R vers (0, 1)."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -20.0, 20.0)))


def _logit(a: float, low: float, high: float) -> float:
    """Inverse de la sigmoïde affine sur (low, high)."""
    span = high - low
    frac = np.clip((a - low) / span, 1e-6, 1.0 - 1e-6)
    return float(np.log(frac / (1.0 - frac)))


def pack_theta(params: R1C1Params) -> np.ndarray:
    """Paramètres d'optimisation (non contraints)."""
    return np.array(
        [
            _logit(params.a, A_BOUNDS[0], A_BOUNDS[1]),
            np.log(max(params.g_solar, GAIN_MIN)),
            np.log(max(params.g_heating, GAIN_MIN)),
            np.log(max(params.process_noise_std, 1e-6)),
        ],
        dtype=float,
    )


def unpack_theta(theta: np.ndarray, dt_seconds: float) -> R1C1Params:
    """Reconstruit un R1C1 à partir de ``theta``."""
    a_frac = _sigmoid(float(theta[0]))
    a = A_BOUNDS[0] + (A_BOUNDS[1] - A_BOUNDS[0]) * a_frac
    return R1C1Params(
        a=float(a),
        g_solar=float(np.exp(theta[1])),
        g_heating=float(np.exp(theta[2])),
        dt_seconds=dt_seconds,
        process_noise_std=float(np.exp(theta[3])),
        sensor_noise_std=SENSOR_NOISE_STD,
    )


def filter_r1c1(
    y: np.ndarray,
    u: np.ndarray,
    params: R1C1Params,
) -> KalmanResult:
    """Kalman R1C1 sur une série (NaN gérés dans le filtre).

    Parameters
    ----------
    y : ndarray
        Mesures.
    u : ndarray
        ``[T_ext, S, P]``.
    params : R1C1Params
        Dynamique.

    Returns
    -------
    KalmanResult
        Filtrage et log-vraisemblance.
    """
    ad, bd = discretize(params)
    y = np.asarray(y, dtype=float).reshape(-1)
    finite = np.isfinite(y)
    x0 = np.array([float(y[finite][0]) if finite.any() else 20.0])
    p0 = np.array([[P0_SCALE]])
    q = np.array([[params.process_noise_std**2]])
    r = np.array([[params.sensor_noise_std**2]])
    c = np.array([[1.0]])
    return run_kalman(y, u, ad, bd, c, q, r, x0, p0)


def negative_loglik(theta: np.ndarray, y: np.ndarray, u: np.ndarray, dt: float) -> float:
    """NLL pour ``scipy.optimize`` (somme des innovations)."""
    params = unpack_theta(theta, dt)
    result = filter_r1c1(y, u, params)
    if result.n_obs < 10:
        return 1e12
    return float(-result.loglik)


def fit_r1c1(
    y: np.ndarray,
    u: np.ndarray,
    dt_seconds: float = 300.0,
    initial: R1C1Params | None = None,
) -> tuple[R1C1Params, dict]:
    """Estime le R1C1 par PEM (max. vraisemblance des innovations).

    Parameters
    ----------
    y : ndarray
        Température salon.
    u : ndarray
        Entrées ``(n, 3)``.
    dt_seconds : float
        Pas.
    initial : R1C1Params, optional
        Départ de l'optimiseur.

    Returns
    -------
    params : R1C1Params
        Paramètres estimés.
    info : dict
        NLL, succès, incertitudes approximatives.

    Raises
    ------
    RuntimeError
        Si l'optimiseur échoue sans point admissible.
    """
    if initial is None:
        initial = R1C1Params(
            a=0.97,
            g_solar=1e-5,
            g_heating=1e-3,
            dt_seconds=dt_seconds,
            process_noise_std=0.08,
            sensor_noise_std=SENSOR_NOISE_STD,
        )
    else:
        initial = replace(initial, dt_seconds=dt_seconds)

    theta0 = pack_theta(initial)
    opt = minimize(
        negative_loglik,
        theta0,
        args=(y, u, dt_seconds),
        method="BFGS",
        options={"maxiter": 80, "gtol": 1e-4},
    )
    params = unpack_theta(opt.x, dt_seconds)
    stderr = _stderr_from_hess(opt, dt_seconds)
    info = {
        "success": bool(opt.success),
        "message": str(opt.message),
        "nll": float(opt.fun),
        "n_iter": int(opt.nit),
        "stderr": stderr,
        "sensor_noise_std_fixed": SENSOR_NOISE_STD,
    }
    if not opt.success and not np.isfinite(opt.fun):
        msg = f"identification R1C1 impossible : {opt.message}"
        raise RuntimeError(msg)
    return params, info


def _stderr_from_hess(opt: object, dt_seconds: float) -> dict[str, float] | None:
    """Écart-types via hessienne inverse (espace d'origine, delta method)."""
    hess_inv = getattr(opt, "hess_inv", None)
    if hess_inv is None:
        return None
    cov = np.asarray(hess_inv, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != 4:
        return None
    theta = np.asarray(getattr(opt, "x"), dtype=float)
    se_th = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    a_low, a_high = A_BOUNDS
    a_frac = _sigmoid(float(theta[0]))
    da_dtheta = (a_high - a_low) * a_frac * (1.0 - a_frac)
    a = a_low + (a_high - a_low) * a_frac
    se_a = float(abs(da_dtheta) * se_th[0])
    g_s = float(np.exp(theta[1]))
    g_h = float(np.exp(theta[2]))
    q = float(np.exp(theta[3]))
    # tau = -dt / log(a) / 3600 ; d tau / da = dt / (a log(a)^2) / 3600
    log_a = np.log(a)
    dtau_da = dt_seconds / (a * log_a * log_a) / 3600.0
    return {
        "a": se_a,
        "g_solar": float(g_s * se_th[1]),
        "g_heating": float(g_h * se_th[2]),
        "process_noise_std": float(q * se_th[3]),
        "tau_hours": float(abs(dtau_da) * se_a),
    }
