"""PEM R2C2 : mêmes innovations Kalman que le R1C1.

On estime des constantes de temps, pas des R bruts. Sans borne, le
filtre explique la série par un bruit de modèle et R part à l'infini
(deux τ ~ 140 h). Ici τ_air, τ_masse et C_m/C_a restent dans une
fourchette de pièce, et σ_w est fixé.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.optimize import minimize

from basic_mpc.identification.pem import P0_SCALE, SENSOR_NOISE_STD
from basic_mpc.models.kalman import KalmanResult, run_kalman
from basic_mpc.models.r2c2 import R2C2Params, discretize

# Fourchette grey-box : air d'une pièce, murs plus lents. Pas un mur infini.
TAU_AIR_H = (0.5, 6.0)
TAU_MASS_H = (8.0, 72.0)
CM_BOUNDS = (4.0, 25.0)
G_BOUNDS = (1e-10, 1e-2)
Q_FIXED = 0.05


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


def _unit(z: float) -> float:
    """Sigmoïde : un réel vers (0, 1)."""
    return float(1.0 / (1.0 + np.exp(-np.clip(z, -20.0, 20.0))))


def _log_span(z: float, low: float, high: float) -> float:
    """Pose z dans (low, high), à l'échelle log."""
    frac = _unit(z)
    log_v = np.log(low) + (np.log(high) - np.log(low)) * frac
    return float(np.exp(log_v))


def _inv_log_span(value: float, low: float, high: float) -> float:
    """Inverse de ``_log_span``."""
    frac = (np.log(value) - np.log(low)) / (np.log(high) - np.log(low))
    frac = float(np.clip(frac, 1e-6, 1.0 - 1e-6))
    return float(np.log(frac / (1.0 - frac)))


def _on_edge(value: float, low: float, high: float) -> bool:
    """Vrai si value est collée au bord de l'intervalle log."""
    span = np.log(high) - np.log(low)
    pos = np.log(value) - np.log(low)
    return bool(pos < 0.04 * span or pos > 0.96 * span)


def pack_theta_r2c2(params: R2C2Params) -> np.ndarray:
    """Cinq réels libres : τ_air, τ_masse, C_m/C_a, g_S, g_P."""
    return np.array(
        [
            _inv_log_span(params.tau_air_hours, *TAU_AIR_H),
            _inv_log_span(params.tau_mass_hours, *TAU_MASS_H),
            _inv_log_span(params.cm, *CM_BOUNDS),
            _inv_log_span(params.g_solar, *G_BOUNDS),
            _inv_log_span(params.g_heating, *G_BOUNDS),
        ],
        dtype=float,
    )


def unpack_theta_r2c2(theta: np.ndarray, dt_seconds: float) -> R2C2Params:
    """Reconstruit un R2C2. σ_w est fixé (sinon R s'échappe)."""
    z = np.asarray(theta, dtype=float)
    tau_air_h = _log_span(float(z[0]), *TAU_AIR_H)
    tau_mass_h = _log_span(float(z[1]), *TAU_MASS_H)
    cm = _log_span(float(z[2]), *CM_BOUNDS)
    # C_a = 1 : R_ae = τ_air, R_am = τ_masse / C_m, en secondes.
    rae = tau_air_h * 3600.0
    ram = tau_mass_h * 3600.0 / cm
    return R2C2Params(
        rae=rae,
        ram=ram,
        cm=cm,
        g_solar=_log_span(float(z[3]), *G_BOUNDS),
        g_heating=_log_span(float(z[4]), *G_BOUNDS),
        dt_seconds=dt_seconds,
        process_noise_std=Q_FIXED,
        process_noise_std_mass=Q_FIXED,
        sensor_noise_std=SENSOR_NOISE_STD,
    )


def describe_bounds(params: R2C2Params) -> str:
    """Une ligne pour la figure : la fourchette, et si on est au bord."""
    box = (
        f"bornes : air {TAU_AIR_H[0]:.1f}–{TAU_AIR_H[1]:.0f} h, "
        f"murs {TAU_MASS_H[0]:.0f}–{TAU_MASS_H[1]:.0f} h"
    )
    edges = []
    if _on_edge(params.tau_air_hours, *TAU_AIR_H):
        edges.append("τ_air au bord")
    if _on_edge(params.tau_mass_hours, *TAU_MASS_H):
        edges.append("τ_murs au bord")
    if _on_edge(params.cm, *CM_BOUNDS):
        edges.append("C_m au bord")
    if edges:
        return box + " · " + ", ".join(edges)
    return box


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
        # Départ au milieu d'une pièce : air ~2 h, murs ~20 h.
        initial = R2C2Params(
            rae=2.0 * 3600.0,
            ram=(20.0 * 3600.0) / 10.0,
            cm=10.0,
            g_solar=1e-6,
            g_heating=1e-4,
            dt_seconds=dt_seconds,
            process_noise_std=Q_FIXED,
            process_noise_std_mass=Q_FIXED,
            sensor_noise_std=SENSOR_NOISE_STD,
        )
    else:
        initial = replace(initial, dt_seconds=dt_seconds)

    theta0 = pack_theta_r2c2(initial)
    opt = minimize(
        negative_loglik_r2c2,
        theta0,
        args=(y, u, dt_seconds),
        method="L-BFGS-B",
        options={"maxiter": 80, "ftol": 1e-6},
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
