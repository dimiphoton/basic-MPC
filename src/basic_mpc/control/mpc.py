"""MPC linéaire à horizon glissant, QP condensé (SciPy).

Deux coûts :
- v1 : suivi + hors-bande + effort (poids magiques) ;
- v1.1 : facture HP/HC + inconfort sous T_conf (brief contrôle).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import LinearConstraint, minimize


def move_blocking_matrix(n_pred: int, block_len: int) -> np.ndarray:
    """``P`` constant par blocs de ``block_len`` pas.

    Parameters
    ----------
    n_pred, block_len : int
        Horizon en pas et longueur d'un mouvement.

    Returns
    -------
    ndarray
        ``S`` de shape ``(n_pred, n_moves)`` : ``p_full = S @ p_moves``.
    """
    n_moves = int(np.ceil(n_pred / block_len))
    mat_s = np.zeros((n_pred, n_moves))
    for move in range(n_moves):
        i0 = move * block_len
        i1 = min(n_pred, i0 + block_len)
        mat_s[i0:i1, move] = 1.0
    return mat_s


def condensed_air(
    ad: np.ndarray,
    bd: np.ndarray,
    x0: np.ndarray,
    dist: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """``T_air`` libre (P=0) et réponse impulsionnelle du chauffage.

    Parameters
    ----------
    ad, bd : ndarray
        Dynamique interne.
    x0 : ndarray
        État estimé actuel.
    dist : ndarray
        Prévisions ``[T_ext, S]``, shape ``(n, 2)``.

    Returns
    -------
    free, gamma : ndarray
        ``ta = free + gamma @ p``, ``gamma`` de shape ``(n, n)``.
    """
    n = dist.shape[0]
    bp = bd[:, 2]
    bdd = bd[:, :2]
    x = np.asarray(x0, dtype=float).reshape(-1).copy()
    free = np.empty(n)
    for k in range(n):
        x = ad @ x + bdd @ dist[k]
        free[k] = x[0]
    # h[i] : effet sur T_air i pas après un P=1 au premier pas
    extra = bp.copy()
    resp = np.empty(n)
    resp[0] = extra[0]
    for i in range(1, n):
        extra = ad @ extra
        resp[i] = extra[0]
    gamma = np.zeros((n, n))
    for j in range(n):
        gamma[j:, j] = resp[: n - j]
    return free, gamma


def mpc_first_move(
    x0: np.ndarray,
    t_ext_fc: np.ndarray,
    solar_fc: np.ndarray,
    ad: np.ndarray,
    bd: np.ndarray,
    p_max: float,
    t_set: float,
    t_min: float,
    t_max: float,
    block_len: int,
    q_track: float,
    q_band: float,
    r_u: float,
    p_guess: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """Premier mouvement optimal (receding horizon).

    Prévisions d'extérieur et de solaire : oracle (la météo future du
    scénario). Coût = suivi + hors-bande + effort.

    Parameters
    ----------
    x0 : ndarray
        ``[T_air, T_masse]`` estimés.
    t_ext_fc, solar_fc : ndarray
        Prévisions alignées.
    ad, bd : ndarray
        Modèle interne.
    p_max : float
        Borne haute de P.
    t_set, t_min, t_max : float
        Consigne et bande.
    block_len : int
        Pas par mouvement.
    q_track, q_band, r_u : float
        Poids du coût.
    p_guess : ndarray, optional
        Warm start (mouvements).

    Returns
    -------
    p0, p_moves : float, ndarray
        Commande du premier bloc et séquence optimale.
    """
    dist = np.column_stack(
        [np.asarray(t_ext_fc, dtype=float), np.asarray(solar_fc, dtype=float)]
    )
    n = dist.shape[0]
    if n < 1:
        return 0.0, np.zeros(1)
    free, gamma = condensed_air(ad, bd, x0, dist)
    mat_s = move_blocking_matrix(n, max(1, block_len))
    n_moves = mat_s.shape[1]
    gain = gamma @ mat_s

    if p_guess is None or p_guess.size != n_moves:
        p0_vec = np.full(n_moves, 0.5 * p_max)
    else:
        p0_vec = np.clip(np.asarray(p_guess, dtype=float), 0.0, p_max)

    def cost_and_grad(p_moves: np.ndarray) -> tuple[float, np.ndarray]:
        p_full = mat_s @ p_moves
        ta = free + gain @ p_moves
        err = ta - t_set
        under = np.maximum(t_min - ta, 0.0)
        over = np.maximum(ta - t_max, 0.0)
        cost = (
            q_track * float(err @ err)
            + q_band * float(under @ under + over @ over)
            + r_u * float(p_full @ p_full)
        )
        d_ta = 2.0 * q_track * err + 2.0 * q_band * (over - under)
        grad = gain.T @ d_ta + 2.0 * r_u * (mat_s.T @ p_full)
        return cost, grad

    opt = minimize(
        cost_and_grad,
        p0_vec,
        jac=True,
        bounds=[(0.0, p_max)] * n_moves,
        method="L-BFGS-B",
        options={"maxiter": 50, "ftol": 1e-9},
    )
    p_moves = np.clip(opt.x, 0.0, p_max)
    return float(p_moves[0]), p_moves


def _t_sp_linear_map(
    x0_air: float,
    free: np.ndarray,
    gamma: np.ndarray,
    mat_s: np.ndarray,
    n_band: float,
    p_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """T_sp = T_air(début de pas) + n P / P_max, affine en p_moves.

    Le premier T_air est l'estimé actuel ; les suivants sont l'air *après*
    le pas précédent (effet des P déjà appliqués).
    """
    n = free.size
    n_moves = mat_s.shape[1]
    alpha = n_band / p_max if p_max > 0.0 else 0.0
    gain = gamma @ mat_s
    const = np.empty(n)
    mat_a = np.zeros((n, n_moves))
    const[0] = x0_air
    mat_a[0] = alpha * mat_s[0]
    if n > 1:
        const[1:] = free[:-1]
        mat_a[1:] = gain[:-1] + alpha * mat_s[1:]
    return mat_a, const


def mpc_euro_first_move(
    x0: np.ndarray,
    t_ext_fc: np.ndarray,
    solar_fc: np.ndarray,
    ad: np.ndarray,
    bd: np.ndarray,
    p_max: float,
    t_conf: np.ndarray,
    pi: np.ndarray,
    block_len: int,
    n_band: float,
    t_sp_min: float,
    t_sp_max: float,
    beta: float,
    lambda_comfort: float,
    dt_seconds: float,
    p_guess: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """Premier mouvement : min facture + inconfort, consigne bornée.

    J = Σ π β P Δt_s + λ max(T_conf − T_air, 0)² Δt_h.
    On décide P (QP), T_sp en découle. Pas de pénalité « trop chaud ».

    Parameters
    ----------
    x0 : ndarray
        ``[T_air, T_masse]`` estimés.
    t_ext_fc, solar_fc : ndarray
        Prévisions oracle alignées.
    ad, bd : ndarray
        Modèle interne.
    p_max : float
        Plafond de P.
    t_conf, pi : ndarray
        Confort (°C) et prix (€/kWh) sur l'horizon.
    block_len : int
        Pas par mouvement.
    n_band, t_sp_min, t_sp_max : float
        Bande n et bornes thermostat.
    beta : float
        kWh par (unité de P × seconde).
    lambda_comfort : float
        €·K⁻²·h⁻¹.
    dt_seconds : float
        Pas du plant.
    p_guess : ndarray, optional
        Warm start.

    Returns
    -------
    p0, p_moves : float, ndarray
        Commande du premier bloc et séquence.
    """
    dist = np.column_stack(
        [np.asarray(t_ext_fc, dtype=float), np.asarray(solar_fc, dtype=float)]
    )
    n = dist.shape[0]
    if n < 1 or p_max <= 0.0:
        return 0.0, np.zeros(1)
    free, gamma = condensed_air(ad, bd, x0, dist)
    mat_s = move_blocking_matrix(n, max(1, block_len))
    n_moves = mat_s.shape[1]
    gain = gamma @ mat_s
    t_conf = np.asarray(t_conf, dtype=float).reshape(-1)[:n]
    pi = np.asarray(pi, dtype=float).reshape(-1)[:n]
    dt_h = dt_seconds / 3600.0
    dt_s = float(dt_seconds)
    x0_air = float(np.asarray(x0, dtype=float).reshape(-1)[0])
    mat_tsp, const_tsp = _t_sp_linear_map(
        x0_air, free, gamma, mat_s, n_band, p_max
    )

    if p_guess is None or p_guess.size != n_moves:
        p0_vec = np.full(n_moves, 0.35 * p_max)
    else:
        p0_vec = np.clip(np.asarray(p_guess, dtype=float), 0.0, p_max)

    def cost_and_grad(p_moves: np.ndarray) -> tuple[float, np.ndarray]:
        p_full = mat_s @ p_moves
        ta = free + gain @ p_moves
        under = np.maximum(t_conf - ta, 0.0)
        bill = float(np.dot(pi * beta * dt_s, p_full))
        discomfort = float(lambda_comfort * dt_h * np.dot(under, under))
        d_ta = -2.0 * lambda_comfort * dt_h * under
        grad = gain.T @ d_ta + mat_s.T @ (pi * beta * dt_s)
        return bill + discomfort, grad

    bounds = [(0.0, p_max)] * n_moves
    lb = t_sp_min - const_tsp
    ub = t_sp_max - const_tsp
    constraint = LinearConstraint(mat_tsp, lb, ub)

    opt = minimize(
        cost_and_grad,
        p0_vec,
        jac=True,
        bounds=bounds,
        constraints=[constraint],
        method="SLSQP",
        options={"maxiter": 80, "ftol": 1e-9, "disp": False},
    )
    if not opt.success:
        # Contraintes T_sp parfois vides si l'air est déjà hors [16, 22].
        opt = minimize(
            cost_and_grad,
            p0_vec,
            jac=True,
            bounds=bounds,
            method="L-BFGS-B",
            options={"maxiter": 60, "ftol": 1e-9},
        )
    p_moves = np.clip(opt.x, 0.0, p_max)
    return float(p_moves[0]), p_moves
