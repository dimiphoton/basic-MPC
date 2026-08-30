"""MPC linéaire à horizon glissant, QP condensé, bornes sur P (SciPy)."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


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
