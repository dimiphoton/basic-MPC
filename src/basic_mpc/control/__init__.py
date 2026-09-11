"""Contrôle : MPC à horizon glissant vs thermostat bang-bang / hystérésis."""

from basic_mpc.control.bangbang import bangbang_step, hysteresis_comfort_step
from basic_mpc.control.closed_loop import run_mpc_vs_bangbang
from basic_mpc.control.cout import run_mpc_cout_consigne
from basic_mpc.control.internal import p_to_hold, r2c2_internal_from_plant
from basic_mpc.control.mpc import mpc_euro_first_move, mpc_first_move

__all__ = [
    "bangbang_step",
    "hysteresis_comfort_step",
    "mpc_euro_first_move",
    "mpc_first_move",
    "p_to_hold",
    "r2c2_internal_from_plant",
    "run_mpc_cout_consigne",
    "run_mpc_vs_bangbang",
]
