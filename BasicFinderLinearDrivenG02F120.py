# C:\GitHub\TrapFrequencyAnalysis\BasicFinderLinearDrivenG02F120.py
"""
F120: Build the Jacobian for the map R^20 -> R^{C(3N,2)} that sends the twenty
per-electrode amplitudes of a single extra-drive (DC1..DC10, RF11..RF20) to *all*
two-mode driven couplings g0_ab (Hz), with DC/RF/modes held fixed.

Implementation mirrors F110/F310:
- Use a single Simulation with a center drive at 'point' and ±h probe drives per axis.
- Lock equilibrium and static modes once, evaluate g0 via the pair accessor for consistency.
- Return J (Hz/V), pairs (i<j), and g_center (Hz) at the evaluation point.
"""

import math
import numpy as np
import time

from sim.simulation import Simulation
from trapping_variables import Trapping_Vars
from typing import List, Tuple


def _vec20_to_dc_rf_map(v20):
    """
    Map a 20-vector of per-electrode amplitudes (V) to a dict over:
        DC1..DC10, RF11..RF15, RF16..RF20
    Missing symmetry on purpose: each channel is independent.

    Order convention (match UI/data & keep stable):
        idx:  0   1   2   3   4   5   6    7    8    9   10   11   12   13   14   15   16   17   18   19
        elec: DC1 DC2 DC3 DC4 DC5 DC6 DC7  DC8  DC9  DC10 RF11 RF12 RF13 RF14 RF15 RF16 RF17 RF18 RF19 RF20
    """
    a = [float(x) for x in v20]
    if len(a) != 20:
        raise ValueError(f"_vec20_to_dc_rf_map expects length 20, got {len(a)}")
    return {
        "DC1": a[0],
        "DC2": a[1],
        "DC3": a[2],
        "DC4": a[3],
        "DC5": a[4],
        "DC6": a[5],
        "DC7": a[6],
        "DC8": a[7],
        "DC9": a[8],
        "DC10": a[9],
        "RF11": a[10],
        "RF12": a[11],
        "RF13": a[12],
        "RF14": a[13],
        "RF15": a[14],
        "RF16": a[15],
        "RF17": a[16],
        "RF18": a[17],
        "RF19": a[18],
        "RF20": a[19],
    }


def _copy_tv(tv_in: Trapping_Vars) -> Trapping_Vars:
    """Copy DC map and all non-DC drives (e.g. RF) into a fresh Trapping_Vars."""
    tv = Trapping_Vars()
    # DC
    dc_in = tv_in.get_drive_amplitudes(tv_in.dc_key)
    for el, A in dc_in.items():
        tv.Var_dict[tv.dc_key].set_amplitude_volt(el, A)
    tv._update_pickoff_for_drive(tv.dc_key)
    # non-DC (RF etc.)
    for dk in tv_in.get_drives():
        if dk == tv_in.dc_key:
            continue
        amps = tv_in.get_drive_amplitudes(dk)
        tv.add_driving(dk.label, dk.f_hz, dk.phi, amps)
    return tv


def calculate_jacobian_F120(
    num_ions: int = 3,
    constant_trappingvars=Trapping_Vars,  # instance preferred; class allowed
    point=None,  # 20 DOFs (V): [DC1..DC10, RF11..RF15, RF16..RF20]
    simulation_preset: str = "Simp58_101",
):
    """
    Build the full F120 Jacobian (all pair couplings vs 20 inputs) using the *pair accessor path*
    to ensure exact agreement with the F110 pattern.

    Returns
    -------
    dict with:
      - 'J_Hz_per_V' : (M x 20) array, rows in (i<j) order, units Hz/V
      - 'pairs'      : list[(i,j)] row mapping
      - 'g_center_Hz': (M,) array of g0 at 'point' (Hz)
      - 'K'          : mode-space dimension (3N)
      - 'M'          : number of mode pairs (K choose 2)
    """
    import numpy as np

    if point is None:
        point = [0.0] * 20
    if not (isinstance(point, (list, tuple)) and len(point) == 20):
        raise ValueError(
            "point must be a length-20 list/tuple of floats (DC1..DC10, RF11..RF20)."
        )

    # Accept Trapping_Vars instance or class
    if isinstance(constant_trappingvars, Trapping_Vars):
        tv_in = constant_trappingvars
    elif isinstance(constant_trappingvars, type) and issubclass(
        constant_trappingvars, Trapping_Vars
    ):
        tv_in = constant_trappingvars()
    else:
        raise ValueError(
            "constant_trappingvars must be a Trapping_Vars instance or the Trapping_Vars class."
        )

    # Step size: larger at origin (your convention), otherwise smaller
    is_zero_point = all(abs(float(x)) == 0.0 for x in point)
    h = 0.5 if is_zero_point else 0.05

    # Fresh TV with DC/RF copied; add center and ±h probe drives
    tv = _copy_tv(tv_in)

    dk_center = tv.add_driving("F120_center", 77.0, 0.0, _vec20_to_dc_rf_map(point))

    probe_plus, probe_minus = [], []
    for k in range(20):
        p = list(point)
        p[k] += h
        m = list(point)
        m[k] -= h
        dk_p = tv.add_driving(f"F120_p{k+1}", 1.0 + k, 0.0, _vec20_to_dc_rf_map(p))
        dk_m = tv.add_driving(f"F120_m{k+1}", 101.0 + k, 0.0, _vec20_to_dc_rf_map(m))
        probe_plus.append(dk_p)
        probe_minus.append(dk_m)

    # One Simulation; lock equilibrium and modes once
    sim = Simulation(simulation_preset, tv)
    sim.find_equilib_position_single(int(num_ions))
    sim.get_static_normal_modes_and_freq(
        int(num_ions), normalize=True, sort_by_freq=True
    )

    # Mode count and pair list in canonical (i<j) order
    K = sim.normal_modes_and_frequencies[int(num_ions)]["modes"].shape[1]
    pairs = [(i, j) for i in range(K) for j in range(i + 1, K)]
    M = len(pairs)

    # Warm build for the center drive (Hz) and read center values via pair accessor
    sim.get_g0_matrix(int(num_ions), dk_center)
    g_center = np.empty(M, dtype=float)
    for idx, (i, j) in enumerate(pairs):
        g_center[idx] = sim.get_g0_for_mode_pair(
            num_ions=int(num_ions),
            drive=dk_center,
            mode_i=i,
            mode_j=j,
            units="Hz",
            recompute=False,
        )

    # Build Jacobian columns via central differences, using the *pair* accessor
    J = np.empty((M, 20), dtype=float)
    for k in range(20):
        col = np.empty(M, dtype=float)
        for idx, (i, j) in enumerate(pairs):
            gp = sim.get_g0_for_mode_pair(
                num_ions=int(num_ions),
                drive=probe_plus[k],
                mode_i=i,
                mode_j=j,
                units="Hz",
                recompute=False,
            )
            gm = sim.get_g0_for_mode_pair(
                num_ions=int(num_ions),
                drive=probe_minus[k],
                mode_i=i,
                mode_j=j,
                units="Hz",
                recompute=False,
            )
            col[idx] = (gp - gm) / (2.0 * h)  # Hz / Volt
        J[:, k] = col

    return {
        "J_Hz_per_V": J,
        "pairs": pairs,
        "g_center_Hz": g_center,
        "K": K,
        "M": M,
    }


def _pair_to_index_map(pairs):
    """Map (i,j) with i<j to row index in J/g vectors."""
    return {(i, j): idx for idx, (i, j) in enumerate(pairs)}


def _normalize_bounds_20(amp_bounds):
    """
    Normalize bounds into two length-20 arrays: lower, upper.
    Accepts:
      - float/int B  -> [-B, +B] for all channels
      - (lower, upper) each length-20 iterables
      - dict{name: (lo, hi)} subset; others default to (0,0)
    """
    import numpy as np

    if amp_bounds is None:
        lower = np.zeros(20)
        upper = np.zeros(20)
        return lower, upper

    if isinstance(amp_bounds, (int, float)):
        B = float(amp_bounds)
        return -B * np.ones(20), B * np.ones(20)

    # tuple/list of two arrays
    if (
        isinstance(amp_bounds, (list, tuple))
        and len(amp_bounds) == 2
        and isinstance(amp_bounds[0], (list, tuple, np.ndarray))
        and isinstance(amp_bounds[1], (list, tuple, np.ndarray))
    ):
        lo = np.asarray(amp_bounds[0], float).reshape(-1)
        hi = np.asarray(amp_bounds[1], float).reshape(-1)
        if lo.size != 20 or hi.size != 20:
            raise ValueError("amp_bounds arrays must be length 20.")
        return lo, hi

    # dict per electrode name
    if isinstance(amp_bounds, dict):
        order = [
            "DC1",
            "DC2",
            "DC3",
            "DC4",
            "DC5",
            "DC6",
            "DC7",
            "DC8",
            "DC9",
            "DC10",
            "RF11",
            "RF12",
            "RF13",
            "RF14",
            "RF15",
            "RF16",
            "RF17",
            "RF18",
            "RF19",
            "RF20",
        ]
        lo = np.zeros(20, float)
        hi = np.zeros(20, float)
        for k, name in enumerate(order):
            if name in amp_bounds:
                pair = amp_bounds[name]
                if (not isinstance(pair, (list, tuple))) or len(pair) != 2:
                    raise ValueError(f"Bounds for {name} must be (lo, hi).")
                lo[k] = float(pair[0])
                hi[k] = float(pair[1])
        return lo, hi

    raise ValueError(
        "amp_bounds must be a number, (lo,hi) arrays, or a dict per electrode."
    )


def find_max_coupling_matrix_F120(
    num_ions: int,
    constant_trappingvars,
    point=None,  # length-20 amplitudes (V)
    amp_bounds=None,  # scalar B or (lo,hi) arrays or dict per electrode
    simulation_preset: str = "Simp58_101",
):
    """
    Compute the absolute row-wise maxima of g0_ab (Hz) under box bounds on the 20 inputs.
    Uses the linearization from calculate_jacobian_F120 at 'point':
        g ≈ g_center + J * (x - point)

    Returns
    -------
    dict with:
      - 'Gmax_abs_Hz' : KxK matrix (upper triangle filled) of absolute maxima
      - 'pairs'       : list of (i,j) row mapping for J
      - 'J_Hz_per_V'  : (M x 20) Jacobian (Hz/V)
      - 'g_center_Hz' : (M,) center values (Hz)
      - 'K','M'       : mode and row counts
    """
    import numpy as np

    # Linearization at 'point'
    pack = calculate_jacobian_F120(
        num_ions=num_ions,
        constant_trappingvars=constant_trappingvars,
        point=point,
        simulation_preset=simulation_preset,
    )
    J = pack["J_Hz_per_V"]  # (M x 20)
    pairs = pack["pairs"]  # list of (i<j)
    g0 = pack["g_center_Hz"]  # (M,)
    K, M = pack["K"], pack["M"]

    # Bounds over x (absolute, not delta). We need bounds over delta := x - point
    lo_abs, hi_abs = _normalize_bounds_20(amp_bounds)
    p = np.zeros(20) if point is None else np.asarray(point, float).reshape(20)
    lo = lo_abs - p
    hi = hi_abs - p

    # For each row a:=J_row, b:=g0_row, maximize |b + a·delta| over box delta∈[lo,hi]
    Gmax = np.zeros((K, K), float)
    for row_idx, (i, j) in enumerate(pairs):
        a = J[row_idx, :]  # shape (20,)
        b = g0[row_idx]

        # Maximize (b + a·delta)
        x_pos = np.where(a >= 0.0, hi, lo)
        val_pos = b + float(a @ x_pos)

        # Maximize -(b + a·delta)  ≡  minimize (b + a·delta)
        x_neg = np.where(a <= 0.0, hi, lo)  # choose opposite corners
        val_neg = -(b + float(a @ x_neg))

        Gmax[i, j] = max(abs(val_pos), abs(val_neg))

    return {
        "Gmax_abs_Hz": Gmax,
        "pairs": pairs,
        "J_Hz_per_V": J,
        "g_center_Hz": g0,
        "K": K,
        "M": M,
    }


def solve_F120_for_targets(
    mode_pair_targets,  # list of ((i,j), target_Hz) with i<j
    num_ions: int,
    constant_trappingvars,
    point=None,  # length-20 amplitudes (V) about which J is taken
    amp_bounds=None,  # same formats as in find_max_coupling_matrix_F120
    l2_reg: float = 0.0,  # ridge parameter (Hz/V)^2 on Δx
    minimize_rest: bool = False,  # project along nullspace of J_sub to reduce rest
    rest_l2: float = 1e-8,  # tiny Tikhonov on nullspace solve
    clip_to_bounds: bool = True,
    simulation_preset: str = "Simp58_101",
):
    """
    Solve for Δx (20-vector, V) so that J_sub Δx ≈ y, where
      y = targets_Hz - g_center_sub.

    If minimize_rest=True, adjust along the nullspace of J_sub to reduce ||J_rest Δx||_2.

    Returns
    -------
    dict with:
      - 'amplitudes_vector'        : x = point + Δx (length-20)
      - 'amplitudes_by_electrode'  : dict {DC1..DC10, RF11..RF20}
      - 'predicted_targets_Hz'     : list of ((i,j), g_pred_ij)
      - 'predicted_rest_rms_Hz'    : float
      - 'J_Hz_per_V', 'pairs', 'g_center_Hz', 'K', 'M'
    """
    import numpy as np

    # Linearization at 'point'
    pack = calculate_jacobian_F120(
        num_ions=num_ions,
        constant_trappingvars=constant_trappingvars,
        point=point,
        simulation_preset=simulation_preset,
    )
    J = pack["J_Hz_per_V"]  # (M x 20)
    pairs = pack["pairs"]
    g0 = pack["g_center_Hz"]
    K, M = pack["K"], pack["M"]
    pair2idx = _pair_to_index_map(pairs)

    # Targets subset
    targ_rows = []
    y = []
    for (i, j), tval in mode_pair_targets:
        if i >= j:
            raise ValueError(f"Target pair must satisfy i<j, got {(i,j)}.")
        r = pair2idx.get((i, j))
        if r is None:
            raise ValueError(f"Pair {(i,j)} not in pairs.")
        targ_rows.append(r)
        y.append(float(tval) - g0[r])
    targ_rows = np.asarray(targ_rows, int)
    y = np.asarray(y, float)  # shape (R,)

    J_sub = J[targ_rows, :]  # (R x 20)

    # Ridge least squares for Δx
    # Δx = argmin ||J_sub Δx - y||^2 + λ ||Δx||^2
    lam = float(l2_reg)
    if J_sub.size == 0:
        dx = np.zeros(20, float)
    else:
        A = J_sub.T @ J_sub
        if lam > 0.0:
            A = A + lam * np.eye(20)
        b = J_sub.T @ y
        dx = np.linalg.solve(A, b)

    # Optional nullspace tweak to reduce off-target couplings
    if minimize_rest and J_sub.size > 0:
        # Nullspace of J_sub via SVD
        U, S, Vt = np.linalg.svd(J_sub, full_matrices=False)
        tol = max(J_sub.shape) * np.max(S) * 1e-12
        r = np.sum(S > tol)
        N = Vt[r:, :].T  # (20 x d), basis for nullspace
        if N.size > 0:
            J_rest = np.delete(J, targ_rows, axis=0)  # (M-R x 20)
            # Minimize ||J_rest (dx + N z)||^2 + μ ||z||^2
            A2 = N.T @ (J_rest.T @ J_rest) @ N + float(rest_l2) * np.eye(N.shape[1])
            b2 = -N.T @ (J_rest.T @ J_rest) @ dx
            z = np.linalg.solve(A2, b2)
            dx = dx + N @ z

    # Clip to bounds if requested
    p = np.zeros(20) if point is None else np.asarray(point, float).reshape(20)
    x = p + dx

    if amp_bounds is not None and clip_to_bounds:
        lo_abs, hi_abs = _normalize_bounds_20(amp_bounds)
        x = np.minimum(np.maximum(x, lo_abs), hi_abs)
        dx = x - p

    # Predicted full g (Hz)
    g_pred = g0 + J @ dx

    # Report targets and rest RMS
    pred_targets = [
        ((i, j), float(g_pred[pair2idx[(i, j)]])) for (i, j), _ in mode_pair_targets
    ]
    mask_rest = np.ones(M, dtype=bool)
    mask_rest[targ_rows] = False
    rest_rms = (
        float(np.sqrt(np.mean(g_pred[mask_rest] ** 2))) if np.any(mask_rest) else 0.0
    )

    # Amplitudes mapping
    amps = _vec20_to_dc_rf_map(x)

    return {
        "amplitudes_vector": x,
        "amplitudes_by_electrode": amps,
        "predicted_targets_Hz": pred_targets,
        "predicted_rest_rms_Hz": rest_rms,
        "J_Hz_per_V": J,
        "pairs": pairs,
        "g_center_Hz": g0,
        "K": K,
        "M": M,
    }


def main2():
    # --- Hard-coded configuration (mirror F101 style) ---
    t0 = time.time()
    preset = "Simp58_101"
    num_ions = 3
    rf_freq_hz = 25_500_000
    rf1_v = 377.0
    rf2_v = 377.0

    # 20-channel linearization point (DC1..DC10, RF11..RF15, RF16..RF20)
    point = [0.0] * 20

    # Build trapping variables with RF only (F120 adds center/probe drives internally)
    tv = Trapping_Vars()
    tv.apply_dc_twist_endcaps(twist=0.28, endcaps=3)
    tv.add_driving("RF", rf_freq_hz, 0.0, {"RF1": rf1_v, "RF2": rf2_v})

    # Solve a small target set to put the path through its paces
    vars = solve_F120_for_targets(
        mode_pair_targets=[((0, 1), 100.0), ((3, 5), 200.0), ((0, 3), 0.0)],
        num_ions=num_ions,
        constant_trappingvars=tv,
        point=point,
        minimize_rest=False,
        simulation_preset=preset,
    )

    print("\nF120 solve for targets:")
    from pprint import pprint

    pprint(vars)

    print(f"\nElapsed: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main2()
