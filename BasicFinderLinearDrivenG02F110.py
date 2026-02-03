# C:\GitHub\TrapFrequencyAnalysis\BasicFinderLinearDrivenG02F110.py
"""
F110: Build the Jacobian for the map R^5 -> R^{C(3N,2)} that sends the five
symmetric DC pair amplitudes (extra-drive channel) to *all* two-mode driven
couplings g0_ab (Hz), with DC/RF/modes held fixed.

We do central differences using a single Simulation that contains:
- one center drive at 'point'
- ±h probe drives for each of the 5 input axes

Return:
- J (M x 5) in Hz/V, where rows correspond to (i,j) with 0 <= i < j < K = 3N
- pairs: list of (i,j) giving the row order
- g_center: length-M vector of g0 at 'point' (Hz), same row order as J
"""
print("hi")
import math
import numpy as np
import time

from sim.simulation import Simulation
from trapping_variables import Trapping_Vars
from typing import List, Tuple


def _vec5_to_symmetric_dc_map(v5):
    a1, a2, a3, a4, a5 = [float(x) for x in v5]
    return {
        "DC1": a1,
        "DC10": a1,
        "DC2": a2,
        "DC9": a2,
        "DC3": a3,
        "DC8": a3,
        "DC4": a4,
        "DC7": a4,
        "DC5": a5,
        "DC6": a5,
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


def _flatten_upper_triangle(G):
    import numpy as np

    K = G.shape[0]
    vec, pairs = [], []
    for i in range(K):
        for j in range(i + 1, K):
            vec.append(G[i, j])  # <-- no symmetrization
            pairs.append((i, j))
    return np.asarray(vec, float), pairs


def calculate_jacobian_F110(
    num_ions: int = 3,
    constant_trappingvars=Trapping_Vars,  # instance preferred; class allowed
    point=[0.0, 0.0, 0.0, 0.0, 0.0],  # 5 symmetric DOFs (V): (1=10,2=9,3=8,4=7,5=6)
    simulation_preset: str = "Simp58_101",
):
    """
    Build the full F110 Jacobian (all pair couplings vs 5 inputs) using the *pair accessor path*
    to ensure exact agreement with F101.

    Returns
    -------
    dict with:
      - 'J_Hz_per_V' : (M x 5) array, rows in (i<j) order, units Hz/V
      - 'pairs'      : list[(i,j)] row mapping
      - 'g_center_Hz': (M,) array of g0 at 'point' (Hz)
      - 'K'          : number of modes (3N)
      - 'M'          : number of pairs (K*(K-1)//2)
    """
    # Validate inputs
    if not isinstance(point, (list, tuple)) or len(point) != 5:
        raise ValueError(
            "point must be a length-5 list/tuple of floats (symmetric DC DOFs)."
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
    is_zero_point = all(abs(x) == 0.0 for x in point)
    h = 0.5 if is_zero_point else 0.05

    # Fresh TV with DC/RF copied; add center and ±h probe drives
    tv = _copy_tv(tv_in)

    dk_center = tv.add_driving(
        "F110_center", 77.0, 0.0, _vec5_to_symmetric_dc_map(point)
    )

    probe_plus, probe_minus = [], []
    for k in range(5):
        p = list(point)
        p[k] += h
        m = list(point)
        m[k] -= h
        dk_p = tv.add_driving(
            f"F110_p{k+1}", 1.0 + k, 0.0, _vec5_to_symmetric_dc_map(p)
        )
        dk_m = tv.add_driving(
            f"F110_m{k+1}", 101.0 + k, 0.0, _vec5_to_symmetric_dc_map(m)
        )
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
    # (We build the matrix once to populate cache, then read per pair to mirror F101.)
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
    J = np.empty((M, 5), dtype=float)
    for k in range(5):
        # Warm build matrices for caches (once per drive)
        sim.get_g0_matrix(int(num_ions), probe_plus[k])
        sim.get_g0_matrix(int(num_ions), probe_minus[k])

        # Row k: derivative for each (i,j)
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


def _pair_indices(num_ions: int) -> List[Tuple[int, int]]:
    """Return the (i,j) mode index pairs in the same order used by F110: i<j over 0..3N-1."""
    dim = 3 * num_ions
    pairs = []
    for i in range(dim):
        for j in range(i + 1, dim):
            pairs.append((i, j))
    return pairs


def _rowwise_linear_max(row: np.ndarray, bounds: List[Tuple[float, float]]) -> float:
    """
    For linear objective g(u) = row · u with box bounds, the max is achieved
    by taking u_j = upper if row_j >= 0 else lower.
    """
    total = 0.0
    for w, (lo, hi) in zip(row, bounds):
        total += (hi * w) if w >= 0 else (lo * w)
    return total


def find_max_coupling_matrix_F110(
    num_ions: int = 3,
    constant_trappingvars: Trapping_Vars = Trapping_Vars,
    point: List[float] = None,
    input_bounds: List[Tuple[float, float]] = None,
    simulation_preset: str = "Simp58_101",
):
    """
    Maximize each pairwise coupling g0_{i,j}(u) over a 5-D box of inputs and
    return a 3N x 3N matrix M with the upper triangle (i<j) set to the max value.

    Uses g(u) = J u + c, where:
      - J (Hz/V) and g_center (Hz) come from calculate_jacobian_F110 at 'point'
      - c = g_center - J @ point  (vector over all pairs)

    Args:
        num_ions: N (so K=3N modes).
        constant_trappingvars: fixed DC/RF Trapping_Vars (instance or class).
        point: 5-vector (Volts) where the Jacobian is evaluated (default zeros).
        input_bounds: list of 5 (lo, hi) tuples in Volts (default [(-0.5,0.5)]*5).
        simulation_preset: Simulation preset string.

    Returns:
        M: np.ndarray (K x K), upper triangle filled with per-pair maxima (Hz).
    """
    if point is None:
        point = [0.0, 0.0, 0.0, 0.0, 0.0]
    if input_bounds is None:
        input_bounds = [(-0.5, 0.5)] * 5
    if len(point) != 5:
        raise ValueError("point must be length-5.")
    if len(input_bounds) != 5:
        raise ValueError("input_bounds must be 5 (lo, hi) tuples.")

    # 1) Get full F110 Jacobian & baseline at 'point'
    out = calculate_jacobian_F110(
        num_ions=num_ions,
        constant_trappingvars=constant_trappingvars,
        point=point,
        simulation_preset=simulation_preset,
    )
    J = out["J_Hz_per_V"]  # (M x 5) Hz/V
    g_center = out["g_center_Hz"]  # (M,) Hz
    pairs = out["pairs"]  # list[(i,j)]
    K = out["K"]
    M_pairs = out["M"]

    # 2) Intercept vector c = g(point) - J @ point
    point_vec = np.asarray(point, float)
    c_vec = g_center - J @ point_vec  # (M,)

    # 3) Row-wise box maximization of J u + c:
    #    For each row w, choose u_k = hi if w_k >= 0 else lo.
    lo = np.array([b[0] for b in input_bounds], float)
    hi = np.array([b[1] for b in input_bounds], float)
    choose = (J >= 0.0).astype(float)  # (M x 5)
    u_star = choose * hi + (1.0 - choose) * lo  # (M x 5) row-wise optimal inputs
    g_max_vec = (J * u_star).sum(axis=1) + c_vec  # (M,)

    # 4) Fill the K x K matrix (upper triangle only)
    Mmat = np.zeros((K, K), float)
    if len(pairs) != M_pairs or M_pairs != J.shape[0]:
        raise RuntimeError("Pair ordering mismatch in F110 outputs.")
    for idx, (i, j) in enumerate(pairs):
        Mmat[i, j] = g_max_vec[idx]

    return Mmat


import numpy as np
from typing import List, Tuple, Dict


def solve_F110_for_targets(
    mode_pair_targets: List[Tuple[Tuple[int, int], float]],
    num_ions: int = 3,
    constant_trappingvars=Trapping_Vars,  # instance preferred; class allowed
    point: List[float] = None,  # 5-vector (Volts); absolute inputs
    simulation_preset: str = "Simp58_101",
    minimize_rest: bool = False,  # try to reduce all other g0's
    reg_l2: float = 0.0,  # optional Tikhonov on u (Volts^2); 0 = off
    svd_tol: float = 1e-10,  # rank/NS tolerance
) -> Dict:
    """
    Solve for inputs u ∈ R^5 so that selected pairwise couplings reach desired values,
    using the linear F110 model: g(u) = J u + c  (Hz).

    - Primary objective: hit the specified targets exactly (when feasible).
    - Secondary (optional): minimize ||g_rest(u)||_2 without changing the achieved targets.

    Returns a dict with:
      'u_star'            : list[5] Volt
      'achieved_Hz'       : dict{(i,j): value} for requested pairs at u_star
      'achieved_err_Hz'   : dict{(i,j): target - achieved}
      'rank'              : rank of constraint matrix
      'nullspace_dim'     : dimension of the nullspace used for rest minimization
      'pred_rest_norm_Hz' : ||g_rest(u_star)||_2
    """
    if point is None:
        point = [0.0, 0.0, 0.0, 0.0, 0.0]
    if len(point) != 5:
        raise ValueError("point must be a length-5 vector (Volts).")
    if not mode_pair_targets:
        raise ValueError(
            "mode_pair_targets must be a non-empty list of ((i,j), target_Hz)."
        )

    # 1) Build F110 mapping: g(u) = J u + c
    out = calculate_jacobian_F110(
        num_ions=num_ions,
        constant_trappingvars=constant_trappingvars,
        point=point,
        simulation_preset=simulation_preset,
    )
    J = out["J_Hz_per_V"]  # (M x 5) Hz/V
    pairs = out["pairs"]  # list[(i,j)]
    g_point = out["g_center_Hz"]  # (M,) g(point)
    point_vec = np.asarray(point, float)

    # c = g(point) - J @ point
    c = g_point - J @ point_vec  # (M,)

    # Row index map for quick lookup
    row_of: Dict[Tuple[int, int], int] = {p: k for k, p in enumerate(pairs)}

    # 2) Build constraint system A u = b for requested pairs
    targets = []
    rows = []
    for (i, j), tgt in mode_pair_targets:
        if i > j:
            i, j = j, i
        if (i, j) not in row_of:
            raise ValueError(f"Requested pair {(i,j)} is out of range.")
        k = row_of[(i, j)]
        rows.append(k)
        targets.append(float(tgt))
    A = J[rows, :]  # (m x 5)
    b = np.asarray(targets, float) - c[rows]  # desired (J u) values

    # 3) Primary solve: exact targets (when feasible) with min-norm u (and optional L2 reg)
    # Solve (A^T A + reg I) u = A^T b  (if reg_l2>0), else use min-norm least squares.
    m = A.shape[0]
    if reg_l2 > 0:
        ATA = A.T @ A + reg_l2 * np.eye(5)
        ATb = A.T @ b
        u0 = np.linalg.solve(ATA, ATb)  # R^5
    else:
        # min-norm LS; if A has full row-rank (m<=5), this hits targets exactly
        u0, resid, rnk, s = np.linalg.lstsq(A, b, rcond=None)

    # 4) Optional: minimize rest ||J_rest u + c_rest|| with u constrained to keep A u = b
    u = u0.copy()
    null_dim = 0
    pred_rest_norm = None
    if minimize_rest and m < 5:
        # Nullspace of A
        U, S, VT = np.linalg.svd(A, full_matrices=True)
        r = np.sum(S > svd_tol)
        null_dim = 5 - r
        if null_dim > 0:
            N = VT[r:, :].T  # (5 x null_dim)
            # Rest objective: minimize ||J_rest (u0 + N z) + c_rest||_2
            mask = np.ones(J.shape[0], dtype=bool)
            mask[np.array(rows, int)] = False
            J_rest = J[mask, :]  # (M-m x 5)
            c_rest = c[mask]  # (M-m,)
            y = J_rest @ u0 + c_rest  # (M-m,)
            B = J_rest @ N  # (M-m x null_dim)
            # Solve min_z ||B z + y||_2  -> z* = - B^+ y
            if B.size > 0:
                z_star, *_ = np.linalg.lstsq(B, -y, rcond=None)
                u = u0 + N @ z_star  # stays feasible: A u = b
                pred_rest_norm = float(np.linalg.norm(B @ z_star + y))
        else:
            pred_rest_norm = float(np.linalg.norm((J @ u + c)[mask]))
    else:
        # just report rest norm from u0
        mask = np.ones(J.shape[0], dtype=bool)
        mask[np.array(rows, int)] = False
        pred_rest_norm = float(np.linalg.norm((J @ u + c)[mask]))

    # 5) Final outputs and diagnostics
    g_pred = J @ u + c  # all pairs (Hz)
    achieved = {}
    achieved_err = {}
    for (i, j), tgt in mode_pair_targets:
        if i > j:
            i, j = j, i
        k = row_of[(i, j)]
        val = float(g_pred[k])
        achieved[(i, j)] = val
        achieved_err[(i, j)] = float(tgt - val)

    # Rank of A (use same tol)
    rank = int(np.linalg.matrix_rank(A, tol=svd_tol))

    return {
        "u_star": u.tolist(),
        "achieved_Hz": achieved,
        "achieved_err_Hz": achieved_err,
        "rank": rank,
        "nullspace_dim": null_dim,
        "pred_rest_norm_Hz": pred_rest_norm,
    }


def main():
    # --- Hard-coded configuration (mirror F101 style) ---
    time1 = time.time()
    preset = "Simp58_101"
    num_ions = 4
    rf_freq_hz = 25500000
    rf1_v = 377.0
    rf2_v = 377.0
    point = [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]  # symmetric DC pair inputs (1=10, 2=9, 3=8, 4=7, 5=6)

    # Build trapping variables with RF only (extra-drive probes are added inside F110)
    tv = Trapping_Vars()
    tv.apply_dc_twist_endcaps(twist=0.28, endcaps=3)
    tv.add_driving("RF", rf_freq_hz, 0.0, {"RF1": rf1_v, "RF2": rf2_v})

    # Run F110 Jacobian builder
    out = calculate_jacobian_F110(
        num_ions=num_ions,
        constant_trappingvars=tv,
        point=point,
        simulation_preset=preset,
    )

    J = out["J_Hz_per_V"]  # (M x 5)
    pairs = out["pairs"]  # length M list of (i,j)
    g_center = out["g_center_Hz"]  # (M,)
    K = out["K"]
    M = out["M"]

    # --- Print a quick report ---
    print("=== F110 sanity ===")
    print(f"Preset: {preset}")
    print(f"N ions: {num_ions}  |  K modes: {K}  |  M pairs: {M}")
    print(f"Jacobian shape: {J.shape} (Hz/V)")
    # Column norms tell you how effective each symmetric channel is
    col_norms = (J**2).sum(axis=0) ** 0.5
    print("Column L2 norms (Hz/V):", "  ".join(f"{n:.3e}" for n in col_norms))
    # Baseline coupling stats at 'point'
    print(
        f"Center g0 stats at point={point}: min={g_center.min():.3e} Hz, "
        f"max={g_center.max():.3e} Hz, mean={g_center.mean():.3e} Hz"
    )

    # Show first 8 rows as a spot-check
    print("\nFirst 8 rows (pair: sensitivities [Hz/V]):")
    show_rows = min(8, M)
    for r in range(show_rows):
        i, j = pairs[r]
        sens = "  ".join(f"{x:.3e}" for x in J[r, :])
        print(f"({i:2d},{j:2d})  {sens}")

    print(out)
    time2 = time.time()
    print("time:", time2 - time1)

    maxmatrix = find_max_coupling_matrix_F110(
        num_ions=num_ions,
        constant_trappingvars=tv,
        point=point,
        input_bounds=[(-1, 1)] * 5,
    )

    print("\nMax coupling matrix (upper triangle):")

    def print_bordered_matrix(matrix):
        s = [[str(e) for e in row] for row in matrix]
        lens = [max(map(len, col)) for col in zip(*s)]
        fmt = ' │ '.join('{{:{}}}'.format(x) for x in lens)

        # Top border
        print('┌─' + '─┬─'.join('─' * l for l in lens) + '─┐')

        for i, row in enumerate(s):
            print('│ ' + fmt.format(*row) + ' │')
            if i < len(s) - 1:
                print('├─' + '─┼─'.join('─' * l for l in lens) + '─┤')

        # Bottom border
        print('└─' + '─┴─'.join('─' * l for l in lens) + '─┘')

    print_bordered_matrix(maxmatrix)


def main2():
    # --- Hard-coded configuration (mirror F101 style) ---
    time1 = time.time()
    preset = "Simp58_101"
    num_ions = 4
    rf_freq_hz = 25500000
    rf1_v = 377.0
    rf2_v = 377.0
    point = [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]  # symmetric DC pair inputs (1=10, 2=9, 3=8, 4=7, 5=6)

    # Build trapping variables with RF only (extra-drive probes are added inside F110)
    tv = Trapping_Vars()
    tv.apply_dc_twist_endcaps(twist=0.28, endcaps=3)
    tv.add_driving("RF", rf_freq_hz, 0.0, {"RF1": rf1_v, "RF2": rf2_v})

    vars = solve_F110_for_targets(mode_pair_targets=[((0, 1), 100.0), ((3, 5), 200.0), ((0,3), 0)], num_ions=3, constant_trappingvars=tv, minimize_rest=False)
    print("\nF110 solve for targets:")
    print(vars)


if __name__ == "__main__":
    main2()
