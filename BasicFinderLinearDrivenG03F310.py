# C:\GitHub\TrapFrequencyAnalysis\BasicFinderLinearDrivenG03F310.py
"""
F310: Build the Jacobian for the map R^5 -> R^{C(3N,2) * (3N)} that sends the five
symmetric DC pair amplitudes (extra-drive channel) to *all* three-mode-driven
couplings g0_{a,b,c} (Hz), with DC/RF/modes held fixed.

We replicate F110’s pattern (center + ±h probes in the extra-drive channel), but:
- at each probe we evaluate the full driven cubic tensor G3 := get_driven_g_0_3_tensor(...),
- rows correspond to triples (a<b, c), flattened in canonical order.

Returns:
- J (T x 5) in Hz/V, rows correspond to triples (a<b, c)
- triples: list of (a, b, c) giving the row order
- g_center: length-T vector of g0^{(3)} at 'point' (Hz), same row order as J
- K: number of modes (3N)
- T: number of triples (= K*(K-1)//2 * K)
"""
print("Loading BasicFinderLinearDrivenG03F310.py")
import math
import time
import numpy as np
from typing import List, Tuple

from sim.simulation import Simulation
from trapping_variables import Trapping_Vars

print("hi")
def _vec5_to_symmetric_dc_map(v5):
    """
    Map a 5-vector (1↔10, 2↔9, 3↔8, 4↔7, 5↔6) into the symmetric DC electrode map.
    """
    a1, a2, a3, a4, a5 = [float(x) for x in v5]
    return {
        "DC1": a1,  "DC10": a1,
        "DC2": a2,  "DC9":  a2,
        "DC3": a3,  "DC8":  a3,
        "DC4": a4,  "DC7":  a4,
        "DC5": a5,  "DC6":  a5,
    }


def _copy_tv(tv_in: Trapping_Vars) -> Trapping_Vars:
    """
    Copy DC map and all non-DC drives (e.g., RF) into a fresh Trapping_Vars.
    Extra-drive probes for F310 are added on top of this copy.
    """
    tv = Trapping_Vars()
    # DC
    dc_in = tv_in.get_drive_amplitudes(tv_in.dc_key)
    for el, A in dc_in.items():
        tv.Var_dict[tv.dc_key].set_amplitude_volt(el, A)
    tv._update_pickoff_for_drive(tv.dc_key)
    # non-DC drives (RF, etc.)
    for dk in tv_in.get_drives():
        if dk == tv_in.dc_key:
            continue
        amps = tv_in.get_drive_amplitudes(dk)
        tv.add_driving(dk.label, dk.f_hz, dk.phi, amps)
    return tv


def calculate_jacobian_F310(
    num_ions: int = 3,
    constant_trappingvars=Trapping_Vars,  # instance preferred; class allowed
    point: List[float] = None,  # 5 symmetric DOFs (V): (1=10,2=9,3=8,4=7,5=6)
    simulation_preset: str = "Simp58_101",
):
    """
    Build the full F310 Jacobian (all triples (a<b,c) vs 5 inputs) using the
    driven cubic accessor path `get_driven_g_0_3_tensor`.

    Returns
    -------
    dict with:
      - 'J_Hz_per_V' : (T x 5) array, rows in (a<b,c) canonical order, units Hz/V
      - 'triples'    : list[(a,b,c)] row mapping
      - 'g_center_Hz': (T,) array of g0^{(3)} at 'point' (Hz)
      - 'K'          : number of modes (3N)
      - 'T'          : number of triples (= K*(K-1)//2 * K)
    """
    if point is None:
        point = [0.0, 0.0, 0.0, 0.0, 0.0]
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

    # Step size rule: larger at origin, smaller otherwise (match F110 convention)
    is_zero_point = all(abs(x) == 0.0 for x in point)
    h = 0.5 if is_zero_point else 0.05

    # Fresh TV with DC/RF copied; add center and ±h probe drives in extra-drive channel
    tv = _copy_tv(tv_in)

    dk_center = tv.add_driving(
        "F310_center", 77.0, 0.0, _vec5_to_symmetric_dc_map(point)
    )

    probe_plus, probe_minus = [], []
    for k in range(5):
        p = list(point)
        p[k] += h
        m = list(point)
        m[k] -= h
        dk_p = tv.add_driving(
            f"F310_p{k+1}", 1.0 + k, 0.0, _vec5_to_symmetric_dc_map(p)
        )
        dk_m = tv.add_driving(
            f"F310_m{k+1}", 101.0 + k, 0.0, _vec5_to_symmetric_dc_map(m)
        )
        probe_plus.append(dk_p)
        probe_minus.append(dk_m)

    # One Simulation; lock equilibrium and modes once
    sim = Simulation(simulation_preset, tv)
    sim.find_equilib_position_single(int(num_ions))
    sim.get_static_normal_modes_and_freq(
        int(num_ions), normalize=True, sort_by_freq=True
    )

    # Mode count
    K = sim.normal_modes_and_frequencies[int(num_ions)]["modes"].shape[1]

    # Canonical triple list: (a<b) for the pair, then iterate c over 0..K-1
    triples = [(a, b, c) for a in range(K) for b in range(a + 1, K) for c in range(K)]
    T = len(triples)

    # --- Center tensor in Hz and baseline vector ---
    # Warm build for center; returns full (K x K x K) in Hz
    G_center = sim.get_driven_g_0_3_tensor(int(num_ions), dk_center)
    # Flatten to match triples order
    g_center = np.fromiter(
        (G_center[a, b, c] for (a, b, c) in triples), dtype=float, count=T
    )

    # --- Jacobian columns via central differences over 5 inputs ---
    J = np.empty((T, 5), dtype=float)
    for k in range(5):
        Gp = sim.get_driven_g_0_3_tensor(int(num_ions), probe_plus[k])
        Gm = sim.get_driven_g_0_3_tensor(int(num_ions), probe_minus[k])

        # central diff for each (a,b,c)
        col = np.fromiter(
            ((Gp[a, b, c] - Gm[a, b, c]) / (2.0 * h) for (a, b, c) in triples),
            dtype=float,
            count=T,
        )
        J[:, k] = col

    return {
        "J_Hz_per_V": J,
        "triples": triples,
        "g_center_Hz": g_center,
        "K": K,
        "T": T,
    }


def find_max_coupling_tensor_F310(
    num_ions: int,
    constant_trappingvars,
    bounds5,
    point=None,
    simulation_preset: str = "Simp58_101",
    return_argmax: bool = True,
):
    """
    Row-wise (per triple (a<b, c)) linear box maximization of |J row · u + g0| over u in bounds5.

    Parameters
    ----------
    num_ions : int
        Number of ions (K = 3*num_ions modes).
    constant_trappingvars : Trapping_Vars or type(Trapping_Vars)
        Baseline TV (instance or class); DC/RF copied, extra-drive added here.
    bounds5 : (lo, hi) or list/tuple of 5 pairs
        Box constraints in volts for the five symmetric DC DOFs.
        Examples: (-1, 1) or [(-1,1)]*5 or [(-0.5,0.5), ... five entries ...].
    point : length-5 list/tuple
        Center operating point (V) for the 5 inputs (1↔10, 2↔9, 3↔8, 4↔7, 5↔6).
    simulation_preset : str
        Simulation preset name.
    return_argmax : bool
        If True, also return the maximizing 5-vector per triple (T x 5).

    Returns
    -------
    dict with:
      - 'Gmax_Hz' : (K x K x K) tensor of per-(a<b,c) maximal |g| (symmetric in a,b).
      - 'Ustar_V' : (T x 5) array of argmax inputs (if return_argmax=True), row-ordered like 'triples'.
      - 'triples' : list of (a,b,c) defining the row order used internally.
      - 'K'       : number of modes.
    """
    # Normalize bounds
    if (
        isinstance(bounds5, (tuple, list))
        and len(bounds5) == 2
        and all(isinstance(x, (int, float)) for x in bounds5)
    ):
        lo, hi = bounds5
        bounds = [(float(lo), float(hi))] * 5
    else:
        if not (isinstance(bounds5, (list, tuple)) and len(bounds5) == 5):
            raise ValueError(
                "bounds5 must be a (lo, hi) pair or a length-5 list/tuple of (lo, hi) pairs."
            )
        bounds = [(float(lo), float(hi)) for (lo, hi) in bounds5]

    # Build Jacobian and baseline at 'point'
    out = calculate_jacobian_F310(
        num_ions=num_ions,
        constant_trappingvars=constant_trappingvars,
        point=point if point is not None else [0.0] * 5,
        simulation_preset=simulation_preset,
    )
    J = out["J_Hz_per_V"]  # (T x 5)
    g0 = out["g_center_Hz"]  # (T,)
    triples = out["triples"]  # list[(a,b,c)]
    K = out["K"]
    T = len(triples)

    # Prebuild two corners that extremize a linear form row-wise:
    # For maximizing J·u: choose u_k = hi if J_k > 0 else lo
    # For minimizing J·u: choose u_k = lo if J_k > 0 else hi
    Ustar = np.zeros((T, 5), dtype=float) if return_argmax else None
    Gmax = np.zeros((K, K, K), dtype=float)

    lo_vec = np.array([b[0] for b in bounds], dtype=float)
    hi_vec = np.array([b[1] for b in bounds], dtype=float)

    # Row-wise maximize absolute value of affine function
    for t in range(T):
        row = J[t, :]  # (5,)
        choose_hi_for_max = row >= 0.0
        u_plus = np.where(choose_hi_for_max, hi_vec, lo_vec)  # maximizes J·u
        u_minus = np.where(choose_hi_for_max, lo_vec, hi_vec)  # minimizes J·u

        g_plus = g0[t] + float(np.dot(row, u_plus))
        g_minus = g0[t] + float(np.dot(row, u_minus))

        if abs(g_plus) >= abs(g_minus):
            g_best = g_plus
            u_best = u_plus
        else:
            g_best = g_minus
            u_best = u_minus

        a, b, c = triples[t]
        # Symmetric placement in (a,b)
        Gmax[a, b, c] = abs(g_best)
        Gmax[b, a, c] = abs(g_best)

        if return_argmax:
            Ustar[t, :] = u_best

    result = {"Gmax_Hz": Gmax, "triples": triples, "K": K}
    if return_argmax:
        result["Ustar_V"] = Ustar
    return result


def solve_F310_for_targets(
    num_ions: int,
    constant_trappingvars,
    target_tensor_Hz: np.ndarray,
    bounds5 = (-1.0, 1.0),
    point=None,
    simulation_preset: str = "Simp58_101",
    rest_penalty: float = 0.0,
    clip_to_bounds: bool = True,
):
    """
    Solve for a single 5-vector u (symmetric DC inputs) that best matches selected
    g0^{(3)} targets in the linear model g ≈ g0 + J u.

    Parameters
    ----------
    num_ions : int
        Number of ions (K = 3*num_ions).
    constant_trappingvars : Trapping_Vars or type(Trapping_Vars)
        Baseline TV (instance or class).
    target_tensor_Hz : (K x K x K) array
        Desired g0^{(3)}(a,b,c) in Hz. Use np.nan for entries to ignore.
        Only (a<b) are meaningful; (b,a,c) is mirrored.
    bounds5 : (lo,hi) or list/tuple of 5 pairs
        Box constraints (V) for the five inputs.
    point : length-5 list/tuple
        Operating point (V) around which the Jacobian is computed.
    simulation_preset : str
        Simulation preset name.
    rest_penalty : float
        Ridge penalty λ ≥ 0 for ||u||^2 (helps conditioning). λ=0 gives plain LS.
    clip_to_bounds : bool
        If True, clip the solved u to the provided bounds.

    Returns
    -------
    dict with:
      - 'u_solution_V' : (5,) optimal inputs (possibly clipped).
      - 'G_pred_Hz'    : (K x K x K) predicted tensor using linear model at u.
      - 'residual_rms' : RMS residual over selected rows (Hz).
      - 'selected_rows': number of constrained (non-NaN) targets used.
      - 'triples'      : list of (a,b,c) row order used internally.
      - 'K'            : number of modes.
    """
    # Normalize bounds
    if (
        isinstance(bounds5, (tuple, list))
        and len(bounds5) == 2
        and all(isinstance(x, (int, float)) for x in bounds5)
    ):
        lo, hi = bounds5
        bounds = [(float(lo), float(hi))] * 5
    else:
        if not (isinstance(bounds5, (list, tuple)) and len(bounds5) == 5):
            raise ValueError(
                "bounds5 must be a (lo, hi) pair or a length-5 list/tuple of (lo, hi) pairs."
            )
        bounds = [(float(lo), float(hi)) for (lo, hi) in bounds5]
    lo_vec = np.array([b[0] for b in bounds], dtype=float)
    hi_vec = np.array([b[1] for b in bounds], dtype=float)

    # Jacobian and baseline at 'point'
    out = calculate_jacobian_F310(
        num_ions=num_ions,
        constant_trappingvars=constant_trappingvars,
        point=point if point is not None else [0.0] * 5,
        simulation_preset=simulation_preset,
    )
    J = out["J_Hz_per_V"]  # (T x 5)
    g0 = out["g_center_Hz"]  # (T,)
    triples = out["triples"]
    K = out["K"]
    T = len(triples)

    # Validate target tensor
    if not (
        isinstance(target_tensor_Hz, np.ndarray) and target_tensor_Hz.shape == (K, K, K)
    ):
        raise ValueError(
            f"target_tensor_Hz must be a numpy array with shape {(K, K, K)}."
        )

    # Build selection mask and target vector in row order
    sel_rows = []
    t_vec = []
    for t in range(T):
        a, b, c = triples[t]
        tgt = target_tensor_Hz[a, b, c]
        if np.isnan(tgt):
            continue
        sel_rows.append(t)
        t_vec.append(float(tgt))
    if len(sel_rows) == 0:
        raise ValueError(
            "No valid targets found (all NaN). Provide at least one (a<b,c) target."
        )

    A = J[sel_rows, :]  # (M x 5)
    y = np.asarray(t_vec) - g0[sel_rows]  # desired delta (Hz)

    # Ridge-regularized least squares: minimize ||A u - y||^2 + λ ||u||^2
    # Solve (A^T A + λ I) u = A^T y
    ATA = A.T @ A
    ATy = A.T @ y
    if rest_penalty > 0.0:
        ATA = ATA + rest_penalty * np.eye(5)

    try:
        u = np.linalg.solve(ATA, ATy)
    except np.linalg.LinAlgError:
        # Fallback to lstsq if singular
        u, *_ = np.linalg.lstsq(ATA, ATy, rcond=None)

    # Optional clipping to bounds
    if clip_to_bounds:
        u = np.minimum(hi_vec, np.maximum(lo_vec, u))

    # Predicted full tensor using linear model
    g_pred_rows = g0 + J @ u  # (T,)
    G_pred = np.zeros((K, K, K), dtype=float)
    for t in range(T):
        a, b, c = triples[t]
        val = g_pred_rows[t]
        G_pred[a, b, c] = val
        G_pred[b, a, c] = val

    # Residual on selected rows
    pred_sel = g_pred_rows[sel_rows]
    resid = pred_sel - np.asarray(t_vec)
    residual_rms = float(np.sqrt(np.mean(resid**2))) if len(resid) > 0 else 0.0

    return {
        "u_solution_V": u,
        "G_pred_Hz": G_pred,
        "residual_rms": residual_rms,
        "selected_rows": len(sel_rows),
        "triples": triples,
        "K": K,
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
    # Sanity checks for F310 on a tiny case (2 ions)
    np.set_printoptions(precision=4, suppress=True)

    num_ions = 2
    tv0 = Trapping_Vars()  # baseline TV; DC/RF copied by _copy_tv inside
    point = [0.0, 0.0, 0.0, 0.0, 0.0]  # center operating point for the 5 symmetric DOFs
    bounds = (-0.5, 0.5)  # box bounds for each of the five inputs (V)

    print("=== F310: calculate_jacobian_F310 ===")
    out = calculate_jacobian_F310(
        num_ions=num_ions,
        constant_trappingvars=tv0,
        point=point,
        simulation_preset="Simp58_101",
    )
    J = out["J_Hz_per_V"]  # (T x 5)
    g0 = out["g_center_Hz"]  # (T,)
    triples = out["triples"]  # list of (a,b,c)
    K = out["K"]
    T = out["T"]

    print(f"K (modes) = {K}")
    print(f"T (triples) = {T}")
    print(f"J shape = {J.shape}, g0 shape = {g0.shape}")
    print("First 5 triples:", triples[:5])
    print("First 3 rows of J:\n", J[:3, :])
    print("First 5 entries of g0 (Hz):", g0[:5])

    print("\n=== F310: find_max_coupling_tensor_F310 ===")
    max_out = find_max_coupling_tensor_F310(
        num_ions=num_ions,
        constant_trappingvars=tv0,
        bounds5=bounds,
        point=point,
        simulation_preset="Simp58_101",
        return_argmax=True,
    )
    Gmax = max_out["Gmax_Hz"]  # (K x K x K)
    Ustar = max_out.get("Ustar_V")  # (T x 5), only when return_argmax=True
    triples2 = max_out["triples"]
    K2 = max_out["K"]
    assert K2 == K, "Mode count mismatch."

    max_idx = np.unravel_index(np.nanargmax(Gmax), Gmax.shape)
    print(f"Gmax tensor shape = {Gmax.shape}")
    print(f"Global max |g| = {Gmax[max_idx]:.6g} Hz at (a,b,c) = {max_idx}")
    if Ustar is not None:
        print("U* (first 3 rows):\n", Ustar[:3, :])
    print("triples[0..4]:", triples2[:5])

    print("\n=== F310: solve_F310_for_targets ===")
    # Build a sparse target tensor: match g0 for all rows except bump two triples by +10 Hz.
    target = np.full((K, K, K), np.nan, dtype=float)
    bump_indices = [0, 1] if T > 1 else [0]
    for ti in bump_indices:
        a, b, c = triples[ti]
        target[a, b, c] = g0[ti] + 10.0  # ask for +10 Hz at these two triples

    solve_out = solve_F310_for_targets(
        num_ions=num_ions,
        constant_trappingvars=tv0,
        target_tensor_Hz=target,
        bounds5=bounds,
        point=point,
        simulation_preset="Simp58_101",
        rest_penalty=1e-6,
        clip_to_bounds=True,
    )
    u_sol = solve_out["u_solution_V"]  # (5,)
    G_pred = solve_out["G_pred_Hz"]  # (K x K x K)
    rms = solve_out["residual_rms"]
    print("u_solution_V:", u_sol)
    for ti in bump_indices:
        a, b, c = triples[ti]
        print(
            f"Predicted g at {a,b,c}: {G_pred[a,b,c]:.6g} Hz (target {target[a,b,c]:.6g} Hz)"
        )
    print(f"Residual RMS over selected rows: {rms:.6g} Hz")
