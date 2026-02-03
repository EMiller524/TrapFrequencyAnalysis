# C:\GitHub\TrapFrequencyAnalysis\BasicFinderLinearDrivenG03F320.py
"""
F320: Build the Jacobian for the map R^20 -> R^{C(3N,2) * (3N)} that sends the twenty
per-electrode amplitudes of a single extra-drive (DC1..DC10, RF11..RF20) to *all*
three-mode driven couplings g0_{a,b,c} (Hz), with DC/RF/modes held fixed.

Pattern matches F310 (cubic/tensor output) and F120 (20 independent input channels):
- Use one Simulation; add a center extra-drive at 'point' and ±h probe drives per input.
- Lock equilibrium and static modes once.
- Evaluate g0^{(3)} via the driven cubic accessor, flatten rows in canonical (a<b, c) order.
- Return J (Hz/V), triples, g_center (Hz), K (=3N) and T (=C(K,2)*K).
"""

import math
import time
import numpy as np
from typing import List, Tuple

from sim.simulation import Simulation
from trapping_variables import Trapping_Vars

# Canonical 20-channel order (must match F120 everywhere the vector<->dict map is used)
ELECTRODE_ORDER_20: Tuple[str, ...] = (
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
)


def _vec20_to_dc_rf_map(v20):
    """
    Map a 20-vector of per-electrode amplitudes (V) to a dict over:
        DC1..DC10, RF11..RF15, RF16..RF20

    Canonical order (must match ELECTRODE_ORDER_20 and F120):
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
    """
    Copy DC map and all non-DC drives (e.g., RF) into a fresh Trapping_Vars.
    Extra-drive probes for F320 are added on top of this copy.
    (Pattern identical to F310.)
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


def calculate_jacobian_F320(
    num_ions: int = 3,
    constant_trappingvars=Trapping_Vars,  # instance preferred; class allowed
    point=None,  # 20 DOFs (V): [DC1..DC10, RF11..RF15, RF16..RF20]
    simulation_preset: str = "Simp58_101",
):
    """
    Build the full F320 Jacobian (all triples (a<b,c) vs 20 inputs) using the
    driven cubic accessor path `get_driven_g_0_3_tensor`, with a single Simulation:
      - add a center extra-drive at 'point' and ±h probe drives per input channel,
      - lock equilibrium and static modes once,
      - flatten rows in canonical (a<b, c) order.

    Returns
    -------
    dict with:
      - 'J_Hz_per_V' : (T x 20) array, rows in (a<b,c) canonical order, units Hz/V
      - 'triples'    : list[(a,b,c)] row mapping
      - 'g_center_Hz': (T,) array of g0^{(3)} at 'point' (Hz)
      - 'K'          : number of modes (3N)
      - 'T'          : number of triples (= K*(K-1)//2 * K)
    """
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

    # Step size rule: larger at origin, smaller otherwise (match F110/F120/F310 convention)
    is_zero_point = all(abs(float(x)) == 0.0 for x in point)
    h = 0.5 if is_zero_point else 0.05

    # Fresh TV with DC/RF copied; add center and ±h probe drives
    tv = _copy_tv(tv_in)

    dk_center = tv.add_driving("F320_center", 77.0, 0.0, _vec20_to_dc_rf_map(point))

    probe_plus, probe_minus = [], []
    for k in range(20):
        p = list(point)
        p[k] += h
        m = list(point)
        m[k] -= h
        dk_p = tv.add_driving(f"F320_p{k+1}", 1.0 + k, 0.0, _vec20_to_dc_rf_map(p))
        dk_m = tv.add_driving(f"F320_m{k+1}", 101.0 + k, 0.0, _vec20_to_dc_rf_map(m))
        probe_plus.append(dk_p)
        probe_minus.append(dk_m)

    # One Simulation; lock equilibrium and modes once
    sim = Simulation(simulation_preset, tv)
    N = int(num_ions)
    sim.find_equilib_position_single(N)
    sim.get_static_normal_modes_and_freq(N, normalize=True, sort_by_freq=True)

    # Mode count and canonical triple list (a<b, then c)
    K = sim.normal_modes_and_frequencies[N]["modes"].shape[1]
    triples = [(a, b, c) for a in range(K) for b in range(a + 1, K) for c in range(K)]
    T = len(triples)

    # Center tensor (Hz) and baseline vector in triples order
    G_center = sim.get_driven_g_0_3_tensor(N, dk_center)
    g_center = np.fromiter(
        (G_center[a, b, c] for (a, b, c) in triples), dtype=float, count=T
    )

    # Jacobian columns via central differences over 20 inputs
    J = np.empty((T, 20), dtype=float)
    for k in range(20):
        Gp = sim.get_driven_g_0_3_tensor(N, probe_plus[k])
        Gm = sim.get_driven_g_0_3_tensor(N, probe_minus[k])
        J[:, k] = np.fromiter(
            ((Gp[a, b, c] - Gm[a, b, c]) / (2.0 * h) for (a, b, c) in triples),
            dtype=float,
            count=T,
        )

    return {
        "J_Hz_per_V": J,
        "triples": triples,
        "g_center_Hz": g_center,
        "K": K,
        "T": T,
    }


# --- helpers ---------------------------------------------------------------


def _triple_to_index_map(triples):
    """Map (a,b,c) with a<b to row index in flattened triples vector."""
    return {t: idx for idx, t in enumerate(triples)}


def _normalize_bounds_20(amp_bounds):
    """
    Normalize bounds into two length-20 arrays: lower, upper.

    Accepts:
      - float/int B                  -> [-B, +B] for all channels
      - (lower, upper) arrays        -> each length-20
      - dict{name: (lo, hi)} subset  -> others default to (0,0)

    Channel order follows ELECTRODE_ORDER_20.
    """
    import numpy as np

    if amp_bounds is None:
        lower = np.zeros(20)
        upper = np.zeros(20)
        return lower, upper

    if isinstance(amp_bounds, (int, float)):
        B = float(amp_bounds)
        return -B * np.ones(20), B * np.ones(20)

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

    if isinstance(amp_bounds, dict):
        lo = np.zeros(20, float)
        hi = np.zeros(20, float)
        for k, name in enumerate(ELECTRODE_ORDER_20):
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


# --- F320: row-wise max over a box, returned as a K×K×K tensor -------------

def find_max_coupling_matrix_F320(
    num_ions: int,
    constant_trappingvars,
    point=None,                   # length-20 amplitudes (V)
    amp_bounds=None,              # scalar B or (lo,hi) arrays or dict per electrode
    simulation_preset: str = "Simp58_101",
    fill_lower_triangle: bool = False,   # keep False to match UI: color only a<b
):
    """
    Compute the absolute row-wise maxima of g0_{a,b,c} (Hz) under box bounds on the 20 inputs,
    using the linearization from calculate_jacobian_F320 at 'point':
        g ≈ g_center + J * (x - point)

    Returns
    -------
    dict with:
      - 'G3max_abs_Hz' : K×K×K tensor, with entries set for a<b (and optionally mirrored)
      - 'triples'      : list of (a,b,c) row mapping for J (a<b)
      - 'J_Hz_per_V'   : (T × 20) Jacobian (Hz/V), T = C(K,2)*K
      - 'g_center_Hz'  : (T,) center values (Hz)
      - 'K','T'        : mode count and triple row count
    """
    import numpy as np

    # Linearization at 'point'
    pack = calculate_jacobian_F320(
        num_ions=num_ions,
        constant_trappingvars=constant_trappingvars,
        point=point,
        simulation_preset=simulation_preset,
    )
    J = pack["J_Hz_per_V"]            # (T x 20)
    triples = pack["triples"]         # list of (a<b,c)
    g0 = pack["g_center_Hz"]          # (T,)
    K, T = pack["K"], pack["T"]

    # Bounds over x; convert to bounds over delta := x - point
    lo_abs, hi_abs = _normalize_bounds_20(amp_bounds)
    p = np.zeros(20) if point is None else np.asarray(point, float).reshape(20)
    lo = lo_abs - p
    hi = hi_abs - p

    # Build K×K×K tensor of per-row maxima
    Gmax = np.zeros((K, K, K), float)

    for row_idx, (a, b, c) in enumerate(triples):
        arow = J[row_idx, :]          # shape (20,)
        b0 = g0[row_idx]

        # Maximize (b0 + arow·delta) and its negative for absolute value
        x_pos = np.where(arow >= 0.0, hi, lo)
        val_pos = b0 + float(arow @ x_pos)

        x_neg = np.where(arow <= 0.0, hi, lo)
        val_neg = -(b0 + float(arow @ x_neg))

        val = max(abs(val_pos), abs(val_neg))
        Gmax[a, b, c] = val
        if fill_lower_triangle:
            Gmax[b, a, c] = val  # optional symmetry fill for display
    return {
        "G3max_abs_Hz": Gmax,
        "triples": triples,
        "J_Hz_per_V": J,
        "g_center_Hz": g0,
        "K": K,
        "T": T,
    }


# --- F320: targeted solve for triples ((a,b,c), target_Hz) ------------------


def solve_F320_for_targets(
    mode_triple_targets,  # list of ((a,b,c), target_Hz) with a<b
    num_ions: int,
    constant_trappingvars,
    point=None,  # length-20 amplitudes (V) about which J is taken
    amp_bounds=None,  # same formats as in find_max_coupling_matrix_F320
    l2_reg: float = 0.0,  # ridge parameter (Hz/V)^2 on Δx
    minimize_rest: bool = False,  # project along nullspace of J_sub to reduce rest
    rest_l2: float = 1e-8,  # tiny Tikhonov on nullspace solve
    clip_to_bounds: bool = True,
    simulation_preset: str = "Simp58_101",
):
    """
    Solve for Δx (20-vector, V) so that J_sub Δx ≈ y, where
      y = targets_Hz - g_center_sub
    using the F320 (G03) linear model.

    Returns
    -------
    dict with:
      - 'amplitudes_vector'        : x = point + Δx (length-20)
      - 'amplitudes_by_electrode'  : dict {DC1..DC10, RF11..RF20}
      - 'predicted_targets_Hz'     : list of ((a,b,c), g_pred_abc)
      - 'predicted_rest_rms_Hz'    : float
      - 'predicted_tensor_Hz'      : K×K×K tensor of predicted g (upper triangle in a<b)
      - 'J_Hz_per_V', 'triples', 'g_center_Hz', 'K', 'T'
    """
    import numpy as np

    # Linearization at 'point'
    pack = calculate_jacobian_F320(
        num_ions=num_ions,
        constant_trappingvars=constant_trappingvars,
        point=point,
        simulation_preset=simulation_preset,
    )
    J = pack["J_Hz_per_V"]  # (T x 20)
    triples = pack["triples"]  # list of (a<b,c)
    g0 = pack["g_center_Hz"]  # (T,)
    K, T = pack["K"], pack["T"]
    triple2idx = _triple_to_index_map(triples)

    # Targets subset
    targ_rows = []
    y = []
    for (a, b, c), tval in mode_triple_targets:
        if a >= b:
            raise ValueError(f"Target triple must satisfy a<b, got {(a,b,c)}.")
        r = triple2idx.get((a, b, c))
        if r is None:
            raise ValueError(f"Triple {(a,b,c)} not in triples.")
        targ_rows.append(r)
        y.append(float(tval) - g0[r])
    targ_rows = np.asarray(targ_rows, int)
    y = np.asarray(y, float)  # shape (R,)

    J_sub = J[targ_rows, :]  # (R x 20)

    # Ridge least squares for Δx
    lam = float(l2_reg)
    if J_sub.size == 0:
        dx = np.zeros(20, float)
    else:
        A = J_sub.T @ J_sub
        if lam > 0.0:
            A = A + lam * np.eye(20)
        bvec = J_sub.T @ y
        dx = np.linalg.solve(A, bvec)

    # Optional nullspace tweak to reduce off-target couplings
    if minimize_rest and J_sub.size > 0:
        U, S, Vt = np.linalg.svd(J_sub, full_matrices=False)
        tol = max(J_sub.shape) * np.max(S) * 1e-12
        r = np.sum(S > tol)
        N = Vt[r:, :].T  # (20 x d), basis for nullspace
        if N.size > 0:
            J_rest = np.delete(J, targ_rows, axis=0)  # (T-R x 20)
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

    # Predicted full g (Hz), back into K×K×K tensor (only a<b set)
    g_pred_vec = g0 + J @ dx
    G_pred = np.zeros((K, K, K), float)
    for idx, (a, b, c) in enumerate(triples):
        G_pred[a, b, c] = g_pred_vec[idx]

    # Report targets and rest RMS
    pred_targets = [
        ((a, b, c), float(G_pred[a, b, c])) for (a, b, c), _ in mode_triple_targets
    ]
    mask_rest = np.ones(T, dtype=bool)
    mask_rest[targ_rows] = False
    rest_rms = (
        float(np.sqrt(np.mean((g_pred_vec[mask_rest]) ** 2)))
        if np.any(mask_rest)
        else 0.0
    )

    # Amplitudes mapping
    amps = _vec20_to_dc_rf_map(x)

    return {
        "amplitudes_vector": x,
        "amplitudes_by_electrode": amps,
        "predicted_targets_Hz": pred_targets,
        "predicted_rest_rms_Hz": rest_rms,
        "predicted_tensor_Hz": G_pred,
        "J_Hz_per_V": J,
        "triples": triples,
        "g_center_Hz": g0,
        "K": K,
        "T": T,
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

    # Build trapping variables with RF only (F320 adds center/probe drives internally)
    tv = Trapping_Vars()
    tv.apply_dc_twist_endcaps(twist=0.28, endcaps=3)
    tv.add_driving("RF", rf_freq_hz, 0.0, {"RF1": rf1_v, "RF2": rf2_v})

    print("[F320] Building Jacobian at the requested point…")
    pack = calculate_jacobian_F320(
        num_ions=num_ions,
        constant_trappingvars=tv,
        point=point,
        simulation_preset=preset,
    )
    J = pack["J_Hz_per_V"]  # (T x 20)
    triples = pack["triples"]  # list of (a<b,c)
    g0 = pack["g_center_Hz"]  # (T,)
    K, T = pack["K"], pack["T"]

    print(f"[F320] K = {K} (3N), T = {T} (= C(K,2)*K), J shape = {J.shape}")
    g0_rms = float(np.sqrt(np.mean(g0**2))) if T else 0.0
    print(
        f"[F320] g_center stats (Hz): min={g0.min():.3e}, max={g0.max():.3e}, rms={g0_rms:.3e}"
    )

    # Row-wise absolute max under box bounds
    print(
        f"\n[F320] Computing row-wise max |g₀^{'{'}3{'}'}| over per-channel box ±1.0 V …"
    )
    fmax = find_max_coupling_matrix_F320(
        num_ions=num_ions,
        constant_trappingvars=tv,
        point=point,
        amp_bounds=1.0,  # ±1 V on each of 20 channels
        simulation_preset=preset,
    )
    G3 = fmax["G3max_abs_Hz"]  # K x K x K (set on a<b)
    # Report top-10 triples (a<b,c) by descending |g| max
    vals = []
    for idx, (a, b, c) in enumerate(triples):
        vals.append((G3[a, b, c], a, b, c))
    vals.sort(key=lambda t: -t[0])
    print("[F320] Top |g| rows under bounds (Hz):")
    for v, a, b, c in vals[:10]:
        print(f"  (a,b,c)=({a},{b},{c})  |g|max≈ {v:.3e} Hz")

    # Solve for a few target triples
    targets = [((0, 1, 0), 100.0), ((2, 3, 1), -80.0), ((3, 5, 4), 50.0)]
    print("\n[F320] Solving for target triples (Hz) via linear model…")
    sol = solve_F320_for_targets(
        mode_triple_targets=targets,
        num_ions=num_ions,
        constant_trappingvars=tv,
        point=point,
        amp_bounds=1.0,  # keep within ±1 V
        l2_reg=0.0,
        minimize_rest=False,
        clip_to_bounds=True,
        simulation_preset=preset,
    )

    # Print 20-channel solution neatly
    x = sol["amplitudes_vector"]
    print("\n[F320] Solution amplitudes (V):")
    for k, name in enumerate(ELECTRODE_ORDER_20):
        print(f"  {name:>4}: {x[k]: .6f} V")

    # Achieved vs target
    pred = dict(sol["predicted_targets_Hz"])
    print("\n[F320] Achieved vs target (Hz):")
    for (a, b, c), tgt in targets:
        print(
            f"  ({a},{b},{c}): target={tgt:.6e}, achieved={pred[(a,b,c)]:.6e}, "
            f"error={pred[(a,b,c)]-tgt:.6e}"
        )

    print(f"\n[F320] Predicted rest RMS: {sol['predicted_rest_rms_Hz']:.3e} Hz")
    print(f"[F320] Elapsed: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main2()
