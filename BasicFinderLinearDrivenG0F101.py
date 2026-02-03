# Minimal sanity check: init Simulation, add RF + one extra drive,
# compute modes, then print g0 for a single mode pair.
print("hi")

import sim
from sim.simulation import Simulation
import trapping_variables
import time
from trapping_variables import Trapping_Vars


# takes in number of ions and some point,
def calculate_jacobian_F101(
    num_ions: int = 3,
    constant_trappingvars=Trapping_Vars,  # may be a Trapping_Vars instance or the class
    point=[0.0, 0.0, 0.0, 0.0, 0.0],
    output_g0_coupoling=(0, 1),
    simulation_preset: str = "Simp58_101",
):
    """
    Compute the Jacobian (1 x 5) of F101 at 'point' via central differences using a
    single Simulation that contains all ± step probe drives.

    Inputs
    ------
    num_ions: int
        Number of ions (defines K=3N).
    constant_trappingvars:
        Either a preconfigured Trapping_Vars instance (preferred), or the class
        Trapping_Vars (in which case a fresh, empty instance will be created).
        This holds the DC and RF that are to remain constant.
    point: list[5] of float
        The 5 symmetric DC DOFs for the extra drive: (DC1=DC10, DC2=DC9, DC3=DC8,
        DC4=DC7, DC5=DC6), in volts.
        If exactly [0,0,0,0,0], we use a larger step (±0.5 V); otherwise ±0.05 V.
    output_g0_coupoling: tuple(int,int)
        Mode indices (i,j) whose driven g0 coupling (in Hz) is the scalar output.

    Returns
    -------
    list[list[float]]
        2D list shaped (1,5): the Jacobian row d g0[i,j]/d v_k in units of Hz/Volt.
        Columns correspond to the 5 symmetric DC DOFs in the order above.
    """
    import copy
    from sim.simulation import Simulation
    from trapping_variables import Trapping_Vars

    # --- validate / coerce inputs ---
    if not isinstance(point, (list, tuple)) or len(point) != 5:
        raise ValueError(
            "point must be a length-5 list/tuple of floats (symmetric DC DOFs)."
        )
    mode_i, mode_j = map(int, output_g0_coupoling)

    # Trapping_Vars: accept instance or class
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

    # --- make a clean copy so we don't mutate caller's tv ---
    tv = Trapping_Vars()

    # copy DC amplitudes
    dc_in = tv_in.get_drive_amplitudes(tv_in.dc_key)
    for el, A in dc_in.items():
        tv.Var_dict[tv.dc_key].set_amplitude_volt(el, A)
    tv._update_pickoff_for_drive(tv.dc_key)

    # copy all non-DC drives (e.g., RF) exactly
    for dk in tv_in.get_drives():
        if dk == tv_in.dc_key:
            continue
        amps = tv_in.get_drive_amplitudes(dk)
        tv.add_driving(dk.label, dk.f_hz, dk.phi, amps)

    # helper: map 5-vector -> symmetric DC dict for a probe drive
    def vec5_to_symmetric_dc_map(v5):
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

    label_0 = "Probe_0"
    dk_0 =  tv.add_driving(label_0, 77.0, 0.0, vec5_to_symmetric_dc_map(point))

    # choose step size
    is_zero_point = all(abs(x) == 0.0 for x in point)
    h = 0.5 if is_zero_point else 0.05

    # Build 10 probe drives (±h in each of 5 coords) with distinct frequencies
    probe_pairs = []  # [(dk_plus, dk_minus, step), ...] length 5
    base_f_plus = 1.0  # Hz
    base_f_minus = 101.0  # Hz (kept distinct)
    for k in range(5):
        p = list(point)
        p[k] += h
        m = list(point)
        m[k] -= h
        label_p = f"Probe_p{k+1}"
        label_m = f"Probe_m{k+1}"
        dk_p = tv.add_driving(
            label_p, base_f_plus + k, 0.0, vec5_to_symmetric_dc_map(p)
        )
        dk_m = tv.add_driving(
            label_m, base_f_minus + k, 0.0, vec5_to_symmetric_dc_map(m)
        )
        probe_pairs.append((dk_p, dk_m, h))

    # --- one Simulation containing all probes ---
    # NOTE: use the preset you normally use in this repo; if you pass in a tv instance that already
    # defines your trap, this preset string should match your environment.
    preset_name = simulation_preset
    sim = Simulation(preset_name, tv)

    # Equilibrium and modes once (held fixed for all probes)
    sim.find_equilib_position_single(int(num_ions))
    sim.get_static_normal_modes_and_freq(
        int(num_ions), normalize=True, sort_by_freq=True
    )

    # Optional but harmless: build all center fits (so every probe is ready)
    sim.update_center_polys()

    g0_center = sim.get_g0_for_mode_pair(
        num_ions, dk_0, mode_i, mode_j, units="Hz", recompute=True
    )

    # Central differences: derivative_k = [g0(p_k+) - g0(p_k-)] / (2h)
    jac = [0.0] * 5
    for k, (dk_p, dk_m, step) in enumerate(probe_pairs):
        g_plus = sim.get_g0_for_mode_pair(
            num_ions, dk_p, mode_i, mode_j, units="Hz", recompute=True
        )
        g_minus = sim.get_g0_for_mode_pair(
            num_ions, dk_m, mode_i, mode_j, units="Hz", recompute=True
        )
        jac[k] = (g_plus - g_minus) / (2.0 * step)

    # Second-difference (linearity) diagnostics per axis
    residuals = []
    for k, (dk_p, dk_m, step) in enumerate(probe_pairs):
        g_plus  = sim.get_g0_for_mode_pair(num_ions, dk_p, mode_i, mode_j, units="Hz", recompute=False)
        g_minus = sim.get_g0_for_mode_pair(num_ions, dk_m, mode_i, mode_j, units="Hz", recompute=False)
        r_k = g_plus - 2.0*g0_center + g_minus     # should be ~0 if linear
        residuals.append(r_k)

    # Intercept check (should be ~0 if baseline is zero-coupling)
    intercept_est = g0_center - sum(jac[k]*point[k] for k in range(5))

    # (Optional) print or return these as a side channel
    print(f"linearity residuals (Hz): {residuals}")
    print(f"intercept_est (Hz): {intercept_est}")

    # Return as 2D list (1 x 5), units Hz / Volt
    return [jac]


def solve_for_target_F101(
    target_Hz: float,
    num_ions: int = 3,
    constant_trappingvars=Trapping_Vars,
    point=[0.0, 0.0, 0.0, 0.0, 0.0],
    output_g0_coupoling=(0, 1),
    simulation_preset: str = "Simp58_101",
):
    """
    Find a 5-vector of extra-drive voltages (symmetric DC pairs) that achieves
    the desired two-mode coupling target_Hz for the given mode pair.

    Uses the linear model g(u) = s^T u + c with:
      s (1x5) from calculate_jacobian_F101 (Hz/Volt),
      c inferred from g(point) - s^T point.

    Returns the *minimum-norm* update from 'point':
      u* = point + ((target_Hz - g(point)) / ||s||^2) * s

    Parameters
    ----------
    target_Hz : float
        Desired g0 coupling in Hz for the given mode pair.
    num_ions : int
        Number of ions (defines K = 3N).
    constant_trappingvars :
        Trapping_Vars instance (preferred) with DC/RF fixed for this config,
        or the Trapping_Vars class (a fresh instance will be created).
    point : list[5] of float
        Starting 5-vector (DC1=DC10, DC2=DC9, DC3=DC8, DC4=DC7, DC5=DC6) in Volts.
    output_g0_coupoling : tuple(int,int)
        Mode indices (i,j) for which g0 is targeted.

    Returns
    -------
    list[5] of float
        The voltages (in Volts) for the five symmetric DOFs achieving target_Hz,
        as the minimum-norm solution from 'point'.
    """
    import numpy as np
    from sim.simulation import Simulation
    from trapping_variables import Trapping_Vars

    # --- 1) Jacobian s (1x5) in Hz/V ---
    J = calculate_jacobian_F101(
        num_ions=num_ions,
        constant_trappingvars=constant_trappingvars,
        point=point,
        output_g0_coupoling=output_g0_coupoling,
        simulation_preset=simulation_preset,
    )  # returns [[s1, s2, s3, s4, s5]]
    s = np.asarray(J[0], dtype=float)  # shape (5,)

    # Guard against degenerate sensitivity
    norm2 = float(np.dot(s, s))
    if norm2 == 0.0 or not np.isfinite(norm2):
        raise RuntimeError(
            "Jacobian has zero/invalid norm; target is not controllable with these inputs."
        )

    # --- 2) Evaluate g(point) once to infer intercept (no bounds; single drive) ---
    # Accept instance or class for tv
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

    # Copy tv_in (DC + all non-DC drives) and add one probe at 'point'
    tv = Trapping_Vars()
    # copy DC amplitudes
    dc_in = tv_in.get_drive_amplitudes(tv_in.dc_key)
    for el, A in dc_in.items():
        tv.Var_dict[tv.dc_key].set_amplitude_volt(el, A)
    tv._update_pickoff_for_drive(tv.dc_key)
    # copy non-DC drives (e.g., RF)
    for dk in tv_in.get_drives():
        if dk == tv_in.dc_key:
            continue
        amps = tv_in.get_drive_amplitudes(dk)
        tv.add_driving(dk.label, dk.f_hz, dk.phi, amps)

    # helper to map 5-vector -> symmetric DC dict
    def vec5_to_symmetric_dc_map(v5):
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

    mode_i, mode_j = map(int, output_g0_coupoling)

    # one probe drive at the central point (frequency is bookkeeping only)
    dk0 = tv.add_driving("Probe_center", 77.0, 0.0, vec5_to_symmetric_dc_map(point))

    # One Simulation; modes fixed by DC+RF (extra drive is not pseudo-ized)
    sim = Simulation("Simp58_101", tv)
    sim.find_equilib_position_single(int(num_ions))
    sim.get_static_normal_modes_and_freq(
        int(num_ions), normalize=True, sort_by_freq=True
    )

    g_center = sim.get_g0_for_mode_pair(
        num_ions=num_ions,
        drive=dk0,
        mode_i=mode_i,
        mode_j=mode_j,
        units="Hz",
        recompute=True,
    )

    # --- 3) Minimum-norm solution from 'point' that hits target ---
    delta = float(target_Hz) - float(g_center)
    u_star = np.asarray(point, dtype=float) + (delta / norm2) * s
    return u_star.tolist()


def maximize_coupling_within_bounds_F101(
    bounds=None,  # list of 5 (lo, hi) tuples in Volts
    num_ions: int = 3,
    constant_trappingvars=Trapping_Vars,  # instance or class
    point=[0.0, 0.0, 0.0, 0.0, 0.0],  # reference point u0 (Volts)
    output_g0_coupoling=(0, 1),  # (mode_i, mode_j)
    verify_with_sim: bool = False,  # optionally evaluate g(u*) with SIM to confirm
):
    """
    Maximize the scalar F101 output g0_{i,j}(u) over a 5-D box of input voltages.

    bounds : list[(lo, hi)] length 5, in Volts. If None, defaults to [(-1,1)]*5.
    Returns dict with:
      - 'u_max' (list[5] Volts), 'g_max_pred_Hz' (float),
      - 'u_min', 'g_min_pred_Hz',
      - 's_Hz_per_V' (list[5]) and 'g_at_point_Hz' (float),
      - optionally 'g_max_verified_Hz' if verify_with_sim=True.
    """
    import numpy as np
    from sim.simulation import Simulation
    from trapping_variables import Trapping_Vars

    if bounds is None:
        bounds = [(-1.0, 1.0)] * 5
    if len(bounds) != 5:
        raise ValueError("bounds must be a length-5 list of (lo, hi) tuples.")

    # 1) Get Jacobian row s (Hz/V) at 'point' (central differences, single SIM with probes)
    J = calculate_jacobian_F101(
        num_ions=num_ions,
        constant_trappingvars=constant_trappingvars,
        point=point,
        output_g0_coupoling=output_g0_coupoling,
    )  # [[s1, s2, s3, s4, s5]]
    s = np.asarray(J[0], dtype=float)  # shape (5,)

    # 2) Evaluate g(point) once to infer intercept c = g(point) - s^T point
    # Accept instance or class for tv
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

    # copy tv_in (DC + all non-DC drives), add one drive at 'point'
    tv = Trapping_Vars()
    dc_in = tv_in.get_drive_amplitudes(tv_in.dc_key)
    for el, A in dc_in.items():
        tv.Var_dict[tv.dc_key].set_amplitude_volt(el, A)
    tv._update_pickoff_for_drive(tv.dc_key)
    for dk in tv_in.get_drives():
        if dk == tv_in.dc_key:
            continue
        amps = tv_in.get_drive_amplitudes(dk)
        tv.add_driving(dk.label, dk.f_hz, dk.phi, amps)

    def vec5_to_symmetric_dc_map(v5):
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

    mode_i, mode_j = map(int, output_g0_coupoling)
    dk0 = tv.add_driving("Probe_center", 77.0, 0.0, vec5_to_symmetric_dc_map(point))

    sim = Simulation("Simp58_101", tv)
    sim.find_equilib_position_single(int(num_ions))
    sim.get_static_normal_modes_and_freq(
        int(num_ions), normalize=True, sort_by_freq=True
    )

    g_point = sim.get_g0_for_mode_pair(
        num_ions=num_ions,
        drive=dk0,
        mode_i=mode_i,
        mode_j=mode_j,
        units="Hz",
        recompute=True,
    )
    c = float(g_point) - float(np.dot(s, np.asarray(point, dtype=float)))

    # 3) Box maximization of a linear form: choose bound by sign of s_k
    lo = np.asarray([float(b[0]) for b in bounds], dtype=float)
    hi = np.asarray([float(b[1]) for b in bounds], dtype=float)
    if not np.all(lo <= hi):
        raise ValueError("Each bounds entry must satisfy lo <= hi.")

    u_max = np.where(s >= 0.0, hi, lo)  # pick upper if s_k >= 0 else lower
    u_min = np.where(s >= 0.0, lo, hi)  # opposite for min

    g_max_pred = float(np.dot(s, u_max) + c)
    g_min_pred = float(np.dot(s, u_min) + c)

    out = {
        "u_max": u_max.tolist(),
        "g_max_pred_Hz": g_max_pred,
        "u_min": u_min.tolist(),
        "g_min_pred_Hz": g_min_pred,
        "s_Hz_per_V": s.tolist(),
        "g_at_point_Hz": float(g_point),
    }

    # 4) Optional verification by evaluating g(u_max) once
    if verify_with_sim:
        dk_star = tv.add_driving(
            "Probe_u_max", 79.0, 0.0, vec5_to_symmetric_dc_map(u_max)
        )
        g_max_verified = sim.get_g0_for_mode_pair(
            num_ions=num_ions,
            drive=dk_star,
            mode_i=mode_i,
            mode_j=mode_j,
            units="Hz",
            recompute=True,
        )
        out["g_max_verified_Hz"] = float(g_max_verified)

    return out


# find inputs that produced a desired g0_coupling without any bounds
# then do it with bounds
# then find the range of possible g_0 that can be produced with bounds on the inputs

# then do the same basic jacobian fucn but looking at all (3N choose 2) coupolings. Call it F110
# now make a finder for F110, where we request a target g0 coupling matrix and it finds the inputs to produce it
# make a finder that lets the user specify x number of coupolings to target and have the rest be as low as possible.


def main():
    # --- Hard-coded settings ---
    preset = "Simp58_101"
    num_ions = 3
    rf_freq_hz =    25500000
    rf1_v = 377.0
    rf2_v = 377.0
    extra_freq_hz = 1
    mode_i = 0
    mode_j = 1

    # Symmetric extra-drive map (pairs: 1=10, 2=9, 3=8, 4=7, 5=6)
    extra_map = {
        "DC1": +0.175,
        "DC2": +0.060,
        "DC3": 0.0,
        "DC4": -0.60,
        "DC5": -0.175,
        "DC10": +0.175,
        "DC9": +0.060,
        "DC8": 0.0,
        "DC7": -0.60,
        "DC6": -0.175,
    }

    # --- Build trapping variables and drives ---
    tv = Trapping_Vars()
    tv.apply_dc_twist_endcaps(twist = 0.28, endcaps= 3)
    tv.add_driving("RF", rf_freq_hz, 0.0, {"RF1": rf1_v, "RF2": rf2_v})
    extra_drive = tv.add_driving("ExtraDrive1", extra_freq_hz, 0.0, extra_map)

    # --- Simulation and modes ---
    sim = Simulation(preset, tv)

    sim.find_equilib_position_single(num_ions)

    sim.get_static_normal_modes_and_freq(int(num_ions), normalize=True, sort_by_freq=True)

    # --- Query g0 and print ---
    g0_hz = sim.get_g0_for_mode_pair(
        num_ions=num_ions,
        drive=extra_drive,
        mode_i=mode_i,
        mode_j=mode_j,
        units="Hz",
        recompute=True,  # force build once for this drive/ion count
    )

    freqs = sim.normal_modes_and_frequencies[num_ions]["frequencies_Hz"]
    print("=== g0 sanity ===")
    print(f"Preset: {preset}")
    print(f"N ions: {num_ions}  |  K modes: {len(freqs)}")
    print(
        f"Modes ({mode_i},{mode_j}) -> ({freqs[mode_i]:.3f} Hz, {freqs[mode_j]:.3f} Hz)"
    )
    print(
        f"Drive: {getattr(extra_drive, 'label', 'ExtraDrive1')} @ {extra_freq_hz:.1f} Hz"
    )
    print(f"g0 = {g0_hz:.6e} Hz")

    print("hi, hi, hi")    
    time2 = time.time()
    print("Calculating Jacobian...")
    tv = Trapping_Vars()
    tv.apply_dc_twist_endcaps(twist = 0.28, endcaps= 3)
    tv.add_driving("RF", rf_freq_hz, 0.0, {"RF1": rf1_v, "RF2": rf2_v})

    result = solve_for_target_F101(
        target_Hz=100.0,
        num_ions=8,
        constant_trappingvars=tv,
        point=[0.0, 0.0, 0.0, 0.0, 0.0],
        output_g0_coupoling=(0, 1),
        simulation_preset="Simp58_101",
    )
    print(result)
    time3 = time.time()
    print("Time taken to solve: ", time3 - time2)

    out1 = maximize_coupling_within_bounds_F101(
        bounds=[(-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1)],
        output_g0_coupoling=(0, 1),
        num_ions=3,
        constant_trappingvars=tv,
        verify_with_sim=False,
    )
    out2 = maximize_coupling_within_bounds_F101(
        bounds=[(-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1)],
        output_g0_coupoling=(2, 6),
        num_ions=3,
        constant_trappingvars=tv,
        verify_with_sim=False,
    )
    
    print(out1)
    print(" ")
    print(out2)


if __name__ == "__main__":
    main()
