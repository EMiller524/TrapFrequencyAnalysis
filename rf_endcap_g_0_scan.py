"""
Scan RF frequency/amplitude settings and, for each, search the DC endcap voltage
that yields a ~5 µm nearest-neighbor spacing for a 3‑ion chain. Then report the
(g0) parametric coupling between modes 1 and 2 (by index) and those modes'
frequencies.

Assumptions & notes
-------------------
- Twist is held fixed; only the endcap voltage is varied for spacing control.
- Target spacing is interpreted as the *average nearest-neighbor spacing* of the
  3 ions, measured along the chain axis (found via PCA of the equilibrium
  positions).
- Modes are indexed in **0-based** Python indices throughout; reporting focuses
  on indices (1, 2) to match prior usage like g0[1][2]. Adjust as needed.
- We rebuild the pipeline via `sim._smoke_test_new_stack(n_ions=3, poly_deg=4)`
  after each endcap update for robustness.
- g0 returned by `get_g0_matrix` is assumed in s^-1; we also report in Hz by
  dividing by 2π.

How to use
----------
Place this file in the repo root. Run:

    python rf_endcap_g0_scan.py

Or import and call `run_rf_endcap_scan(...)` from your own script.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np

# Repo-local imports (adjust if your module paths differ)
from trapping_variables import Trapping_Vars
from sim.simulation import Simulation


@dataclass
class RFSetting:
    label: str
    freq_hz: float
    phase_rad: float
    amps_volts: Dict[str, float]


@dataclass
class ScanResult:
    rf_label: str
    rf_freq_hz: float
    rf_amps_volts: Dict[str, float]
    found_endcaps_v: float
    achieved_avg_nn_um: float
    mode_i: int
    mode_j: int
    f_i_hz: float
    f_j_hz: float
    g0_ij_s_inv: float
    g0_ij_hz: float


def _principal_axis(points_m: np.ndarray) -> np.ndarray:
    """Return unit vector for the chain axis via PCA of 3×3 positions."""
    pts = np.asarray(points_m, dtype=float)
    if pts.shape != (3, 3):
        raise ValueError(f"Expected (3,3) positions, got {pts.shape}")
    centered = pts - pts.mean(axis=0, keepdims=True)
    cov = centered.T @ centered
    vals, vecs = np.linalg.eigh(cov)
    axis = vecs[:, np.argmax(vals)]
    # Ensure a consistent orientation (e.g., positive z projection preferred)
    if axis[2] < 0:
        axis = -axis
    return axis / np.linalg.norm(axis)


def _avg_nearest_neighbor_spacing_um(points_m: np.ndarray) -> Tuple[float, List[float]]:
    """Project positions onto the chain axis and compute the two NN spacings (µm).

    Returns (avg_spacing_um, [s01_um, s12_um]) where s01 and s12 are consecutive
    neighbor spacings along the axis.
    """
    axis = _principal_axis(points_m)
    proj = (points_m @ axis).astype(float)
    order = np.argsort(proj)
    sorted_proj = proj[order]
    # consecutive spacings in meters
    s01 = sorted_proj[1] - sorted_proj[0]
    s12 = sorted_proj[2] - sorted_proj[1]
    s_um = [1e6 * float(s01), 1e6 * float(s12)]
    return float(np.mean(s_um)), s_um


def _rebuild_and_measure(sim: Simulation, n_ions: int = 3) -> Tuple[np.ndarray, float]:
    """Run the pipeline and return (positions_m, avg_nn_um)."""
    sim._smoke_test_new_stack(n_ions=n_ions, poly_deg=4)
    pos = sim.ion_equilibrium_positions.get(n_ions)
    if pos is None:
        raise RuntimeError("No equilibrium positions cached after smoke test.")
    avg_um, _ = _avg_nearest_neighbor_spacing_um(pos)
    return pos, avg_um


def _coarse_endcap_sweep(
    sim: Simulation,
    tv: Trapping_Vars,
    twist_v: float,
    targets_v: Iterable[float],
    target_um: float,
) -> Tuple[List[Tuple[float, float]], Tuple[float, float]]:
    """Probe spacing vs. endcap voltage on a coarse grid.

    Returns (samples, best) where samples is a list of (V_endcap, avg_um) and
    best is the (V, avg_um) closest to target.
    """
    samples: List[Tuple[float, float]] = []
    best_V, best_um = None, float("inf")

    for V in targets_v:
        tv.clear_dc_modifications()
        tv.apply_dc_twist_endcaps(twist=twist_v, endcaps=float(V))
        _, s_um = _rebuild_and_measure(sim)
        samples.append((float(V), float(s_um)))
        if abs(s_um - target_um) < abs(best_um - target_um):
            best_V, best_um = float(V), float(s_um)
    return samples, (best_V, best_um)


def _find_endcaps_for_spacing(
    sim: Simulation,
    tv: Trapping_Vars,
    twist_v: float = 0.275,
    target_lo_um: float = 4.95,
    target_hi_um: float = 5.05,
    start_v: float = 2.5,
    v_lo: float = 0.0,
    v_hi: float = 5.0,
    max_steps: int = 10,
    verbose: bool = True,
) -> Tuple[float, float]:
    """Binary split around a seeded region to hit ~5 µm spacing.

    Procedure (per user spec):
      - Start at 4.0 V. Measure avg nearest-neighbor spacing.
      - If spacing > 5.0 µm, move to (4 + (8-4)/2) = 6.0 V (stronger endcaps to shrink spacing).
      - If spacing < 5.0 µm, move to (4 / 2) = 2.0 V (weaker endcaps to expand spacing).
      - Thereafter, select the reasonable half-interval and split in half again.
      - Stop after at most 6 steps or as soon as spacing ∈ [target_lo_um, target_hi_um].

    Returns (endcaps_v, achieved_avg_um).
    """
    target_mid = 0.5 * (target_lo_um + target_hi_um)

    # Clamp start within [v_lo, v_hi]
    V = min(max(start_v, v_lo), v_hi)
    lo, hi = float(v_lo), float(v_hi)

    last_V, last_s = None, None

    for step in range(1, max_steps + 1):
        tv.clear_dc_modifications()
        tv.apply_dc_twist_endcaps(twist=twist_v, endcaps=V)
        _, s_um = _rebuild_and_measure(sim)

        if verbose:
            print(
                f"[bisect] step {step}: V={V:.4f} V  ->  avg={s_um:.3f} µm  (band {target_lo_um:.1f}–{target_hi_um:.1f})"
            )

        # Accept if inside band
        if target_lo_um <= s_um <= target_hi_um:
            return V, s_um

        # Decide which half to keep based on whether we need to increase or decrease spacing
        if (
            s_um > target_mid
        ):  # too large spacing -> need stronger confinement -> increase V
            lo = V
            V = 0.5 * (V + hi)
        else:  # spacing too small -> need weaker confinement -> decrease V
            hi = V
            V = 0.5 * (lo + V)

        last_V, last_s = V, s_um

    # If we didn't land in-band, return the last tested voltage & spacing
    if verbose:
        print(f"[bisect] max steps reached; using V={last_V:.4f} V (≈ {last_s:.3f} µm)")
    tv.clear_dc_modifications()
    tv.apply_dc_twist_endcaps(twist=twist_v, endcaps=float(last_V))
    _rebuild_and_measure(sim)
    return float(last_V), float(last_s)


def run_rf_endcap_scan(
    rf_settings: Iterable[RFSetting],
    geometry_name: str = "NISTMock",
    twist_v: float = 0.275,
    target_spacing_um: float = 5.0,
    spacing_tol_um: float = 0.05,
    extradrive_label: str = "ExtraDrive1",
    extradrive_freq_hz: float = 28_000.0,
    extradrive_phase_rad: float = 0.0,
    extradrive_dc_map_volts: Dict[str, float] | None = None,
    mode_i: int = 1,
    mode_j: int = 2,
) -> List[ScanResult]:
    """For each RF setting, tune endcaps to hit target spacing and report g0( i,j )."""
    if extradrive_dc_map_volts is None:
        # Default to the map used in your example.
        extradrive_dc_map_volts = {
            "DC1": 0.175,
            "DC2": 0.060,
            "DC3": 0.0,
            "DC4": -0.60,
            "DC5": -0.175,
            "DC10": 0.175,
            "DC9": 0.060,
            "DC8": 0.0,
            "DC7": -0.60,
            "DC6": -0.175,
        }

    results: List[ScanResult] = []

    for rf in rf_settings:
        # Fresh TV/Simulation per RF config for isolation.
        tv = Trapping_Vars()
        rf_key = tv.add_driving(rf.label, rf.freq_hz, rf.phase_rad, rf.amps_volts)
        tv.clear_dc_modifications()
        tv.apply_dc_twist_endcaps(twist=twist_v, endcaps=2.0)  # initial guess
        extradrive_key = tv.add_driving(
            extradrive_label,
            extradrive_freq_hz,
            extradrive_phase_rad,
            extradrive_dc_map_volts,
        )

        sim = Simulation(geometry_name, tv)

        # Tune endcaps for target spacing
        endcaps_v, achieved_um = _find_endcaps_for_spacing(
            sim,
            tv,
            twist_v=twist_v,
            target_lo_um=4.95,
            target_hi_um=5.05,
            start_v=2.5,
            v_lo=0.0,
            v_hi=5.0,
            max_steps=10,
            verbose=True,
        )

        # Make sure modes are computed for n_ions=3
        sim.get_static_normal_modes_and_freq(3)
        modes_dict = sim.normal_modes_and_frequencies.get(3)
        if modes_dict is None or "frequencies_Hz" not in modes_dict:
            raise RuntimeError("Frequencies not available after mode computation.")
        freqs_hz = np.asarray(modes_dict["frequencies_Hz"], dtype=float)
        n_modes = freqs_hz.size
        if max(mode_i, mode_j) >= n_modes:
            raise IndexError(
                f"Requested mode indices ({mode_i},{mode_j}) exceed available modes {n_modes}."
            )

        # g0 matrix for the chosen modulation drive
        g0_mat = sim.get_g0_matrix(3, extradrive_key)
        g0_s_inv = float(g0_mat[mode_i][mode_j])
        g0_hz = g0_s_inv / (2.0 * math.pi)

        results.append(
            ScanResult(
                rf_label=rf.label,
                rf_freq_hz=rf.freq_hz,
                rf_amps_volts=rf.amps_volts,
                found_endcaps_v=endcaps_v,
                achieved_avg_nn_um=achieved_um,
                mode_i=mode_i,
                mode_j=mode_j,
                f_i_hz=float(freqs_hz[mode_i]),
                f_j_hz=float(freqs_hz[mode_j]),
                g0_ij_s_inv=g0_s_inv,
                g0_ij_hz=g0_hz,
            )
        )

    return results


def _pretty_print_results(results: List[ScanResult]) -> None:
    if not results:
        print("(no results)")
        return
    print("\n=== RF scan results (target spacing ~5 µm) ===")
    for r in results:
        print(
            f"\n{r.rf_label}: f_RF={r.rf_freq_hz/1e6:.3f} MHz, amps={r.rf_amps_volts}\n"
            f"  endcaps={r.found_endcaps_v:.4f} V  |  achieved avg NN = {r.achieved_avg_nn_um:.3f} µm\n"
            f"  modes ({r.mode_i},{r.mode_j}): f_i={r.f_i_hz/1e6:.6f} MHz, f_j={r.f_j_hz/1e6:.6f} MHz\n"
            f"  g0(i,j) = {r.g0_ij_s_inv:.3e} s^-1  ({r.g0_ij_hz:.3e} Hz)\n"
            f"  const = {math.sqrt(r.f_i_hz/1e6 * r.f_j_hz/1e6) * 173 * abs(r.g0_ij_hz) / 1000:.6f}"
        )
    for r in results:
        print(
            f"  const = {80.7 - math.sqrt(r.f_i_hz/1e6 * r.f_j_hz/1e6) * 173 * abs(r.g0_ij_hz) / 1000:.6f}"
        )


if __name__ == "__main__":
    # Example RF settings to scan. Edit to your actual pairings.
    # rf_settings = [
    #     RFSetting(
    #         "RF", float(f_mhz) * 1_000_000.0, 0.0, {"RF1": float(v), "RF2": float(v)}
    #     )
    #     for f_mhz in range(15, 35 + 1, 4)  # MHz: 15, 19, 23, 27, 31, 35
    #     for v in range(150, 350 + 1, 40)  # Volts: 150, 190, 230, 270, 310, 350
    # ]

    rf_settings = [
        RFSetting("RF", 17_500_000.0, 0.0, {"RF1": 175.0, "RF2": 175.0}),
        RFSetting("RF", 25_500_000.0, 0.0, {"RF1": 377.0, "RF2": 377.0}),
    ]
    
    
    results = run_rf_endcap_scan(
        rf_settings,
        geometry_name="NISTMock",
        twist_v=0.275,
        target_spacing_um=5.0,
        spacing_tol_um=0.05,
        extradrive_label="ExtraDrive1",
        extradrive_freq_hz=28_000.0,
        extradrive_phase_rad=0.0,
        extradrive_dc_map_volts=None,
        mode_i=1,
        mode_j=2,
    )

    _pretty_print_results(results)
