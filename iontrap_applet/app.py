# import os
# import sys
# import json
# import time
# from typing import Any, Dict, List, Tuple

# import numpy as np
# import pandas as pd
# import streamlit as st


# if "res" not in st.session_state:
#     st.session_state["res"] = None
# if "cfg" not in st.session_state:
#     st.session_state["cfg"] = None


# def _to_jsonable(obj):
#     import numpy as np, json

#     if isinstance(obj, np.ndarray):
#         return obj.tolist()
#     if isinstance(obj, (np.floating, np.integer)):
#         return obj.item()
#     if isinstance(obj, (list, tuple)):
#         return [_to_jsonable(x) for x in obj]
#     if isinstance(obj, dict):
#         return {k: _to_jsonable(v) for k, v in obj.items()}
#     try:
#         json.dumps(obj)
#         return obj
#     except Exception:
#         return str(obj)


# def _eq_from_ion_equilibrium_positions(sim, num_ions):
#     """
#     Return sim.ion_equilibrium_positions[num_ions] if it exists and is (N,3).
#     Does not call any compute methods.
#     """
#     import numpy as np

#     d = getattr(sim, "ion_equilibrium_positions", None)
#     if isinstance(d, dict) and (num_ions in d):
#         arr = np.asarray(d[num_ions], dtype=float)
#         if arr.ndim == 2 and arr.shape[1] == 3:
#             return arr
#     return None


# # ------------------------------------------------------------
# # App configuration
# # ------------------------------------------------------------
# st.set_page_config(page_title="Ion Trap Resonance Explorer", layout="wide")

# st.title("🔬 Ion Trap Resonance Explorer")
# st.caption(
#     "Quick UI to tweak trapping variables, choose ion count, and inspect equilibria, modes, frequencies, and resonant couplings."
# )

# # ------------------------------------------------------------
# # Sidebar inputs
# # ------------------------------------------------------------
# with st.sidebar:
#     st.header("Project / Imports")
#     repo_path = st.text_input(
#         "Path to TrapFrequencyAnalysis repo (so we can import Simulation, Trapping_Vars)",
#         value=".",
#         help="Use an absolute path or a relative path from where you run `streamlit run app.py`.",
#     )
#     add_repo_to_sys_path = st.checkbox("Add the path above to PYTHONPATH", value=True)

#     st.divider()
#     st.header("Trap / Simulation Setup")
#     preset = st.selectbox(
#         "Simulation preset (name passed to Simulation)",
#         options=["Simp58_101", "NISTMock", "Hyper_2", "Custom"],
#         index=0,
#     )
#     if preset == "Custom":
#         preset = st.text_input("Custom preset string", value="Simp58_101")

#     num_ions = st.number_input(
#         "Number of ions", min_value=1, max_value=60, value=3, step=1
#     )
#     poly_deg = st.selectbox(
#         "Polynomial degree for fits", options=[2, 3, 4, 5, 6], index=2
#     )

#     st.subheader("RF Drive")
#     rf_freq = st.number_input(
#         "RF frequency (Hz)",
#         min_value=1.0,
#         value=25_500_000.0,
#         step=1_000.0,
#         format="%.3f",
#     )
#     rf_amp1 = st.number_input("RF1 amplitude (V)", value=377.0, step=0.5)
#     rf_amp2 = st.number_input("RF2 amplitude (V)", value=377.0, step=0.5)

#     st.subheader("Extra Drive (optional)")
#     use_extra = st.checkbox("Enable Extra Drive", value=True)
#     extra_freq = st.number_input(
#         "Extra Drive frequency (Hz)",
#         min_value=0.0,
#         value=250_000.0,
#         step=100.0,
#         format="%.3f",
#     )
#     default_extra_map = {
#         "DC1": 0.175,
#         "DC2": 0.060,
#         "DC3": 0.0,
#         "DC4": -0.60,
#         "DC5": -0.175,
#         "DC10": 0.175,
#         "DC9": 0.060,
#         "DC8": 0.0,
#         "DC7": -0.60,
#         "DC6": -0.175,
#     }
#     extra_map_json = st.text_area(
#         "Extra Drive DC map (JSON dict)",
#         value=json.dumps(default_extra_map, indent=2),
#         height=160,
#         help="Map electrode names to amplitudes (in volts). Example keys: DC1..DC10.",
#     )

#     st.subheader("DC Geometry")
#     twist = st.number_input("apply_dc_twist_endcaps: twist", value=0.275, step=0.01)
#     endcaps = st.number_input(
#         "apply_dc_twist_endcaps: endcaps", min_value=0, max_value=20, value=3, step=1
#     )

#     st.divider()
#     st.header("Resonance Scan")
#     tol_Hz = st.number_input(
#         "Resonance tolerance ± (Hz)",
#         min_value=0.0,
#         value=1_000.0,
#         step=10.0,
#         format="%.3f",
#     )
#     orders_pick = st.multiselect("Coupling orders", options=[2, 3, 4], default=[2, 3])

#     run_btn = st.button("Compute", type="primary")

# # ------------------------------------------------------------
# # Utilities: imports, hashing, and compute wrapper
# # ------------------------------------------------------------


# def _ensure_imports(repo_path: str, add_to_sys_path: bool = True):
#     """Ensure we can import Simulation and Trapping_Vars from the repo.
#     Returns (Simulation, Trapping_Vars) types if successful; otherwise raises.
#     """
#     if add_to_sys_path:
#         abs_repo = os.path.abspath(repo_path)
#         if abs_repo not in sys.path:
#             sys.path.insert(0, abs_repo)

#     # Try a few plausible import paths, fall back gracefully.
#     last_err = None
#     # Simulation probe list
#     for sim_mod in ("sim.simulation", "simulation", "Simulation", "simulation_fitting"):
#         try:
#             sim = __import__(sim_mod, fromlist=["Simulation"])
#             Simulation = getattr(sim, "Simulation")
#             break
#         except Exception as e:
#             last_err = e
#             Simulation = None
#             continue
#     if Simulation is None:
#         raise ImportError(
#             f"Could not import Simulation from repo at {repo_path}: {last_err}"
#         )

#     # Trapping_Vars is likely defined in same module or a voltage interface module
#     Trapping_Vars = None

#     # Trapping_Vars probe list
#     for vmod in (
#         "trapping_variables",
#         "sim.simulation",
#         "voltage_interfaceMixin",
#         "voltage_fitsMixin",
#     ):
#         try:
#             m = __import__(vmod, fromlist=["Trapping_Vars"])
#             Trapping_Vars = getattr(m, "Trapping_Vars")
#             break
#         except Exception:
#             continue
#     if Trapping_Vars is None:
#         # Some projects expose it from simulation as well
#         try:
#             Trapping_Vars = getattr(sim, "Trapping_Vars")
#         except Exception as e:  # noqa: BLE001
#             raise ImportError(
#                 "Could not import Trapping_Vars; set PYTHONPATH or adjust app imports."
#             ) from e

#     return Simulation, Trapping_Vars


# def _hashable_cfg(cfg: Dict[str, Any]) -> str:
#     """Stable hash key for caching, based on a JSON dump."""
#     return json.dumps(cfg, sort_keys=True, separators=(",", ":"))


# def _extract_eq_positions(sim, nmf, num_ions):
#     import numpy as np

#     # 1) Try nmf dict keys
#     for key in ("equilibrium_positions", "eq_positions", "equilibrium", "x_eq"):
#         if key in nmf:
#             arr = np.asarray(nmf[key])
#             if arr.ndim == 2 and arr.shape[1] == 3:
#                 return arr

#     # 2) Try attributes on sim (dict keyed by num_ions or direct array)
#     for attr in (
#         "equilibrium_positions",
#         "eq_positions",
#         "equilibrium",
#         "V_min_positions",
#         "Vmin_positions",
#         "V_minima",
#     ):
#         if hasattr(sim, attr):
#             obj = getattr(sim, attr)
#             if isinstance(obj, dict) and num_ions in obj:
#                 arr = np.asarray(obj[num_ions])
#                 if arr.ndim == 2 and arr.shape[1] == 3:
#                     return arr
#             elif isinstance(obj, (list, np.ndarray)):
#                 arr = np.asarray(obj)
#                 if arr.ndim == 2 and arr.shape[1] == 3:
#                     return arr

#     # 3) Last resort: call a method that returns positions
#     for meth in (
#         "get_equilibrium_positions",
#         "find_V_min",
#         "find_Vmin",
#         "find_V_min_positions",
#     ):
#         if hasattr(sim, meth):
#             try:
#                 ret = getattr(sim, meth)(num_ions)
#                 arr = np.asarray(ret)
#                 if arr.ndim == 2 and arr.shape[1] == 3:
#                     return arr
#             except Exception:
#                 pass

#     return None


# def _scan_equilibrium_candidates(sim, nmf, num_ions):
#     import numpy as np

#     candidates = []

#     # (A) Look inside nmf dict
#     for key in (
#         "equilibrium_positions",
#         "eq_positions",
#         "equilibrium",
#         "x_eq",
#         "V_min_positions",
#     ):
#         if key in nmf:
#             try:
#                 arr = np.asarray(nmf[key])
#                 if (
#                     arr.ndim == 2
#                     and arr.shape[1] == 3
#                     and np.issubdtype(arr.dtype, np.number)
#                 ):
#                     candidates.append((f"nmf.{key}", arr))
#             except Exception:
#                 pass

#     # (B) Look on sim as attributes (either dict keyed by num_ions or a 2D array)
#     for attr in (
#         "equilibrium_positions",
#         "eq_positions",
#         "equilibrium",
#         "V_min_positions",
#         "Vmin_positions",
#         "V_minima",
#         "x_eq",
#     ):
#         if hasattr(sim, attr):
#             try:
#                 obj = getattr(sim, attr)
#                 if isinstance(obj, dict) and num_ions in obj:
#                     arr = np.asarray(obj[num_ions])
#                     if (
#                         arr.ndim == 2
#                         and arr.shape[1] == 3
#                         and np.issubdtype(arr.dtype, np.number)
#                     ):
#                         candidates.append((f"sim.{attr}[{num_ions}]", arr))
#                 else:
#                     arr = np.asarray(obj)
#                     if (
#                         arr.ndim == 2
#                         and arr.shape[1] == 3
#                         and np.issubdtype(arr.dtype, np.number)
#                     ):
#                         candidates.append((f"sim.{attr}", arr))
#             except Exception:
#                 pass

#     # (C) As a last resort, try calling a method that returns positions
#     for meth in (
#         "get_equilibrium_positions",
#         "find_V_min",
#         "find_Vmin",
#         "find_V_min_positions",
#     ):
#         if hasattr(sim, meth):
#             try:
#                 ret = getattr(sim, meth)(num_ions)
#                 arr = np.asarray(ret)
#                 if (
#                     arr.ndim == 2
#                     and arr.shape[1] == 3
#                     and np.issubdtype(arr.dtype, np.number)
#                 ):
#                     candidates.append((f"sim.{meth}()", arr))
#             except Exception:
#                 pass

#     # Deduplicate by label
#     seen = set()
#     uniq = []
#     for lab, arr in candidates:
#         if lab not in seen:
#             uniq.append((lab, arr))
#             seen.add(lab)
#     return uniq


# @st.cache_data(show_spinner=False)
# def compute_result(cfg_key: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
#     """Core compute: build trap, run stack, collect outputs. Cached by cfg_key."""
#     Simulation, Trapping_Vars = _ensure_imports(
#         cfg["repo_path"], cfg["add_repo_to_sys_path"]
#     )  # may raise

#     # Build Trapping_Vars and drives
#     tv = Trapping_Vars()
#     rf = tv.add_driving(
#         "RF", cfg["rf_freq"], 0.0, {"RF1": cfg["rf_amp1"], "RF2": cfg["rf_amp2"]}
#     )
#     tv.apply_dc_twist_endcaps(twist=cfg["twist"], endcaps=int(cfg["endcaps"]))

#     extra_key = None
#     if cfg["use_extra"]:
#         try:
#             extra_map = (
#                 json.loads(cfg["extra_map_json"]) if cfg["extra_map_json"] else {}
#             )
#             assert isinstance(extra_map, dict)
#         except Exception as e:  # noqa: BLE001
#             raise ValueError(
#                 f"Extra drive map must be a JSON dict. Parse error: {e}"
#             ) from e
#         extra_key = tv.add_driving("ExtraDrive1", cfg["extra_freq"], 0.0, extra_map)

#     # Build Simulation and run base pipeline
#     sim = Simulation(cfg["preset"], tv)

#     # This is the known entrypoint seen in your example; adjust if your API differs.
#     if hasattr(sim, "_smoke_test_new_stack"):
#         sim._smoke_test_new_stack(
#             n_ions=cfg["num_ions"], poly_deg=cfg["poly_deg"]
#         )  # noqa: SLF001
#     else:
#         # Fallbacks if the project uses a different entrypoint
#         if hasattr(sim, "get_static_normal_modes_and_freq"):
#             sim.get_static_normal_modes_and_freq(
#                 cfg["num_ions"], normalize=True, sort_by_freq=True
#             )
#         else:
#             raise RuntimeError(
#                 "Simulation does not have _smoke_test_new_stack or get_static_normal_modes_and_freq."
#             )

#     # Frequencies & modes
#     nmf = sim.normal_modes_and_frequencies[cfg["num_ions"]]
#     freqs = np.asarray(nmf.get("frequencies_Hz"), dtype=float)
#     modes = np.asarray(nmf.get("modes"))  # (3N, 3N) with modes as columns

#     # Equilibrium positions: scan for candidates and pick first by default
#     eq_positions = _eq_from_ion_equilibrium_positions(sim, cfg["num_ions"])

#     # Resonances
#     drives_arg = None  # let the sim discover all non-DC drives
#     try:
#         out_res = sim.collect_resonant_couplings(
#             num_ions=cfg["num_ions"],
#             tol_Hz=cfg["tol_Hz"],
#             orders=tuple(cfg["orders"]),
#             drives=drives_arg,
#         )
#     except Exception as e:  # noqa: BLE001
#         # Provide a clearer message if the method is missing
#         raise RuntimeError(
#             "collect_resonant_couplings failed. Ensure StaticCoupolingMixin with this method is mixed into Simulation."
#         ) from e

#     return {
#         "frequencies_Hz": freqs,
#         "modes": modes,
#         "eq_positions": eq_positions,
#         "resonances": out_res,
#     }


# # ------------------------------------------------------------
# # Run compute when requested
# # ------------------------------------------------------------
# if run_btn:
#     cfg: Dict[str, Any] = {
#         "repo_path": repo_path,
#         "add_repo_to_sys_path": add_repo_to_sys_path,
#         "preset": preset,
#         "num_ions": int(num_ions),
#         "poly_deg": int(poly_deg),
#         "rf_freq": float(rf_freq),
#         "rf_amp1": float(rf_amp1),
#         "rf_amp2": float(rf_amp2),
#         "use_extra": bool(use_extra),
#         "extra_freq": float(extra_freq),
#         "extra_map_json": extra_map_json,
#         "twist": float(twist),
#         "endcaps": int(endcaps),
#         "tol_Hz": float(tol_Hz),
#         "orders": list(map(int, orders_pick)),
#     }

#     cfg_key = _hashable_cfg(cfg)

#     try:
#         t0 = time.time()
#         res = compute_result(cfg_key, cfg)
#         dt = time.time() - t0
#     except Exception as e:
#         st.error(f"❌ Compute failed: {e}")
#         res = None
#     else:
#         st.session_state["res"] = res
#         st.session_state["cfg"] = cfg

#     st.success(f"✅ Done in {dt:.3f} s")

#     # --------------------------------------------------------
#     # Layout: three columns for quick glance
#     # --------------------------------------------------------
#     c1, c2 = st.columns([1, 1])

#     with c1:
#         st.subheader("Equilibrium positions (µm)")
#         arr_to_show = res.get("eq_positions")

#         if arr_to_show is None:
#             st.info(
#                 f"No entry in sim.ion_equilibrium_positions for {cfg['num_ions']} ions. "
#                 "Make sure your pipeline populates it (your _smoke_test_new_stack already should)."
#             )
#         else:
#             import numpy as np

#             df_eq = (
#                 pd.DataFrame(np.asarray(arr_to_show), columns=["x", "y", "z"])
#                 .reset_index()
#                 .rename(columns={"index": "ion"})
#             )
#             # Show in µm if values look like meters
#             scale = (
#                 1e6 if np.nanmax(np.abs(df_eq[["x", "y", "z"]].values)) < 1e-2 else 1.0
#             )
#             if scale != 1.0:
#                 df_eq[["x", "y", "z"]] = df_eq[["x", "y", "z"]] * scale
#             st.caption("source: sim.ion_equilibrium_positions[num_ions]")
#             st.dataframe(df_eq, use_container_width=True)

#         st.subheader("Frequencies (Hz)")
#         df_f = pd.DataFrame(
#             {
#                 "mode": np.arange(len(res["frequencies_Hz"])),
#                 "f_Hz": res["frequencies_Hz"],
#             }
#         )
#         st.dataframe(df_f, use_container_width=True)
#         st.bar_chart(df_f, x="mode", y="f_Hz", use_container_width=True)

#     with c2:
#         st.subheader("Normal modes |1| (columns are modes)")
#         modes = res["modes"]
#         if modes is None or modes.size == 0:
#             st.info("Mode matrix not available.")
#         else:
#             abs_modes = np.abs(modes)
#             # Build coordinate labels 1..3N grouped by ion
#             ncoords = abs_modes.shape[0]
#             labels = [f"q{idx+1}" for idx in range(ncoords)]
#             df_modes = pd.DataFrame(abs_modes, index=labels)
#             st.dataframe(df_modes, use_container_width=True, height=360)

#             # --- NEW: Per-mode participation bar plot ---
#         st.subheader("Mode participation (per ion)")

#         if modes is not None and modes.size != 0:
#             import numpy as np
#             import matplotlib.pyplot as plt

#             freqs = np.asarray(res.get("frequencies_Hz"), dtype=float)
#             M = modes.shape[1]  # 3N
#             N = modes.shape[0] // 3

#             # Mode selector (aligns with freqs and modes columns)
#             mode_labels = [
#                 f"{i}: {freqs[i]:.2f} Hz" if i < len(freqs) else f"{i}"
#                 for i in range(M)
#             ]
#             mode_idx = st.selectbox(
#                 "Select mode",
#                 options=list(range(M)),
#                 format_func=lambda i: mode_labels[i],
#                 index=0,
#                 key="mode_select",
#             )
#             plot_abs = st.checkbox("Plot absolute participation |·|", value=True)

#             # Pull the selected eigenvector (columns are modes)
#             v = np.asarray(modes[:, mode_idx]).reshape(3 * N)
#             # Build per-ion xyz arrays
#             x_part = v[0::3]
#             y_part = v[1::3]
#             z_part = v[2::3]

#             if plot_abs:
#                 x_show, y_show, z_show = np.abs(x_part), np.abs(y_part), np.abs(z_part)
#                 y_label = "Participation (|component|)"
#             else:
#                 x_show, y_show, z_show = x_part, y_part, z_part
#                 y_label = "Participation (component)"

#             # Grouped bar positions:
#             # For each ion i we want three bars packed together (x,y,z) with no gap,
#             # and a visible gap between ions. Do that by spacing groups by (3 + gap),
#             # and placing bars at offsets 0,1,2 within the group.
#             group_gap = 0.6
#             base = np.arange(N) * (3.0 + group_gap)
#             pos_x = base + 0.0
#             pos_y = base + 1.0
#             pos_z = base + 2.0
#             width = 1.0  # contiguous inside each ion group

#             # X tick labels: ion index and (x,y,z) if we have them
#             eq = res.get("eq_positions")
#             if eq is not None:
#                 eq = np.asarray(eq, dtype=float)
#                 # Decide units: if coords look like meters, show µm
#                 scale = 1e6 if np.nanmax(np.abs(eq)) < 1e-2 else 1.0
#                 unit = "µm" if scale == 1e6 else "arb"
#                 disp = eq * scale
#                 tick_labels = [
#                     f"{i+1}\n({disp[i,0]:.2f}, {disp[i,1]:.2f}, {disp[i,2]:.2f}) {unit}"
#                     for i in range(N)
#                 ]
#             else:
#                 tick_labels = [f"{i+1}" for i in range(N)]

#             # Plot
#             fig, ax = plt.subplots(figsize=(min(10, max(6, N * 0.6)), 4.8))

#             # Consistent colors for axes
#             col_x = "#1f77b4"  # x
#             col_y = "#ff7f0e"  # y
#             col_z = "#2ca02c"  # z

#             ax.bar(pos_x, x_show, width=width, label="x", color=col_x)
#             ax.bar(pos_y, y_show, width=width, label="y", color=col_y)
#             ax.bar(pos_z, z_show, width=width, label="z", color=col_z)

#             # Title & labels
#             f_str = f"{freqs[mode_idx]:.2f} Hz" if mode_idx < len(freqs) else "n/a"
#             ax.set_title(f"Mode {mode_idx} — f = {f_str}")
#             ax.set_ylabel(y_label)

#             # Set group-centered ticks (middle bar of each group)
#             ax.set_xticks(base + 1.0)
#             ax.set_xticklabels(tick_labels, rotation=0, ha="center")

#             # Nice grid & limits
#             ax.grid(axis="y", linestyle=":", alpha=0.4)
#             if not plot_abs:
#                 # symmetric y-limits around zero for signed view
#                 ymax = np.nanmax(np.abs([x_show, y_show, z_show]))
#                 ax.set_ylim(-1.1 * ymax, 1.1 * ymax)

#             ax.legend(ncols=3, loc="upper right", frameon=False)
#             st.pyplot(fig, use_container_width=True)
#         else:
#             st.info("Mode matrix not available, so participation plot can’t be drawn.")

#         st.subheader("Resonant couplings")
#         out_res = res["resonances"]
#         # Flatten useful bits for display
#         rows: List[Dict[str, Any]] = []
#         try:
#             R = out_res["resonances"]
#             for order in (2, 3, 4):
#                 for item in R.get(order, []):
#                     row = {
#                         "order": order,
#                         "modes": tuple(item.get("modes", [])),
#                         "target_Hz": item.get("target_Hz"),
#                         "detune_Hz": item.get("delta_Hz"),
#                     }
#                     if order == 2:
#                         # Try to show g0_Hz (single or list) and which drive(s)
#                         row["g0_Hz"] = item.get("g0_Hz") or item.get("g0_Hz_by_drive")
#                         if "drive_resonances" in item:
#                             row["drives"] = [
#                                 getattr(d.get("drive"), "label", str(d.get("drive")))
#                                 for d in item["drive_resonances"]
#                             ]
#                     elif order == 3:
#                         row["g3_Hz"] = item.get("g3_Hz")
#                     elif order == 4:
#                         row["g4_Hz"] = item.get("g4_Hz")
#                     rows.append(row)
#         except Exception:
#             st.info("Resonance output schema not recognized; showing raw JSON below.")
#             rows = []

#         if rows:
#             df_rows = pd.DataFrame(rows)
#             st.dataframe(
#                 df_rows.sort_values(["order", "detune_Hz"], ascending=[True, True]),
#                 use_container_width=True,
#             )
#         else:
#             st.write("(No flattened rows to display.)")

#     st.divider()
#     # Raw resonance JSON
#     st.subheader("Raw resonance JSON (for debugging / export)")
#     st.code(json.dumps(_to_jsonable(out_res), indent=2))

#     # Download buttons
#     st.download_button(
#         label="Download frequencies.json",
#         data=json.dumps(_to_jsonable(res["frequencies_Hz"]), indent=2),
#         file_name="frequencies.json",
#         mime="application/json",
#     )

#     st.download_button(
#         label="Download resonances.json",
#         data=json.dumps(_to_jsonable(out_res), indent=2),
#         file_name="resonances.json",
#         mime="application/json",
#     )

# else:
#     st.info("Set parameters in the sidebar and click **Compute**.")


import os
import sys
import json
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st


# Persist last results and config across reruns
if "res" not in st.session_state:
    st.session_state["res"] = None
if "cfg" not in st.session_state:
    st.session_state["cfg"] = None


def _to_jsonable(obj):
    import numpy as np, json

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)


def _eq_from_ion_equilibrium_positions(sim, num_ions):
    """
    Return sim.ion_equilibrium_positions[num_ions] if it exists and is (N,3).
    Does not call any compute methods.
    """
    import numpy as np

    d = getattr(sim, "ion_equilibrium_positions", None)
    if isinstance(d, dict) and (num_ions in d):
        arr = np.asarray(d[num_ions], dtype=float)
        if arr.ndim == 2 and arr.shape[1] == 3:
            return arr
    return None


# ------------------------------------------------------------
# App configuration
# ------------------------------------------------------------
st.set_page_config(page_title="Ion Trap Resonance Explorer", layout="wide")

st.title("🔬 Ion Trap Resonance Explorer")
st.caption(
    "Quick UI to tweak trapping variables, choose ion count, and inspect equilibria, modes, frequencies, and resonant couplings."
)

# ------------------------------------------------------------
# Sidebar inputs (as a FORM so edits don't apply until 'Compute')
# ------------------------------------------------------------
with st.sidebar:
    st.header("Project / Imports")
    with st.form("params"):
        repo_path = st.text_input(
            "Path to TrapFrequencyAnalysis repo (so we can import Simulation, Trapping_Vars)",
            value=".",
            help="Use an absolute path or a relative path from where you run `streamlit run app.py`.",
        )
        add_repo_to_sys_path = st.checkbox(
            "Add the path above to PYTHONPATH", value=True
        )

        st.divider()
        st.header("Trap / Simulation Setup")
        preset = st.selectbox(
            "Simulation preset (name passed to Simulation)",
            options=["Simp58_101", "NISTMock", "Hyper_2", "Custom"],
            index=0,
        )
        if preset == "Custom":
            preset = st.text_input("Custom preset string", value="Simp58_101")

        num_ions = st.number_input(
            "Number of ions", min_value=1, max_value=60, value=3, step=1
        )
        poly_deg = st.selectbox(
            "Polynomial degree for fits", options=[2, 3, 4, 5, 6], index=2
        )

        st.subheader("RF Drive")
        rf_freq = st.number_input(
            "RF frequency (Hz)",
            min_value=1.0,
            value=25_500_000.0,
            step=1_000.0,
            format="%.3f",
        )
        rf_amp1 = st.number_input("RF1 amplitude (V)", value=377.0, step=0.5)
        rf_amp2 = st.number_input("RF2 amplitude (V)", value=377.0, step=0.5)

        st.subheader("Extra Drive (optional)")
        use_extra = st.checkbox("Enable Extra Drive", value=True)
        extra_freq = st.number_input(
            "Extra Drive frequency (Hz)",
            min_value=0.0,
            value=250_000.0,
            step=100.0,
            format="%.3f",
        )
        default_extra_map = {
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
        extra_map_json = st.text_area(
            "Extra Drive DC map (JSON dict)",
            value=json.dumps(default_extra_map, indent=2),
            height=160,
            help="Map electrode names to amplitudes (in volts). Example keys: DC1..DC10.",
        )

        st.subheader("DC Geometry")
        twist = st.number_input("apply_dc_twist_endcaps: twist", value=0.275, step=0.01)
        endcaps = st.number_input(
            "apply_dc_twist_endcaps: endcaps",
            min_value=0,
            max_value=20,
            value=3,
            step=1,
        )

        st.divider()
        st.header("Resonance Scan")
        tol_Hz = st.number_input(
            "Resonance tolerance ± (Hz)",
            min_value=0.0,
            value=1_000.0,
            step=10.0,
            format="%.3f",
        )
        orders_pick = st.multiselect(
            "Coupling orders", options=[2, 3, 4], default=[2, 3]
        )

        run_btn = st.form_submit_button("Compute", type="primary")


# ------------------------------------------------------------
# Utilities: imports, hashing, and compute wrapper
# ------------------------------------------------------------
def _ensure_imports(repo_path: str, add_to_sys_path: bool = True):
    """Ensure we can import Simulation and Trapping_Vars from the repo.
    Returns (Simulation, Trapping_Vars) types if successful; otherwise raises.
    """
    if add_to_sys_path:
        abs_repo = os.path.abspath(repo_path)
        if abs_repo not in sys.path:
            sys.path.insert(0, abs_repo)

    # Try a few plausible import paths, fall back gracefully.
    last_err = None
    for sim_mod in ("sim.simulation", "simulation", "Simulation", "simulation_fitting"):
        try:
            sim = __import__(sim_mod, fromlist=["Simulation"])
            Simulation = getattr(sim, "Simulation")
            break
        except Exception as e:
            last_err = e
            Simulation = None
            continue
    if Simulation is None:
        raise ImportError(
            f"Could not import Simulation from repo at {repo_path}: {last_err}"
        )

    Trapping_Vars = None
    for vmod in (
        "trapping_variables",
        "sim.simulation",
        "voltage_interfaceMixin",
        "voltage_fitsMixin",
    ):
        try:
            m = __import__(vmod, fromlist=["Trapping_Vars"])
            Trapping_Vars = getattr(m, "Trapping_Vars")
            break
        except Exception:
            continue
    if Trapping_Vars is None:
        try:
            Trapping_Vars = getattr(sim, "Trapping_Vars")
        except Exception as e:  # noqa: BLE001
            raise ImportError(
                "Could not import Trapping_Vars; set PYTHONPATH or adjust app imports."
            ) from e

    return Simulation, Trapping_Vars


def _hashable_cfg(cfg: Dict[str, Any]) -> str:
    """Stable hash key for caching, based on a JSON dump."""
    return json.dumps(cfg, sort_keys=True, separators=(",", ":"))


def _extract_eq_positions(sim, nmf, num_ions):
    import numpy as np

    # 1) Try nmf dict keys
    for key in ("equilibrium_positions", "eq_positions", "equilibrium", "x_eq"):
        if key in nmf:
            arr = np.asarray(nmf[key])
            if arr.ndim == 2 and arr.shape[1] == 3:
                return arr

    # 2) Try attributes on sim (dict keyed by num_ions or direct array)
    for attr in (
        "equilibrium_positions",
        "eq_positions",
        "equilibrium",
        "V_min_positions",
        "Vmin_positions",
        "V_minima",
    ):
        if hasattr(sim, attr):
            obj = getattr(sim, attr)
            if isinstance(obj, dict) and num_ions in obj:
                arr = np.asarray(obj[num_ions])
                if arr.ndim == 2 and arr.shape[1] == 3:
                    return arr
            elif isinstance(obj, (list, np.ndarray)):
                arr = np.asarray(obj)
                if arr.ndim == 2 and arr.shape[1] == 3:
                    return arr

    # 3) Last resort: call a method that returns positions
    for meth in (
        "get_equilibrium_positions",
        "find_V_min",
        "find_Vmin",
        "find_V_min_positions",
    ):
        if hasattr(sim, meth):
            try:
                ret = getattr(sim, meth)(num_ions)
                arr = np.asarray(ret)
                if arr.ndim == 2 and arr.shape[1] == 3:
                    return arr
            except Exception:
                pass

    return None


def _scan_equilibrium_candidates(sim, nmf, num_ions):
    import numpy as np

    candidates = []

    # (A) Look inside nmf dict
    for key in (
        "equilibrium_positions",
        "eq_positions",
        "equilibrium",
        "x_eq",
        "V_min_positions",
    ):
        if key in nmf:
            try:
                arr = np.asarray(nmf[key])
                if (
                    arr.ndim == 2
                    and arr.shape[1] == 3
                    and np.issubdtype(arr.dtype, np.number)
                ):
                    candidates.append((f"nmf.{key}", arr))
            except Exception:
                pass

    # (B) Look on sim as attributes
    for attr in (
        "equilibrium_positions",
        "eq_positions",
        "equilibrium",
        "V_min_positions",
        "Vmin_positions",
        "V_minima",
        "x_eq",
    ):
        if hasattr(sim, attr):
            try:
                obj = getattr(sim, attr)
                if isinstance(obj, dict) and num_ions in obj:
                    arr = np.asarray(obj[num_ions])
                    if (
                        arr.ndim == 2
                        and arr.shape[1] == 3
                        and np.issubdtype(arr.dtype, np.number)
                    ):
                        candidates.append((f"sim.{attr}[{num_ions}]", arr))
                else:
                    arr = np.asarray(obj)
                    if (
                        arr.ndim == 2
                        and arr.shape[1] == 3
                        and np.issubdtype(arr.dtype, np.number)
                    ):
                        candidates.append((f"sim.{attr}", arr))
            except Exception:
                pass

    # (C) Try a method
    for meth in (
        "get_equilibrium_positions",
        "find_V_min",
        "find_Vmin",
        "find_V_min_positions",
    ):
        if hasattr(sim, meth):
            try:
                ret = getattr(sim, meth)(num_ions)
                arr = np.asarray(ret)
                if (
                    arr.ndim == 2
                    and arr.shape[1] == 3
                    and np.issubdtype(arr.dtype, np.number)
                ):
                    candidates.append((f"sim.{meth}()", arr))
            except Exception:
                pass

    # Deduplicate by label
    seen = set()
    uniq = []
    for lab, arr in candidates:
        if lab not in seen:
            uniq.append((lab, arr))
            seen.add(lab)
    return uniq


@st.cache_data(show_spinner=False)
def compute_result(cfg_key: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Core compute: build trap, run stack, collect outputs. Cached by cfg_key."""
    Simulation, Trapping_Vars = _ensure_imports(
        cfg["repo_path"], cfg["add_repo_to_sys_path"]
    )  # may raise

    # Build Trapping_Vars and drives
    tv = Trapping_Vars()
    rf = tv.add_driving(
        "RF", cfg["rf_freq"], 0.0, {"RF1": cfg["rf_amp1"], "RF2": cfg["rf_amp2"]}
    )
    tv.apply_dc_twist_endcaps(twist=cfg["twist"], endcaps=int(cfg["endcaps"]))

    if cfg["use_extra"]:
        try:
            extra_map = (
                json.loads(cfg["extra_map_json"]) if cfg["extra_map_json"] else {}
            )
            assert isinstance(extra_map, dict)
        except Exception as e:  # noqa: BLE001
            raise ValueError(
                f"Extra drive map must be a JSON dict. Parse error: {e}"
            ) from e
        tv.add_driving("ExtraDrive1", cfg["extra_freq"], 0.0, extra_map)

    # Build Simulation and run base pipeline
    sim = Simulation(cfg["preset"], tv)

    # Entry point (adjust if your API differs)
    if hasattr(sim, "_smoke_test_new_stack"):
        sim._smoke_test_new_stack(
            n_ions=cfg["num_ions"], poly_deg=cfg["poly_deg"]
        )  # noqa: SLF001
    elif hasattr(sim, "get_static_normal_modes_and_freq"):
        sim.get_static_normal_modes_and_freq(
            cfg["num_ions"], normalize=True, sort_by_freq=True
        )
    else:
        raise RuntimeError(
            "Simulation does not have _smoke_test_new_stack or get_static_normal_modes_and_freq."
        )

    # Frequencies & modes
    nmf = sim.normal_modes_and_frequencies[cfg["num_ions"]]
    freqs = np.asarray(nmf.get("frequencies_Hz"), dtype=float)
    modes = np.asarray(nmf.get("modes"))  # (3N, 3N) with modes as columns

    # Equilibrium positions
    eq_positions = _eq_from_ion_equilibrium_positions(sim, cfg["num_ions"])

    # Resonances
    drives_arg = None  # let the sim discover all non-DC drives
    try:
        out_res = sim.collect_resonant_couplings(
            num_ions=cfg["num_ions"],
            tol_Hz=cfg["tol_Hz"],
            orders=tuple(cfg["orders"]),
            drives=drives_arg,
        )
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "collect_resonant_couplings failed. Ensure StaticCoupolingMixin with this method is mixed into Simulation."
        ) from e

    return {
        "frequencies_Hz": freqs,
        "modes": modes,
        "eq_positions": eq_positions,
        "resonances": out_res,
    }


# ------------------------------------------------------------
# Optional: warn if inputs changed since last compute
# ------------------------------------------------------------
pending_cfg = {
    "repo_path": repo_path,
    "add_repo_to_sys_path": add_repo_to_sys_path,
    "preset": preset,
    "num_ions": int(num_ions),
    "poly_deg": int(poly_deg),
    "rf_freq": float(rf_freq),
    "rf_amp1": float(rf_amp1),
    "rf_amp2": float(rf_amp2),
    "use_extra": bool(use_extra),
    "extra_freq": float(extra_freq),
    "extra_map_json": extra_map_json,
    "twist": float(twist),
    "endcaps": int(endcaps),
    "tol_Hz": float(tol_Hz),
    "orders": list(map(int, orders_pick)),
}

# ------------------------------------------------------------
# Run compute only when requested; otherwise render from cache
# ------------------------------------------------------------
if run_btn:
    cfg = dict(pending_cfg)  # snapshot
    cfg_key = _hashable_cfg(cfg)
    try:
        t0 = time.time()
        res = compute_result(cfg_key, cfg)
        dt = time.time() - t0
    except Exception as e:
        st.error(f"❌ Compute failed: {e}")
        res = None
    else:
        st.session_state["res"] = res
        st.session_state["cfg"] = cfg
        st.success(f"✅ Done in {dt:.3f} s")

# Always render from cached result (if any)
res = st.session_state.get("res")
cfg_cached = st.session_state.get("cfg")

if cfg_cached and _hashable_cfg(pending_cfg) != _hashable_cfg(cfg_cached):
    st.warning("Inputs changed since last compute — press **Compute** to update.")

if res is not None:
    # --------------------------------------------------------
    # Layout: two columns for quick glance
    # --------------------------------------------------------
    c1, c2 = st.columns([1, 1])

    with c1:
        st.subheader("Equilibrium positions (µm)")
        arr_to_show = res.get("eq_positions")

        if arr_to_show is None:
            st.info(
                f"No entry in sim.ion_equilibrium_positions for {pending_cfg['num_ions']} ions. "
                "Make sure your pipeline populates it (your _smoke_test_new_stack already should)."
            )
        else:
            df_eq = (
                pd.DataFrame(np.asarray(arr_to_show), columns=["x", "y", "z"])
                .reset_index()
                .rename(columns={"index": "ion"})
            )
            # Show in µm if values look like meters
            scale = (
                1e6 if np.nanmax(np.abs(df_eq[["x", "y", "z"]].values)) < 1e-2 else 1.0
            )
            if scale != 1.0:
                df_eq[["x", "y", "z"]] = df_eq[["x", "y", "z"]] * scale
            st.caption("source: sim.ion_equilibrium_positions[num_ions]")
            st.dataframe(df_eq, use_container_width=True)

        st.subheader("Frequencies (Hz)")
        df_f = pd.DataFrame(
            {
                "mode": np.arange(len(res["frequencies_Hz"])),
                "f_Hz": res["frequencies_Hz"],
            }
        )
        st.dataframe(df_f, use_container_width=True)
        st.bar_chart(df_f, x="mode", y="f_Hz", use_container_width=True)

    with c2:
        st.subheader("Normal modes |1| (columns are modes)")
        modes = res["modes"]
        if modes is None or modes.size == 0:
            st.info("Mode matrix not available.")
        else:
            abs_modes = np.abs(modes)
            ncoords = abs_modes.shape[0]
            labels = [f"q{idx+1}" for idx in range(ncoords)]
            df_modes = pd.DataFrame(abs_modes, index=labels)
            st.dataframe(df_modes, use_container_width=True, height=360)

        # --- Per-mode participation bar plot ---
        st.subheader("Mode participation (per ion)")
        if modes is not None and modes.size != 0:
            import matplotlib.pyplot as plt

            freqs = np.asarray(res.get("frequencies_Hz"), dtype=float)
            M = modes.shape[1]  # 3N
            N = modes.shape[0] // 3

            # Mode selector (aligns with freqs and modes columns)
            mode_labels = [
                f"{i}: {freqs[i]:.2f} Hz" if i < len(freqs) else f"{i}"
                for i in range(M)
            ]
            mode_idx = st.selectbox(
                "Select mode",
                options=list(range(M)),
                format_func=lambda i: mode_labels[i],
                index=0,
                key="mode_select",
            )
            plot_abs = st.checkbox("Plot absolute participation |·|", value=True)

            # Pull the selected eigenvector (columns are modes)
            v = np.asarray(modes[:, mode_idx]).reshape(3 * N)
            # Build per-ion xyz arrays
            x_part = v[0::3]
            y_part = v[1::3]
            z_part = v[2::3]

            if plot_abs:
                x_show, y_show, z_show = np.abs(x_part), np.abs(y_part), np.abs(z_part)
                y_label = "Participation (|component|)"
            else:
                x_show, y_show, z_show = x_part, y_part, z_part
                y_label = "Participation (component)"

            # Grouped bar positions: no space inside (x,y,z), gap between ions
            group_gap = 0.6
            base = np.arange(N) * (3.0 + group_gap)
            pos_x = base + 0.0
            pos_y = base + 1.0
            pos_z = base + 2.0
            width = 1.0

            # X tick labels: ion index and (x,y,z) if available
            eq = res.get("eq_positions")
            if eq is not None:
                eq = np.asarray(eq, dtype=float)
                scale = 1e6 if np.nanmax(np.abs(eq)) < 1e-2 else 1.0
                unit = "µm" if scale == 1e6 else "arb"
                disp = eq * scale
                tick_labels = [
                    f"{i+1}\n({disp[i,0]:.2f}, {disp[i,1]:.2f}, {disp[i,2]:.2f}) {unit}"
                    for i in range(N)
                ]
            else:
                tick_labels = [f"{i+1}" for i in range(N)]

            fig, ax = plt.subplots(figsize=(min(10, max(6, N * 0.6)), 4.8))

            col_x = "#1f77b4"  # x
            col_y = "#ff7f0e"  # y
            col_z = "#2ca02c"  # z

            ax.bar(pos_x, x_show, width=width, label="x", color=col_x)
            ax.bar(pos_y, y_show, width=width, label="y", color=col_y)
            ax.bar(pos_z, z_show, width=width, label="z", color=col_z)

            f_str = f"{freqs[mode_idx]:.2f} Hz" if mode_idx < len(freqs) else "n/a"
            ax.set_title(f"Mode {mode_idx} — f = {f_str}")
            ax.set_ylabel(y_label)

            ax.set_xticks(base + 1.0)
            ax.set_xticklabels(tick_labels, rotation=0, ha="center")

            ax.grid(axis="y", linestyle=":", alpha=0.4)
            if not plot_abs:
                ymax = np.nanmax(np.abs([x_show, y_show, z_show]))
                ax.set_ylim(-1.1 * ymax, 1.1 * ymax)

            ax.legend(ncols=3, loc="upper right", frameon=False)
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("Mode matrix not available, so participation plot can’t be drawn.")

        st.subheader("Resonant couplings")
        out_res = res["resonances"]
        rows: List[Dict[str, Any]] = []
        try:
            R = out_res["resonances"]
            for order in (2, 3, 4):
                for item in R.get(order, []):
                    row = {
                        "order": order,
                        "modes": tuple(item.get("modes", [])),
                        "target_Hz": item.get("target_Hz"),
                        "detune_Hz": item.get("delta_Hz"),
                    }
                    if order == 2:
                        row["g0_Hz"] = item.get("g0_Hz") or item.get("g0_Hz_by_drive")
                        if "drive_resonances" in item:
                            row["drives"] = [
                                getattr(d.get("drive"), "label", str(d.get("drive")))
                                for d in item["drive_resonances"]
                            ]
                    elif order == 3:
                        row["g3_Hz"] = item.get("g3_Hz")
                    elif order == 4:
                        row["g4_Hz"] = item.get("g4_Hz")
                    rows.append(row)
        except Exception:
            st.info("Resonance output schema not recognized; showing raw JSON below.")
            rows = []

        if rows:
            df_rows = pd.DataFrame(rows)
            st.dataframe(
                df_rows.sort_values(["order", "detune_Hz"], ascending=[True, True]),
                use_container_width=True,
            )
        else:
            st.write("(No flattened rows to display.)")

    st.divider()
    st.subheader("Raw resonance JSON (for debugging / export)")
    st.code(json.dumps(_to_jsonable(res["resonances"]), indent=2))

    st.download_button(
        label="Download frequencies.json",
        data=json.dumps(_to_jsonable(res["frequencies_Hz"]), indent=2),
        file_name="frequencies.json",
        mime="application/json",
    )
    st.download_button(
        label="Download resonances.json",
        data=json.dumps(_to_jsonable(res["resonances"]), indent=2),
        file_name="resonances.json",
        mime="application/json",
    )

else:
    st.info("Set parameters in the sidebar and click **Compute**.")
