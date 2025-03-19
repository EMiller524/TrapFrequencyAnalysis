from functools import reduce
import math
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import dataextractor
import consts
import Electrode
import time
import matplotlib.cm as cm
import concurrent.futures
from SimulationMix.simulation_ploting import sim_ploting
from SimulationMix.simulation_hessianfitting import sim_hessian
from SimulationMix.simulation_normalfitting import sim_normalfitting
from numba import njit, prange
import multiprocessing
from multiprocessing import Pool, cpu_count
from joblib import Parallel, delayed


multiprocessing.set_start_method("spawn", force=True)


def init_electrode(electrode_name, dataset, electrode_vars):
    return electrode_name, Electrode.Electrode(
        electrode_name, dataset, electrode_vars.get_vars(electrode_name)
    )


class Simulation(sim_ploting, sim_hessian, sim_normalfitting):
    def __init__(self, dataset, variables=consts.Electrode_vars()):
        """
        Initialize the Electrode object.
        """
        print("initializing simulation")
        self.dataset = dataset
        self.file_path = "C:\\GitHub\\TrapFrequencyAnalysis\\Data\\" + dataset + "\\"

        # make a dictionary of electrodes with each name in consts.electrode_names as the key and an electrode class as the value
        self.electrode_vars = variables

        self.electrodes = {}
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    init_electrode, electrode, self.dataset, self.electrode_vars
                )
                for electrode in consts.electrode_names
            ]
            for future in concurrent.futures.as_completed(futures):
                electrode_name, electrode_instance = future.result()
                self.electrodes[electrode_name] = electrode_instance

        # self.valid_points = self.get_valid_points()
        print("hiiii")
        self.total_voltage_df = None
        self.get_total_voltage_at_all_points()

        print("simulation initialized")

    def get_variables(self, electrode):
        return self.electrode_vars.get_vars(electrode)

    def get_valid_points(self):
        non_empty_dfs = []
        for electrode in self.electrodes:
            if self.electrodes[electrode].get_dataframe() is not None:
                non_empty_dfs.append(self.electrodes[electrode].get_dataframe())

        point_sets = [set(zip(df["x"], df["y"], df["z"])) for df in non_empty_dfs]

        common_points = set.intersection(*point_sets)

        return np.array(sorted(common_points))

    # def get_total_voltage_at_all_points(self):
    #     # get the data for each electrode
    #     dfs = []
    #     for electrode in consts.electrode_names:
    #         if self.electrodes[electrode].get_dataframe() is None:
    #             print(f"no data for {electrode}")
    #         else:
    #             dfs.append(self.electrodes[electrode].get_dataframe())

    #     # Step 1: Find the intersection of all unique (x, y, z) combinations
    #     common_keys = reduce(
    #         lambda left, right: pd.merge(left, right, on=["x", "y", "z"]),
    #         [df[["x", "y", "z"]] for df in dfs],
    #     )

    #     # Step 2: Concatenate all dataframes
    #     merged_df = pd.concat(dfs, ignore_index=True)

    #     # Step 3: Filter merged_df to only keep common (x, y, z) keys
    #     filtered_df = merged_df.merge(common_keys, on=["x", "y", "z"])

    #     # Step 4: Group by (x, y, z) and sum CalcV
    #     master_df = filtered_df.groupby(["x", "y", "z"], as_index=False)["CalcV"].sum()

    #     self.total_voltage_df = master_df

    def get_total_voltage_at_all_points(self):
        """🚀 Ultra-fast method to sum CalcV over all electrodes using Pandas groupby().sum() 🚀"""

        print("Computing total voltage at all points...")

        # ✅ Step 1: Extract all electrode data **before processing**
        dfs = []
        for elec in consts.electrode_names:
            electrode_data = self.electrodes[elec].get_dataframe()
            if electrode_data is not None:
                dfs.append(
                    electrode_data[["x", "y", "z", "CalcV"]]
                )  # Keep only necessary columns

        if not dfs:
            print("No valid electrode data found.")
            return None

        # ✅ Step 2: Concatenate all DataFrames at once (Fast!)
        merged_df = pd.concat(dfs, ignore_index=True)  # No manual merging needed!

        # ✅ Step 3: Group by (x, y, z) and sum CalcV efficiently
        self.total_voltage_df = merged_df.groupby(["x", "y", "z"], as_index=False)[
            "CalcV"
        ].sum()

        print("Total voltage computation complete.")

    # --- Main Optimized Function ---
    # def get_total_voltage_at_all_points(self): ## this really shouldnt take that long its not that hard, just need to sum blank columbs over 1-11 electrodes...
    #     """Ultra-fast method to sum CalcV across all electrodes using NumPy & Multiprocessing."""
    #     print("Computing total voltage at all points...")

    #     # ✅ Step 1: Extract data **before multiprocessing** to prevent reinitialization
    #     dfs = []
    #     for elec in consts.electrode_names:
    #         electrode_data = self.electrodes[elec].get_dataframe()
    #         if electrode_data is not None:
    #             dfs.append(electrode_data[["x", "y", "z", "CalcV"]].to_numpy())

    #     if not dfs:
    #         print("No valid electrode data found.")
    #         return None

    #     # ✅ Step 2: Convert (x, y, z) into structured NumPy arrays for fast set operations
    #     struct_dfs = [to_structured_array(df) for df in dfs]

    #     print("Hi3")
    #     # ✅ Step 3: Parallelize Intersection Calculation (WITHOUT REINITIALIZING ELECTRODES)
    #     num_workers = min(cpu_count(), len(struct_dfs))
    #     with Pool(num_workers) as pool:
    #         chunk_size = max(1, len(struct_dfs) // num_workers)
    #         struct_dfs_chunks = [
    #             struct_dfs[i : i + chunk_size]
    #             for i in range(0, len(struct_dfs), chunk_size)
    #         ]
    #         results = pool.map(fast_intersection, struct_dfs_chunks)
    #     print("hi4")
    #     # Merge results from different workers
    #     common_points = fast_intersection(results)

    #     print("Hi5")
    #     # 🚀 Step 4: **Optimized Filtering Using Binary Search**
    #     sorted_common_points = parallel_sort(common_points)
    #     print("Hi5.5") ## bad
    #     dfs = [
    #         df.to_numpy() for df in dfs
    #     ]  # Convert all to NumPy arrays before chunking
    #     chunk_size = max(1, len(dfs) // (cpu_count() or 1))
    #     dfs_chunks = [dfs[i:i + chunk_size] for i in range(0, len(dfs), chunk_size)]

    #     with Pool(cpu_count()) as pool:
    #         filtered_dfs = pool.starmap(fast_filter_common, [(chunk, sorted_common_points) for chunk in dfs_chunks])

    #     print("Hi6")
    #     # ✅ Step 5: Convert to NumPy Arrays & Sum `CalcV`
    #     merged_data = np.vstack(filtered_dfs)
    #     x_vals, y_vals, z_vals, calcV_vals = (
    #         merged_data[:, 0],
    #         merged_data[:, 1],
    #         merged_data[:, 2],
    #         merged_data[:, 3],
    #     )
    #     print("Hi7")
    #     # ✅ Ensure all values are NumPy arrays
    #     x_vals = np.asarray(x_vals).flatten()
    #     y_vals = np.asarray(y_vals).flatten()
    #     z_vals = np.asarray(z_vals).flatten()
    #     calcV_vals = np.asarray(calcV_vals).flatten()

    #     # ✅ Use column_stack instead of vstack
    #     xyz_vals = np.column_stack((x_vals, y_vals, z_vals))

    #     # ✅ Ensure all values have correct shapes
    #     assert xyz_vals.shape[1] == 3, "Error: xyz_vals should have shape (N, 3)"
    #     assert calcV_vals.shape[0] == xyz_vals.shape[0], "Error: Mismatch in shapes of coordinates and values."

    #     # ✅ Compute unique values
    #     unique_coords, indices = np.unique(xyz_vals, axis=0, return_inverse=True)

    #     # ✅ Compute sum correctly
    #     summed_calcV = np.bincount(indices, weights=calcV_vals)

    #     print("Hi8")
    #     # ✅ Step 6: Convert to DataFrame (Final Output)
    #     self.total_voltage_df = pd.DataFrame(
    #         {
    #             "x": unique_coords[:, 0],
    #             "y": unique_coords[:, 1],
    #             "z": unique_coords[:, 2],
    #             "CalcV": summed_calcV,
    #         }
    #     )

    #     print("Total voltage computation complete.")


# --- Helper Functions (Must Be Top-Level for Multiprocessing) ---


def to_structured_array(df):
    """Converts a NumPy array to a structured array for fast intersections."""
    return np.core.records.fromarrays(df[:, :3].T, names="x,y,z")


def fast_intersection(arrays):
    """Performs fast intersection of multiple structured arrays."""
    result = arrays[0]
    for arr in arrays[1:]:
        result = np.intersect1d(result, arr, assume_unique=True)
    return result


def filter_common(df, common_points):
    """Filters a DataFrame to keep only common (x, y, z) points."""
    struct_df = to_structured_array(df)
    mask = np.isin(struct_df, common_points)
    return df[mask]


def fast_filter_common(df, sorted_common_points):
    """Filters a DataFrame using binary search instead of isin()."""
    struct_df = to_structured_array(df)
    indices = np.searchsorted(sorted_common_points, struct_df)
    mask = (indices < len(sorted_common_points)) & (
        sorted_common_points[indices] == struct_df
    )
    return df[mask]


def parallel_sort(arr, num_chunks=cpu_count()):
    """Sort large NumPy arrays in parallel using chunks."""
    chunks = np.array_split(arr, num_chunks)
    sorted_chunks = Parallel(n_jobs=num_chunks)(
        delayed(np.sort)(chunk) for chunk in chunks
    )
    return np.concatenate(sorted_chunks)


# ✅ Make sure all simulation execution happens **inside** `if __name__ == "__main__"`
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)  # ✅ Prevent reinitialization

    elec_vars = consts.get_electrodvars_w_twist(377, 28000000 * 2 * math.pi, 0, 0)
    elec_vars0 = consts.get_electrodvars_w_twist(377, 28000000 * 2 * math.pi, .275, 3)

    print("Starting simulation...")

    # test_sim0 = Simulation("Simp58_101", elec_vars0)
    # test_sim0.get_main_report("Test_58_101_rfonly")
    tstart = time.time()
    test_sim1 = Simulation("Simp58_101", elec_vars)
    test_sim1.get_main_report("Test_58_101_twist_etc_test13")
    tstop = time.time()
    print("Time taken: " + str(tstop - tstart))

    # test_sim.get_main_report("Test_hyper2_lookarond100_poly4")
    # test_sim.compute_all_frequencies()

    # test_sim = Simulation("Simp58_2", elec_vars)
    # width = 140

    # freqsss_2 = [
    #     test_sim.get_wy_wz_at_point_withR2_fit(
    #         0, 0, 0, look_around=int(width * 0.05), polyfit=2
    #     )
    #     + [int(width * 0.05)],
    #     test_sim.get_wy_wz_at_point_withR2_fit(
    #         0, 0, 0, look_around=int(width * 0.1), polyfit=2
    #     )
    #     + [int(width * 0.1)],
    #     test_sim.get_wy_wz_at_point_withR2_fit(
    #         0, 0, 0, look_around=int(width * 0.16), polyfit=2
    #     )
    #     + [int(width * 0.16)],
    #     test_sim.get_wy_wz_at_point_withR2_fit(
    #         0, 0, 0, look_around=int(width * 0.165), polyfit=2
    #     )
    #     + [int(width * 0.165)],
    #     test_sim.get_wy_wz_at_point_withR2_fit(
    #         0, 0, 0, look_around=int(width * 0.17), polyfit=2
    #     )
    #     + [int(width * 0.17)],
    #     test_sim.get_wy_wz_at_point_withR2_fit(
    #         0, 0, 0, look_around=int(width * 0.18), polyfit=2
    #     )
    #     + [int(width * 0.18)],
    #     test_sim.get_wy_wz_at_point_withR2_fit(
    #         0, 0, 0, look_around=int(width * 0.2), polyfit=2
    #     )
    #     + [int(width * 0.2)],
    #     test_sim.get_wy_wz_at_point_withR2_fit(
    #         0, 0, 0, look_around=int(width * 0.3), polyfit=2
    #     )
    #     + [int(width * 0.3)],
    #     test_sim.get_wy_wz_at_point_withR2_fit(
    #         0, 0, 0, look_around=int(width * 0.4), polyfit=2
    #     )
    #     + [int(width * 0.4)],
    #     test_sim.get_wy_wz_at_point_withR2_fit(
    #         0, 0, 0, look_around=int(width * 0.5), polyfit=2
    #     )
    #     + [int(width * 0.5)],
    #     test_sim.get_wy_wz_at_point_withR2_fit(
    #         0, 0, 0, look_around=int(width * 0.6), polyfit=2
    #     )
    #     + [int(width * 0.6)],
    # ]

    # print("HIIIIIIIIIIII2.022.0")

    # freqsss_4 = [
    #     test_sim.get_wy_wz_at_point_withR2_fit(
    #         0, 0, 0, look_around=int(width * 0.05), polyfit=4
    #     )
    #     + [int(width * 0.05)],
    #     test_sim.get_wy_wz_at_point_withR2_fit(
    #         0, 0, 0, look_around=int(width * 0.1), polyfit=4
    #     )
    #     + [int(width * 0.1)],
    #     test_sim.get_wy_wz_at_point_withR2_fit(
    #         0, 0, 0, look_around=int(width * 0.16), polyfit=4
    #     )
    #     + [int(width * 0.16)],
    #     test_sim.get_wy_wz_at_point_withR2_fit(
    #         0, 0, 0, look_around=int(width * 0.165), polyfit=4
    #     )
    #     + [int(width * 0.165)],
    #     test_sim.get_wy_wz_at_point_withR2_fit(
    #         0, 0, 0, look_around=int(width * 0.17), polyfit=4
    #     )
    #     + [int(width * 0.17)],
    #     test_sim.get_wy_wz_at_point_withR2_fit(
    #         0, 0, 0, look_around=int(width * 0.18), polyfit=4
    #     )
    #     + [int(width * 0.18)],
    #     test_sim.get_wy_wz_at_point_withR2_fit(
    #         0, 0, 0, look_around=int(width * 0.2), polyfit=4
    #     )
    #     + [int(width * 0.2)],
    #     test_sim.get_wy_wz_at_point_withR2_fit(
    #         0, 0, 0, look_around=int(width * 0.3), polyfit=4
    #     )
    #     + [int(width * 0.3)],
    #     test_sim.get_wy_wz_at_point_withR2_fit(
    #         0, 0, 0, look_around=int(width * 0.4), polyfit=4
    #     )
    #     + [int(width * 0.4)],
    #     test_sim.get_wy_wz_at_point_withR2_fit(
    #         0, 0, 0, look_around=int(width * 0.5), polyfit=4
    #     )
    #     + [int(width * 0.5)],
    #     test_sim.get_wy_wz_at_point_withR2_fit(
    #         0, 0, 0, look_around=int(width * 0.6), polyfit=4
    #     )
    #     + [int(width * 0.6)],
    # ]

    # freq_yz_diff_2 = []
    # look_around_2 = []
    # for freq in freqsss_2:
    #     freq_yz_diff_2.append(freq[0] - freq[1])
    #     look_around_2.append(freq[2])

    # freq_yz_diff_4 = []
    # look_around_4 = []
    # for freq in freqsss_4:
    #     freq_yz_diff_4.append(freq[0] - freq[1])
    #     look_around_4.append(freq[2])

    # # Now we plot Wy - Wz vs look around for 2, and 4 (2 plots on graph)
    # plt.plot(look_around_2, freq_yz_diff_2, 'o', label="Polyfit 2")
    # plt.plot(look_around_4, freq_yz_diff_4, 'o', label="Polyfit 4")
    # plt.title("Frequency Difference (Hz) vs Look Around for various deg fits")
    # plt.xlabel("Look Around")
    # plt.ylabel("Frequency Difference (Hz)")
    # plt.legend()

    # plt.show()

    #####################################################

    # # Then for each look around we plot wy-wz for 2 and for 4 (1 plot)
    # for idx in range(len(freq_yz_diff_2)):

    # freq_x = []
    # freq_y = []
    # freq_z = []
    # look_around = []
    # for freq in freqsss:
    #     freq_x.append(freq[0])
    #     freq_y.append(freq[1])
    #     freq_z.append(freq[2])
    #     look_around.append(freq[3])

    # # plt.plot(look_around, freq_x, label="x")
    # plt.plot(look_around, freq_y, label="y")
    # plt.plot(look_around, freq_z, label="z")
    # plt.xlabel("Look Around")
    # plt.ylabel("Frequency (Hz)")
    # plt.legend()
    # plt.show()


# elec_vars = consts.Electrode_vars()
# elec_vars.set_vars("RF12", [377, 28000000 * 2 * math.pi, 0, 0])
# test_sim = Simulation("Simp58_3", elec_vars)
# test_sim.get_full_report("Test_new_58_201")
# print(test_sim.get_frequencys_at_point_xyz(0, 0, 0))


# elec_vars.set_vars("DC1", [-1, 0, 0, 0])
# elec_vars.set_vars("DC5", [-1, 0, 0, 0])
# elec_vars.set_vars("DC6", [-1, 0, 0, 0])
# elec_vars.set_vars("DC10", [-1, 0, 0, 0])

# elec_vars.set_vars("DC2", [-1, 0, 0, 0])
# elec_vars.set_vars("DC4", [-1, 0, 0, 0])
# elec_vars.set_vars("DC7", [-1, 0, 0, 0])
# elec_vars.set_vars("DC9", [-1, 0, 0, 0])

# elec_vars.set_vars("DC8", [-1, 0, 0, 0])
# elec_vars.set_vars("DC3", [-1, 0, 0, 0])


# test_sim = Simulation("Simp89", elec_vars)
# test_sim.get_full_report("Simp89(000)_377_28_DC_Grounded_NotCaught_RF_test")
# test_sim = Simulation("Simp875", elec_vars)
# test_sim.get_full_report("Simp875(000)_377_28_DC_Grounded_NotCaught_RF_test")
# # test_sim = Simulation("Simp85", elec_vars)
# test_sim.get_full_report("Simp85(000)_377_28_DC_Grounded_NotCaught_RF_test")
# test_sim = Simulation("Simp80", elec_vars)
# test_sim.get_full_report("Simp80(000)_377_28_DC_Grounded_NotCaught_RF_test")
# test_sim = Simulation("Simp75", elec_vars)
# test_sim.get_full_report("Simp75(000)_377_28_DC_Grounded_NotCaught_RF_test")
# fig1 = test_sim.plot_value_in_blank_direction(0,0,0,"y",'Emag')
# plt.show()

# test_sim = Simulation("Simp90", elec_vars)
# test_sim.get_full_report("Simp90(000)_377_28_DC_Grounded_NotCaught_RF_test")


# fig1 = test_sim.plot_potential_in_xyz_directions(0, 0, 0)
# fig2 = test_sim.plot_potential_in_xyz_directions(0, 0, 0, 0.0003, 0.00005, 0.00005)s

# fig3 = test_sim.plot_freq_in_xyz_directions(0, 0, 0)
# fig4 = test_sim.plot_freq_in_xyz_directions(0, 0, 0, 0.0003, 0.00005, 0.00005)
# plt.show()


# Other todo
# Get the data again from comsol (grounded vs not??, added vs not??) (Yes)
# Clean up code, make the "catch RF" more robust
# Make a ploting class?
# Find ways to make the code run faster!!!
# find out if the (+) and (-) x^4 coeffs on the quartic fit is an error... (Ughhhhhhhhh)

# Hessian bullshit
# freq = test_sim.get_frequencys_at_point_hess(0, .00005, 0.00002)
# print('For point 0, 0, 0')
# for i in range(3):
#     print("Frequency in direction " + str(freq[i][1]) + " is " + str(freq[i][0]) + " Hz")
