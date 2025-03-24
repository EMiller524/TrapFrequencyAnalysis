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
from scipy.interpolate import griddata
from scipy.interpolate import RBFInterpolator


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
        timesimstart = time.time()
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
        self.total_voltage_df = None
        self.get_total_voltage_at_all_points()

        timesimstop = time.time()
        print("simulation initialized in " + str(timesimstop - timesimstart) + " seconds")
        

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

        # print("Computing total voltage at all points...")
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

        # ✅ Step 3: Group by (x, y, z) and sum CalcV efficiently ##CurrentBottleNeck##
        self.total_voltage_df = merged_df.groupby(["x", "y", "z"], as_index=False)[
            "CalcV"
        ].sum()

        # print("Total voltage computation complete.")

    def find_V_min(self):
        """
        
        Finds and returns the point with the minimum total voltage.
        To catch errors, the minimum 100 points are found. If they are all close to each other, then the minimum is found.
        If there are outliers, they are thrown out, then the minimum is found.
        
        MAKE FASTER!!
        
        """
        timesimstart = time.time()
        
        if self.total_voltage_df is None:
            print("Total voltage data not available.")
            return None

        # Sort by CalcV to find the minimum values
        sorted_df = self.total_voltage_df.sort_values(by="CalcV").head(100)

        # Check proximity of the top 100 minimum points
        points = sorted_df[["x", "y", "z"]].values
        calcV_values = sorted_df["CalcV"].values

        # Calculate distances between points
        distances = np.linalg.norm(points[:, np.newaxis] - points, axis=2)

        # Identify outliers based on distance threshold
        threshold = np.percentile(distances, 95)
        close_points_mask = np.all(distances < threshold, axis=1)
        

        # Filter out outliers
        filtered_points = points[close_points_mask]
        filtered_calcV = calcV_values[close_points_mask]

        # Find the minimum point among the filtered points
        min_index = np.argmin(filtered_calcV)
        min_point = filtered_points[min_index]


        # Define the cutout grid range (4 points in each direction, total 9x9x9)
        step_size = 0.0001  # Adjust for desired resolution
        grid_x = np.linspace(min_point[0] - .000015, min_point[0] + .000015, 15)
        grid_y = np.linspace(min_point[1] - .000005, min_point[1] + .000005, 15)
        grid_z = np.linspace(min_point[2] - .000005, min_point[2] + .000005, 15)
        

        # Create a data frame that includes only points within this range
        df_cutout = self.total_voltage_df[
            (self.total_voltage_df["x"].between(grid_x.min(), grid_x.max())) &
            (self.total_voltage_df["y"].between(grid_y.min(), grid_y.max())) &
            (self.total_voltage_df["z"].between(grid_z.min(), grid_z.max()))
        ]

        # Extract cutout points and voltage values
        cutout_points = df_cutout[["x", "y", "z"]].values
        cutout_voltages = df_cutout["CalcV"].values

        if len(cutout_points) < 10:
            print("Warning: Not enough points in cutout. Returning dataset minimum.")
            return min_point


        # Generate fine grid for interpolation
        fine_grid_x, fine_grid_y, fine_grid_z = np.mgrid[
            grid_x.min():grid_x.max():30j,
            grid_y.min():grid_y.max():30j,
            grid_z.min():grid_z.max():30j
        ]


        # Interpolate using quadratic interpolation (griddata with cubic method) ## takes long time##
        fine_interp = griddata(
            cutout_points, cutout_voltages, 
            (fine_grid_x, fine_grid_y, fine_grid_z), 
            method='linear'
        )
        
        ## just changed ##
        
        # rbf = RBFInterpolator(cutout_points, cutout_voltages, kernel='thin_plate_spline')
        # fine_interp = rbf(np.vstack((fine_grid_x.ravel(), fine_grid_y.ravel(), fine_grid_z.ravel())).T)
        # fine_interp = fine_interp.reshape(fine_grid_x.shape)


        # Find the minimum value in the interpolated grid
        min_index = np.unravel_index(np.nanargmin(fine_interp), fine_interp.shape)
        refined_min_point = np.array([
            fine_grid_x[min_index], fine_grid_y[min_index], fine_grid_z[min_index]
        ])        

        # Round refined_min_point to the nearest (1e-9 meters)
        refined_min_point = np.round(refined_min_point, decimals=9)
        
        timestop = time.time()
        print("find_V_min took: " + str(timestop - timesimstart) + " seconds")

        

        return refined_min_point, min_point

    def get_principal_freq_at_min(self):
        min1, min2 = self.find_V_min()
        eigenfreq, axialfreq, eigendir = self.get_wy_wz_wx_at_point_withR3_fit(
            min1[0], min1[1], min1[2], look_around=50, polyfit=4
        )

        # Convert eigenvectors into a readable format
        eigendir_readable = {
            f"Direction {i+1}": f"({vec[0]:.3f}, {vec[1]:.3f}, {vec[2]:.3f})"
            for i, vec in enumerate(eigendir.T)  # Transpose so each column is an eigenvector
    }

        return eigenfreq, eigendir_readable

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


def recreate_old_data(rfamp, rffreq, twist, endcaps):

    x_pos = []
    Radial1 = []
    Radial2 = []
    Axial = []
    x_push = 0
    while True:
        test_sim_l = Simulation(
            "Simp58_101",
            consts.get_electrodvars_w_twist_and_push(rfamp, rffreq, twist=twist, endcaps=endcaps, pushx = x_push),
        )
        min_l = test_sim_l.find_V_min()
        print("min_l_x: ", min_l[0][0])

        test_sim_r = Simulation(
            "Simp58_101",
            consts.get_electrodvars_w_twist_and_push(rfamp, rffreq, twist=twist, endcaps=endcaps, pushx=-x_push),
        )
        min_r = test_sim_r.find_V_min()
        # print("min_r: ", min_r)

        print()

        if min_l is None or min_r is None:
            print("No minimum found, stopping.")
            break
        if min_l[0][0] > 0.0002:
            print("Minimum too far from origin, stopping.")
            break
        if x_push > 3:
            print("Push too large, stopping.")
            break
        else:
            x_pos.append(min_l[0][0])
            freqs_l = test_sim_l.get_principal_freq_at_min()
            Radial1.append(freqs_l[0][1])
            Radial2.append(freqs_l[0][2])
            Axial.append(freqs_l[0][0])

            x_pos.append(min_r[0][0])
            freqs_r = test_sim_r.get_principal_freq_at_min()
            Radial1.append(freqs_r[0][1])
            Radial2.append(freqs_r[0][2])
            Axial.append(freqs_r[0][0])

            x_push += 0.2

    # Plot x_pos vs Radial1 and Radial2 and Axial but have Radial 1 and 2 share an axis and Axial on the other
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.scatter(x_pos, Radial1, label="Radial1", color="red")
    ax1.scatter(x_pos, Radial2, label="Radial2", color="blue")
    ax2.scatter(x_pos, Axial, label="Axial", color="green")
    ax1.set_xlabel("x_pos")
    ax1.set_ylabel("Radial Frequency (Hz)")
    ax2.set_ylabel("Axial Frequency (Hz)")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.show()

# ✅ Make sure all simulation execution happens **inside** `if __name__ == "__main__"`
if __name__ == "__main__":
    tstart = time.time()
    multiprocessing.set_start_method("spawn", force=True)  # ✅ Prevent reinitialization

    elec_var0 = consts.get_electrodvars_w_twist(377, 28000000 * 2 * math.pi, 0, 0)
    elec_var1 = consts.get_electrodvars_w_twist(377, 25500000 * 2 * math.pi, -.275, 2.5)
    elec_var2 = consts.get_electrodvars_w_twist_and_push(377, 28000000 * 2 * math.pi, -.275, 2, 2)


    test_sim_2nd = Simulation(
        "Simp58_101",
        elec_var1,
    )
    
    print(test_sim_2nd.get_principal_freq_at_min())


    # min_normal = test_sim_normal.find_V_min()[0]
    # miny1 = test_sim_y_push_1.find_V_min()[0]
    # miny2 = test_sim_y_push_2.find_V_min()[0]
    # miny3 = test_sim_y_push_3.find_V_min()[0]
    # minz1 = test_sim_z_push_1.find_V_min()[0]
    # minz2 = test_sim_z_push_2.find_V_min()[0]
    # minz3 = test_sim_z_push_3.find_V_min()[0]

    # freqs_normal = test_sim_normal.get_principal_freq_at_min()[0]
    # freqs_y1 = test_sim_y_push_1.get_principal_freq_at_min()[0]
    # freqs_y2 = test_sim_y_push_2.get_principal_freq_at_min()[0]
    # freqs_y3 = test_sim_y_push_3.get_principal_freq_at_min()[0]
    # freqs_z1 = test_sim_z_push_1.get_principal_freq_at_min()[0]
    # freqs_z2 = test_sim_z_push_2.get_principal_freq_at_min()[0]
    # freqs_z3 = test_sim_z_push_3.get_principal_freq_at_min()[0]

    # # Print results
    # print("normal min: " + str(min_normal) + "freqs: " + str(freqs_normal))
    # print("y push 1 min: " + str(miny1) + "freqs: " + str(freqs_y1))
    # print("y push 2 min: " + str(miny2) + "freqs: " + str(freqs_y2))
    # print("y push 3 min: " + str(miny3) + "freqs: " + str(freqs_y3))
    # print("z push 1 min: " + str(minz1) + "freqs: " + str(freqs_z1))
    # print("z push 2 min: " + str(minz2) + "freqs: " + str(freqs_z2))
    # print("z push 3 min: " + str(minz3) + "freqs: " + str(freqs_z3))

    tstop = time.time()
    print("Time taken: " + str(tstop - tstart))

