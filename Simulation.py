from functools import reduce
import math
import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import dataextractor
import constants
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
import numexpr as ne
import electrode_vars as evars


multiprocessing.set_start_method("spawn", force=True)


# def init_electrode(electrode_name, dataset, electrode_vars):
#     return electrode_name, Electrode.Electrode(
#         electrode_name, dataset, electrode_vars.get_vars(electrode_name)
#     )


class Simulation(sim_ploting, sim_hessian, sim_normalfitting):
    def __init__(self, dataset, variables=evars.Electrode_vars()):
        """
        Initialize the Electrode object.
        """
        timesimstart = time.time()
        print("initializing simulation")
        self.dataset = dataset
        self.file_path = "C:\\GitHub\\TrapFrequencyAnalysis\\Data\\" + dataset + "\\"

        self.electrode_vars = variables

        self.total_voltage_df = None

        if os.path.exists(self.file_path + "combined_dataframe.csv"):
            self.total_voltage_df = pd.read_pickle(
                self.file_path + "combined_dataframe.csv"
            )
        else:
            print("making combined dataframe")
            print(self.file_path + "combined_dataframe.csv")
            self.total_voltage_df = dataextractor.make_simulation_dataframe(
                self.file_path
            )

        self.update_total_voltage()

        timesimstop = time.time()
        print(
            "simulation initialized in " + str(timesimstop - timesimstart) + " seconds"
        )

    def get_variables(self, electrode):
        return self.electrode_vars.get_vars(electrode)

    def update_total_voltage(self):
        """Update the total voltage DataFrame by forming
        an equation based on the electorde_varaibles and
        applying to the column "TotalV" in the dataframe."""

        # self.data["CalcV"] = ne.evaluate(
        #     "(amp**2 * Q * (Ex**2 + Ey**2 + Ez**2)) / (4 * M * freq**2)",
        #     local_dict={
        #     "amp": amp,
        #     "Q": Q,
        #     "Ex": self.data["Ex"],
        #     "Ey": self.data["Ey"],
        #     "Ez": self.data["Ez"],
        #     "M": M,
        #     "freq": freq,
        #     },
        #     )
        evaluation_final = ""
        eval_ex_str = ""
        eval_ey_str = ""
        eval_ez_str = ""
        for electro in [
            "RF1",
            "RF2",
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
        ]:
            evaluation_final += (
                electro + "_V * " + str(self.electrode_vars.get_offset(electro)) + " + "
            )
            eval_ex_str += (
                electro
                + "_Ex * "
                + str(self.electrode_vars.get_amplitude(electro))
                + " + "
            )
            eval_ey_str += (
                electro
                + "_Ey * "
                + str(self.electrode_vars.get_amplitude(electro))
                + " + "
            )
            eval_ez_str += (
                electro
                + "_Ez * "
                + str(self.electrode_vars.get_amplitude(electro))
                + " + "
            )
        eval_ex_str = eval_ex_str[:-3]  # remove last " + "
        eval_ey_str = eval_ey_str[:-3]  # remove last " + "
        eval_ez_str = eval_ez_str[:-3]  # remove last " + "

        evaluation_pseudo_part = (
            "("
            + str(constants.ion_charge)
            + " * (("
            + eval_ex_str
            + ")**2 + ("
            + eval_ey_str
            + ")**2 + ("
            + eval_ez_str
            + ")**2) / (4 * "
            + str(constants.ion_mass)
            + " * "
            + str(self.electrode_vars.get_frequency("RF1") ** 2)
            + "))"
        )

        evaluation_final += evaluation_pseudo_part

        # print("evaluation_final: ", evaluation_final)

        self.total_voltage_df["TotalV"] = ne.evaluate(
            evaluation_final,
            local_dict={
                "RF1_V": self.total_voltage_df["RF1_V"],
                "RF2_V": self.total_voltage_df["RF2_V"],
                "DC1_V": self.total_voltage_df["DC1_V"],
                "DC2_V": self.total_voltage_df["DC2_V"],
                "DC3_V": self.total_voltage_df["DC3_V"],
                "DC4_V": self.total_voltage_df["DC4_V"],
                "DC5_V": self.total_voltage_df["DC5_V"],
                "DC6_V": self.total_voltage_df["DC6_V"],
                "DC7_V": self.total_voltage_df["DC7_V"],
                "DC8_V": self.total_voltage_df["DC8_V"],
                "DC9_V": self.total_voltage_df["DC9_V"],
                "DC10_V": self.total_voltage_df["DC10_V"],
                "RF1_Ex": self.total_voltage_df["RF1_Ex"],
                "RF2_Ex": self.total_voltage_df["RF2_Ex"],
                "DC1_Ex": self.total_voltage_df["DC1_Ex"],
                "DC2_Ex": self.total_voltage_df["DC2_Ex"],
                "DC3_Ex": self.total_voltage_df["DC3_Ex"],
                "DC4_Ex": self.total_voltage_df["DC4_Ex"],
                "DC5_Ex": self.total_voltage_df["DC5_Ex"],
                "DC6_Ex": self.total_voltage_df["DC6_Ex"],
                "DC7_Ex": self.total_voltage_df["DC7_Ex"],
                "DC8_Ex": self.total_voltage_df["DC8_Ex"],
                "DC9_Ex": self.total_voltage_df["DC9_Ex"],
                "DC10_Ex": self.total_voltage_df["DC10_Ex"],
                "RF1_Ey": self.total_voltage_df["RF1_Ey"],
                "RF2_Ey": self.total_voltage_df["RF2_Ey"],
                "DC1_Ey": self.total_voltage_df["DC1_Ey"],
                "DC2_Ey": self.total_voltage_df["DC2_Ey"],
                "DC3_Ey": self.total_voltage_df["DC3_Ey"],
                "DC4_Ey": self.total_voltage_df["DC4_Ey"],
                "DC5_Ey": self.total_voltage_df["DC5_Ey"],
                "DC6_Ey": self.total_voltage_df["DC6_Ey"],
                "DC7_Ey": self.total_voltage_df["DC7_Ey"],
                "DC8_Ey": self.total_voltage_df["DC8_Ey"],
                "DC9_Ey": self.total_voltage_df["DC9_Ey"],
                "DC10_Ey": self.total_voltage_df["DC10_Ey"],
                "RF1_Ez": self.total_voltage_df["RF1_Ez"],
                "RF2_Ez": self.total_voltage_df["RF2_Ez"],
                "DC1_Ez": self.total_voltage_df["DC1_Ez"],
                "DC2_Ez": self.total_voltage_df["DC2_Ez"],
                "DC3_Ez": self.total_voltage_df["DC3_Ez"],
                "DC4_Ez": self.total_voltage_df["DC4_Ez"],
                "DC5_Ez": self.total_voltage_df["DC5_Ez"],
                "DC6_Ez": self.total_voltage_df["DC6_Ez"],
                "DC7_Ez": self.total_voltage_df["DC7_Ez"],
                "DC8_Ez": self.total_voltage_df["DC8_Ez"],
                "DC9_Ez": self.total_voltage_df["DC9_Ez"],
                "DC10_Ez": self.total_voltage_df["DC10_Ez"],
            },
        )

        return

    def change_electrode_variables(self, new_vars: evars.Electrode_vars):
        self.electrode_vars = new_vars
        self.update_total_voltage()

    def get_valid_points(self):
        non_empty_dfs = []
        for electrode in self.electrodes:
            if self.electrodes[electrode].get_dataframe() is not None:
                non_empty_dfs.append(self.electrodes[electrode].get_dataframe())

        point_sets = [set(zip(df["x"], df["y"], df["z"])) for df in non_empty_dfs]

        common_points = set.intersection(*point_sets)

        return np.array(sorted(common_points))

    def find_V_min(self, step_size=0.000003):
        """

        Finds and returns the point with the minimum total voltage.
        To catch errors, the minimum 100 points are found. If they are all close to each other, then the minimum is found.
        If there are outliers, they are thrown out, then the minimum is found.

        MAKE FASTER!!

        """

        if self.total_voltage_df is None:
            print("Total voltage data not available.")
            return None
        time1 = time.time()

        # Sort by CalcV to find the minimum values

        def find_nsmallest_df(df, colname, n=100):
            # 1) Extract column values
            arr = df[colname].to_numpy()

            # 2) Grab the indices of the n smallest values (unordered)
            idx = np.argpartition(arr, n)[:n]

            # 3) Sort those n rows by their actual values so the final result is ascending
            idx_sorted = idx[np.argsort(arr[idx])]

            # 4) Index back into the DataFrame
            return df.iloc[idx_sorted]

        # Usage:
        sorted_df = find_nsmallest_df(self.total_voltage_df, "TotalV", n=100)

        # Check proximity of the top 1000 minimum points
        points = sorted_df[["x", "y", "z"]].values
        calcV_values = sorted_df["TotalV"].values

        # Calculate distances between points
        distances = np.linalg.norm(points[:, np.newaxis] - points, axis=2)

        # Calculate average distance from each point to all other points
        average_distances = np.mean(distances, axis=1)

        # Identify outliers based on average distance threshold
        threshold = np.percentile(average_distances, 80)
        outliers_mask = average_distances > threshold

        # Filter out outliers
        filtered_points = points[~outliers_mask]
        filtered_calcV = calcV_values[~outliers_mask]

        if len(filtered_points) < 70:
            print("Many min points removed")
            print("Total points removed: " + str(len(points) - len(filtered_points)))

        # Find the minimum point among the filtered points
        min_index = np.argmin(filtered_calcV)
        min_point = filtered_points[min_index]

        # Now i want to get a R3 fit around "min_point" and find the minimum of that fit

        ## Get the surrounding points for R3 fit using stepsize as the cutoff
        cutout_of_df = self.total_voltage_df[
            (self.total_voltage_df["x"].between(min_point[0] - (5 * step_size), min_point[0] + (5 * step_size)))
            & (self.total_voltage_df["y"].between(min_point[1] - step_size, min_point[1] + step_size))
            & (self.total_voltage_df["z"].between(min_point[2] - step_size, min_point[2] + step_size))
        ]


        voltage_vals = cutout_of_df["TotalV"].values
        xyz_vals_uncentered = cutout_of_df[["x", "y", "z"]].values

        # Make the Point of interest the origin (0,0,0) and move the other points accordingly
        xyz_vals_centered = xyz_vals_uncentered - min_point

        poly = PolynomialFeatures(degree=2, include_bias=True)
        X_poly = poly.fit_transform(xyz_vals_centered)

        # Fit the model
        model = LinearRegression()
        model.fit(X_poly, voltage_vals)

        # (4) Extract coefficients
        # model.coef_ is length-10 if include_bias=True for 3D data; also consider model.intercept_
        c0 = model.intercept_
        c1, c2, c3, c4, c5, c6, c7, c8, c9 = model.coef_[1:]  # skipping the bias column's coef

        # (5) Solve gradient=0 for (x, y, z) in the centered frame
        H = np.array([
            [2*c4,   c5,   c6],
            [c5,    2*c7,  c8],
            [c6,     c8,  2*c9]
        ])
        linear_terms = np.array([c1, c2, c3])

        delta_xyz_centered = np.linalg.solve(H, -linear_terms)

        # (6) Shift back to original coordinates
        best_fit_minimum = min_point + delta_xyz_centered

        time5 = time.time()
        # print("Total time taken: ", time5 - time1)

        # print("best_fit_minimum: ", best_fit_minimum)
        # print("min_point: ", min_point)

        return best_fit_minimum, min_point

    def get_principal_freq_at_min(self, getmintoo = False, fitdeg = 4, look_around = 5):
        min1, min2 = self.find_V_min()
        eigenfreq, axialfreq, eigendir = self.get_wy_wz_wx_at_point_withR3_fit(
            min1[0], min1[1], min1[2], look_around=look_around, polyfit=fitdeg
        )

        # Convert eigenvectors into a readable format
        eigendir_readable = {
            f"Direction {i+1}": f"({vec[0]:.3f}, {vec[1]:.3f}, {vec[2]:.3f})"
            for i, vec in enumerate(
                eigendir.T
            )  # Transpose so each column is an eigenvector
        }
        if getmintoo:
            return eigenfreq, eigendir_readable, min1, min2

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



def recreate_old_data(rfamp, rffreq, twist, endcaps, push_stop = 1, step_size = 0.1):

    x_pos = []
    Radial1 = []
    Radial2 = []
    Axial = []
    x_push = 0
    test_sim = Simulation("Simp58_101")
    while True:
        if x_push >= push_stop:
            print("Push too large, stopping.")
            break
        test_sim.change_electrode_variables(evars.get_electrodvars_w_twist_and_push(
            rfamp, rffreq, twist, endcaps, pushx=x_push
        ))
        
        freqs, eigendir, minreal, min_snap = test_sim.get_principal_freq_at_min(getmintoo=True)

        x_pos.append(minreal[0])
        Radial1.append(freqs[1])
        Radial2.append(freqs[2])
        Axial.append(freqs[0])
        
        test_sim.change_electrode_variables(evars.get_electrodvars_w_twist_and_push(
            rfamp, rffreq, twist, endcaps, pushx=-x_push
        ))

        freqs, eigendir, minreal, min_snap = test_sim.get_principal_freq_at_min(getmintoo=True)

        x_pos.append(minreal[0])
        Radial1.append(freqs[1])
        Radial2.append(freqs[2])
        Axial.append(freqs[0])

        x_push += step_size
        print(x_push)

    # Plot x_pos vs Radial1 and Radial2 and Axial but have Radial 1 and 2 share an axis and Axial on the other
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.scatter(x_pos, Radial1, label="Radial", color="red")
    ax1.scatter(x_pos, Radial2, label="Radial", color="red")
    ax2.scatter(x_pos, Axial, label="Axial", color="green")
    ax1.set_xlabel("x_pos")
    ax1.set_ylabel("Radial Frequency (Hz)")
    ax2.set_ylabel("Axial Frequency (Hz)")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax1.set_ylim(ax1.get_ylim()[0] * .995, ax1.get_ylim()[1] * 1.005)
    ax2.set_ylim(ax2.get_ylim()[0] * .98, ax2.get_ylim()[1] * 1.02)
    return fig

# ✅ Make sure all simulation execution happens **inside** `if __name__ == "__main__"`
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)  # ✅ Prevent reinitialization

    # elec_var0 = consts.get_electrodvars_w_twist(377, 25500000 * 2 * math.pi, 0, 0)
    # elec_var1 = consts.get_electrodvars_w_twist(
    #     377, 25500000 * 2 * math.pi, -0.275, 2.5
    # )
    # elec_var2 = consts.get_electrodvars_w_twist_and_push(
    #     377, 28000000 * 2 * math.pi, -0.275, 2, pushx = 1
    # )

    tstart = time.time()
    
    testsim1 = Simulation("Simp58_101", evars.get_electrodvars_w_twist(377, 25500000 * 2 * math.pi, 0, 0))
    for i in range(100):
        if i%2 == 0:
            testsim1.change_electrode_variables(evars.get_electrodvars_w_twist(377, 25500000 * 2 * math.pi, -0.275, 2.5))
        else:
            testsim1.change_electrode_variables(evars.get_electrodvars_w_twist(277, 28000000 * 2 * math.pi, -0.275, 2))
        testsim1.get_principal_freq_at_min(look_around=10)
    
    # fig = recreate_old_data(377, 25500000 * 2 * math.pi, -0.275, 2, push_stop=1.5, step_size=0.3)
        
    tstop = time.time()

    plt.show()
    # test_sim1.change_electrode_variables(elec_var_new)
    # test_sim1.recreate_old_data(377, 28000000 * 2 * math.pi, -0.275, 2)
    
    # for push in [.1,.2,.3,.4,.5,.6,.7,.8,.9, 1, 2, 3]:
    #     print("Push: " + str(push))
        
    #     test_sim1.change_electrode_variables(
    #         consts.get_electrodvars_w_twist_and_push(
    #             377, 28000000 * 2 * math.pi, -0.275, 2, pushx=push))
        
    #     mins = test_sim1.find_V_min()
    #     print(mins[0][0], mins[1][0])



    # for fit in [2,4]:
    #     for lookaround in [3, 5, 10, 15, 20, 30, 40, 50, 60]:
    #         freqs = test_sim1.get_wy_wz_wx_at_point_withR3_fit(0,0,0, look_around=lookaround, polyfit=fit)[0]
    #         print("Fit, lookaround: " + str(fit) + ", " + str(lookaround) + "--> Val: " + str((abs(freqs[2] - freqs[1]))))


    # [5,10,15,20,30,40,50,60,90]

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

    print("Time taken: " + str(tstop - tstart))
