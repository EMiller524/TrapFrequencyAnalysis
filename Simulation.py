'''
This file will contain the main base of the class simulation
'''

import math
import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import dataextractor
import constants
import time
import matplotlib.cm as cm
from SimulationMix.simulation_ploting import sim_ploting
from SimulationMix.simulation_fitting import sim_normalfitting
from multiprocessing import Pool, cpu_count
from joblib import Parallel, delayed
from scipy.interpolate import griddata
import numexpr as ne
import electrode_vars as evars


class Simulation(sim_ploting, sim_normalfitting):
    def __init__(self, dataset, variables=evars.Electrode_vars()):
        """
        Initialize the simulation object by making sure the data is all extracted and finding total voltages if given electodes
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

    def get_elec_variables(self, electrode):
        return self.electrode_vars.get_vars(electrode)

    def update_total_voltage(self):
        """Update the total voltage DataFrame by forming
        an equation based on the electorde_varaibles and
        applying to the column "TotalV" in the dataframe."""

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
        '''
        Change the electrode variables of the simulation to new_vars and update the total voltage DataFrame.
        '''
        self.electrode_vars = new_vars
        self.update_total_voltage()

    def find_V_min(self, step_size=5):
        """
        Finds and returns the point with the minimum total voltage.
        To catch errors, the minimum 100 points are found. If they are all close to each other, then the minimum is found.
        If there are outliers, they are thrown out, then the minimum is found.
        step_size is used to determine the cutout size for the R3 fit around the minimum point.
        
        args:
            step_size (float): The step size(in microns) to use for the cutout around the minimum point. Default is 5 microns.
            Note if step size is too small an error will be thrown (must be over 5)
        """

        if step_size <= 4.9:
            raise ValueError("Step size must be greater than 0.")
        
        step_size = step_size * 1e-6  # Convert microns to meters for calculations

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

        # print("Total time taken to find min: ", time5 - time1)

        # print("best_fit_minimum: ", best_fit_minimum)
        # print("min_point: ", min_point)
        return best_fit_minimum, min_point

    def get_principal_freq_at_min(self, getmintoo = False, fitdeg = 4, look_around = 5):
        '''
        This fuction finds the principal frequency at the minimum point of the total voltage.
        To do so the minimum in the potential well is found and then at this point a R3 fit is performed to find the eigenfrequencies and eigenvectors.
        
        args:
            getmintoo (bool): If True, return the minimum points as well. Default is False.
            fitdeg (int): The degree of the polynomial to fit to the data. Default is 4.
            look_around (float): The size of the cutout around the minimum point to use for the R3 fit. Default is 5 microns.
        
        returns:
            eigenfreq (list): A list of the eigenfrequencies in increasing order.
            eigendir (dict): A dictionary of the eigenvectors in a readable format.
            min1 (array): The best fit minimum point (x,y,z) in meters.
            min2 (array): The original minimum point (x,y,z) in meters.
        '''
        min1, min2 = self.find_V_min()
        eigenfreq, axialfreq, eigendir = self.get_freqs_at_point_withR3_fit(
            min1[0], min1[1], min1[2], look_around=look_around, polyfit=fitdeg
        )

        # Convert eigenfreq to a numpy array for sorting
        eigenfreq = np.array(eigenfreq)

        # Sort eigenfreq in increasing order and eigendir accordingly
        sorted_indices = np.argsort(eigenfreq)
        eigenfreq = eigenfreq[sorted_indices].tolist()  # Convert back to list if needed
        eigendir = eigendir[:, sorted_indices]  # Sort eigendir columns based on sorted indices

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
