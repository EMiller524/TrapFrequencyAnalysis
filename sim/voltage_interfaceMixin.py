# sim/voltage_interface.py
from __future__ import annotations
import time
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from trapping_variables import drive_colname  # single source of truth for column names


class VoltageInterfaceMixin:
    """
    Methods that *read* voltage data from the grid/DF to compute values/gradients/minima.
    Expects `self` to have:
      - self.total_voltage_df : pandas.DataFrame with columns x,y,z and per-drive TotalV columns
      - self.trapVariables    : Trapping_Vars (for drive_colname(dc_key) aka 'Static_TotalV')
    """

    def find_V_min(self, step_size=5):
        """
        Finds and returns the point with the minimum total voltage.

        To catch errors, the minimum 100 points are found. If they are all close to each other, then the minimum is found.
        If there are outliers, they are thrown out, then the minimum is found.
        then the points around this min are fit to a quadratic to find the best fit min

        step_size is used to determine the cutout size for the R3 fit around the minimum point.

        args:
            step_size (float): The step size(in microns) to use for the cutout around the minimum point. Default is 5 microns.
            Note if step size is too small an error will be thrown (must be over 5)

        returns:
            the best fit minimum point (x,y,z) in meters and the dataframe minimum
        """

        if step_size <= 4.9:
            raise ValueError("Step size must be greater than 5.")

        step_size = step_size * 1e-6  # Convert microns to meters for calculations

        if self.total_voltage_df is None:
            print("Total voltage data not available.")
            return None
        time1 = time.time()

        # Sort by TotalV to find the minimum values

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
            (
                self.total_voltage_df["x"].between(
                    min_point[0] - (5 * step_size), min_point[0] + (5 * step_size)
                )
            )
            & (
                self.total_voltage_df["y"].between(
                    min_point[1] - step_size, min_point[1] + step_size
                )
            )
            & (
                self.total_voltage_df["z"].between(
                    min_point[2] - step_size, min_point[2] + step_size
                )
            )
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
        c1, c2, c3, c4, c5, c6, c7, c8, c9 = model.coef_[
            1:
        ]  # skipping the bias column's coef

        # (5) Solve gradient=0 for (x, y, z) in the centered frame
        H = np.array([[2 * c4, c5, c6], [c5, 2 * c7, c8], [c6, c8, 2 * c9]])
        linear_terms = np.array([c1, c2, c3])

        delta_xyz_centered = np.linalg.solve(H, -linear_terms)

        # (6) Shift back to original coordinates
        best_fit_minimum = min_point + delta_xyz_centered

        time5 = time.time()

        # print("Total time taken to find min: ", time5 - time1)

        # print("best_fit_minimum: ", best_fit_minimum)
        # print("min_point: ", min_point)
        return best_fit_minimum, min_point

    def find_V_trap_at_point_fast_and_dirty(self, x, y, z, starting_step=0.49):
        """
        Lol dont use this

        Finds and returns the potential of the trap at a given point (x,y,z).
        The function takes in the coordinates of the desired point in space and returns the potential (V).

        We do this fast by using the df to find the 8 closest points and avergaing them.
        """
        # starting_step = starting_step * 1e-6  # Convert microns to meters for calculations

        ## Get the surrounding points for R3 fit using stepsize as the cutoff
        cutout_of_df = pd.DataFrame()

        # while len(cutout_of_df) < 10:
        #     cutout_of_df = self.total_voltage_df[
        #     (self.total_voltage_df["x"].between(x - (20 * starting_step), x + (20 * starting_step)))
        #     & (self.total_voltage_df["y"].between(y - starting_step, y + starting_step))
        #     & (self.total_voltage_df["z"].between(z - starting_step, z + starting_step))
        #     ]

        #     starting_step += (.01 * 1e-6)
        # print(starting_step, "************************************************************************************")
        starting_step = (
            starting_step * 1e-6
        )  # Convert microns to meters for calculations
        cutout_of_df = self.total_voltage_df[
            (
                self.total_voltage_df["x"].between(
                    x - (20 * starting_step), x + (20 * starting_step)
                )
            )
            & (self.total_voltage_df["y"].between(y - starting_step, y + starting_step))
            & (self.total_voltage_df["z"].between(z - starting_step, z + starting_step))
        ]

        voltage_vals = cutout_of_df["TotalV"].values
        print(len(voltage_vals))
        avg_V = np.mean(voltage_vals)

        return avg_V

    def find_V_trap_at_point(self, x, y, z, starting_step=0.4, derivs=False):
        """
        Finds and returns the potential of the trap at a given point (x,y,z).
        The function takes in the coordinates of the desired point in space and returns the potential (V).

        To do this we find the closest 50 values in the dataframe to the point (x,y,z) and then use these points to find the potential at the point.
        """

        starting_step = (
            starting_step * 1e-6
        )  # Convert microns to meters for calculations

        ## Get the surrounding points for R3 fit using stepsize as the cutoff
        cutout_of_df = pd.DataFrame()

        len_df = len(self.total_voltage_df)

        while len(cutout_of_df) < 20:
            cutout_of_df = self.total_voltage_df[
                (
                    self.total_voltage_df["x"].between(
                        x - (10 * starting_step), x + (10 * starting_step)
                    )
                )
                & (
                    self.total_voltage_df["y"].between(
                        y - starting_step, y + starting_step
                    )
                )
                & (
                    self.total_voltage_df["z"].between(
                        z - starting_step, z + starting_step
                    )
                )
            ]

            starting_step += 5 * 1e-6

        voltage_vals = cutout_of_df["TotalV"].values
        # print(len(cutout_of_df), " points found in cutout")
        # print("voltage_vals: ", voltage_vals)
        xyz_vals_uncentered = cutout_of_df[["x", "y", "z"]].values

        # Make the Point of interest the origin (0,0,0) and move the other points accordingly
        xyz_vals_centered = xyz_vals_uncentered - [x, y, z]
        t1 = time.time()

        poly = PolynomialFeatures(degree=3, include_bias=True)
        X_poly = poly.fit_transform(xyz_vals_centered)

        # Fit the model
        model = LinearRegression()
        model.fit(X_poly, voltage_vals)
        t2 = time.time()

        # find and print out the r2 of the fit
        r2 = model.score(X_poly, voltage_vals)
        # print("R-squared of the fit:", r2)

        # Get derivatives (d/dx, d/dy, d/dz) at point
        derivatives = model.coef_[1:4]

        # find the value of the fit at the origin
        Vvalue_at_point = model.predict(poly.transform([[0, 0, 0]]))
        # print("Time taken to find V at point: ", t2 - t1)

        if derivs:
            # Return the derivatives at the point
            return Vvalue_at_point[0], derivatives
        return Vvalue_at_point[0]
        # Calculate the potential energy of the ions
