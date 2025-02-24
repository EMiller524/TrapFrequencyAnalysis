from functools import reduce
import math
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import dataextractor
import os
import consts
import Electrode
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import v_interpolator
import sympy as sp  # Import sympy
import time


class Simulation:
    def __init__(self, dataset, variables=consts.Electrode_vars()):
        """
        Initialize the Electrode object.

        Parameters:
        - name (str): The name of the electrode.
        - voltage (float): The applied voltage.
        - data (pd.DataFrame, optional): A pandas DataFrame containing relevant electrode data.
        """
        print("initializing simulation")
        self.dataset = dataset
        self.file_path = "C:\\GitHub\\TrapFrequencyAnalysis\\Data\\" + dataset + "\\"

        # make a dictionary of electrodes with each name in consts.electrode_names as the key and an electrode class as the value
        self.electrode_vars = variables

        self.electrodes = {}
        for electrode in consts.electrode_names:
            self.electrodes[electrode] = Electrode.Electrode(
                electrode, self.dataset, self.electrode_vars.get_vars(electrode)
            )

        self.valid_points = self.get_valid_points()

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

    def get_total_voltage_at_all_points(self):
        # get the data for each electrode
        dfs = []
        for electrode in consts.electrode_names:
            if self.electrodes[electrode].get_dataframe() is None:
                print (f"no data for {electrode}")
            else:
                dfs.append(self.electrodes[electrode].get_dataframe())

        # Step 1: Find the intersection of all unique (x, y, z) combinations
        common_keys = reduce(lambda left, right: pd.merge(left, right, on=['x', 'y', 'z']), [df[['x', 'y', 'z']] for df in dfs])

        # Step 2: Concatenate all dataframes
        merged_df = pd.concat(dfs, ignore_index=True)

        # Step 3: Filter merged_df to only keep common (x, y, z) keys
        filtered_df = merged_df.merge(common_keys, on=['x', 'y', 'z'])

        # Step 4: Group by (x, y, z) and sum CalcV
        master_df = filtered_df.groupby(['x', 'y', 'z'], as_index=False)['CalcV'].sum()

        self.total_voltage_df = master_df

    def fit_v_at_point(self, x0, y0, z0, grid_size = 7):
        """
        Compute the Hessian matrix at (x0, y0, z0) using a cubic polynomial fit
        on a local 7x7x7 grid of points.

        Parameters:
        - x0, y0, z0 (float): The point where Hessian is computed.
        - grid_size (int): The number of points along each axis (default 7).

        Returns:
        - H (np.array): 3x3 Hessian matrix at (x0, y0, z0).
        """
        if self.total_voltage_df is None:
            print("Total voltage data is not available.")
            return None

        # Ensure the grid size is odd to center on (x0, y0, z0)
        if grid_size % 2 == 0:
            grid_size += 1

        # Find the closest grid points to (x0, y0, z0)
        df = self.total_voltage_df.copy()

        # Get unique grid points along each axis
        x_vals = np.sort(df["x"].unique())
        y_vals = np.sort(df["y"].unique())
        z_vals = np.sort(df["z"].unique())

        # Find the index of the closest grid points to (x0, y0, z0)
        x_idx = np.searchsorted(x_vals, x0)
        y_idx = np.searchsorted(y_vals, y0)
        z_idx = np.searchsorted(z_vals, z0)

        # Get indices for the surrounding grid
        half_grid = grid_size // 2
        x_range = x_vals[
            max(0, x_idx - half_grid) : min(len(x_vals), x_idx + half_grid + 1)
        ]
        y_range = y_vals[
            max(0, y_idx - half_grid) : min(len(y_vals), y_idx + half_grid + 1)
        ]
        z_range = z_vals[
            max(0, z_idx - half_grid) : min(len(z_vals), z_idx + half_grid + 1)
        ]

        # Filter the dataframe to include only points within this grid
        df_grid = df[
            (df["x"].isin(x_range)) & (df["y"].isin(y_range)) & (df["z"].isin(z_range))
        ]

        if len(df_grid) < grid_size**3:
            print(
                "Warning: Not enough points in the selected grid. Reducing grid size."
            )

        # Prepare input features
        X = df_grid[["x", "y", "z"]]
        y = df_grid["CalcV"]

        # Create cubic polynomial features
        poly = PolynomialFeatures(degree=3, include_bias=True)
        X_poly = poly.fit_transform(X)

        # Fit the model
        model = LinearRegression()
        model.fit(X_poly, y)
        coeff = model.coef_

        y_pred = model.predict(X_poly)
        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)

        print(f"R²: {r2:.4f}, MSE: {mse:.4f}")

        return model, poly

    def get_hessian_at_point(self, x, y, z):

        """
        Compute the Hessian of the polynomial regression model at a given point (x, y, z).
        
        Args:
            model: Trained LinearRegression model.
            poly: PolynomialFeatures instance used for feature transformation.
            point: Tuple (x, y, z) where the Hessian is computed.
        
        Returns:
            Hessian matrix (3x3 numpy array)
        """
        model, poly = self.fit_v_at_point(x, y, z)

        feature_names = poly.get_feature_names_out(["x", "y", "z"])
        coef_dict = dict(zip(feature_names, model.coef_))

        # Initialize Hessian as a 3x3 zero matrix
        H = np.zeros((3, 3))

        # Second-order partial derivatives
        H[0, 0] = 2 * coef_dict.get("x^2", 0) + 2 * coef_dict.get("x^2 y", 0) * y + 2 * coef_dict.get("x^2 z", 0) * z
        H[1, 1] = 2 * coef_dict.get("y^2", 0) + 2 * coef_dict.get("y^2 x", 0) * x + 2 * coef_dict.get("y^2 z", 0) * z
        H[2, 2] = 2 * coef_dict.get("z^2", 0) + 2 * coef_dict.get("z^2 x", 0) * x + 2 * coef_dict.get("z^2 y", 0) * y

        # Mixed partial derivatives
        H[0, 1] = H[1, 0] = coef_dict.get("x y", 0) + 2 * coef_dict.get("x y z", 0) * z
        H[0, 2] = H[2, 0] = coef_dict.get("x z", 0) + 2 * coef_dict.get("x y z", 0) * y
        H[1, 2] = H[2, 1] = coef_dict.get("y z", 0) + 2 * coef_dict.get("x y z", 0) * x

        print("X dir freq: " + str(math.sqrt((consts.ion_charge / consts.ion_mass) * abs(H[0,0])) * 2 * math.pi))
        print("Y dir freq: " + str(math.sqrt((consts.ion_charge / consts.ion_mass) * abs(H[1,1])) * 2 * math.pi))

        print("Z dir freq: " + str(math.sqrt((consts.ion_charge / consts.ion_mass) * abs(H[2,2])) * 2 * math.pi))

        print (H)
        return H

    def diagonalize_hessian(self, H):
        """
        Diagonalize the Hessian matrix and return the eigenvalues and eigenvectors.
        
        Args:
            H (np.array): 3x3 Hessian matrix.
        
        Returns:
            Tuple (eigenvalues, eigenvectors)
        """
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(H)

        # Sort eigenvalues in ascending order
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        return eigenvalues, eigenvectors

    def get_frequencys_at_point(self, x, y, z):
        # call get_hessian_at_point and then diagonalize the hessian
        # calculate the frequencys from the diagonalized hessian and the principlae directions of the hessian
        # return the frequencys and the principlae directions

        eigenvalues, eigenvectors = self.diagonalize_hessian(self.get_hessian_at_point(x, y, z))
        Q = consts.ion_charge
        M = consts.ion_mass

        frequencys_and_directions = []
        # Calculate frequencies
        for i in range(3):
            frequency = math.sqrt((Q/M) * abs(eigenvalues[i])) * 2 * math.pi
            direction = eigenvectors[i]
            frequencys_and_directions.append((frequency, direction))

        return frequencys_and_directions

    def get_electrode(self, name):
        return self.electrodes[name]


elec_vars = consts.Electrode_vars()
elec_vars.set_vars("RF12", [377, 28000000 * 2 * math.pi, 0, 0])
elec_vars.set_vars("DC2", [0, 0, 0, 0])
elec_vars.set_vars("DC4", [0, 0, 0, 0])

test_sim = Simulation("Simplified1", elec_vars)

# calcualte the time this takes
start = time.time()
print(test_sim.get_frequencys_at_point(0,0,0))
end = time.time()
print("time took: ", end - start)
# print(test_sim.get_total_voltage_at_point(0, 0, 0))
# deriv = test_sim.get_voltage_second_derivative_at_point(0, 0, 0, plot_fit=True)
# print("deriv", deriv)
# freqa = test_sim.calcualte_frequencys(0, 0, 0)
# print(freqa)

# test_sim.plot_potential_in_principal_directions(0, 0, 0)


# print(test_sim.calcualte_frequencys(0, 0, 0))
# print(test_sim.calcualte_frequencys(0.1, 0, 0))
# print(test_sim.calcualte_frequencys(0.2, 0, 0))
# print(test_sim.calcualte_frequencys(-0.2, 0, 0))
# print(test_sim.calcualte_frequencys(-0.1, 0, 0))
