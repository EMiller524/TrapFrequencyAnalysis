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
import matplotlib.cm as cm
import matplotlib.colors as colors
import concurrent.futures


def init_electrode(electrode_name, dataset, electrode_vars):
    return electrode_name, Electrode.Electrode(
        electrode_name, dataset, electrode_vars.get_vars(electrode_name)
    )

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

    def fit_v_at_point(self, x0, y0, z0, grid_size = 9, show = True):
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

        if len(df_grid) < (grid_size**2):
            print(
                "Warning: Not enough points in the selected grid. Reducing grid size."
            )

        # Prepare input features
        X = df_grid[["x", "y", "z"]]
        y = df_grid["CalcV"]

        # Create cubic polynomial features
        poly = PolynomialFeatures(degree=3, include_bias=False)
        X_poly = poly.fit_transform(X)

        # Fit the model
        model = LinearRegression()
        model.fit(X_poly, y)
        coeff = model.coef_

        y_pred = model.predict(X_poly)
        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)

        if r2 < .99:
            print(f"Warning: Low R² value ({r2:.4f}) for the polynomial fit.")
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

        # print("X dir freq: " + str(math.sqrt((consts.ion_charge / consts.ion_mass) * abs(H[0,0])) * 2 * math.pi))
        # print("Y dir freq: " + str(math.sqrt((consts.ion_charge / consts.ion_mass) * abs(H[1,1])) * 2 * math.pi))

        # print("Z dir freq: " + str(math.sqrt((consts.ion_charge / consts.ion_mass) * abs(H[2,2])) * 2 * math.pi))

        print ("Hessian at: " + str((x,y,z)) + str(H))
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

    def get_frequencys_at_point_hess(self, x, y, z):
        # call get_hessian_at_point and then diagonalize the hessian
        # calculate the frequencys from the diagonalized hessian and the principlae directions of the hessian
        # return the frequencys and the principlae directions

        eigenvalues, eigenvectors = self.diagonalize_hessian(self.get_hessian_at_point(x, y, z))
        Q = consts.ion_charge
        M = consts.ion_mass

        frequencys_and_directions = []
        # Calculate frequencies
        for i in range(3):
            frequency = math.sqrt((Q/M) * abs(eigenvalues[i])) / (math.pi * 2)
            direction = eigenvectors[i]
            frequencys_and_directions.append((frequency, direction))

        return frequencys_and_directions

    def get_frequencys_at_point_xyz(self, x, y, z, look_around=10):
        """
        Compute the frequencies in the x, y, and z directions by fitting a cubic polynomial
        to voltage values along each axis separately and extracting the second derivative.

        Args:
            x, y, z (float): The point of interest.
            look_around (int): Number of points to consider in each direction (default 5).

        Returns:
            list: [freq_x, freq_y, freq_z]
        """
        if self.total_voltage_df is None:
            print("Total voltage data is not available.")
            return None

        def fit_and_get_second_derivative(axis_values, voltage_values, target_value):
            """Fit a cubic polynomial and return the second derivative at the target value."""
            if len(axis_values) < 4:
                print("Not enough points for a cubic fit along one axis.")
                return None
            # print("axis values: " + str(axis_values))
            # print("voltage values: " + str(voltage_values))
            # print("target value: " + str(target_value))

            # Fit a cubic polynomial
            coeffs = np.polyfit(axis_values, voltage_values, 4)
            # print a readable version of the fit
            # if the target_value = 0
            # if target_value == 0:
            #     print("Fit: "  + str(coeffs[0]) + "x^3 + " + str(coeffs[1]) + "x^2 + " + str(coeffs[2]) + "x + " + str(coeffs[3]))

            poly_derivative = np.polyder(coeffs, 2) 
            second_derivative_at_target = np.polyval(poly_derivative, target_value)

            # plot the fit of the polynomial
            should_plot = False
            if should_plot:
                x = np.linspace(min(axis_values), max(axis_values), 100)
                y = np.polyval(coeffs, x)
                plt.plot(x, y, 'r-')
                plt.scatter(axis_values, voltage_values, color='b')
                plt.xlabel("Axis Value")
                plt.ylabel("Voltage Value")
                plt.title("Cubic Fit along Axis")
                plt.show()

            return second_derivative_at_target

        df = self.total_voltage_df.copy()
        Q = consts.ion_charge
        M = consts.ion_mass
        frequencies = []

        for axis in ["x", "y", "z"]:
            # Get axis values and voltage values around the point of interest while holding other coordinates constant
            if axis == "x":
                filtered_df = df[(df["y"] == y) & (df["z"] == z)]
            elif axis == "y":
                filtered_df = df[(df["x"] == x) & (df["z"] == z)]
            else:  # axis == 'z'
                filtered_df = df[(df["x"] == x) & (df["y"] == y)]

            sorted_df = filtered_df.sort_values(by=[axis])
            axis_vals = sorted_df[axis].values
            voltage_vals = sorted_df["CalcV"].values

            closest_idx = np.searchsorted(axis_vals, eval(axis))
            start_idx = max(0, closest_idx - look_around)
            end_idx = min(len(axis_vals), closest_idx + look_around + 1)
            selected_axis_vals = axis_vals[start_idx:end_idx]
            selected_voltage_vals = voltage_vals[start_idx:end_idx]

            # print("")
            # print("Axis is in the " + axis + " direction")
            second_derivative = fit_and_get_second_derivative(
                selected_axis_vals, selected_voltage_vals, eval(axis)
            )

            if second_derivative is None:
                frequencies.append(None)
                print(f"Could not calculate frequency along {axis} axis.")
            else:
                # Frequency Equation
                # print("Second derivative in " + axis + " direction: " + str(second_derivative) + " V/m^2")
                # freq = second_derivative
                freq = math.sqrt((Q / M) * abs(second_derivative)) / (2 * math.pi) ###############################################################################################################
                frequencies.append(freq)

        return frequencies

    def plot_potential_in_xyz_directions(self, x, y, z, x_cutoff = 1, y_cutoff = 1, z_cutoff = 1):
        # plots all the potential with x varying and y,z as inputed. And the same for y and z
        # this plot will be shown all together side by side in 3 different plots, but one window and formatted nicely

        if self.total_voltage_df is None:
            print("Total voltage data is not available.")
            return None

        df = self.total_voltage_df.copy()

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))

        for axis, ax in zip(["x", "y", "z"], axs):
            # Get axis values and voltage values around the point of interest while holding other coordinates constant
            if axis == "x":
                filtered_df = df[
                    (df["x"] > (-x_cutoff))
                    & (df["x"] < x_cutoff)
                    & (df["y"] == y)
                    & (df["z"] == z)
                ]
            elif axis == "y":
                filtered_df = df[
                    (df["y"] > (-y_cutoff))
                    & (df["y"] < y_cutoff)
                    & (df["x"] == x)
                    & (df["z"] == z)
                ]
            else:  # axis == 'z'
                filtered_df = df[
                    (df["z"] > (-z_cutoff))
                    & (df["z"] < z_cutoff)
                    & (df["x"] == x)
                    & (df["y"] == y)
                ]

            # fit the potential vs the axis values to a 4th degree polynomial
            coeffs = np.polyfit(filtered_df[axis], filtered_df["CalcV"], 4)
            poly = np.poly1d(coeffs)
            fitted_values = poly(filtered_df[axis])

            # plot the fitted curve
            ax.plot(filtered_df[axis] * 1000, fitted_values, 'g--')

            # put the fit equation in legend
            fit_equation = f"{coeffs[0]:.2e}x^4"
            ax.legend([fit_equation], loc='upper left', bbox_to_anchor=(0, 1), fontsize='small', frameon=False)

            # now plot filtered df with axis as the x axis and CalcV as the y axis
            # ax.plot(filtered_df[axis] * 1000, filtered_df["CalcV"], 'b-')
            ax.scatter(filtered_df[axis] * 1000, filtered_df["CalcV"], color='r', s=10)
            ax.set_xlabel(f"{axis} (mm)")
            ax.set_ylabel("PseudoPotential (V)")
            ax.set_title(f"Calculated PseudoV along {axis} axis", fontsize=12)

        plt.tight_layout(pad=.1)
        return fig

    def plot_freq_in_xyz_directions(self, x, y, z, x_cutoff = 1, y_cutoff = 1, z_cutoff = 1):
        # plots all the potential with x varying and y,z as inputed. And the same for y and z
        # this plot will be shown all together side by side in 3 different plots, but one window and formatted nicely

        if self.total_voltage_df is None:
            print("Total voltage data is not available.")
            return None

        df = self.total_voltage_df.copy()

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))

        for axis, ax in zip(["x", "y", "z"], axs):
            # Get axis values and voltage values around the point of interest while holding other coordinates constant
            if axis == "x":
                filtered_df = df[
                    (df["x"] > (-x_cutoff))
                    & (df["x"] < x_cutoff)
                    & (df["y"] == y)
                    & (df["z"] == z)
                ]
            elif axis == "y":
                filtered_df = df[
                    (df["y"] > (-y_cutoff))
                    & (df["y"] < y_cutoff)
                    & (df["x"] == x)
                    & (df["z"] == z)
                ]
            else:  # axis == 'z'
                filtered_df = df[
                    (df["z"] > (-z_cutoff))
                    & (df["z"] < z_cutoff)
                    & (df["x"] == x)
                    & (df["y"] == y)
                ]
            # now calculate the frequency at each point in the filtered df
            freqs = []
            for i in range(len(filtered_df)):
                freqs.append(self.get_frequencys_at_point_xyz(filtered_df.iloc[i]["x"], filtered_df.iloc[i]["y"], filtered_df.iloc[i]["z"])[["x", "y", "z"].index(axis)])
            # print(str(axis) + str(freqs))

            # now fit the frequencys vs the axis values
            ax.plot(filtered_df[axis] * 1000, freqs, "b-")  # Convert x-axis values to mm
            ax.scatter(filtered_df[axis] * 1000, freqs, color="r", s=10)  # Convert x-axis values to mm
            ax.set_xlabel(f"{axis} (mm)")  # Update x-axis label
            ax.set_ylabel(f"Freq in the {axis} direction")
            ax.set_title(f"Freq in the {axis} dir, vs the {axis} axis" , fontsize=12)

        plt.tight_layout(pad=.1)
        return fig

    def plot_frequency_in_principal_directions_everywhere(self):
        return None
        """
        Plots frequency vectors in the principal directions at each valid point using a 3D quiver plot.

        - The magnitude of each vector corresponds to the frequency in that direction.
        - The direction of the vector corresponds to the principal axis.
        - The color of each vector represents its magnitude.
        """

        if self.valid_points is None or len(self.valid_points) == 0:
            print("No valid points available for plotting.")
            return

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Lists to store vector components for quiver plot
        x_vals, y_vals, z_vals = [], [], []
        u_vals, v_vals, w_vals = [], [], []
        freq_mags = []  # Color by magnitude
        i = 0
        for point in self.valid_points:
            i = i + 1
            if i % 1000 == 0:
                print(
                    "______________________________________________________________________________"
                    + str(i / 380000)
                )
            x, y, z = point
            if x < -.000005 or x > .000005 or y < -.000005 or y > .000005 or z < -.00001 or z > .00001:
                continue

            frequencys_and_directions = self.get_frequencys_at_point_hess(x, y, z)

            for frequency, direction in frequencys_and_directions:
                x_vals.append(x)
                y_vals.append(y)
                z_vals.append(z)

                # Normalize direction vector and scale by frequency
                norm_dir = np.array(direction) / np.linalg.norm(direction)
                scaled_dir = norm_dir * frequency

                u_vals.append(scaled_dir[0])
                v_vals.append(scaled_dir[1])
                w_vals.append(scaled_dir[2])
                freq_mags.append(frequency)

        # Convert lists to numpy arrays
        x_vals, y_vals, z_vals = np.array(x_vals), np.array(y_vals), np.array(z_vals)
        u_vals, v_vals, w_vals = np.array(u_vals), np.array(v_vals), np.array(w_vals)
        freq_mags = np.array(freq_mags)

        # Normalize frequency values to use with colormap
        norm = colors.Normalize(vmin=min(freq_mags), vmax=max(freq_mags))
        cmap = cm.get_cmap('RdBu')  # Choose a colormap
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        quiver_colors = sm.to_rgba(freq_mags)  # Convert frequencies to RGBA colors

        # Create quiver plot
        quiver = ax.quiver(x_vals, y_vals, z_vals, u_vals, v_vals, w_vals, 
                    length=0.000001, normalize=True, color=quiver_colors, linewidth=0.5)
        # Add color bar
        cbar = fig.colorbar(quiver, ax=ax, shrink=0.6)
        cbar.set_label("Frequency Magnitude")

        # Labels and title
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title("Principal Frequency Vectors at Each Valid Point")

        plt.show()

    def get_electrode(self, name):
        return self.electrodes[name]


elec_vars = consts.Electrode_vars()
elec_vars.set_vars("RF12", [377, 28000000 * 2 * math.pi, 0, 0])
# elec_vars.set_vars("RF2", [377, 28000000 * 2 * math.pi, 0, 0])
# elec_vars.set_vars("DC1", [0, 0, 0, 0])
# elec_vars.set_vars("DC5", [0, 0, 0, 0])
# elec_vars.set_vars("DC6", [0, 0, 0, 0])
# elec_vars.set_vars("DC10", [0, 0, 0, 0])

test_sim = Simulation("Simplified2", elec_vars)

print(test_sim.get_frequencys_at_point_xyz(0, 0, 0))

freq = test_sim.get_frequencys_at_point_hess(0, .00005, 0.00002)
print('For point 0, 0, 0')
for i in range(3):
    print("Frequency in direction " + str(freq[i][1]) + " is " + str(freq[i][0]) + " Hz")

fig1 = test_sim.plot_potential_in_xyz_directions(0, 0, 0)
fig2 = test_sim.plot_potential_in_xyz_directions(0, 0, 0, 0.0003, 0.00005, 0.00005)

fig3 = test_sim.plot_freq_in_xyz_directions(0, 0, 0)
fig4 = test_sim.plot_freq_in_xyz_directions(0, 0, 0, 0.0003, 0.00005, 0.00005)

# Things to plot: (And make it like a total report output save to file type thing)
    # Ex,y,z in x,y,z directions
    # Emag in x,y,z directions
    # PseudoPot in x,y,z directions (3 graphs)
    # Freq in x,y,z as a fucntion of postion in x,y,z (9 graphs!!!)
    # Freq in x,y,z as a function of position in x,y,z with a cutoff of 0.0003, 0.00005, 0.00005 (9 graphs!!!)
    # Radial freq vs x axis

# Other todo
    # Get the data again from comsol (grounded vs not??, added vs not??)
    # Clean up code, make the "catch RF" more robust
    # Make a ploting class?
    # Find ways to make the code run faster!!!
    # find out if the (+) and (-) x^4 coeffs on the quartic fit is an error... 

plt.show()
