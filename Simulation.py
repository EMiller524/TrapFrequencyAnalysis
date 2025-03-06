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
from matplotlib.backends.backend_pdf import PdfPages
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
                print(f"no data for {electrode}")
            else:
                dfs.append(self.electrodes[electrode].get_dataframe())

        # Step 1: Find the intersection of all unique (x, y, z) combinations
        common_keys = reduce(
            lambda left, right: pd.merge(left, right, on=["x", "y", "z"]),
            [df[["x", "y", "z"]] for df in dfs],
        )

        # Step 2: Concatenate all dataframes
        merged_df = pd.concat(dfs, ignore_index=True)

        # Step 3: Filter merged_df to only keep common (x, y, z) keys
        filtered_df = merged_df.merge(common_keys, on=["x", "y", "z"])

        # Step 4: Group by (x, y, z) and sum CalcV
        master_df = filtered_df.groupby(["x", "y", "z"], as_index=False)["CalcV"].sum()

        self.total_voltage_df = master_df

    def fit_v_at_point(self, x0, y0, z0, grid_size=5, show=False):
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

        if r2 < 0.99:
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
        H[0, 0] = (
            2 * coef_dict.get("x^2", 0)
            + 2 * coef_dict.get("x^2 y", 0) * y
            + 2 * coef_dict.get("x^2 z", 0) * z
        )
        H[1, 1] = (
            2 * coef_dict.get("y^2", 0)
            + 2 * coef_dict.get("y^2 x", 0) * x
            + 2 * coef_dict.get("y^2 z", 0) * z
        )
        H[2, 2] = (
            2 * coef_dict.get("z^2", 0)
            + 2 * coef_dict.get("z^2 x", 0) * x
            + 2 * coef_dict.get("z^2 y", 0) * y
        )

        # Mixed partial derivatives
        H[0, 1] = H[1, 0] = coef_dict.get("x y", 0) + 2 * coef_dict.get("x y z", 0) * z
        H[0, 2] = H[2, 0] = coef_dict.get("x z", 0) + 2 * coef_dict.get("x y z", 0) * y
        H[1, 2] = H[2, 1] = coef_dict.get("y z", 0) + 2 * coef_dict.get("x y z", 0) * x

        # print("X dir freq: " + str(math.sqrt((consts.ion_charge / consts.ion_mass) * abs(H[0,0])) * 2 * math.pi))
        # print("Y dir freq: " + str(math.sqrt((consts.ion_charge / consts.ion_mass) * abs(H[1,1])) * 2 * math.pi))

        # print("Z dir freq: " + str(math.sqrt((consts.ion_charge / consts.ion_mass) * abs(H[2,2])) * 2 * math.pi))

        print("Hessian at: " + str((x, y, z)) + str(H))
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

        eigenvalues, eigenvectors = self.diagonalize_hessian(
            self.get_hessian_at_point(x, y, z)
        )
        Q = consts.ion_charge
        M = consts.ion_mass

        frequencys_and_directions = []
        # Calculate frequencies
        for i in range(3):
            frequency = math.sqrt((Q / M) * abs(eigenvalues[i])) / (math.pi * 2)
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
                plt.plot(x, y, "r-")
                plt.scatter(axis_values, voltage_values, color="b")
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
                freq = (
                    math.copysign(1, second_derivative)
                    * math.sqrt((Q / M) * abs(second_derivative))
                    / (2 * math.pi)
                )  ###############################################################################################################
                frequencies.append(freq)

        return frequencies

    def plot_potential_in_xyz_directions(
        self, x, y, z, x_cutoff=10, y_cutoff=10, z_cutoff=10
    ):
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
            ax.plot(filtered_df[axis] * 1000, fitted_values, "g--")

            # put the fit equation in legend
            fit_equation = f"{coeffs[0]:.2e}x^4"
            ax.legend(
                [fit_equation],
                loc="upper left",
                bbox_to_anchor=(0, 1),
                fontsize="small",
                frameon=False,
            )

            # now plot filtered df with axis as the x axis and CalcV as the y axis
            # ax.plot(filtered_df[axis] * 1000, filtered_df["CalcV"], 'b-')
            ax.scatter(filtered_df[axis] * 1000, filtered_df["CalcV"], color="r", s=10)
            ax.set_xlabel(f"{axis} (mm)")
            ax.set_ylabel("PseudoPotential (V)")
            ax.set_title(f"Calculated PseudoV along {axis} axis", fontsize=12)

        plt.tight_layout(pad=0.1)
        return fig

    def plot_freq_in_xyz_directions(
        self, x, y, z, x_cutoff=10, y_cutoff=10, z_cutoff=10
    ):
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
                freqs.append(
                    self.get_frequencys_at_point_xyz(
                        filtered_df.iloc[i]["x"],
                        filtered_df.iloc[i]["y"],
                        filtered_df.iloc[i]["z"],
                    )[["x", "y", "z"].index(axis)]
                )
            # print(str(axis) + str(freqs))

            # now fit the frequencys vs the axis values
            ax.plot(
                filtered_df[axis] * 1000, freqs, "b-"
            )  # Convert x-axis values to mm
            ax.scatter(
                filtered_df[axis] * 1000, freqs, color="r", s=5
            )  # Convert x-axis values to mm
            ax.set_xlabel(f"{axis} (mm)")  # Update x-axis label
            ax.set_ylabel(f"Freq in the {axis} direction")
            ax.set_title(f"Freq in the {axis} dir, vs the {axis} axis", fontsize=12)

        plt.tight_layout(pad=0.1)
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
            if (
                x < -0.000005
                or x > 0.000005
                or y < -0.000005
                or y > 0.000005
                or z < -0.00001
                or z > 0.00001
            ):
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
        cmap = cm.get_cmap("RdBu")  # Choose a colormap
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        quiver_colors = sm.to_rgba(freq_mags)  # Convert frequencies to RGBA colors

        # Create quiver plot
        quiver = ax.quiver(
            x_vals,
            y_vals,
            z_vals,
            u_vals,
            v_vals,
            w_vals,
            length=0.000001,
            normalize=True,
            color=quiver_colors,
            linewidth=0.5,
        )
        # Add color bar
        cbar = fig.colorbar(quiver, ax=ax, shrink=0.6)
        cbar.set_label("Frequency Magnitude")

        # Labels and title
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title("Principal Frequency Vectors at Each Valid Point")

        plt.show()

    def plot_value_in_blank_direction(
        self, x, y, z, direction, value, x_cutoff=1, y_cutoff=1, z_cutoff=1
    ):
        """
        Returns a fig that plots value vs direction starting at the point(x,y,z) and moving in the direction.
        For example, if direction = "x" and value = "CalcV", then the graph will be of the potential vs x.
        """
        if self.total_voltage_df is None:
            return None

        df = self.total_voltage_df.copy()

        fig, ax = plt.subplots(figsize=(8, 6))

        if direction == "x":
            x_label = "X (mm)"
            filtered_df = df[
                (df["x"] > (-x_cutoff))
                & (df["x"] < x_cutoff)
                & (df["y"] == y)
                & (df["z"] == z)
            ]
        elif direction == "y":
            x_label = "Y (mm)"
            filtered_df = df[
                (df["y"] > (-y_cutoff))
                & (df["y"] < y_cutoff)
                & (df["x"] == x)
                & (df["z"] == z)
            ]
        elif direction == "z":
            x_label = "Z (mm)"
            filtered_df = df[
                (df["z"] > (-z_cutoff))
                & (df["z"] < z_cutoff)
                & (df["x"] == x)
                & (df["y"] == y)
            ]

        valuess = None
        val_name = value
        fit = True
        # Plot the value vs direction if value is a valid column
        if value in filtered_df.columns:
            valuess = filtered_df[value]
            if value == "CalcV":
                val_name = "PseudoPotential"
        elif value == "EMag":
            valuess = filtered_df.apply(
                lambda row: math.sqrt(row["Ex"] ** 2 + row["Ey"] ** 2 + row["Ez"] ** 2),
                axis=1,
            )
        elif value in ["Wx", "Wy", "Wz"]:
            fit = False
            axis_index = ["x", "y", "z"].index(value[1])
            valuess = []
            for i in range(len(filtered_df)):
                valuess.append(
                    self.get_frequencys_at_point_xyz(
                        filtered_df.iloc[i]["x"],
                        filtered_df.iloc[i]["y"],
                        filtered_df.iloc[i]["z"],
                    )[axis_index]
                )
        elif value == "Wr":
            fit = False
            # get the frequency in the y and z directions using get_frequencys_at_point_xyz
            valuess = []
            for i in range(len(filtered_df)):
                Wy = self.get_frequencys_at_point_xyz(
                    filtered_df.iloc[i]["x"],
                    filtered_df.iloc[i]["y"],
                    filtered_df.iloc[i]["z"],
                )[["x", "y", "z"].index("y")]
                Wz = self.get_frequencys_at_point_xyz(
                    filtered_df.iloc[i]["x"],
                    filtered_df.iloc[i]["y"],
                    filtered_df.iloc[i]["z"],
                )[["x", "y", "z"].index("z")]
                Wr = math.sqrt((Wy**2) / 2 + (Wz**2) / 2)
                valuess.append(Wr)
        else:
            return None

        ax.scatter(filtered_df[direction] * 1000, valuess, s=5, c="blue")

        if fit:
            # now fit the data to a 4th degree polynomial
            coeffs = np.polyfit(filtered_df[direction], valuess, 4)
            poly = np.poly1d(coeffs)
            fitted_values = poly(filtered_df[direction])
            ax.plot(filtered_df[direction] * 1000, fitted_values, "g--")

            # put the x^4 coeff in the legend and the r^2 and mse values
            r2 = r2_score(valuess, fitted_values)
            fit_equation_x4 = f"{coeffs[0]:.2e}x^4"
            mse = mean_squared_error(valuess, fitted_values)
            mse_normalized = mse * 1000000 / (np.max(valuess) - np.min(valuess)) ** 2
            r2 = r2_score(valuess, fitted_values)
            ax.legend(
                [fit_equation_x4, f"R²: {r2:.4f}, RMSE: {mse_normalized:.4f}"],
                loc="upper left",
                bbox_to_anchor=(0, 1),
                fontsize="small",
                frameon=True,
            )

        # Set labels and title
        ax.set_xlabel(x_label)
        ax.set_ylabel(val_name)
        ax.set_title(f"{val_name} vs {str(direction)}")
        ax.grid(True, alpha=0.2)

        return fig

    def get_full_report(self, name):
        # plots may things
        # Plot 1-3 PseudoPot in x,y,z directions (3 graphs)
        # Plot 4-6 Wx vs x Wy vs y Wz Vs z (3 graphs)
        # Plot 7 Wy vs x Wz vs x and We vs x (1 graph)

        plot1 = test_sim.plot_value_in_blank_direction(0, 0, 0, "x", "CalcV")
        
        plot21 = test_sim.plot_value_in_blank_direction(0, 0, 0, "y", "CalcV")
        plot22 = test_sim.plot_value_in_blank_direction(0, 0, 0, "z", "CalcV")

        plot3 = test_sim.plot_value_in_blank_direction(0, 0, 0, "x", "Wx")
        
        
        fig41 = test_sim.plot_value_in_blank_direction(0, 0, 0, "y", "Wy", y_cutoff=0.001)
        fig42 = test_sim.plot_value_in_blank_direction(0, 0, 0, "z", "Wz", z_cutoff=0.001)

        # plot fig51 and fig52 on the same plot
        plot4 = plt.figure()
        ax = plot4.add_subplot(111)
        ax.scatter(fig41.axes[0].collections[0].get_offsets()[:, 0], fig41.axes[0].collections[0].get_offsets()[:, 1], label="Wy", s=5)
        ax.scatter(fig42.axes[0].collections[0].get_offsets()[:, 0], fig42.axes[0].collections[0].get_offsets()[:, 1], label="Wz", s=5)
        ax.legend()
        ax.set_xlabel("Y, Z (mm)")
        ax.set_ylabel("Frequency in given direction")
        ax.set_title("Wy, Wz vs Y, Z")
        

        plot5 = plt.figure()
        ax = plot5.add_subplot(111)

        # Recompute values directly instead of extracting from previous figures
        df = test_sim.total_voltage_df.copy()
        filtered_df = df[
            (df["x"] > -1) & (df["x"] < 1) & (df["y"] == 0) & (df["z"] == 0)
        ]

        # Compute values
        Wy_values = [
            test_sim.get_frequencys_at_point_xyz(row["x"], row["y"], row["z"])[1]
            for _, row in filtered_df.iterrows()
        ]
        Wz_values = [
            test_sim.get_frequencys_at_point_xyz(row["x"], row["y"], row["z"])[2]
            for _, row in filtered_df.iterrows()
        ]
        Wr_values = [
            math.sqrt((Wy**2) / 2 + (Wz**2) / 2) for Wy, Wz in zip(Wy_values, Wz_values)
        ]

        ax.scatter(filtered_df["x"] * 1000, Wy_values, label="Wy", s=5)
        ax.scatter(filtered_df["x"] * 1000, Wz_values, label="Wz", s=5)
        ax.scatter(filtered_df["x"] * 1000, Wr_values, label="Wr", s=5)

        ax.legend()
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Frequency")
        ax.set_title("Wy, Wz, Wr vs X")

        # saves plot 1-7 in a pdf with the name, name, and opens up a file dialgog thing to ask the user where to save
        pdf_file = "repts/" + name + ".pdf"
        with PdfPages(pdf_file) as pdf:
            pdf.savefig(plot1)
            pdf.savefig(plot21)
            pdf.savefig(plot22)
            pdf.savefig(plot3)
            pdf.savefig(plot4)
            pdf.savefig(plot5)
            # pdf.savefig(plot6)
            # pdf.savefig(plot7)

        print(f"Figures saved in {pdf_file}")

    def get_electrode(self, name):
        return self.electrodes[name]


elec_vars = consts.Electrode_vars()
elec_vars.set_vars("RF12", [377, 28000000 * 2 * math.pi, 0, 0])
# elec_vars.set_vars("RF2", [377, 28000000 * 2 * math.pi, 0, 0])
# elec_vars.set_vars("DC1", [0, 0, 0, 0])
# elec_vars.set_vars("DC5", [0, 0, 0, 0])
# elec_vars.set_vars("DC6", [0, 0, 0, 0])
# elec_vars.set_vars("DC10", [0, 0, 0, 0])

test_sim = Simulation("Simp90", elec_vars)

print(test_sim.get_frequencys_at_point_xyz(0, 0, 0))

test_sim.get_full_report("Simp90(000)_377_28_DC_Grounded_NotCaught_RF_test")

# fig1 = test_sim.plot_potential_in_xyz_directions(0, 0, 0)
# fig2 = test_sim.plot_potential_in_xyz_directions(0, 0, 0, 0.0003, 0.00005, 0.00005)

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
