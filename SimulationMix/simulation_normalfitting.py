import math
from statistics import LinearRegression
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
import consts

class sim_normalfitting:
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
