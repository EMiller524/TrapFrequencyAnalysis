import math
import random
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import consts

class sim_normalfitting:

    def get_frequencys_at_point_xyz(self, x, y, z, look_around=1000, polyfit = 4):
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

        def fit_and_get_second_derivative(axis_values, voltage_values, target_value, plot = False):
            """Fit a quartic polynomial and return the second derivative at the target value."""

            should_plot = False

            if len(axis_values) < 4:
                print("Not enough points for a quartic fit along one axis.")
                return None

            # Fit a quartic  polynomial
            coeffs = np.polyfit(axis_values, voltage_values, polyfit)
            # print(coeffs[0])

            # print the coeffs
            # if abs(coeffs[0]) < abs(coeffs[2]):
            #     print(coeffs)

            # If r2 is less than 0.99999, print the r2 value and target value
            # r2 = r2_score(voltage_values, np.polyval(coeffs, axis_values))
            # if r2 < .99999 and plot:
            #     print("r2 value: " + str(r2))
            #     print("target value: " + str(target_value))
            #     should_plot = False

            # Takes the second derivative of the polynomial at the given point
            poly_derivative = np.polyder(coeffs, 2)
            second_derivative_at_target = np.polyval(poly_derivative, target_value)

            # plot the fit of the polynomial if desired
            # random_number = random.randint(0, 1000)
            # if random_number < 3:
            #     should_plot = True
            should_plot = True
            plot = False
            if should_plot and plot and polyfit == 4:
                x = np.linspace(min(axis_values), max(axis_values), 1000)
                y = np.polyval(coeffs, x)
                plt.plot(x*1000, y, "r-", label="Total Fit")
                plt.plot(x*1000, coeffs[0] * np.power(x, 4), "g--", label="4th degree")
                plt.plot(x*1000, coeffs[2] * np.power(x, 2), "b--", label="2nd degree")
                plt.scatter(axis_values*1000, voltage_values, s=2, color="b")
                plt.xlabel("Axis Value (mm)")
                plt.ylabel("Voltage Value")
                plt.title("4th degree fit along Axis")
                plt.legend()
                plt.show()

            return second_derivative_at_target

        df = self.total_voltage_df.copy()
        Q = consts.ion_charge
        M = consts.ion_mass
        frequencies = []

        for axis in ["x", "y", "z"]:
            # Get axis values and voltage values around the point of interest while holding other coordinates constant
            ploot = True
            if axis == "x":
                filtered_df = df[(df["y"] == y) & (df["z"] == z)]
                ploot = False
            elif axis == "y":
                filtered_df = df[(df["x"] == x) & (df["z"] == z)]
            else:  # axis == 'z'
                filtered_df = df[(df["x"] == x) & (df["y"] == y)]

            sorted_df = filtered_df.sort_values(by=[axis])
            axis_vals = sorted_df[axis].values
            voltage_vals = sorted_df["CalcV"].values

            # Get the points of interest
            closest_idx = np.searchsorted(axis_vals, eval(axis))
            start_idx = max(0, closest_idx - look_around)
            end_idx = min(len(axis_vals), closest_idx + look_around + 1)
            selected_axis_vals = axis_vals[start_idx:end_idx]
            selected_voltage_vals = voltage_vals[start_idx:end_idx]

            # Get the second derivative
            second_derivative = fit_and_get_second_derivative(
                selected_axis_vals, selected_voltage_vals, eval(axis), plot = ploot
            )

            if second_derivative is None:
                frequencies.append(None)
                print(f"Could not calculate frequency along {axis} axis.")
            else:
                freq = (
                    math.copysign(1, second_derivative)
                    * math.sqrt((Q / M) * abs(second_derivative))
                    / (2 * math.pi)
                )
                frequencies.append(freq)

        return frequencies

    def get_wy_wz_at_point_withR2_fit (self, x, y, z, look_around=100, polyfit = 4):
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

        def fit_and_get_second_derivative_withR2(axis_values, voltage_values, plot = False):
            """Fit a quadratic polynomial and return the second derivative at the target value."""

            if len(axis_values) < 4:
                print("Not enough points for a quadratic fit along one axis.")
                return None

            # Create cubic polynomial features
            poly = PolynomialFeatures(degree= polyfit, include_bias=True)
            X_poly = poly.fit_transform(axis_values)

            # Fit the model
            model = LinearRegression()
            model.fit(X_poly, voltage_values)
            coeff = model.coef_
            # print(coeff)

            # get the second derivative of the polynomial at the origin
            ydd = 2 * coeff[3]
            zdd = 2 * coeff[5]

            y_pred = model.predict(X_poly)
            r2 = r2_score(voltage_vals, y_pred)
            mse = mean_squared_error(voltage_vals, y_pred)
            err = (("r2", r2),('MSE', mse))

            if r2 < .99:
                print(f"Warning: Low R² value ({r2:.4f}) for the polynomial fit.")
                print(f"R²: {r2:.4f}, MSE: {mse:.4f}")

            # 3D plot
            plot = False
            if plot:
                fig = plt.figure(figsize=(12, 8))
                ax = fig.add_subplot(111, projection='3d')

                # Scatter plot of the actual data points
                ax.scatter(axis_values[:, 0], axis_values[:, 1], voltage_values, color='red', label='Actual Data')

                # Generate a grid of values for the surface plot
                x_range = np.linspace(min(axis_values[:, 0]), max(axis_values[:, 0]), 30)
                z_range = np.linspace(min(axis_values[:, 1]), max(axis_values[:, 1]), 30)
                x_grid, z_grid = np.meshgrid(x_range, z_range)

                # Compute the total polynomial fit
                X_grid_poly = poly.transform(np.c_[x_grid.ravel(), z_grid.ravel()])
                total_fit = model.predict(X_grid_poly).reshape(x_grid.shape)

                # Compute the second-degree surface (X², Y², XY terms only)
                second_degree_terms = coeff[3] * x_grid**2 + coeff[4] * x_grid * z_grid + coeff[5] * z_grid**2

                # Compute the fourth-degree surface (X⁴, Y⁴, X²Y², etc.)
                fourth_degree_terms = np.zeros_like(x_grid)

                if polyfit >= 4:
                    fourth_degree_terms = (coeff[9] * x_grid**4 + coeff[10] * x_grid**3 * z_grid +
                                        coeff[11] * x_grid**2 * z_grid**2 + coeff[12] * x_grid * z_grid**3 +
                                        coeff[13] * z_grid**4)

                # Plot the total polynomial fit
                ax.plot_surface(x_grid, z_grid, total_fit, color='blue', alpha=0.6, edgecolor='k', label='Total Fit')

                # Plot the x^2 and x^4 terms separately
                if polyfit >= 4:
                    # Plot the second-degree surface
                    ax.plot_surface(x_grid, z_grid, second_degree_terms, color='red', alpha=0.6, edgecolor='k', label='2nd Degree Terms')

                    # Plot the fourth-degree surface
                    ax.plot_surface(x_grid, z_grid, fourth_degree_terms, color='orange', alpha=0.6, edgecolor='k', label='4th Degree Terms')

                ax.set_xlabel('X Axis')
                ax.set_ylabel('Z Axis')
                ax.set_zlabel('Voltage')
                ax.set_title(f'3D Polynomial Fit (Degree {polyfit}) - Contributions Breakdown')
                ax.legend()
                # ax.text(0.02, 0.02, s = f"R²: {err[0][1]:.4f}, MSE: {err[1][1]:.4f}", transform=ax.transAxes, fontsize=10, verticalalignment='bottom')
                plt.show()
            return ydd, zdd

        df = self.total_voltage_df.copy()
        Q = consts.ion_charge
        M = consts.ion_mass

        filtered_df = df[(df["x"] == x)]

        look_around_val = look_around * 0.0000005

        cutout_of_df = filtered_df[
            (filtered_df["y"].between(y - look_around_val, y + look_around_val))
            & (filtered_df["z"].between(z - look_around_val, z + look_around_val))
        ]

        voltage_vals = cutout_of_df["CalcV"].values
        yz_vals_uncentered = cutout_of_df[["y", "z"]].values

        # Make the Point of interest the origin (0,0,0) and move the other points accordingly
        yz_vals_centered = yz_vals_uncentered - [y, z]

        # Get the second derivative
        ydd, zdd = fit_and_get_second_derivative_withR2(
            yz_vals_centered,
            voltage_vals)

        if ydd is None or zdd is None:
            print(f"Could not calculate frequency :(")
            return None
        else:
            wy = (
                math.copysign(1, ydd)
                * math.sqrt((Q / M) * abs(ydd))
                / (2 * math.pi)
            )

            wz = (
                math.copysign(1, zdd)
                * math.sqrt((Q / M) * abs(zdd))
                / (2 * math.pi)
            )

        return [wy, wz]

    def get_wy_wz_wx_at_point_withR3_fit (self, x, y, z, look_around=20, polyfit = 4):

        """
        Compute the frequencies in the x, y, and z directions by fitting a cubic polynomial
        to voltage values along each axis separately and extracting the second derivative.

        Args:
            x, y, z (float): The point of interest.
            look_around (int): Number of points to consider in each direction (default 5).

        Returns:
            list: [freq_x, freq_y, freq_z]
        """
        look_around = look_around * 0.000001
        if self.total_voltage_df is None:
            print("Total voltage data is not available.")
            return None

        def fit_and_get_second_derivative_withR3(
            axis_values, voltage_values, plot=False
        ):
            """Fit a quadratic polynomial and return the second derivative at the target value."""

            if len(axis_values) < 4:
                print("Not enough points for a quadratic fit along one axis.")
                return None

            # Create cubic polynomial features
            poly = PolynomialFeatures(degree=polyfit, include_bias=True)
            X_poly = poly.fit_transform(axis_values)

            # Fit the model
            model = LinearRegression()
            model.fit(X_poly, voltage_values)
            coeff = model.coef_
            # print(poly.get_feature_names_out())
            # print(coeff)

            # get the second derivative of the polynomial at the origin
            dd_xx = 2 * coeff[4]
            dd_yy = 2 * coeff[7]
            dd_zz = 2 * coeff[9]
            dd_xy = coeff[5]
            dd_xz = coeff[6]
            dd_yz = coeff[8]

            axial_dds = (dd_xx, dd_yy, dd_zz)

            hessian = np.array([[dd_xx, dd_xy, dd_xz],
                                [dd_xy, dd_yy, dd_yz],
                                [dd_xz, dd_yz, dd_zz]])

            diagonal_hessian = np.diagonal(hessian)
            # print(diagonal_hessian)
            # get the principal directions and eigenvalues of the hessian
            eigenvalues, eigenvectors = np.linalg.eig(hessian)
            # print(eigenvalues)
            # print(eigenvectors)

            y_pred = model.predict(X_poly)
            r2 = r2_score(voltage_vals, y_pred)
            mse = mean_squared_error(voltage_vals, y_pred)
            err = (("r2", r2), ("MSE", mse))

            if r2 < 0.99:
                print(f"Warning: Low R² value ({r2:.4f}) for the polynomial fit.")
                print(f"R²: {r2:.4f}, MSE: {mse:.4f}")

            # # 3D plot
            # plot = False
            # if plot:
            #     fig = plt.figure(figsize=(12, 8))
            #     ax = fig.add_subplot(111, projection="3d")

            #     # Scatter plot of the actual data points
            #     ax.scatter(
            #         axis_values[:, 0],
            #         axis_values[:, 1],
            #         voltage_values,
            #         color="red",
            #         label="Actual Data",
            #     )

            #     # Generate a grid of values for the surface plot
            #     x_range = np.linspace(
            #         min(axis_values[:, 0]), max(axis_values[:, 0]), 30
            #     )
            #     z_range = np.linspace(
            #         min(axis_values[:, 1]), max(axis_values[:, 1]), 30
            #     )
            #     x_grid, z_grid = np.meshgrid(x_range, z_range)

            #     # Compute the total polynomial fit
            #     X_grid_poly = poly.transform(np.c_[x_grid.ravel(), z_grid.ravel()])
            #     total_fit = model.predict(X_grid_poly).reshape(x_grid.shape)

            #     # Compute the second-degree surface (X², Y², XY terms only)
            #     second_degree_terms = (
            #         coeff[3] * x_grid**2
            #         + coeff[4] * x_grid * z_grid
            #         + coeff[5] * z_grid**2
            #     )

            #     # Compute the fourth-degree surface (X⁴, Y⁴, X²Y², etc.)
            #     fourth_degree_terms = np.zeros_like(x_grid)

            #     if polyfit >= 4:
            #         fourth_degree_terms = (
            #             coeff[9] * x_grid**4
            #             + coeff[10] * x_grid**3 * z_grid
            #             + coeff[11] * x_grid**2 * z_grid**2
            #             + coeff[12] * x_grid * z_grid**3
            #             + coeff[13] * z_grid**4
            #         )

            #     # Plot the total polynomial fit
            #     ax.plot_surface(
            #         x_grid,
            #         z_grid,
            #         total_fit,
            #         color="blue",
            #         alpha=0.6,
            #         edgecolor="k",
            #         label="Total Fit",
            #     )

            #     # Plot the x^2 and x^4 terms separately
            #     if polyfit >= 4:
            #         # Plot the second-degree surface
            #         ax.plot_surface(
            #             x_grid,
            #             z_grid,
            #             second_degree_terms,
            #             color="red",
            #             alpha=0.6,
            #             edgecolor="k",
            #             label="2nd Degree Terms",
            #         )

            #         # Plot the fourth-degree surface
            #         ax.plot_surface(
            #             x_grid,
            #             z_grid,
            #             fourth_degree_terms,
            #             color="orange",
            #             alpha=0.6,
            #             edgecolor="k",
            #             label="4th Degree Terms",
            #         )

            #     ax.set_xlabel("X Axis")
            #     ax.set_ylabel("Z Axis")
            #     ax.set_zlabel("Voltage")
            #     ax.set_title(
            #         f"3D Polynomial Fit (Degree {polyfit}) - Contributions Breakdown"
            #     )
            #     ax.legend()
            #     # ax.text(0.02, 0.02, s = f"R²: {err[0][1]:.4f}, MSE: {err[1][1]:.4f}", transform=ax.transAxes, fontsize=10, verticalalignment='bottom')
            #     plt.show()
            return eigenvalues, eigenvectors, axial_dds

        df = self.total_voltage_df.copy()
        Q = consts.ion_charge
        M = consts.ion_mass

        cutout_of_df = df[
            (df["x"].between(x - (5*look_around), x + (5*look_around)))
            & (df["y"].between(y - look_around, y + look_around))
            & (df["z"].between(z - look_around, z + look_around))
        ]

        voltage_vals = cutout_of_df["CalcV"].values
        xyz_vals_uncentered = cutout_of_df[["x", "y", "z"]].values

        # Make the Point of interest the origin (0,0,0) and move the other points accordingly
        xyz_vals_centered = xyz_vals_uncentered - [x, y, z]

        # Get the second derivative
        eigenval, eigenvec, axialdds = fit_and_get_second_derivative_withR3(xyz_vals_centered, voltage_vals)
        xx, yy, zz = axialdds

        if xx is None or yy is None or zz is None:
            print(f"Could not calculate frequency :(")
            return None
        else:
            w1 = (
                math.copysign(1, eigenval[0])
                * math.sqrt((Q / M) * abs(eigenval[0]))
                / (2 * math.pi)
            )

            w2 = (
                math.copysign(1, eigenval[1])
                * math.sqrt((Q / M) * abs(eigenval[1]))
                / (2 * math.pi)
            )

            w3 = (
                math.copysign(1, eigenval[2])
                * math.sqrt((Q / M) * abs(eigenval[2]))
                / (2 * math.pi)
            )

            wx = (
                math.copysign(1, xx)
                * math.sqrt((Q / M) * abs(xx))
                / (2 * math.pi)
            )

            wy = math.copysign(1, yy) * math.sqrt((Q / M) * abs(yy)) / (2 * math.pi)

            wz = math.copysign(1, zz) * math.sqrt((Q / M) * abs(zz)) / (2 * math.pi)

        return [w1, w2, w3], [wx, wy, wz], eigenvec

    # def get_frequencies_at_point


#############################################
# import numpy as np
# import pandas as pd
# from joblib import Parallel, delayed
# from numba import njit
# import consts


# class sim_normalfitting:
#     def compute_all_frequencies(self, look_around=20, n_jobs=-1):
#         """
#         🚀 Compute Wx, Wy, Wz at ALL points in the dataset efficiently.
#         Uses parallel execution & pre-sorting to handle 100-200M points efficiently.

#         Args:
#             look_around (int): Number of neighboring points to consider for fitting.
#             n_jobs (int): Number of CPU cores to use (-1 = all).

#         Returns:
#             DataFrame with ['x', 'y', 'z', 'Wx', 'Wy', 'Wz']
#         """
#         if self.total_voltage_df is None:
#             print("Total voltage data is not available.")
#             return None

#         df = self.total_voltage_df
#         Q = consts.ion_charge
#         M = consts.ion_mass

#         # ✅ Pre-sort the entire DataFrame ONCE
#         df_sorted = df.sort_values(by=["x", "y", "z"]).reset_index(drop=True)

#         # ✅ Convert to NumPy arrays (removes slow DataFrame slicing)
#         x_vals = df_sorted["x"].values
#         y_vals = df_sorted["y"].values
#         z_vals = df_sorted["z"].values
#         voltage_vals = df_sorted["CalcV"].values

#         # ✅ Extract unique points (removes duplicates)
#         points = df_sorted[["x", "y", "z"]].drop_duplicates().values

#         # ✅ Run frequency calculations in parallel for all points (fixed batching)
#         results = Parallel(n_jobs=n_jobs, batch_size=1)(
#             delayed(compute_frequencies_batch)(
#                 point, x_vals, y_vals, z_vals, voltage_vals, Q, M, look_around
#             )
#             for point in points
#         )

#         # ✅ Create final DataFrame
#         results_df = pd.DataFrame(points, columns=["x", "y", "z"])
#         results_df[["Wx", "Wy", "Wz"]] = results

#         return results_df


# def compute_frequencies_batch(
#     point, x_vals, y_vals, z_vals, voltage_vals, Q, M, look_around
# ):
#     """
#     🚀 Computes Wx, Wy, Wz for a single (x, y, z) point efficiently using NumPy arrays.
#     """
#     x, y, z = point
#     second_derivatives = []
#     print("Computing frequencies for point:", point)

#     for axis, axis_vals in zip(["x", "y", "z"], [x_vals, y_vals, z_vals]):
#         # ✅ Filter only relevant slice of data (NumPy indexing)
#         if axis == "x":
#             mask = (y_vals == y) & (z_vals == z)
#             target_value = x
#         elif axis == "y":
#             mask = (x_vals == x) & (z_vals == z)
#             target_value = y
#         else:  # axis == "z"
#             mask = (x_vals == x) & (y_vals == y)
#             target_value = z

#         axis_filtered = axis_vals[mask]
#         voltage_filtered = voltage_vals[mask]

#         if len(axis_filtered) < 3:  # Ensure enough points for polyfit
#             second_derivatives.append(0.0)
#             continue

#         # ✅ Find closest index
#         closest_idx = np.searchsorted(axis_filtered, target_value)

#         # ✅ Select `look_around` points
#         start_idx = max(0, closest_idx - look_around)
#         end_idx = min(len(axis_filtered), closest_idx + look_around + 1)

#         selected_axis_vals = axis_filtered[start_idx:end_idx]
#         selected_voltage_vals = voltage_filtered[start_idx:end_idx]

#         # ✅ Compute second derivative using polyfit
#         second_derivative = polyfit_wrapper(
#             selected_axis_vals, selected_voltage_vals, target_value
#         )

#         second_derivatives.append(second_derivative)

#     # ✅ Compute frequencies using fast Numba function
#     return fast_compute_frequencies(*second_derivatives, Q, M)


# def polyfit_wrapper(x_vals, y_vals, target_x):
#     """🚀 Compute the second derivative at a target point using stable np.polyfit()."""
#     if len(np.unique(x_vals)) < 3:  # Ensure at least 3 unique x values
#         return 0.0  # Return 0 if not enough points for a fit

#     # ✅ Fit quadratic polynomial
#     coeffs = np.polyfit(x_vals, y_vals, 2)

#     # ✅ Compute second derivative at the target value
#     second_derivative_coeffs = np.polyder(coeffs, 2)
#     second_derivative_at_target = np.polyval(second_derivative_coeffs, target_x)

#     return second_derivative_at_target


# @njit
# def fast_compute_frequencies(Wx_dd, Wy_dd, Wz_dd, Q, M):
#     """🚀 Numba-accelerated computation of frequencies after polyfit is done."""
#     frequencies = np.zeros(3)
#     frequencies[0] = np.sqrt((Q / M) * abs(Wx_dd)) / (2 * np.pi)
#     frequencies[1] = np.sqrt((Q / M) * abs(Wy_dd)) / (2 * np.pi)
#     frequencies[2] = np.sqrt((Q / M) * abs(Wz_dd)) / (2 * np.pi)
#     return frequencies
