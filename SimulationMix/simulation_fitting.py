'''
This file contians the sim_normalfitting class
'''

from itertools import combinations_with_replacement
import math
import random
import time
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import constants

class sim_normalfitting:
    '''
    This class will be inherited by "simulation"
    This calss will hold most all fucntions that deal with fitting the data except for a few frontend funcs
    '''

    def get_frequencys_at_point_xyz(self, x, y, z, look_around=1000, polyfit = 4):
        """
        Compute the frequencies in the x, y, and z directions by fitting a 1D polynomial
        to voltage values along each axis separately and extracting the second derivative.
        
        * * Note * * Lookaround here is still in number of points

        Args:
            x, y, z (float): The point of interest.
            look_around (int): Number of points to consider in each direction (default 5).

        Returns:
            list: [freq_x, freq_y, freq_z]
        """

        look_around = look_around * 0.000001  # Convert microns to meters

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

        Q = constants.ion_charge
        M = constants.ion_mass
        frequencies = []

        for axis in ["x", "y", "z"]:
            # Get axis values and voltage values around the point of interest while holding other coordinates constant
            ploot = True
            if axis == "x":
                filtered_df = self.total_voltage_df[
                    (self.total_voltage_df["y"] == y)
                    & (self.total_voltage_df["z"] == z)
                ]
                ploot = False
            elif axis == "y":
                filtered_df = self.total_voltage_df[
                    (self.total_voltage_df["x"] == x)
                    & (self.total_voltage_df["z"] == z)
                ]
            else:  # axis == 'z'
                filtered_df = self.total_voltage_df[
                    (self.total_voltage_df["x"] == x)
                    & (self.total_voltage_df["y"] == y)
                ]

            sorted_df = filtered_df.sort_values(by=[axis])
            axis_vals = sorted_df[axis].values
            voltage_vals = sorted_df["TotalV"].values

            # Get the points of interest
            target_value = eval(axis)
            start_value = target_value - look_around
            end_value = target_value + look_around
            selected_axis_vals = axis_vals[(axis_vals >= start_value) & (axis_vals <= end_value)]
            selected_voltage_vals = voltage_vals[(axis_vals >= start_value) & (axis_vals <= end_value)]

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

    def get_wy_wz_at_point_withR2_fit (self, x, y, z, look_around=5, polyfit = 4):
        """
        Compute the frequencies in the y, and z directions by fitting a 2 dim polynomial to yz plane of
        voltage values and extract the second derivative.

        Args:
            x, y, z (float): The point of interest.
            look_around (int): Number of microns to look around in each direction (default 5).

        Returns:
            list: [freq_y, freq_z]
        """

        look_around = look_around * 0.000001  # Convert microns to meters

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

        Q = constants.ion_charge
        M = constants.ion_mass

        filtered_df = self.total_voltage_df[(self.total_voltage_df["x"] == x)]

        cutout_of_df = filtered_df[
            (filtered_df["y"].between(y - look_around, y + look_around))
            & (filtered_df["z"].between(z - look_around, z + look_around))
        ]

        voltage_vals = cutout_of_df["TotalV"].values
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
            wy = constants.freq_calcualtion(ydd)

            wz = constants.freq_calcualtion(zdd)

        return [wy, wz]

    def get_freqs_at_point_withR3_fit (self, x, y, z, look_around=5, polyfit = 4, return_coefs = False):
        """
        Compute the princapl and x, y, and z dir freqs by fitting a 3dim polynomial
        to the total voltage values in a given neighborhood around the point

        Args:
            x, y, z (float): The point of interest.
            look_around (int): Number of microns to look in each direction (default 5).

        Returns:
            list: [freq_x, freq_y, freq_z], 
            list: [principal_freq_x, principal_freq_y, principal_freq_z], 
            list[list]: eigenvecs = [eigenvec1 = [xcomp, ycomp, zcomp], 2, 3]
        """
        time1 = time.time()

        # Convert look_around from microns to meters
        look_around = look_around * 0.000001

        # Check if the total_voltage_df is available
        if self.total_voltage_df is None:
            print("Total voltage data is not available.")
            return None

        def fit_and_get_second_derivative_withR3(
            axis_values, voltage_values
        ):
            """
            Fit a 3dim polynomial determened by the degree (polyfit) and return:
            1. The eigenvalues of the Hessian matrix (3 values)
            2. The eigenvectors of the Hessian matrix (3x3 matrix)
            3. The second derivatives along the x, y, and z axes (3 values)
            """

            if len(axis_values) < 4:
                print("Not enough points for a quadratic fit along one axis.")
                return None

            # print(f"Number of data points considered: {len(axis_values)}")

            # TODO: Check bound fitting params to be physical
            # Create cubic polynomial features
            poly = PolynomialFeatures(degree=polyfit, include_bias=True)
            X_poly = poly.fit_transform(axis_values)

            # Fit the model
            model = LinearRegression()
            model.fit(X_poly, voltage_values)            
            coeff = model.coef_
            # print(poly.get_feature_names_out())
            # print(coeff)

            ## normalized fit ########################################################################
            scaler = StandardScaler()
            axis_values_scaled = scaler.fit_transform(axis_values)  # shape (n_samples, 3)
            X_poly_scaled = poly.fit_transform(axis_values_scaled)

            model_scaled = LinearRegression()
            model_scaled.fit(X_poly_scaled, voltage_values)
            normalized_coeff = model_scaled.coef_
            
            # Get transformed standardized input
            X_poly_scaled = poly.fit_transform(scaler.transform(axis_values))

            # Term-wise contribution (absolute)
            contributions = np.abs(X_poly_scaled * normalized_coeff)  # shape: (n_samples, n_terms)

            # Mean contribution of each term
            mean_contribution = contributions.mean(axis=0)
            
            # Step 3: Normalize so they sum to 1000
            normalized_weights = (mean_contribution / mean_contribution.sum()) * 1000

            # Now you can rank or normalize these
            relative_importance = mean_contribution / mean_contribution.sum()
            
            term_stds = X_poly_scaled.std(axis=0)
            scaled_coeffs = normalized_coeff * term_stds
            ## end normalized fit ####################################################################

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

            # get the principal directions and eigenvalues of the hessian
            eigenvalues, eigenvectors = np.linalg.eig(hessian)
            # print(eigenvalues)
            # print(eigenvectors)

            # Run error checking on the polynomial fit
            y_pred = model.predict(X_poly)
            r2 = r2_score(voltage_vals, y_pred)
            mse = mean_squared_error(voltage_vals, y_pred)
            err = (("r2", r2), ("MSE", mse))

            if r2 < 0.999:
                print(f"Warning: Low R² value ({r2:.4f}) for the polynomial fit.")
                print(f"R²: {r2:.4f}, MSE: {mse:.4f}")

            # TODO: Check if the 2nd deg terms fit well, IE: r2>.99

            if return_coefs:
                return eigenvalues, eigenvectors, axial_dds, relative_importance*10000
            else:
                return eigenvalues, eigenvectors, axial_dds

        time2 = time.time()

        df = self.total_voltage_df

        x_arr = df["x"].to_numpy()
        y_arr = df["y"].to_numpy()
        z_arr = df["z"].to_numpy()

        xmin, xmax = x - 5*look_around, x + 5*look_around
        ymin, ymax = y - look_around,  y + look_around
        zmin, zmax = z - look_around,  z + look_around

        mask = (
            (x_arr >= xmin) & (x_arr <= xmax) &
            (y_arr >= ymin) & (y_arr <= ymax) &
            (z_arr >= zmin) & (z_arr <= zmax)
        )

        cutout_of_df = df.iloc[mask]

        time3 = time.time()

        voltage_vals = cutout_of_df["TotalV"].values
        xyz_vals_uncentered = cutout_of_df[["x", "y", "z"]].values

        # Make the Point of interest the origin (0,0,0) and move the other points accordingly
        xyz_vals_centered = xyz_vals_uncentered - [x, y, z]
        # Get the second derivative
        if return_coefs:
            eigenval, eigenvec, axialdds, coeff = fit_and_get_second_derivative_withR3(xyz_vals_centered, voltage_vals)
        else:
            eigenval, eigenvec, axialdds = fit_and_get_second_derivative_withR3(xyz_vals_centered, voltage_vals)
        xx, yy, zz = axialdds

        if xx is None or yy is None or zz is None:
            print(f"Could not calculate frequency :(")
            return None

        else:
            w1 = constants.freq_calcualtion(eigenval[0])
            w2 = constants.freq_calcualtion(eigenval[1])
            w3 = constants.freq_calcualtion(eigenval[2])

            wx = constants.freq_calcualtion(xx)
            wy = constants.freq_calcualtion(yy)
            wz = constants.freq_calcualtion(zz)

        time4 = time.time()

        # Finding bottlenecks in run time
        # print(f"Time taken for data extraction: {time2 - time1:.4f} seconds")
        # print(f"Time taken for filtering: {time3 - time2:.4f} seconds")
        # print(f"Time taken for fitting: {time4 - time3:.4f} seconds")
        # print(f"Time taken for total: {time4 - time1:.4f} seconds")
        if return_coefs:
            return [w1, w2, w3], [wx, wy, wz], eigenvec, coeff
        else:
            return [w1, w2, w3], [wx, wy, wz], eigenvec
