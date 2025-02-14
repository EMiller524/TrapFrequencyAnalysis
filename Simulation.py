import math
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import dataextractor
import os
import consts
import Electrode


class Simulation:
    def __init__(self, dataset, variables = consts.Electrode_vars()):
        """
        Initialize the Electrode object.

        Parameters:
        - name (str): The name of the electrode.
        - voltage (float): The applied voltage.
        - data (pd.DataFrame, optional): A pandas DataFrame containing relevant electrode data.
        """
        self.dataset = dataset
        self.file_path = (
            "C:\\GitHub\\TrapFrequencyAnalysis\\Data\\" + dataset + "\\")

        # make a dictionary of electrodes with each name in consts.electrode_names as the key and an electrode class as the value
        self.electrode_vars = variables

        self.electrodes = {}
        for electrode in consts.electrode_names:
            self.electrodes[electrode] = Electrode.Electrode(electrode, self.dataset, self.electrode_vars.get_vars(electrode))

        self.valid_points = self.get_valid_points()

    def update_electrode(self, electrode):
        self.electrodes[electrode] = Electrode.Electrode(electrode, self.dataset, self.electrode_vars.get_vars(electrode))

    def set_variables(self, electrode, variables):
        self.electrode_vars.set_vars(electrode, variables)
        self.update_electrode(electrode)

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

    def get_total_voltage_at_point(self, x, y, z):
        V = 0
        for electrode in consts.electrode_names:
            if self.electrodes[electrode].data is None:
                continue
            else:
                V += self.electrodes[electrode].get_potential_at_point_using_var( x, y, z)
        return V

    def get_voltage_second_derivative_at_point(self, x, y, z, plot_fit=False):
        """
        Compute the second derivative of voltage at (x, y, z) in the x, y, and z directions
        by fitting local points to a polynomial and differentiating.

        Args:
            x, y, z (float): The coordinates of the point.

        Returns:
            tuple: (d²V/dx², d²V/dy², d²V/dz²) or (None, None, None) if data is insufficient.
        """
        derivatives = []

        # Define the directions to iterate over
        directions = ["x", "y", "z"]
        indices = [0, 1, 2]  # Corresponding indices in `valid_points`

        for direction, index in zip(directions, indices):
            # Filter points where the other two coordinates are fixed
            filtered_points = self.valid_points[
                (self.valid_points[:, (index + 1) % 3] == [y, z][index != 0])
                & (self.valid_points[:, (index + 2) % 3] == [y, z][index == 0])
            ]

            # Sort by the coordinate of interest
            sorted_points = filtered_points[np.argsort(filtered_points[:, index])]

            # Get index of the desired point
            target_indices = np.where(sorted_points[:, index] == [x, y, z][index])[0]

            if len(target_indices) == 0:
                derivatives.append(None)
                continue  # Skip if no valid data

            target_index = target_indices[0]

            # Select up to 10 points on both sides
            left_index = max(0, target_index - 2)
            right_index = min(len(sorted_points), target_index + 2 + 1)
            points_of_interest = sorted_points[left_index:right_index]

            # Extract the coordinate values and corresponding voltage
            coord_values = points_of_interest[:, index]
            V_values = np.array(
                [self.get_total_voltage_at_point(*point) for point in points_of_interest]
            )

            # print(f"Direction: {direction}")
            # print("Points of Interest:", points_of_interest)
            # print("Voltages:", V_values)

            # Fit a polynomial of degree 3
            fit = np.polyfit(coord_values, V_values, 3)

            # Compute the second derivative at the given coordinate
            second_derivative = np.polyval(np.polyder(fit, 2), [x, y, z][index])
            derivatives.append(second_derivative)

            # **Plot the polynomial fit if requested**
            if plot_fit:
                x_smooth = np.linspace(
                    min(coord_values), max(coord_values), 100
                )  # Smooth x range
                V_fit = np.polyval(fit, x_smooth)  # Evaluate polynomial at smooth x values

                plt.figure(figsize=(6, 4))
                plt.scatter(
                    coord_values, V_values, color="red", label="Data Points"
                )  # Scatter plot
                plt.plot(
                    x_smooth, V_fit, color="blue", label="Polynomial Fit"
                )  # Polynomial curve
                plt.axvline(x, color="green", linestyle="--", label=f"Point at {x}")
                plt.xlabel(f"{direction} Coordinate")
                plt.ylabel("Voltage (V)")
                plt.title(f"Voltage Fit in {direction}-Direction")
                plt.legend()
                plt.grid()
                plt.show()

        return list(derivatives)  # (d²V/dx², d²V/dy², d²V/dz²)

    def get_electrode(self, name):
        return self.electrodes[name]

    def calcualte_frequencys(self, x, y, z):
        derivatives = self.get_voltage_second_derivative_at_point(x, y, z)

        trap_freq_x = math.sqrt(
            ((consts.ion_charge) * abs((derivatives[0]))) / (consts.ion_mass)
        )
        trap_freq_y = math.sqrt(
            ((consts.ion_charge) * (derivatives[1])) / (consts.ion_mass)
        )
        trap_freq_z = math.sqrt(
            ((consts.ion_charge) * (derivatives[2])) / (consts.ion_mass)
        )

        trap_freq_radial = ((trap_freq_y**2) * (trap_freq_z**2))**(1/2)

        return (
            ("axial trap freq", trap_freq_x),
            ("radial trap freq:", trap_freq_radial),
        )

    def plot_potential_in_principal_directions(self, x, y, z):
        """
        Plots the potential at all valid points along the principal directions (x, y, z)
        while keeping the other two coordinates fixed.

        Args:
            x, y, z (float): The coordinates of the reference point.

        Returns:
            None (Displays the plots)
        """
        directions = ["x", "y", "z"]
        indices = [0, 1, 2]  # Corresponding indices in `valid_points`

        fig, axes = plt.subplots(
            1, 3, figsize=(15, 5)
        )  # Create a subplot for each direction

        for i, (direction, index) in enumerate(zip(directions, indices)):
            # Filter points where the other two coordinates are fixed
            filtered_points = self.valid_points[
                (self.valid_points[:, (index + 1) % 3] == [y, z][index != 0])
                & (self.valid_points[:, (index + 2) % 3] == [y, z][index == 0])
            ]

            # Sort by the coordinate of interest
            sorted_points = filtered_points[np.argsort(filtered_points[:, index])]

            if sorted_points.shape[0] == 0:
                print(f"No valid points found in {direction}-direction.")
                continue

            # Extract the coordinate values and corresponding voltage
            coord_values = sorted_points[:, index]
            V_values = np.array(
                [self.get_total_voltage_at_point(*point) for point in sorted_points]
            )

            # Plot the results
            axes[i].scatter(coord_values, V_values, color="red", label="Data Points")
            axes[i].plot(
                coord_values,
                V_values,
                color="blue",
                linestyle="-",
                alpha=0.7,
                label="Interpolated Curve",
            )
            axes[i].axvline(
                [x, y, z][index], color="green", linestyle="--", label=f"Point at {x, y, z}"
            )
            axes[i].set_xlabel(f"{direction} Coordinate")
            axes[i].set_ylabel("Voltage (V)")
            axes[i].set_title(f"Potential vs. {direction}-coordinate")
            axes[i].legend()
            axes[i].grid()

        plt.tight_layout()
        plt.show()


test_sim = Simulation("Simplified1")
test_sim.set_variables("RF12", [0, 0, 0, 0])
test_sim.set_variables("RF1", [1, 27000000, 0, 0])

print(test_sim.get_total_voltage_at_point(0, 0, 0))
deriv = test_sim.get_voltage_second_derivative_at_point(0, 0, 0, plot_fit=False)
print("deriv", deriv)

#test_sim.plot_potential_in_principal_directions(0, 0, 0)
print(test_sim.calcualte_frequencys(0, 0, 0))
