import math
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import r2_score, mean_squared_error
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import RBFInterpolator


class sim_ploting:

    def plot_2d_color_contour_Vraw(self, point, axis):
        # plots the raw total voltage in the plane defined by point, axis
        # from self.total_voltage_df as a 2d contour plot

        # get the data in which axis = point (i.e. if axis = 'x', then we want all points where x = point)
        df = self.total_voltage_df.copy()
        if axis == "x":
            filtered_df = df[df["x"] == point]
            axis1_label = "y (m)"
            axis2_label = "z (m)"
            axis1_data = filtered_df["y"].values
            axis2_data = filtered_df["z"].values
            v_data = filtered_df["CalcV"].values

        elif axis == "y":
            filtered_df = df[df["y"] == point]
            axis1_label = "x (m)"
            axis2_label = "z (m)"
            axis1_data = filtered_df["x"].values
            axis2_data = filtered_df["z"].values
            v_data = filtered_df["CalcV"].values
        elif axis == "z":
            filtered_df = df[df["z"] == point]
            axis1_label = "x (m)"
            axis2_label = "y (m)"
            axis1_data = filtered_df["x"].values
            axis2_data = filtered_df["y"].values
            v_data = filtered_df["CalcV"].values

        # Define grid for interpolation
        xi = np.linspace(axis1_data.min(), axis1_data.max(), 1000)
        yi = np.linspace(axis2_data.min(), axis2_data.max(), 1000)
        X, Y = np.meshgrid(xi, yi)

        # Interpolate z values on the grid
        Z = griddata((axis1_data, axis2_data), v_data, (X, Y), method='nearest')  # Use 'linear' or 'nearest' if needed

        # Plot contour
        plt.figure(figsize=(8, 6))
        contour_filled = plt.contourf(X, Y, Z, levels=50, cmap='viridis')  # Color map
        plt.colorbar(contour_filled)

        # Add contour lines with solid black lines
        log_levels = np.logspace(np.log10(Z.min()+1e-12), np.log10(Z.max()), num=20)  # Avoid log(0)
        contour_lines = plt.contour(X, Y, Z, levels=log_levels, colors='black', linewidths=.5)
        plt.clabel(contour_lines, inline=True, fontsize=8)  # Add labels to contour lines

        # Scatter original points
        plt.scatter(axis1_data, axis2_data, c='black', s=.1)

        # Labels and title
        plt.xlabel(axis1_label)
        plt.ylabel(axis2_label)
        plt.title('2D Contour/Color Plot of TotalVoltage vs Space')

        # Show plot
        plt.show()

        return

    def plot_3d_surface_Vraw(self, point, axis):
        # todo
        return


    def plot_3d_contour_Vraw(self):
        return
        # Copy DataFrame
        df = self.total_voltage_df.copy()
        x_data = df["x"].values
        y_data = df["y"].values
        z_data = df["z"].values
        v_data = df["CalcV"].values

        # Uniformly downsample the data (keep every Nth point)
        N = max(len(x_data) // 1000000, 10)  # Auto-tune to keep ~100K points
        x_sample = x_data[::N]
        y_sample = y_data[::N]
        z_sample = z_data[::N]
        v_sample = v_data[::N]

        print(f"Reduced dataset from {len(x_data)} to {len(x_sample)} points")

        # Define a coarser grid for interpolation
        grid_size = min(30, len(x_sample) // 1000)  # Auto-tune based on sample size
        xi = np.linspace(x_sample.min(), x_sample.max(), grid_size)
        yi = np.linspace(y_sample.min(), y_sample.max(), grid_size)
        zi = np.linspace(z_sample.min(), z_sample.max(), grid_size)

        # Use np.mgrid for memory-efficient mesh generation
        X, Y, Z = np.mgrid[
            xi.min() : xi.max() : grid_size * 1j,
            yi.min() : yi.max() : grid_size * 1j,
            zi.min() : zi.max() : grid_size * 1j,
        ]

        # Compute a reasonable epsilon (based on mean spacing between points)
        avg_spacing = np.mean([np.ptp(xi) / grid_size, np.ptp(yi) / grid_size, np.ptp(zi) / grid_size])
        epsilon = avg_spacing * 2  # Scale factor to control smoothness


        # Use RBF Interpolation (much faster than griddata)
        interp_func = RBFInterpolator(
            np.column_stack((x_sample, y_sample, z_sample)), v_sample, kernel="multiquadric", epsilon=epsilon
        )
        V_interp = interp_func(np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))).reshape(
            X.shape
        )

        # Create figure and 3D axis
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Define contour levels
        levels = np.linspace(np.nanmin(V_interp), np.nanmax(V_interp), 10)

        # Plot multiple 3D contour surfaces
        for level in levels:
            ax.contour3D(X, Y, Z, V_interp, levels=[level], cmap="viridis", alpha=0.6)

        # Scatter original points (downsampled for visualization)
        ax.scatter(
            x_sample[::100],
            y_sample[::100],
            z_sample[::100],
            c=v_sample[::100],
            cmap="inferno",
            s=5,
            label="Downsampled Data",
        )

        # Labels and title
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("3D Contour Plot with Memory Optimization")

        return fig

    def plot_3d_color_Vraw(self):
        # todo
        return

    def plot_1d_Vraw(self, axis, x=0, y=0, z=0):
        return self.plot_value_in_blank_direction(self, x, y, z, axis, "CalcV", x_cutoff=1, y_cutoff=1, z_cutoff=1)

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
                # freqs.append(
                #     self.get_freqs_in_given_dir_at_point(
                #         filtered_df.iloc[i]["x"],
                #         filtered_df.iloc[i]["y"],
                #         filtered_df.iloc[i]["z"],
                #     )[["x", "y", "z"].index(axis)]
                # )
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

    def plot_value_in_blank_direction(
        self, x, y, z, direction, value, x_cutoff=1, y_cutoff=1, z_cutoff=1
    ):
        """
        Returns a fig that plots value vs direction starting at the point(x,y,z) and moving in the direction.
        For example, if direction = "x" and value = "CalcV", then the graph will be of the potential vs x.
        """
        if self.total_voltage_df is None:
            print("Total voltage data is not available.")
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
                # get_freqs_in_given_dir_at_point
                valuess.append(
                    self.get_frequencys_at_point_xyz(
                        filtered_df.iloc[i]["x"],
                        filtered_df.iloc[i]["y"],
                        filtered_df.iloc[i]["z"],
                    )[axis_index]
                )
                # valuess.append(
                #     self.get_freqs_in_given_dir_at_point(
                #         filtered_df.iloc[i]["x"],
                #         filtered_df.iloc[i]["y"],
                #         filtered_df.iloc[i]["z"],
                #     )[axis_index]
                # )
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

                # Wy = self.get_freqs_in_given_dir_at_point(
                #     filtered_df.iloc[i]["x"],
                #     filtered_df.iloc[i]["y"],
                #     filtered_df.iloc[i]["z"],
                # )[["x", "y", "z"].index("y")]
                # Wz = self.get_freqs_in_given_dir_at_point(
                #     filtered_df.iloc[i]["x"],
                #     filtered_df.iloc[i]["y"],
                #     filtered_df.iloc[i]["z"],
                # )[["x", "y", "z"].index("z")]

                Wr = math.sqrt((Wy**2) / 2 + (Wz**2) / 2)
                valuess.append(Wr)
        else:
            return None

        ax.scatter(filtered_df[direction] * 1000, valuess, s=5, c="blue")

        if fit:
            # now fit the data to a 4th degree polynomial
            coeffs = np.polyfit(filtered_df[direction], valuess, 2)
            poly = np.poly1d(coeffs)

            # Now plot the raw data:
            ax.scatter(filtered_df[direction] * 1000, valuess, s=5, c="blue")

            # Now plot the fit as a curve not a line or spline or anything
            x_vals = np.linspace(
                np.min(filtered_df[direction]), np.max(filtered_df[direction]), num=500
            )
            fity = poly(x_vals)
            print(len(x_vals), len(fity))
            ax.plot(x_vals * 1000, fity, c="red")

            # place the fit as a smooth curve equation in purple on the plot

            # put the x^4 coeff in the legend and the r^2 and mse values
            fitted_values = poly(filtered_df[direction])
            r2 = r2_score(valuess, fitted_values)
            # fit_equation_x4 = f"{coeffs[0]:.2e}x^4"
            fit_equation_x4 = coeffs
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

    def get_main_report(self, name):
        # plots may things
        # Plot 1-3 PseudoPot in x,y,z directions (3 graphs)
        # Plot 4-6 Wx vs x Wy vs y Wz Vs z (3 graphs)
        # Plot 7 Wy vs x Wz vs x and We vs x (1 graph)
        print("Hi1")

        plot1 = self.plot_value_in_blank_direction(0, 0, 0, "x", "CalcV")

        plot21 = self.plot_value_in_blank_direction(
            0, 0, 0, "y", "CalcV", y_cutoff=0.000025
        )
        plot22 = self.plot_value_in_blank_direction(
            0, 0, 0, "z", "CalcV", z_cutoff=0.000025
        )

        plot3 = self.plot_value_in_blank_direction(0, 0, 0, "x", "Wx")

        fig41 = self.plot_value_in_blank_direction(0, 0, 0, "y", "Wy")
        print("HIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII")
        fig42 = self.plot_value_in_blank_direction(0, 0, 0, "z", "Wz")

        # plot fig41 and fig42 on the same plot
        plot4 = plt.figure()
        ax = plot4.add_subplot(111)
        ax.scatter(
            fig41.axes[0].collections[0].get_offsets()[:, 0],
            fig41.axes[0].collections[0].get_offsets()[:, 1],
            label="Wy",
            s=5,
        )
        ax.scatter(
            fig42.axes[0].collections[0].get_offsets()[:, 0],
            fig42.axes[0].collections[0].get_offsets()[:, 1],
            label="Wz",
            s=5,
        )
        ax.legend()
        ax.set_xlabel("Y, Z (mm)")
        ax.set_ylabel("Frequency in given direction")
        ax.set_title("Wy, Wz vs Y, Z")
        # ax.set_ylim(1e6, 2.5e6)

        plot5 = plt.figure()
        ax = plot5.add_subplot(111)

        # Recompute values directly instead of extracting from previous figures
        df = self.total_voltage_df.copy()
        filtered_df = df[
            (df["x"] > -1) & (df["x"] < 1) & (df["y"] == 0) & (df["z"] == 0)
        ]

        # Compute values
        Wy_values = [
            self.get_frequencys_at_point_xyz(row["x"], row["y"], row["z"])[1]
            for _, row in filtered_df.iterrows()
        ]
        Wz_values = [
            self.get_frequencys_at_point_xyz(row["x"], row["y"], row["z"])[2]
            for _, row in filtered_df.iterrows()
        ]

        # Wy_values = [
        #     self.get_freqs_in_given_dir_at_point(row["x"], row["y"], row["z"])[1]
        #     for _, row in filtered_df.iterrows()
        # ]
        # Wz_values = [
        #     self.get_freqs_in_given_dir_at_point(row["x"], row["y"], row["z"])[2]
        #     for _, row in filtered_df.iterrows()
        # ]

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

        plot101 = self.plot_value_in_blank_direction(0, 0, 0, "y", "Wx")
        plot102 = self.plot_value_in_blank_direction(0, 0, 0, "y", "Wz")

        plot104 = self.plot_value_in_blank_direction(0, 0, 0, "z", "Wx")
        plot106 = self.plot_value_in_blank_direction(0, 0, 0, "z", "Wy")

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

            pdf.savefig(plot101)
            pdf.savefig(plot102)
            pdf.savefig(plot104)
            pdf.savefig(plot106)

        print(f"Figures saved in {pdf_file}")
