"""
This file will contain the main base of the simulation class
"""

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
from SimulationMix.simulation_Umin import U_energy
from multiprocessing import Pool, cpu_count
from joblib import Parallel, delayed
from scipy.interpolate import griddata
import numexpr as ne
import electrode_vars as evars
from scipy.optimize import minimize, BFGS, basinhopping
from scipy.optimize import differential_evolution
from scipy.interpolate import RBFInterpolator


class Simulation(sim_ploting, sim_normalfitting, U_energy):
    def __init__(self, dataset, variables=evars.Electrode_vars()):
        """
        Initialize the simulation object by making sure the data is all extracted and finding total voltages if given electodes
        
        dataset: the string of the desired data folder
        varaibles: and instance of the Electrode_vars class containing the desired electrode variables
        """
        timesimstart = time.time()
        print("initializing simulation")
        self.dataset = dataset
        self.file_path = "C:\\GitHub\\TrapFrequencyAnalysis\\Data\\" + dataset + "\\"

        self.electrode_vars = variables

        self.total_voltage_df = None

        # get the dataframe if it exists, otherwise make it
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

        # get the total voltage at each point based on the electrode variables
        self.update_total_voltage()

        # vestigial i think???? cause its turned into {} on line 68
        self.ion_equilibrium_positions = {
            i: tuple([[np.inf, np.inf, np.inf] for _ in range(i)])
            for i in range(1, constants.max_ion_in_chain + 1)
        }

        # init other variables
        self.center_fit = None  # Placeholder for the center fit model
        self.ion_equilibrium_positions = {}
        self.ion_eigenvectors = {}
        self.ion_eigenvalues = {}

        timesimstop = time.time()
        print(
            "simulation initialized in " + str(timesimstop - timesimstart) + " seconds"
        )

    def get_elec_variables(self, electrode):
        '''
        Returns the electrode variables used by this simulation for a given electrode.
        '''
        return self.electrode_vars.get_vars(electrode)

    def update_total_voltage(self):
        """Update the total voltage DataFrame by forming
        an equation based on the electorde_varaibles and
        applying to the column "TotalV" in the dataframe.
        
        yes this implementation seems jank but its quick
        again this is place where errors will happen if the geometry evolves
        """

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
                electro
                + "_V * "
                + str(self.electrode_vars.get_DCoffset(electro))
                + " + "
            )
            eval_ex_str += (
                electro
                + "_Ex * "
                + str(self.electrode_vars.get_RFamplitude(electro))
                + " + "
            )
            eval_ey_str += (
                electro
                + "_Ey * "
                + str(self.electrode_vars.get_RFamplitude(electro))
                + " + "
            )
            eval_ez_str += (
                electro
                + "_Ez * "
                + str(self.electrode_vars.get_RFamplitude(electro))
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
            + str(self.electrode_vars.get_RFfrequency("RF1") ** 2)
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
        """
        Change the electrode variables of the simulation to new_vars and update the total voltage DataFrame.
        """
        self.electrode_vars = new_vars
        self.update_total_voltage()

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
        print(len(cutout_of_df), " points found in cutout")
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
        print("R-squared of the fit:", r2)

        # Get derivatives (d/dx, d/dy, d/dz) at point
        derivatives = model.coef_[1:4]

        # find the value of the fit at the origin
        Vvalue_at_point = model.predict(poly.transform([[0, 0, 0]]))
        print("Time taken to find V at point: ", t2 - t1)

        if derivs:
            # Return the derivatives at the point
            return derivatives
        return Vvalue_at_point[0]
        # Calculate the potential energy of the ions

    # Not writen
    def get_ion_mins(self):
        return

    # Unknown (old?)
    def get_U_min(self, number_of_ions: int):
        """
        Minimizes the total potential energy (trap potential + Coulomb)
        for 'number_of_ions' ions.
        """

        def U_func(positions_flat):
            n = len(positions_flat) // 3
            positions = positions_flat.reshape((n, 3))

            U_trap = 0.0
            U_coulomb = 0.0

            for i in range(n):
                x, y, z = positions[i]
                U_trap += self.find_V_trap_at_point(x, y, z) * constants.ion_charge

            epsilon = 1e-12  # small buffer to avoid division-by-zero issues
            for i in range(n):
                for j in range(i + 1, n):
                    dist = np.linalg.norm(positions[i] - positions[j])
                    dist = max(dist, epsilon)
                    U_coulomb += (
                        constants.coulomb_constant * constants.ion_charge**2
                    ) / dist

            total_energy = (
                U_trap + U_coulomb
            ) * 1e25  # scaling for numerical stability
            return total_energy

        def grad_U_func(positions_flat):
            n = len(positions_flat) // 3
            positions = positions_flat.reshape((n, 3))
            grad = np.zeros_like(positions)

            # Coulomb term
            for i in range(n):
                for j in range(i + 1, n):
                    r_i = positions[i]
                    r_j = positions[j]
                    diff = r_i - r_j
                    dist = np.linalg.norm(diff) + 1e-12  # to prevent division by 0

                    force = (
                        (constants.coulomb_constant * constants.ion_charge**2)
                        * diff
                        / dist**3
                    )

                    grad[i] += force  # ∂U/∂r_i
                    grad[j] -= force  # ∂U/∂r_j, opposite direction

            # Trap term (assuming you can get ∂V/∂x, ∂V/∂y, ∂V/∂z)
            for i in range(n):
                x, y, z = positions[i]

                # estimate gradient of trap potential
                dx = 1e-7  # small finite difference
                # dV_dx = (self.find_V_trap_at_point(x + dx*30, y, z) - self.find_V_trap_at_point(x - dx*30, y, z)) / (2 * dx*30)
                # dV_dy = (self.find_V_trap_at_point(x, y + dx, z) - self.find_V_trap_at_point(x, y - dx, z)) / (2 * dx)
                # dV_dz = (self.find_V_trap_at_point(x, y, z + dx) - self.find_V_trap_at_point(x, y, z - dx)) / (2 * dx)
                # print(np.array([dV_dx, dV_dy, dV_dz]))

                # use the reg gucntion
                dV_dx, dV_dy, dV_dz = self.find_V_trap_at_point(x, y, z, derivs=True)

                grad[i] += constants.ion_charge * np.array([dV_dx, dV_dy, dV_dz])

            return grad.flatten() * 1e25  # match scaling of U_func

        init_guess_flat = np.array(
            constants.ion_locations_intial_guess[number_of_ions]
        ).flatten()

        bounds = constants.ion_locations_bounds[number_of_ions]

        result = minimize(
            U_func,
            init_guess_flat,
            method="L-BFGS-B",
            jac=grad_U_func,  # <-- pass the gradient
            bounds=bounds,
            options={"eps": 1e-6, "disp": False, "maxiter": 10000, "maxfun": 100000},
        )

        if not result.success:
            print("Minimization failed:", result.message)
        else:
            print("Minimization successful!")
            print("Final potential energy:", result.fun / 1e20)

        # return just the minimized positions
        return result.x.reshape((number_of_ions, 3))

    # Unknown (old?)
    def get_U_min_new(self, number_of_ions: int):
        """
        Minimizes the total potential energy (trap + Coulomb) for 'number_of_ions' ions
        using:
        1) coordinate rescaling in y,z to reduce stiffness,
        2) offsetting the potential so the initial guess is U=0,
        3) basinhopping for global search, with L-BFGS-B for local minimization.
        """

        # --
        # 1. Choose a rescaling factor for y, z
        #    This helps the optimizer handle extremely stiff Y/Z directions.
        scale_yz = 1e-1

        def scale_positions(x_flat):
            """Take an array of shape (3*n) and scale y,z coords."""
            x = x_flat.copy().reshape((-1, 3))
            x[:, 1] *= scale_yz
            x[:, 2] *= scale_yz
            return x

        def unscale_positions(x_flat_scaled):
            """Inverse of scale_positions."""
            x = x_flat_scaled.copy().reshape((-1, 3))
            x[:, 1] /= scale_yz
            x[:, 2] /= scale_yz
            return x

        # ---------------------------------------------------
        # 2. Define the total potential in unscaled space
        #    We'll shift it so that the initial guess is at U=0.
        def total_potential_unscaled(positions_flat):
            n = len(positions_flat) // 3
            positions = positions_flat.reshape((n, 3))

            U_trap = 0.0
            U_coulomb = 0.0

            # Trap potential
            for i in range(n):
                x, y, z = positions[i]
                U_trap += self.find_V_trap_at_point(x, y, z) * constants.ion_charge

            # Coulomb potential
            epsilon = 1e-12
            for i in range(n):
                for j in range(i + 1, n):
                    dist = np.linalg.norm(positions[i] - positions[j])
                    dist = max(dist, epsilon)
                    U_coulomb += (
                        constants.coulomb_constant * constants.ion_charge**2
                    ) / dist

            return (U_trap + U_coulomb) * 1e25  # scale for numeric stability

        # ---------------------------------------------------
        # 3. Build the scaled + offset version of the potential
        #    We'll store the potential at the initial guess, subtract it so at guess => 0.
        init_guess = np.array(constants.ion_locations_intial_guess[number_of_ions])
        init_guess_flat = init_guess.flatten()
        init_guess_scaled_flat = scale_positions(init_guess_flat).ravel()

        U0 = total_potential_unscaled(init_guess_flat)  # offset

        def U_func_scaled(x_flat_scaled):
            # Unscale back to physical coords
            x_flat = unscale_positions(x_flat_scaled).ravel()
            return total_potential_unscaled(x_flat) - U0  # shift so init guess => 0

        def grad_U_func_scaled(x_flat_scaled):
            """
            Analytical gradient w.r.t. scaled coords
            = dU/dx_scaled. We'll do chain rule:
            dU/dx_scaled = dU/dx_unscaled * dx_unscaled/dx_scaled.
            """
            x_flat = unscale_positions(x_flat_scaled).ravel()
            n = len(x_flat) // 3
            positions = x_flat.reshape((n, 3))
            grad_unscaled = np.zeros_like(positions)

            # Coulomb gradient
            for i in range(n):
                for j in range(i + 1, n):
                    diff = positions[i] - positions[j]
                    dist = np.linalg.norm(diff) + 1e-12
                    force = (
                        (constants.coulomb_constant * constants.ion_charge**2)
                        * diff
                        / dist**3
                    )
                    grad_unscaled[i] += force
                    grad_unscaled[j] -= force

            # Trap gradient
            # Use your direct derivative approach if available
            # (here let's just do small finite differences on trap, or use your find_V_trap_at_point(derivs=True))
            dx = 1e-7
            for i in range(n):
                x, y, z = positions[i]
                # If you have a direct derivative method, call it. E.g.:
                dV_dx, dV_dy, dV_dz = self.find_V_trap_at_point(x, y, z, derivs=True)

                grad_unscaled[i] += constants.ion_charge * np.array(
                    [dV_dx, dV_dy, dV_dz]
                )

            # Scale by 1e25 to match the total potential scaling
            grad_unscaled *= 1e25

            # Now correct for scaling of y,z
            # x_scaled = [x, y*scale_yz, z*scale_yz]
            # => dx/dx_scaled = 1 for x
            # => dy/dy_scaled = scale_yz
            # => dz/dz_scaled = scale_yz
            # So dU/dx_scaled[0] = dU/dx_unscaled[0]
            #    dU/dx_scaled[1] = dU/dx_unscaled[1] * scale_yz
            # etc.
            grad_unscaled[:, 1] *= scale_yz
            grad_unscaled[:, 2] *= scale_yz

            return grad_unscaled.ravel()

        # ---------------------------------------------------
        # 4. Define BOUNDS in scaled space
        raw_bounds = constants.ion_locations_bounds[number_of_ions]
        scaled_bounds = []
        for (xmin, xmax), (ymin, ymax), (zmin, zmax) in raw_bounds:
            scaled_bounds.append((xmin, xmax))  # x
            scaled_bounds.append((ymin * scale_yz, ymax * scale_yz))  # y
            scaled_bounds.append((zmin * scale_yz, zmax * scale_yz))  # z

        # ---------------------------------------------------
        # 5. Set up local minimizer (L-BFGS-B)
        minimizer_kwargs = {
            "method": "L-BFGS-B",
            "jac": grad_U_func_scaled,
            "bounds": scaled_bounds,
            "options": {"disp": False, "maxiter": 10000, "maxfun": 100000, "eps": 1e-7},
        }

        # ---------------------------------------------------
        # 6. Basinhopping (global + local approach)
        #    We'll do multiple basins to try to find global minimum.
        #    You can tweak niter for how many random steps to take.
        niter = 10

        # We'll define a trivial "take_step" or just use the default.
        # The default random step size might be a bit large, so let's define our own:
        class ScaledRandomStep(object):
            def __init__(self, step_size=1e-7):
                self.step_size = step_size

            def __call__(self, x):
                # random step in scaled coords
                return x + np.random.uniform(
                    -self.step_size, self.step_size, size=x.shape
                )

        bh = basinhopping(
            U_func_scaled,
            init_guess_scaled_flat,
            minimizer_kwargs=minimizer_kwargs,
            niter=niter,
            stepsize=5e-7,  # or pass in a custom "take_step"
            disp=True,
        )

        if not bh.lowest_optimization_result.success:
            print(
                "Minimization (basinhopping) failed:",
                bh.lowest_optimization_result.message,
            )
        else:
            print("Minimization successful!")
            print(
                "Final potential energy (shifted) =", bh.lowest_optimization_result.fun
            )
            # unshifted = ...
            # If you want the absolute potential, just add U0 back:
            # absolute_final_energy = bh.lowest_optimization_result.fun + U0

        # ---------------------------------------------------
        # 7. Unscale the final best positions for physical coordinates
        best_x_scaled = bh.lowest_optimization_result.x
        best_positions = unscale_positions(best_x_scaled)

        print("Final positions (m):\n", best_positions)

        return best_positions

    # Unknown (old?)
    def get_smooth_Ufunc_and_UjaxFunc(self, number_of_ions: int, elec_v_model=None):
        """ """
        if elec_v_model is None:
            elec_v_model = self.get_voltage_poly_at_center(polyfit=4, max_pnts=1e6)[0]

        def U_func(positions_flat):
            n = len(positions_flat) // 3
            positions = positions_flat.reshape((n, 3))

            U_trap = 0.0
            U_coulomb = 0.0

            for i in range(n):
                x, y, z = positions[i]
                U_trap += self.find_V_trap_at_point(x, y, z) * constants.ion_charge

            epsilon = 1e-12  # small buffer to avoid division-by-zero issues
            for i in range(n):
                for j in range(i + 1, n):
                    dist = np.linalg.norm(positions[i] - positions[j])
                    dist = max(dist, epsilon)
                    U_coulomb += (
                        constants.coulomb_constant * constants.ion_charge**2
                    ) / dist

            total_energy = (
                U_trap + U_coulomb
            ) * 1e25  # scaling for numerical stability
            return total_energy

        return U_func

    # Unknown (old?)
    def get_U_min_using_polyfit(self, number_of_ions: int):
        """
        This fucnction will minimize the total_U fucntion returned by polu
        """
        pass

    # Important
    def get_voltage_poly_at_center(
        self, lookaround_x=20, lookaround_yz=5, polyfit=4, max_pnts=1e6
    ):
        return self.get_voltage_poly_at_region(
            x_low=-lookaround_x,
            x_high=lookaround_x,
            y_low=-lookaround_yz,
            y_high=lookaround_yz,
            z_low=-lookaround_yz,
            z_high=lookaround_yz,
            max_pnts=max_pnts,
            polyfit=polyfit,
        )

    # Important
    def get_voltage_poly_at_region(
        self,
        x_low=-10,
        x_high=10,
        y_low=-10,
        y_high=10,
        z_low=-10,
        z_high=10,
        max_pnts=1e6,
        polyfit=4,
    ):

        # get initial maxcutout
        cutout_of_df = self.total_voltage_df[
            (self.total_voltage_df["x"].between((x_low * 1e-6), (x_high * 1e-6)))
            & (self.total_voltage_df["y"].between(y_low * 1e-6, y_high * 1e-6))
            & (self.total_voltage_df["z"].between(z_low * 1e-6, z_high * 1e-6))
        ]

        # if len(cutout_of_df) > max_pnts then randomply sample max_pnts points
        if len(cutout_of_df) > max_pnts:
            cutout_of_df = cutout_of_df.sample(n=max_pnts, random_state=1)
        print("len(cutout_of_df): ", len(cutout_of_df))

        voltage_vals = cutout_of_df["TotalV"].values
        print(len(cutout_of_df), " points found in cutout")
        # print("voltage_vals: ", voltage_vals)
        xyz_vals = cutout_of_df[["x", "y", "z"]].values

        poly = PolynomialFeatures(degree=polyfit, include_bias=True)
        X_poly = poly.fit_transform(xyz_vals)

        # Fit the model
        model = LinearRegression()
        model.fit(X_poly, voltage_vals)

        # find and print out the r2 of the fit
        r2 = model.score(X_poly, voltage_vals)
        if r2 < 0.999:
            print(
                "Warning: Low R-squared value for polynomial fit done by get_voltage_poly_at_region"
            )
            print("R-squared of the fit:", r2)

        # return the polynomial
        return model, poly, r2

    # Single Ion Analysis
    def get_principal_freq_at_min(
        self, getall=False, fitdeg=4, look_around=5, return_coefs=False
    ):
        """
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
        """
        min1, min2 = self.find_V_min()
        if return_coefs or getall:
            eigenfreq, axialfreq, eigendir, coefs = self.get_freqs_at_point_withR3_fit(
                min1[0],
                min1[1],
                min1[2],
                look_around=look_around,
                polyfit=fitdeg,
                return_coefs=True,
            )
        else:
            eigenfreq, axialfreq, eigendir = self.get_freqs_at_point_withR3_fit(
                min1[0],
                min1[1],
                min1[2],
                look_around=look_around,
                polyfit=fitdeg,
                return_coefs=False,
            )

        # Convert eigenfreq to a numpy array for sorting
        eigenfreq = np.array(eigenfreq)

        # Sort eigenfreq in increasing order and eigendir accordingly
        sorted_indices = np.argsort(eigenfreq)
        eigenfreq = eigenfreq[sorted_indices].tolist()  # Convert back to list if needed
        eigendir = eigendir[
            :, sorted_indices
        ]  # Sort eigendir columns based on sorted indices

        # Convert eigenvectors into a readable format
        eigendir_readable = {
            f"Direction {i+1}": f"({vec[0]:.3f}, {vec[1]:.3f}, {vec[2]:.3f})"
            for i, vec in enumerate(
                eigendir.T
            )  # Transpose so each column is an eigenvector
        }
        if getall:
            return eigenfreq, eigendir_readable, min1, min2, coefs

        return eigenfreq, eigendir_readable


# print("hi")
# t1 = time.time()

# evaribs = evars.get_electrodvars_w_twist(377, 25500000 * 2 * math.pi, -0.275, 5)
# evaribs.set_frequency("DC3", 80_000)
# test_sim = Simulation(
#     "Simp58_101", evaribs
# )
# test_sim.evaluate_center_poly(0, 0, 0)
# test_sim.find_equilib_positions()
# res = test_sim.ion_equilibrium_positions

# for num_ions in range(1, constants.max_ion_in_chain + 1):
#     # Get equilibrium positions

#     # Get Hessian, 3rd, and 4th tensors
#     hessian = test_sim.get_eq_U_hessian(num_ions)
#     third_tensor = test_sim.get_eq_3rd_der_tensor(num_ions)
#     fourth_tensor = test_sim.get_eq_4th_der_tensor(num_ions)

#     # Print the results
#     print(f"Number of ions: {num_ions}")

# for num_ions in range(1, constants.max_ion_in_chain + 1):
#     test_sim.get_mode_eigenvec_and_val(num_ions)

# eigenval, eigenvec = test_sim.get_mode_eigenvec_and_val(1)
# print(test_sim.get_principal_freq_at_min())
# for val in eigenval:
#     print(val)
#     print(math.sqrt(val / constants.ion_mass) / (2 * math.pi))
# # print(test_sim.ion_eigenvectors)
# for i in range(1,1+1):
#     test_sim.get_3_wise_mode_couplings(i)
#     test_sim.get_4_wise_mode_couplings(i)


# print(test_sim.get_2_wise_mode_couplings(1))
# t2 = time.time()

# # for key in res:
# #     print(key)
# #     print(res[key])


# # print(
# #     test_sim.get_U_using_polyfit_dimensionless(
# #         np.array([[-1e-6,0,0],[1e-6,0,0]]).flatten()
# #     )
# # )
# # print(
# #     test_sim.get_U_using_polyfit_dimensionless(
# #         np.array([[-2e-6, 0, 0], [2e-6, 0, 0]]).flatten()
# #     )
# # )
# # print(
# #     test_sim.get_U_using_polyfit_dimensionless(
# #         np.array([[-4e-6, 0, 0], [4e-6, 0, 0]]).flatten()
# #     )
# # )
# print("Time taken: ", t2 - t1)
