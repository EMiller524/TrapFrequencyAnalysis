import math
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import constants

import numpy as np


class sim_hessian:
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

        # print("Hessian at: " + str((x, y, z)) + str(H))
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

    def get_freq_in_given_dir_at_point(self, x, y, z, direction = (1, 0, 0)):
        # direction is a unit vector in from of a tuple (x, y, z)
        # normalize just in case
        # get the hessian at that point
        # calculate the frequency in the direction of the unit vector by LH multiply by transpose of unit vector and RH multiply by the unit vector
        #return the frequency in that direction
        
        direction = np.array(direction)
        direction = direction / np.linalg.norm(direction)
        
        H = self.get_hessian_at_point(x, y, z)
        # RH mult H with the column vetor of direction
        intermediary = np.dot(H, direction)
        
        # LH mult the result with the row vector of direction
        seconderivative = np.dot(intermediary, direction)
        
        freq = math.sqrt((constants.ion_charge / constants.ion_mass) * abs(seconderivative)) / (2 * math.pi)
        
        return freq
    
    def get_freqs_in_given_dir_at_point(self, x, y, z, directions = ((1, 0, 0), (0, 1, 0), (0, 0, 1))):
        new_x_dir = np.array(directions[0])
        new_y_dir = np.array(directions[1])
        new_z_dir = np.array(directions[2])
        
        new_x_dir = new_x_dir / np.linalg.norm(new_x_dir)
        new_y_dir = new_y_dir / np.linalg.norm(new_y_dir)
        new_z_dir = new_z_dir / np.linalg.norm(new_z_dir)
        
        # if directions are not orthogonal, then the directions are not valid
        if np.dot(new_x_dir, new_y_dir) != 0 or np.dot(new_x_dir, new_z_dir) != 0 or np.dot(new_y_dir, new_z_dir) != 0:
            print("Directions are not orthogonal")
            return None
        
        #get the freqs in the directions
        freq_x = self.get_freq_in_given_dir_at_point(x, y, z, new_x_dir)
        freq_y = self.get_freq_in_given_dir_at_point(x, y, z, new_y_dir)
        freq_z = self.get_freq_in_given_dir_at_point(x, y, z, new_z_dir)
        
        return (freq_x, freq_y, freq_z)
        
        


    def get_princ_frequencys_at_point_hess(self, x, y, z):
        # call get_hessian_at_point and then diagonalize the hessian
        # calculate the frequencys from the diagonalized hessian and the principlae directions of the hessian
        # return the frequencys and the principlae directions

        eigenvalues, eigenvectors = self.diagonalize_hessian(
            self.get_hessian_at_point(x, y, z)
        )
        Q = constants.ion_charge
        M = constants.ion_mass

        frequencys_and_directions = []
        # Calculate frequencies
        for i in range(3):
            frequency = math.sqrt((Q / M) * abs(eigenvalues[i])) / (math.pi * 2)
            direction = eigenvectors[i]
            frequencys_and_directions.append((frequency, direction))

        return frequencys_and_directions
