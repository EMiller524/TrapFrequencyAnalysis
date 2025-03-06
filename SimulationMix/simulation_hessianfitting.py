
from ast import Constant
from asyncio import constants
import math
import numpy as np


class sim_hessian:
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
        Q = Constant.ion_charge
        M = constants.ion_mass

        frequencys_and_directions = []
        # Calculate frequencies
        for i in range(3):
            frequency = math.sqrt((Q / M) * abs(eigenvalues[i])) / (math.pi * 2)
            direction = eigenvectors[i]
            frequencys_and_directions.append((frequency, direction))

        return frequencys_and_directions
