import numpy as np
from scipy.interpolate import RBFInterpolator


class RBF_PotentialInterpolator:
    def __init__(self, df):
        self.points = df[["x", "y", "z"]].values
        self.values = df["CalcV"].values
        self.interpolator = RBFInterpolator(
            self.points, self.values, kernel="cubic"
        )  # Cubic RBF

    def potential(self, x, y, z):
        return self.interpolator(np.array([[x, y, z]]))[0]

    def hessian(self, x, y, z):
        """
        Computes the Hessian matrix (second derivatives) at (x, y, z) analytically.

        Args:
            x, y, z (float): Query point

        Returns:
            np.array: 3x3 Hessian matrix
        """
        hess = self.interpolator(np.array([[x, y, z]]), diff=2)
        return hess[0]  # Extract the first row as a 3x3 matrix


