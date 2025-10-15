"""
This file will contain the class Mode_Calculations
"""

import math
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import r2_score, mean_squared_error
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import RBFInterpolator
import constants
from scipy.optimize import minimize, BFGS, basinhopping


class StaticNormalModes_EigenMixin:
    """
    This class will be inherited by the simulation class and will contain all fucntions used to find modes and their coupling.
    """

    # look over
    def get_mode_eigenvec_and_val(self, num_ions, sort_by="FreqAbs"):
        """
        This function will return the eigenvalues and eigenvectors of the hessian matrix, in the desired order.
        """
        # get hessian
        hessian = self.get_eq_U_hessian(num_ions)

        # diagonalize hessian
        eigvals, eigvecs = np.linalg.eigh(hessian)
        print(len(eigvals))

        # sort eigenvalues and eigenvectors
        if sort_by == "FreqAbs":
            # sort by absolute value of eigenvalues
            indices = np.argsort(np.abs(eigvals))
        else:
            # sort by eigenvalues
            indices = np.argsort(eigvals)

        # sort eigenvalues and eigenvectors
        sorted_eigvals = eigvals[indices]
        sorted_eigvecs = eigvecs[:, indices]
        self.ion_eigenvectors[num_ions] = sorted_eigvecs
        self.ion_eigenvalues[num_ions] = sorted_eigvals
        print("stored", num_ions)
        return sorted_eigvals, sorted_eigvecs

    def get_freq_from_eigenvalue(self, val):
        return math.sqrt(abs(val) / constants.ion_mass) / (2 * math.pi)

    def get_normal_modes_and_freqs(self, num_ions):
        """
        This function will return the normal modes and their frequencies.
        """
        if self.ion_eigenvectors[num_ions] is None or self.ion_eigenvalues[num_ions] is None:
            self.get_mode_eigenvec_and_val(num_ions)
        eigvals = self.ion_eigenvalues[num_ions]
        eigvecs = self.ion_eigenvectors[num_ions]
        freqs = np.array([self.get_freq_from_eigenvalue(val) for val in eigvals])
        return list(zip(freqs, eigvecs.T))
    
    # Two types of normalized? No thats nonsense lol
    def normalize_normal_modes(self, num_ions):
        """
        This function will normalize the normal modes.
        They should pairwise be orthogonal and have norm 1.
        """
        return None