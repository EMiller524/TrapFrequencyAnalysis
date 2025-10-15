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


class StaticCoupolingMixin:
    """
    This class will be inherited by the simulation class and will contain all fucntions used to find modes and their coupling.
    """

    # look over
    def get_3_wise_mode_couplings(self, num_ions):
        """
        This will be done by contracting the 3rd derivative tensor with the eigenvectors.
        """
        tensor = self.get_eq_3rd_der_tensor(num_ions)
        three_wise_couplings = {}

        for modei in range(0, 3 * num_ions):
            eigveci = self.ion_eigenvectors[num_ions][:,modei]
            coupling_i = np.tensordot(tensor, eigveci, axes=([0], [0]))
            for modej in range(modei, 3 * num_ions):
                eigvecj = self.ion_eigenvectors[num_ions][:,modej]
                coupling_ij = np.tensordot(coupling_i, eigvecj, axes=([0], [0]))
                for modek in range(modej, 3 * num_ions):
                    eigveck = self.ion_eigenvectors[num_ions][:,modek]
                    coupling_ijk = np.tensordot(coupling_ij, eigveck, axes=([0], [0]))

                    three_wise_couplings[(modei, modej, modek)] = coupling_ijk
                # contract the tensor with the eigenvector
        return three_wise_couplings

    # look over
    def get_4_wise_mode_couplings(self, num_ions):
        """
        This will be done by contracting the 4th derivative tensor with the eigenvectors.
        """
        tensor = self.get_eq_4th_der_tensor(num_ions)
        four_wise_couplings = {}

        for modei in range(0, 3 * num_ions):
            eigveci = self.ion_eigenvectors[num_ions][:,modei]
            coupling_i = np.tensordot(tensor, eigveci, axes=([0], [0]))
            for modej in range(modei, 3 * num_ions):
                eigvecj = self.ion_eigenvectors[num_ions][:,modej]
                coupling_ij = np.tensordot(coupling_i, eigvecj, axes=([0], [0]))
                for modek in range(modej, 3 * num_ions):
                    eigveck = self.ion_eigenvectors[num_ions][:,modek]
                    coupling_ijk = np.tensordot(coupling_ij, eigveck, axes=([0], [0]))
                    for model in range(modek, 3 * num_ions):
                        eigvecl = self.ion_eigenvectors[num_ions][:,model]
                        coupling_ijkl = np.tensordot(
                            coupling_ijk, eigvecl, axes=([0], [0])
                        )

                        four_wise_couplings[(modei, modej, modek, model)] = (
                            coupling_ijkl
                        )

                # contract the tensor with the eigenvector
        return four_wise_couplings
