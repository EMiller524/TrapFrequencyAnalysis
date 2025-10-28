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
        # print("stored", num_ions)
        return sorted_eigvals, sorted_eigvecs

    def _ensure_modes(self, num_ions: int):
        """Return (eigenvalues, eigenvectors) for num_ions; compute/store if missing."""
        vals = self.ion_eigenvalues.get(num_ions)
        vecs = self.ion_eigenvectors.get(num_ions)
        if vals is None or vecs is None:
            vals, vecs = self.get_mode_eigenvec_and_val(num_ions)  # your existing method
            self.ion_eigenvalues[num_ions] = vals
            self.ion_eigenvectors[num_ions] = vecs
        return self.ion_eigenvalues[num_ions], self.ion_eigenvectors[num_ions]

    def get_freq_from_eigenvalue(self, val):
        return math.sqrt(abs(val) / constants.ion_mass) / (2 * math.pi)

    # Goal save to self.normal_modes_and_frequencies a dict of num_ions: {(modes, frequencies}) with
    # {modes, frequencies} a dict with keys frequencies and modes the values being arrays


    def normalize_modes(self, num_ions: int, vecs=None):
        """
        Helper: column-normalize eigenvectors to unit L2 norm.
        Uses provided `vecs` (3N x 3N) if given; otherwise pulls from
        self.ion_eigenvectors[num_ions]. Returns a new normalized array.
        """
        import numpy as np

        if vecs is None:
            vecs = self.ion_eigenvectors.get(num_ions)
            if vecs is None:
                # populate from your existing routine
                _, vecs = self.get_mode_eigenvec_and_val(num_ions)

        V = np.array(vecs, dtype=float, copy=True)
        for j in range(V.shape[1]):  # columns = modes
            col = V[:, j]
            nrm = np.linalg.norm(col)
            if nrm > 0.0:
                V[:, j] = col / nrm
        # print(V)
        return V


    def eigenvalue_to_freq_hz(self, lam):
        """
        Helper: map eigenvalue(s) λ of the energy Hessian (J/m^2) to secular
        frequency(ies) in Hz: f = sqrt(λ / m) / (2π). Accepts scalar or array.
        """
        import numpy as np, math

        lam = np.asarray(lam, dtype=float)
        lam = np.abs(lam)  # guard tiny negative roundoff
        omega = np.sqrt(lam / constants.ion_mass)
        return omega / (2.0 * math.pi)


    def get_static_normal_modes_and_freq(self, num_ions: int,
                                     normalize: bool = True,
                                     sort_by_freq: bool = True):
        """
        Compute eigenpairs (via your existing get_mode_eigenvec_and_val),
        optionally sort by frequency (ascending), optionally normalize modes,
        then save to: self.normal_modes_and_frequencies[num_ions] = {
            'modes': (3N x 3N) ndarray,          # columns = eigenvectors
            'frequencies_Hz': (3N,) ndarray      # secular frequencies in Hz
        }
        Returns that dict.
        """
        import numpy as np

        if not hasattr(self, "normal_modes_and_frequencies"):
            self.normal_modes_and_frequencies = {}

        # get eigenpairs (vecs: columns are modes)
        if self.ion_eigenvalues.get(num_ions) is not None and self.ion_eigenvectors.get(num_ions) is not None:
            vals = self.ion_eigenvalues[num_ions]
            vecs = self.ion_eigenvectors[num_ions]
        else:
            vals, vecs = self.get_mode_eigenvec_and_val(num_ions)
            self.ion_eigenvalues[num_ions] = vals
            self.ion_eigenvectors[num_ions] = vecs

        # convert to Hz
        f_Hz = self.eigenvalue_to_freq_hz(vals)

        # optional sort by frequency
        if sort_by_freq:
            order = np.argsort(f_Hz)
            f_Hz = f_Hz[order]
            vecs = vecs[:, order]        # keep columns aligned with freqs
            vals = vals[order]           # keep cached eigenvalues consistent

        # optional normalization (column-wise)
        if normalize:
            vecs = self.normalize_modes(num_ions, vecs=vecs)

        # store result dict
        out = {"modes": vecs, "frequencies_Hz": f_Hz}
        self.normal_modes_and_frequencies[num_ions] = out

        return out
