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

        self.populate_normalmodes_in_prinipledir_freq_labels()

        return out

    def compute_principal_directions_from_one_ion(self):
        """
        Determine principal directions from the 1-ion normal modes.

        Behavior:
        - If self.normal_modes_and_frequencies lacks an entry for 1 ion, we compute it.
        - Extract the 3 mode vectors for the single ion (columns of the (3x3) 'modes' matrix),
        which are already sorted by increasing frequency by get_static_normal_modes_and_freq.
        - Store them as 3 unit vectors in self.principal_dirs = [[x1,y1,z1], ...],
        ordered lowest->highest frequency (index 0 is lowest).
        """
        import numpy as np

        # Ensure we have a 1-ion entry
        if (
            1 not in self.normal_modes_and_frequencies
        ):
            self.get_static_normal_modes_and_freq(1)

        modes_1 = self.normal_modes_and_frequencies[1][
            "modes"
        ]  # shape (3,3); columns = mode vectors
        # Defensive: ensure shape is (3,3)
        if modes_1.shape != (3, 3):
            raise ValueError(f"Expected 3x3 modes matrix for one ion, got {modes_1.shape}")

        # Columns are already sorted by freq (ascending) by get_static_normal_modes_and_freq
        P = []
        for j in range(3):
            v = modes_1[:, j].astype(float)
            nrm = np.linalg.norm(v)
            if nrm > 0:
                v = v / nrm
            P.append([float(v[0]), float(v[1]), float(v[2])])

        # Use the attribute name already present in Simulation.__init__
        self.principal_dirs = P  # [[x,y,z], [..], [..]]
        return self.principal_dirs

    def populate_normalmodes_in_prinipledir_freq_labels(self):
        """
        Build a per-ion-count dict like self.normal_modes_and_frequencies, but with
        eigenvectors expressed in the principal-direction basis defined by the 1-ion case.
        Also attach a per-mode label telling which principal axis the mode aligns with most.

        Produces:
        self.normalmodes_in_prinipledir_freq_labels[num_ions] = {
            'modes'          : (3N x 3N) ndarray  (columns are modes in principal basis),
            'frequencies_Hz' : (3N,) ndarray,
            'dir_alignment'  : (3N,) int ndarray with values in {0,1,2}  (0 = lowest-freq principal axis)
        }

        For compatibility with your existing naming, we also mirror into:
        self.principal_dir_normalmodes_andfrequencies[num_ions] = { ...same content... }
        """
        import numpy as np

        # Make sure principal directions exist
        if self.principal_dirs == [[1,0,0],[0,1,0],[0,0,1]]:
            self.compute_principal_directions_from_one_ion()

        # Principal direction matrix P has columns = principal axes (lab components)
        P = np.array(self.principal_dirs, dtype=float).T  # shape (3,3)
        # Guard/renormalize columns
        for k in range(3):
            c = P[:, k]
            nrm = np.linalg.norm(c)
            if nrm > 0:
                P[:, k] = c / nrm

        # For each ion-count already computed, rotate modes blockwise (each ion's x,y,z) into the principal basis
        for num_ions, pack in self.normal_modes_and_frequencies.items():
            vecs_lab = np.array(
                pack["modes"], dtype=float, copy=True
            )  # (3N,3N), columns are modes (lab-basis)
            freqs = np.array(pack["frequencies_Hz"], dtype=float, copy=True)  # (3N,)

            # check 3N3n
            if vecs_lab.shape != (3 * num_ions, 3 * num_ions):
                raise ValueError(
                    f"Expected modes shape {(3*num_ions, 3*num_ions)}; got {vecs_lab.shape}"
                )

            # Rotate each (x,y,z) block: principal_block = P^T * v_lab_block
            vecs_principal = np.zeros_like(vecs_lab)
            for ion in range(num_ions):
                sl = slice(3 * ion, 3 * ion + 3)
                vecs_principal[sl, :] = P.T @ vecs_lab[sl, :]

            # For each (column), find most aligned  principal axis
            # sum squares across ions per axis and take argmax
            dir_alignment = np.zeros(3 * num_ions, dtype=int)
            for j in range(3 * num_ions):
                # block components are [x'(=axis0), y'(=axis1), z'(=axis2)] per ion
                w = np.zeros(3)
                for ion in range(num_ions):
                    sl = slice(3 * ion, 3 * ion + 3)
                    comp = vecs_principal[sl, j]
                    w += comp * comp
                dir_alignment[j] = int(
                    np.argmax(w)
                )  # 0 = lowest-freq axis)

            out = {
                "modes": vecs_principal,
                "frequencies_Hz": freqs,
                "dir_alignment": dir_alignment,
            }

            self.principal_dir_normalmodes_andfrequencies[num_ions] = out

        return self.principal_dir_normalmodes_andfrequencies
