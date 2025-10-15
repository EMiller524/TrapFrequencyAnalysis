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

    # ========== g0: static parametric coupling from a modulation drive ==========

    #TODO Check these derivatives use paper, wrote at 1am and no paper
    def _poly_hessian_at(self, model, poly, r_xyz):
        """
        Analytic Hessian (3x3) of a 3D PolynomialFeatures+LinearRegression fit,
        evaluated at r_xyz (meters). Returns d^2 Phi / d(x,y,z)^2 in V/m^2.
        """
        import numpy as np

        x, y, z = float(r_xyz[0]), float(r_xyz[1]), float(r_xyz[2])
        H = np.zeros((3, 3), dtype=float)

        # powers_: shape (n_terms, 3), rows are exponents (px, py, pz) for each monomial column
        # coef_:    shape (n_terms,), linear regression weights for those columns
        powers = poly.powers_
        print("****************************************************************************************")
        coef = model.coef_
        print(powers, coef)
        for c, (px, py, pz) in zip(coef, powers):
            if c == 0.0:
                continue

            # diagonals
            if px >= 2:
                H[0, 0] += c * px * (px - 1) * (x ** (px - 2)) * (y**py) * (z**pz)
            if py >= 2:
                H[1, 1] += c * py * (py - 1) * (x**px) * (y ** (py - 2)) * (z**pz)
            if pz >= 2:
                H[2, 2] += c * pz * (pz - 1) * (x**px) * (y**py) * (z ** (pz - 2))

            # mixed
            if px >= 1 and py >= 1:
                t = c * px * py * (x ** (px - 1)) * (y ** (py - 1)) * (z**pz)
                H[0, 1] += t
                H[1, 0] += t
            if px >= 1 and pz >= 1:
                t = c * px * pz * (x ** (px - 1)) * (y**py) * (z ** (pz - 1))
                H[0, 2] += t
                H[2, 0] += t
            if py >= 1 and pz >= 1:
                t = c * py * pz * (x**px) * (y ** (py - 1)) * (z ** (pz - 1))
                H[1, 2] += t
                H[2, 1] += t

        return H

    # seems legit
    def _drive_hessians_at_eq(self, num_ions, drive):
        """
        Build a list [H_1, ..., H_N] of 3x3 voltage Hessians (V/m^2) for the given drive,
        evaluated at each ion's *static* equilibrium position (meters).
        """
        # ensure the drive exists in the simulation
        drives = set(self.trapVariables.get_drives())
        if drive not in drives:
            raise ValueError(
                f"Drive {drive} is not present in this simulation's Trapping_Vars."
            )

        # ensure we have a center fit for the drive; build if missing
        if not hasattr(self, "center_fits"):
            self.center_fits = {}
        if drive not in self.center_fits:
            # build (model, poly, r2) for all drives; cheap and robust
            self.update_center_polys()
            if drive not in self.center_fits:
                raise ValueError(f"No center-fit available for drive {drive}.")

        model, poly, _r2 = self.center_fits[drive]

        # get equilibrium positions in SI
        if num_ions not in self.ion_equilibrium_positions:
            # last resort: try to find equilibrium now
            eq = self.find_equilib_positions(num_ions)
            if eq is None:
                raise RuntimeError(
                    f"No equilibrium positions stored for {num_ions} ions, and search failed."
                )
        eq = self.ion_equilibrium_positions[num_ions]  # (N,3) meters

        # evaluate Hessian at each ion
        H_list = []
        for n in range(eq.shape[0]):
            Hn = self._poly_hessian_at(model, poly, eq[n, :])
            H_list.append(Hn)
        return H_list

    # looks good
    def _get_modes_normalized_from_cache(self, num_ions):
        """
        Fetch normalized modes (columns) and angular freqs (rad/s) from
        self.normal_modes_and_frequencies. If absent, compute & store them.
        """
        import numpy as np, math

        if not hasattr(self, "normal_modes_and_frequencies"):
            self.normal_modes_and_frequencies = {}

        entry = self.normal_modes_and_frequencies.get(num_ions)
        if entry is None:
            # compute with your existing helper that normalizes and stores
            entry = self.get_static_normal_modes_and_freq(
                num_ions, normalize=True, sort_by_freq=True
            )

        V = entry["modes"]  # (3N x 3N), columns are normalized eigenvectors
        f_Hz = entry["frequencies_Hz"]  # (3N,)
        omega = 2.0 * math.pi * f_Hz  # rad/s
        return V, omega

    # asdfghj
    def _sum_MtHM_over_ions(self, H_list, V):
        """
        Efficiently compute C = sum_n ( M_n^T H_n M_n ), where
        M_n = V[3n:3n+3, :] is the 3xK block of normalized modes on ion n.
        Returns a (K x K) symmetric array.
        """
        import numpy as np

        N = len(H_list)
        K = V.shape[1]
        C = np.zeros((K, K), dtype=float)
        for n in range(N):
            Mn = V[3 * n : 3 * n + 3, :]  # (3 x K)
            Hn = H_list[n]  # (3 x 3)
            # C_n = M^T H M
            C += Mn.T @ Hn @ Mn
        return C

    def get_g0_matrix(self, num_ions, drive):
        """
        Compute the static parametric coupling matrix g0 (K x K) for the given *modulation* drive,
        where K=3*num_ions is the number of normal modes.

        g0_ab = [ q / (4 m sqrt(ω_a ω_b)) ] * sum_n ξ_{n,a}^T H_phi^(drive)(r_{n,0}) ξ_{n,b}

        - Uses normalized eigenmodes and secular frequencies from self.normal_modes_and_frequencies.
        - Uses the *voltage* center-fit (scalar potential) Hessian for the specified drive.
        - Evaluates the Hessian at *static* equilibrium positions r_{n,0} (SI meters).
        """
        import numpy as np

        # Guard: DC is not a modulation drive (you asked for non-zero, non-max drives)
        if drive == self.trapVariables.dc_key:
            raise ValueError(
                "DC drive is not a modulation drive for g0. Choose a non-DC drive."
            )

        # 1) Hessians of the modulation potential at each ion
        H_list = self._drive_hessians_at_eq(num_ions, drive)  # list of 3x3 (V/m^2)

        # 2) Normalized modes and secular angular frequencies (rad/s)
        V, omega = self._get_modes_normalized_from_cache(
            num_ions
        )  # V: (3N x K), omega: (K,)

        # 3) Sum over ions of M^T H M  (unscaled KxK)
        C = self._sum_MtHM_over_ions(H_list, V)  # V/m^2 in modal basis

        # 4) Scale by q / (4 m sqrt(ω_a ω_b))
        denom = np.sqrt(omega[:, None] * omega[None, :])
        # protect against any accidental zeros (shouldn't happen for trapped modes)
        denom = np.where(denom > 0.0, denom, np.inf)

        scale = constants.ion_charge / (4.0 * constants.ion_mass)
        G0 = scale * (C / denom)  # units: 1/s (rate-like coupling)

        # symmetry cleanup (numerical)
        G0 = 0.5 * (G0 + G0.T)
        return G0

    def find_largest_g0(self, num_ions, drive):
        """
        Compute g0 for `drive` and return the 6 largest |g0_ij| off-diagonal entries.
        Returns:
        {
            'drive': drive,
            'num_ions': num_ions,
            'top_pairs': [
            {
                'modes': (i, j),
                'g0_s_inv': float,                  # coupling in s^-1
                'g0_Hz': float,                     # coupling in Hz
                'f_i_Hz': float, 'f_j_Hz': float,   # secular freqs (Hz)
                'mode_i': np.ndarray,               # normalized eigenvector (3N,)
                'mode_j': np.ndarray
            }, ...
            ]
        }
        """
        import numpy as np, math

        # Ensure normalized modes & frequencies are available/cached
        if (
            not hasattr(self, "normal_modes_and_frequencies")
            or num_ions not in self.normal_modes_and_frequencies
        ):
            # This computes, normalizes (L2=1 per column), and stores in Hz
            self.get_static_normal_modes_and_freq(
                num_ions, normalize=True, sort_by_freq=True
            )

        entry = self.normal_modes_and_frequencies[num_ions]
        V = entry["modes"]  # (3N x 3N), columns = normalized modes
        f_Hz = entry["frequencies_Hz"]  # (3N,)

        # Build g0 matrix (K x K) in s^-1
        G0 = self.get_g0_matrix(num_ions, drive)  # raises if drive missing / no fit

        K = G0.shape[0]
        iu, ju = np.triu_indices(K, k=1)  # off-diagonal, upper triangle
        vals = G0[iu, ju]
        order = np.argsort(np.abs(vals))[::-1]  # by magnitude, descending
        topk = min(6, len(order))

        out = {"drive": drive, "num_ions": num_ions, "top_pairs": []}
        for idx in order[:topk]:
            i = int(iu[idx])
            j = int(ju[idx])
            gij = float(G0[i, j])
            out["top_pairs"].append(
                {
                    "modes": (i, j),
                    "g0_s_inv": gij,
                    "g0_Hz": gij / (2.0 * math.pi),
                    "f_i_Hz": float(f_Hz[i]),
                    "f_j_Hz": float(f_Hz[j]),
                    "mode_i": V[:, i].copy(),
                    "mode_j": V[:, j].copy(),
                }
            )

        return out
