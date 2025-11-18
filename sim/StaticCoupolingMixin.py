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
from itertools import combinations


class StaticCoupolingMixin:
    """
    This class will be inherited by the simulation class and will contain all fucntions used to find modes and their coupling.
    """

    # lofok over
    def get_3_wise_mode_couplings_old(self, num_ions):
        """
        This will be done by contracting the 3rd derivative tensor with the eigenvectors.
        """
        tensor = self.get_eq_3rd_der_tensor(num_ions)
        three_wise_couplings = {}

        for modei in range(0, 3 * num_ions):
            eigveci = self.ion_eigenvectors[num_ions][:, modei]
            coupling_i = np.tensordot(tensor, eigveci, axes=([0], [0]))
            for modej in range(modei, 3 * num_ions):
                eigvecj = self.ion_eigenvectors[num_ions][:, modej]
                coupling_ij = np.tensordot(coupling_i, eigvecj, axes=([0], [0]))
                for modek in range(modej, 3 * num_ions):
                    eigveck = self.ion_eigenvectors[num_ions][:, modek]
                    coupling_ijk = np.tensordot(coupling_ij, eigveck, axes=([0], [0]))

                    three_wise_couplings[(modei, modej, modek)] = coupling_ijk
                # contract the tensor with the eigenvector

        self.inherent_g_0_3_couplings[num_ions] = three_wise_couplings

        return three_wise_couplings

    def get_3_wise_mode_couplingss(self, num_ions: int):
        """
        Compute cubic (3-wise) mode couplings by contracting the 3rd-derivative tensor
        with the mode eigenvectors.
        """
        import numpy as np

        # --- Ensure modes/frequencies exist ---
        if num_ions not in getattr(self, "normal_modes_and_frequencies", {}):
            self.get_static_normal_modes_and_freq(num_ions)

        # Fetch eigenvectors (columns = modes) and make a working copy
        V = np.array(
            self.normal_modes_and_frequencies[num_ions]["modes"], dtype=float, copy=True
        )  # (3N,3N)

        # Unit/normalization check for eigenvectors
        col_norms = np.linalg.norm(V, axis=0)
        if not np.allclose(col_norms, 1.0, rtol=1e-3, atol=1e-8):
            print("Warning")
            print("Warning: Eigenvectors are not normalized!")
            print("Warning")

        # --- Get the 3rd-derivative tensor at equilibrium ---
        T = np.array(
            self.get_eq_3rd_der_tensor(num_ions), dtype=float, copy=False
        )  # (3N,3N,3N)

        # Convert voltage derivatives to energy derivatives if needed: U = q V   NO!
        # q_SI = constants.ion_charge  # C
        # T = q_SI * T  # now J/m^3

        # --- Contract to mode basis: C[a,b,c] ---
        # einsum: i,j,k over real-space, a,b,c over modes (columns)
        C = np.einsum("ijk,ia,jb,kc->abc", T, V, V, V, optimize=True)  # J/m^3

        # # --- Frequencies and mass ---
        # freqs_Hz = np.asarray(
        #     self.normal_modes_and_frequencies[num_ions]["frequencies_Hz"], dtype=float
        # )  # Hz
        # omega = 2.0 * np.pi * freqs_Hz  # rad/s

        # m_SI = constants.ion_mass  # kg

        # denom = (
        #     4.0
        #     * m_SI
        #     * np.sqrt(omega[:, None, None] * omega[None, :, None] * omega[None, None, :])
        # )

        # g3_rad_s = C / denom
        # g3_Hz = g3_rad_s / (2.0 * np.pi)

        # --- Frequencies and mass ---
        freqs_Hz = np.asarray(
            self.normal_modes_and_frequencies[num_ions]["frequencies_Hz"], dtype=float
        )  # Hz
        omega = 2.0 * np.pi * freqs_Hz  # rad/s

        m_SI = constants.ion_mass  # kg

        zero_pnt_lengths_multipliers = np.sqrt(
            constants.hbar**3
            / (
                2.0**3
                * m_SI**3
                * omega[:, None, None]
                * omega[None, :, None]
                * omega[None, None, :]
            )
        )

        g3_rad_s = C * zero_pnt_lengths_multipliers / (6 * constants.hbar)
        g3_Hz = g3_rad_s / (2.0 * np.pi)

        self.inherent_g_0_3_couplings[num_ions] = g3_Hz

        return g3_Hz

    def get_4_wise_mode_couplingss(self, num_ions: int):
        """
        Compute quartic (4-wise) mode couplings by contracting the 4th-derivative tensor
        with the mode eigenvectors.
        """
        import numpy as np

        # --- Ensure modes/frequencies exist ---
        if num_ions not in getattr(self, "normal_modes_and_frequencies", {}):
            self.get_static_normal_modes_and_freq(num_ions)

        # Fetch eigenvectors (columns = modes) and make a working copy
        V = np.array(
            self.normal_modes_and_frequencies[num_ions]["modes"], dtype=float, copy=True
        )  # (3N,3N)

        # Unit/normalization check for eigenvectors
        col_norms = np.linalg.norm(V, axis=0)
        if not np.allclose(col_norms, 1.0, rtol=1e-3, atol=1e-8):
            print("Warning")
            print("Warning: Eigenvectors are not normalized!")
            print("Warning")

        # --- Get the 4th-derivative tensor at equilibrium ---
        Q = np.array(
            self.get_eq_4th_der_tensor(num_ions), dtype=float, copy=False
        )  # (3N,3N,3N,3N)

        # # Convert voltage derivatives to energy derivatives if needed: U = q V   NO!!
        # q_SI = constants.ion_charge  # C
        # Q = q_SI * Q  # now J/m^4

        # --- Contract to mode basis: D[a,b,c,d] ---
        # einsum: i,j,k,l over real-space, a,b,c,d over modes (columns)
        D = np.einsum("ijkl,ia,jb,kc,ld->abcd", Q, V, V, V, V, optimize=True)  # J/m^4

        # # --- Frequencies and mass ---
        # freqs_Hz = np.asarray(
        #     self.normal_modes_and_frequencies[num_ions]["frequencies_Hz"], dtype=float
        # )  # Hz
        # omega = 2.0 * np.pi * freqs_Hz  # rad/s

        # m_SI = constants.ion_mass  # kg

        # # Rate-style normalization (convention analogous to cubic case)
        # # g4 ~ D / (8 m sqrt(ω_a ω_b ω_c ω_d))  -> rad/s
        # denom = (
        #     8.0
        #     * m_SI
        #     * np.sqrt(
        #         omega[:, None, None, None]
        #         * omega[None, :, None, None]
        #         * omega[None, None, :, None]
        #         * omega[None, None, None, :]
        #     )
        # )
        # denom = np.where(denom == 0, np.finfo(float).eps, denom)

        # g4_rad_s = D / denom
        # g4_Hz = g4_rad_s / (2.0 * np.pi)

        # --- Frequencies and mass ---
        freqs_Hz = np.asarray(
            self.normal_modes_and_frequencies[num_ions]["frequencies_Hz"], dtype=float
        )  # Hz
        omega = 2.0 * np.pi * freqs_Hz  # rad/s

        m_SI = constants.ion_mass  # kg

        # Product of four zero-point lengths:
        # x0_a x0_b x0_c x0_d = (ħ^2 / (2^2 m^2)) * 1/sqrt(ω_a ω_b ω_c ω_d)
        zero_pnt_lengths_multipliers = (
            (constants.hbar**2)
            / ((2.0**2) * (m_SI**2))
            * 1.0
            / np.sqrt(
                omega[:, None, None, None]
                * omega[None, :, None, None]
                * omega[None, None, :, None]
                * omega[None, None, None, :]
            )
        )

        # Per-phonon quartic coupling (rad/s), then Hz
        # g4 = (1/24ħ) * D * (x0_a x0_b x0_c x0_d)
        g4_rad_s = (D * zero_pnt_lengths_multipliers) / (24.0 * constants.hbar)
        g4_Hz = g4_rad_s / (2.0 * np.pi)
        # Cache
        self.inherent_g_0_4_couplings[num_ions] = g4_Hz

        return g4_Hz

    def get_4_wise_mode_couplings_old(self, num_ions):
        """
        This will be done by contracting the 4th derivative tensor with the eigenvectors.
        """
        tensor = self.get_eq_4th_der_tensor(num_ions)
        four_wise_couplings = {}

        for modei in range(0, 3 * num_ions):
            eigveci = self.ion_eigenvectors[num_ions][:, modei]
            coupling_i = np.tensordot(tensor, eigveci, axes=([0], [0]))
            for modej in range(modei, 3 * num_ions):
                eigvecj = self.ion_eigenvectors[num_ions][:, modej]
                coupling_ij = np.tensordot(coupling_i, eigvecj, axes=([0], [0]))
                for modek in range(modej, 3 * num_ions):
                    eigveck = self.ion_eigenvectors[num_ions][:, modek]
                    coupling_ijk = np.tensordot(coupling_ij, eigveck, axes=([0], [0]))
                    for model in range(modek, 3 * num_ions):
                        eigvecl = self.ion_eigenvectors[num_ions][:, model]
                        coupling_ijkl = np.tensordot(
                            coupling_ijk, eigvecl, axes=([0], [0])
                        )

                        four_wise_couplings[(modei, modej, modek, model)] = (
                            coupling_ijkl
                        )

                # contract the tensor with the eigenvector
        self.inherent_g_0_4_couplings[num_ions] = four_wise_couplings
        return four_wise_couplings

    # ========== g0: static parametric coupling from a modulation drive ==========

    # TODO Check these derivatives use paper, wrote at 1am and no paper
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
        # print("****************************************************************************************")
        coef = model.coef_
        # print(powers, coef)
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

    # good
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

    # good
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

        # Guard
        if drive == self.trapVariables.dc_key:
            raise ValueError(
                "DC drive is not a modulation drive for g0. Choose a non-DC drive."
            )

        # Hessians of the modulation potential at each ion
        H_list = self._drive_hessians_at_eq(num_ions, drive)  # list of 3x3 (V/m^2)

        # Normalized modes and secular angular frequencies (rad/s)
        V, omega = self._get_modes_normalized_from_cache(
            num_ions
        )  # V: (3N x K), omega: (K,)

        # Sum over ions of M^T H M  (unscaled KxK)
        C = self._sum_MtHM_over_ions(H_list, V)  # V/m^2 in modal basis

        # Scale by q / (4 m sqrt(ω_a ω_b))
        denom = np.sqrt(omega[:, None] * omega[None, :])
        # protect against any accidental zeros (shouldn't happen for trapped modes)
        denom = np.where(denom > 0.0, denom, np.inf)

        scale = constants.ion_charge / (4.0 * constants.ion_mass)
        G0 = scale * (C / denom)  # units: 1/s (rate-like coupling)

        # symmetry cleanup (numerical)
        G0 = 0.5 * (G0 + G0.T)

        # add G0 as the value to a dictionary with key the number of ions and this while dictionary is the value for the key the drive
        self.driven_g_0_2_couplings[drive] = {}
        self.driven_g_0_2_couplings[drive][num_ions] = G0

        return G0

    # quick thing to help check against known results
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
        topk = min(100, len(order))

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

    def find_resonant_mode_combinations(
        self,
        freqs,
        orders=(2, 3, 4),
        drive_freqs=None,
        tol=None,
        rel_tol=1e-3,
        collapse=True,
    ):
        """
        Identify resonant 2-, 3-, and 4-wise mode combinations under a 'difference' condition.

        Resonance condition (one-vs-rest pattern):
            s · ω  ≈  target
        where s has exactly one +1 and (order-1) -1's, ω are mode frequencies, and
        target ∈ {0} ∪ {±Ω_d : Ω_d in drive_freqs}.

        Parameters
        ----------
        freqs : array-like of shape (M,)
            Mode frequencies (same units as drive_freqs).
        orders : tuple[int], default (2,3,4)
            Which tuple sizes to search.
        drive_freqs : array-like or None
            Extra drive frequencies Ω_d. If None or empty, only target=0 is used.
        tol : float or None
            Absolute tolerance in the same units as freqs. If None, uses rel_tol * median(|freqs|).
        rel_tol : float
            Relative tolerance used when tol is None.
        collapse : bool
            If True, keep only the best (min detuning) entry per unordered index-set.

        Returns
        -------
        list[dict]
            Each dict contains:
                - 'order': 2|3|4
                - 'indices': tuple of mode indices (i, j, ...) with i<j<...
                - 'lhs_pos': int (0..order-1) position of the +1 term
                - 'pattern': list of +1/-1 for readability
                - 'value': float = s·ω
                - 'target': float chosen target (0 or ±Ω_d)
                - 'detuning': float = |s·ω - target|
                - 'type': 'internal' (target=0) or 'driven' (target≠0)
        """

        freqs = np.asarray(freqs, dtype=float)
        M = freqs.size
        if M == 0:
            return []
            print("NO FREQ GIVENNNNNN")

        # Resolve tolerance
        if tol is None:
            scale = np.median(np.abs(freqs)) if np.any(freqs) else 1.0
            tol = max(rel_tol * scale, 0.0)

        # Build target set {0} ∪ {±Ω_d}
        drv = [] if drive_freqs is None else list(drive_freqs)
        targets = [0.0]
        for d in drv:
            d = float(d)
            if d != 0.0:
                targets.append(+d)
                targets.append(-d)

        results = []

        def check_tuple(idxs):
            # For each position as the +1 (lhs), others are -1
            local_hits = []
            w = freqs[list(idxs)]
            order = len(idxs)
            for lhs_pos in range(order):
                s = np.full(order, -1.0)
                s[lhs_pos] = +1.0
                val = float(np.dot(s, w))

                # Find best target and detuning
                best_target = None
                best_detune = None
                for t in targets:
                    det = abs(val - t)
                    if (best_detune is None) or (det < best_detune):
                        best_detune = det
                        best_target = t

                if best_detune is not None and best_detune <= tol:
                    local_hits.append({
                        'order': order,
                        'indices': tuple(idxs),   # canonical sorted
                        'lhs_pos': lhs_pos,
                        'pattern': [int(x) for x in s],
                        'value': val,
                        'target': best_target,
                        'detuning': best_detune,
                        'type': 'internal' if best_target == 0.0 else 'driven',
                    })
            return local_hits

        # Enumerate combinations by order
        for k in orders:
            if k < 2 or k > 4:
                continue
            for idxs in combinations(range(M), k):
                results.extend(check_tuple(idxs))

        if not collapse:
            return results

        # Collapse duplicates: keep the smallest-detuning entry per unordered index set & target kind
        best = {}
        for r in results:
            key = (r['order'], r['indices'], r['target'])
            cur = best.get(key)
            if (cur is None) or (r['detuning'] < cur['detuning']):
                best[key] = r

        return list(best.values())

    def collect_resonant_couplings(
        self,
        num_ions: int,
        tol_Hz: float,
        orders=(2, 3, 4),
        drives=None,  # optional: list of DriveKey; defaults to all non-DC drives
    ):
        """
        Pipeline:
        - make sure normal modes & freqs exist
        - compute all (2,3,4)-wise couplings (easy/robust, uses your existing methods)
        - prepare resonance targets (0 and each drive freq)
        - call self.find_resonant_mode_combinations(...)
        - pull just those amplitudes from the full coupling tensors and return

        Returns a dict:
        {
            'num_ions': N,
            'tolerance_Hz': tol_Hz,
            'frequencies_Hz': [...],
            'drives': [{'key': DriveKey, 'f_Hz': ...}, ...],
            'resonances': {
            2: [ { 'modes': (i,j), 'freqs_Hz': (fi,fj), 'signs': (+1,-1),
                    'target_Hz': t, 'sum_Hz': S, 'delta_Hz': d,
                    'g0_Hz_by_drive': [ {'drive': DriveKey, 'g0_Hz': val}, ... ] }, ... ],
            3: [ { 'modes': (i,j,k), ..., 'g3_Hz': value }, ... ],
            4: [ { 'modes': (i,j,k,l), ..., 'g4_Hz': value }, ... ],
            }
        }
        """

        # --- 1) Ensure modes/frequencies exist ---
        if (
            not hasattr(self, "normal_modes_and_frequencies")
            or num_ions not in self.normal_modes_and_frequencies
        ):
            self.get_static_normal_modes_and_freq(
                num_ions, normalize=True, sort_by_freq=True
            )

        entry = self.normal_modes_and_frequencies[num_ions]
        freqs_Hz = np.asarray(entry["frequencies_Hz"], dtype=float)  # shape (K,)

        # --- 2) Compute all couplings (already-implemented paths) ---
        # 2-wise (g0): needs drives; 3/4-wise: inherent (static) tensors
        # Choose drives if not provided: all non-DC drives present in this sim
        if drives is None:
            drives = [
                d for d in self.trapVariables.get_drives() if getattr(d, "f_uHz", 0) != 0
            ]

        # g0 for each drive (matrix KxK, in s^-1)
        g0_by_drive_s = {}
        if 2 in orders and len(drives) > 0:
            for d in drives:
                G0_s = self.get_g0_matrix(num_ions, d)  # s^-1
                g0_by_drive_s[d] = G0_s

        # g3 (Hz): shape (K,K,K)
        g3_Hz = None
        if 3 in orders:
            g3_Hz = self.get_3_wise_mode_couplingss(num_ions)

        # g4 (Hz): shape (K,K,K,K)
        g4_Hz = None
        if 4 in orders:
            g4_Hz = self.get_4_wise_mode_couplingss(num_ions)

        # --- 3) Prepare resonance targets and call the finder ---
        drive_list = [
            {"key": d, "f_Hz": float(getattr(d, "f_uHz", 0) * 1e-6)} for d in drives
        ]
        drive_freqs = [x["f_Hz"] for x in drive_list if x["f_Hz"] != 0.0]

        if not hasattr(self, "find_resonant_mode_combinations"):
            raise AttributeError("Missing find_resonant_mode_combinations(...)")

        # CALL WITH drive_freqs, and remember it returns a LIST
        hits_list = self.find_resonant_mode_combinations(
            freqs=freqs_Hz,
            orders=orders,
            drive_freqs=drive_freqs,  # <-- critical
            tol=tol_Hz,
            rel_tol=1e-3,
            collapse=True,
        )

        # Group into {2:[...],3:[...],4:[...]} and rename fields you use below
        hits = {2: [], 3: [], 4: []}
        for r in hits_list:
            ordk = r["order"]
            hits[ordk].append(
                {
                    "idxs": r["indices"],
                    "signs": r["pattern"],
                    "sum_Hz": r["value"],
                    "target_Hz": r["target"],
                    "delta_Hz": r["detuning"],
                    "type": r["type"],
                    "lhs_pos": r["lhs_pos"],
                }
            )

        # Expected structure of `hits`:
        # {
        #   2: [ {'idxs':(i,j), 'signs':(s1,s2), 'sum_Hz':S, 'target_Hz':t, 'delta_Hz':d}, ... ],
        #   3: [ {'idxs':(i,j,k), 'signs':(...), 'sum_Hz':S, 'target_Hz':t, 'delta_Hz':d}, ... ],
        #   4: [ {'idxs':(i,j,k,l), ...}, ... ],
        # }

        # --- 4) Organize the output: attach amplitudes from the full tensors ---
        out = {
            "num_ions": int(num_ions),
            "tolerance_Hz": float(tol_Hz),
            "frequencies_Hz": freqs_Hz.tolist(),
            "drives": drive_list,
            "resonances": {2: [], 3: [], 4: []},
        }

        # --- 4a) ORDER-2: dedup across ±target, attach only matching drives ---
        if 2 in hits:
            agg = {}  # key: ((i,j), target_abs_Hz) -> rec
            for h in hits[2]:
                i, j = h["idxs"]
                pair = (min(i, j), max(i, j))
                tgt_abs = 0.0 if h["target_Hz"] == 0.0 else abs(h["target_Hz"])
                key = (pair, tgt_abs)

                rec = agg.get(key)
                if rec is None:
                    rec = {
                        "modes": pair,
                        "freqs_Hz": (float(freqs_Hz[pair[0]]), float(freqs_Hz[pair[1]])),
                        "target_Hz": tgt_abs,  # 0.0 for internal, otherwise |drive|
                        "delta_Hz": float(h["delta_Hz"]),
                        "sum_Hz": float(h["sum_Hz"]),  # representative sign pattern
                        "g0_Hz": None,  # set below
                        "drive_resonances": [],  # [{'drive':..., 'f_Hz':..., 'g0_Hz':...}, ...]
                    }
                    agg[key] = rec
                else:
                    # keep the tightest detuning; keep the sum with smallest |value|
                    rec["delta_Hz"] = min(rec["delta_Hz"], float(h["delta_Hz"]))
                    if abs(h["sum_Hz"]) < abs(rec["sum_Hz"]):
                        rec["sum_Hz"] = float(h["sum_Hz"])

                # If driven (target>0), attach only drives whose freq ~= |target| within tol
                if tgt_abs > 0.0 and len(g0_by_drive_s) > 0:
                    for d in drive_list:
                        if abs(d["f_Hz"] - tgt_abs) <= tol_Hz:
                            G0_s = g0_by_drive_s.get(d["key"])
                            if G0_s is None:
                                continue
                            g0_Hz_val = float(G0_s[pair[0], pair[1]] / (2.0 * math.pi))
                            # avoid duplicate drive entries when both sign patterns hit
                            if not any(
                                dr["drive"] is d["key"] for dr in rec["drive_resonances"]
                            ):
                                rec["drive_resonances"].append(
                                    {
                                        "drive": d["key"],
                                        "f_Hz": d["f_Hz"],
                                        "g0_Hz": g0_Hz_val,
                                    }
                                )

            # Finalize records: single number if one drive, list if multiple; sort by f_Hz
            for rec in agg.values():
                if len(rec["drive_resonances"]) == 1:
                    rec["g0_Hz"] = rec["drive_resonances"][0]["g0_Hz"]
                elif len(rec["drive_resonances"]) >= 2:
                    rec["drive_resonances"].sort(key=lambda x: x["f_Hz"])
                    rec["g0_Hz"] = [dr["g0_Hz"] for dr in rec["drive_resonances"]]
                out["resonances"][2].append(rec)

        # order-3: single scalar amplitude from g3_Hz
        if 3 in hits and g3_Hz is not None:
            for h in hits[3]:
                i, j, k = h["idxs"]
                out["resonances"][3].append(
                    {
                        "modes": (int(i), int(j), int(k)),
                        "freqs_Hz": (
                            float(freqs_Hz[i]),
                            float(freqs_Hz[j]),
                            float(freqs_Hz[k]),
                        ),
                        "signs": tuple(int(s) for s in h["signs"]),
                        "sum_Hz": float(h["sum_Hz"]),
                        "target_Hz": float(h["target_Hz"]),
                        "delta_Hz": float(h["delta_Hz"]),
                        "g3_Hz": float(g3_Hz[i, j, k]),
                    }
                )

        # order-4: single scalar amplitude from g4_Hz
        if 4 in hits and g4_Hz is not None:
            for h in hits[4]:
                i, j, k, l = h["idxs"]
                out["resonances"][4].append(
                    {
                        "modes": (int(i), int(j), int(k), int(l)),
                        "freqs_Hz": (
                            float(freqs_Hz[i]),
                            float(freqs_Hz[j]),
                            float(freqs_Hz[k]),
                            float(freqs_Hz[l]),
                        ),
                        "signs": tuple(int(s) for s in h["signs"]),
                        "sum_Hz": float(h["sum_Hz"]),
                        "target_Hz": float(h["target_Hz"]),
                        "delta_Hz": float(h["delta_Hz"]),
                        "g4_Hz": float(g4_Hz[i, j, k, l]),
                    }
                )

        return out
