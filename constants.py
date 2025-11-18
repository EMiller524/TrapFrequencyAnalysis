"""
This file holds all physical and semantic constants

Electode Naming Diagram:

    Y,Z Plane: 
        DC_Blade_A--------RF2
        |                  |
        |                  |    (+Z dir is up)
        |                  |    (+Y dir is right)
        |                  |
        RF1-----------DC_Blade_B

    DC Blades from -X to +X:
        DC_Blade_A ==> --- DC5  DC4  DC3  DC2  DC1 ---
        DC_Blade_B ==> --- DC6  DC7  DC8  DC9  DC10 ---

"""

# TODO make this read from a config file ST a UI could eazily edit

import math

import numpy as np

INCLUDE_ALL_RF_PSEUDOPOTENTIALS = False  # False keeps current behavior

ion_mass = 2.885 * (10 ** (-25))  # kg Yb+
# ion_mass = 1.5e-26  # kg be
ion_charge = 1.60217662 * (10 ** (-19))  # C
epsilon_0 = 8.854187817e-12  # F/m

electrode_names = (
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
    "RF1",
    "RF2",
)

center_region_x_um = 100 # microns
center_region_y_um = 10 # microns
center_region_z_um = center_region_y_um


# TODO: The values should be applied differently, the current implemetation is flawed
# What should happen is that each pairwise combination of electodes has a capacitance and then
# that should be used to calculate the pickoff multiplier
# what we have right now is the RF to [blank] pickoffs
# ideally we have a 12 by 12 matrix of capacitances (Should be symmetric)
trap_capcitence_per_electrode_PF = {
    "DC1": 0.155,
    "DC2": 0.068,
    "DC3": 0.125,
    "DC4": 0.075,
    "DC5": 0.157,
    "DC6": 0.157,
    "DC7": 0.076,
    "DC8": 0.125,
    "DC9": 0.068,
    "DC10": 0.155,
    "RF1": 0.01,
    "RF2": 0.01,
}

ground_capacitor_PF = 1000  # PF

electrode_RF_pickoff_amp_multipliers = {}
for electrode in trap_capcitence_per_electrode_PF:
    electrode_RF_pickoff_amp_multipliers[electrode] = trap_capcitence_per_electrode_PF[
        electrode
    ] / (trap_capcitence_per_electrode_PF[electrode] + ground_capacitor_PF)


def freq_calcualtion(secondderivative):
    return (
        math.copysign(1, secondderivative)
        * math.sqrt((ion_charge / ion_mass) * abs(secondderivative))
        / (2 * math.pi)
    )


ion_electrode_dis = 0.00025

hbar = 1.0545718e-34  # J*s

max_ion_in_chain = 10

coulomb_constant = 8.9875517873681764 * (10**9)  # N m^2 / C^2

# Should be a flat list/array of length 3n
ion_locations_intial_guess = {}

# Should be length 3n list of (low, high)
ion_locations_bounds = {}

# Length in harmonic apoximation
Z = 1
typical_axial_freq = 225000 * 6.28  # Hz ((Rad/sec))
length_harmonic_approximation = (
    (Z**2 * ion_charge**2) / (4 * 3.1416 * epsilon_0 * ion_mass * typical_axial_freq**2)
) ** (1 / 3)

print("Length in harmonic approximation: ", length_harmonic_approximation)

# Intial equilib positions nomallized by l:
ion_locations_intial_guess[1] = [[0, 0, 0]]
ion_locations_intial_guess[2] = [[-0.62996, 0, 0], [0.62996, 0, 0]]
ion_locations_intial_guess[3] = [[-1.3772, 0, 0], [0, 0, 0], [1.3772, 0, 0]]
ion_locations_intial_guess[4] = [
    [-1.4368, 0, 0],
    [-0.55438, 0, 0],
    [-0.55438, 0, 0],
    [1.4368, 0, 0],
]
ion_locations_intial_guess[5] = [
    [-1.7429, 0, 0],
    [-0.8221, 0, 0],
    [0, 0, 0],
    [0.8221, 0, 0],
    [1.7429, 0, 0],
]
ion_locations_intial_guess[6] = [
    [-1.9, 0, 0],
    [-1.1, 0, 0],
    [-0.3, 0, 0],
    [0.3, 0, 0],
    [1.1, 0, 0],
    [1.9, 0, 0],
]
ion_locations_intial_guess[7] = [
    [-2.0, 0, 0],
    [-1.3, 0, 0],
    [-0.7, 0, 0],
    [0, 0, 0],
    [0.7, 0, 0],
    [1.3, 0, 0],
    [2.0, 0, 0],
]
ion_locations_intial_guess[8] = [
    [-2.1, 0, 0],
    [-1.5, 0, 0],
    [-0.9, 0, 0],
    [-0.3, 0, 0],
    [0.3, 0, 0],
    [0.9, 0, 0],
    [1.5, 0, 0],
    [2.1, 0, 0],
]
ion_locations_intial_guess[9] = [
    [-2.2, 0, 0],
    [-1.7, 0, 0],
    [-1.1, 0, 0],
    [-0.5, 0, 0],
    [0, 0, 0],
    [0.5, 0, 0],
    [1.1, 0, 0],
    [1.7, 0, 0],
    [2.2, 0, 0],
]
ion_locations_intial_guess[10] = [
    [-2.3, 0, 0],
    [-1.9, 0, 0],
    [-1.3, 0, 0],
    [-0.7, 0, 0],
    [-0.1, 0, 0],
    [0.1, 0, 0],
    [0.7, 0, 0],
    [1.3, 0, 0],
    [1.9, 0, 0],
    [2.3, 0, 0],
]

for i in range(1, max_ion_in_chain + 1):
    ion_locations_bounds[i] = [
        (
            -200e-6 / length_harmonic_approximation,
            200e-6 / length_harmonic_approximation,
        ),
        (-5e-6 / length_harmonic_approximation, 5e-6 / length_harmonic_approximation),
        (-5e-6 / length_harmonic_approximation, 5e-6 / length_harmonic_approximation),
    ] * (i)

    # make sure the initial guess is a float
    for pnt in range(len(ion_locations_intial_guess[i])):
        ion_locations_intial_guess[i][pnt][0] = (
            float(ion_locations_intial_guess[i][pnt][0]))


# for i in range(1, max_ion_in_chain + 1):
#     for pnt in range(len(ion_locations_intial_guess[i])):
#         ion_locations_intial_guess[i][pnt][0] = (
#             ion_locations_intial_guess[i][pnt][0] * length_harmonic_approximation
#         )
#     ion_locations_bounds[i] = [
#         (-200e-6, 200e-6),
#         (-1e-6, 1e-6),
#         (-1e-6, 1e-6),
#     ] * (i)

# print("Constants loaded")


# testing

# print("Ion locations intial guess: ", ion_locations_intial_guess[6])
# print("Ion locations intial guess: ", ion_locations_intial_guess[7])
# print("Ion locations intial guess: ", ion_locations_intial_guess[8])
# print("Ion locations intial guess: ", ion_locations_intial_guess[9])
# print("Ion locations intial guess: ", ion_locations_intial_guess[10])
# # print("Ion locations bounds: ", ion_locations_bounds)
# init_guess_flatened = np.array(ion_locations_intial_guess[3]).flatten()
# print("init_guess_flatened: ", init_guess_flatened)
