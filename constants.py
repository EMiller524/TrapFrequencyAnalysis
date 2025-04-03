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

import math


ion_mass = 2.885 * (10 **(-25)) #kg
ion_charge = 1.60217662  * (10 **(-19)) #C

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
    "RF2")


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
    "RF2": 0.01
}

ground_capacitor_PF = 1000 # PF

electrode_RF_pickoff_amp_multipliers = {}
for electrode in trap_capcitence_per_electrode_PF:
    electrode_RF_pickoff_amp_multipliers[electrode] = (
        trap_capcitence_per_electrode_PF[electrode] / (trap_capcitence_per_electrode_PF[electrode] + ground_capacitor_PF)
    )


def freq_calcualtion(secondderivative):
    return (
        math.copysign(1, secondderivative)
        * math.sqrt((ion_charge / ion_mass) * abs(secondderivative))
        / (2 * math.pi)
    )

