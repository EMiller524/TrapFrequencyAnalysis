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


trap_capcitence_per_electrode = {
    "DC1": 0.01,
    "DC2": 0.01,
    "DC3": 0.01,
    "DC4": 0.01,
    "DC5": 0.01,
    "DC6": 0.01,
    "DC7": 0.01,
    "DC8": 0.01,
    "DC9": 0.01,
    "DC10": 0.01,
    "RF1": 0.01,
    "RF2": 0.01
}

def freq_calcualtion(secondderivative):
    return (
        math.copysign(1, secondderivative)
        * math.sqrt((ion_charge / ion_mass) * abs(secondderivative))
        / (2 * math.pi)
    )
