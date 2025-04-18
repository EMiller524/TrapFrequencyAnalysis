"""
Run code here
"""

import math
import time
from matplotlib import pyplot as plt
from simulation import Simulation
import experiment_funcs
import electrode_vars as evars
import constants


tstart = time.time()

# electrodes = evars.get_electrodvars_w_twist(377, 25500000 * 2 * math.pi, -.275, 2.5)
# evars.add_trap_capacitence_to_electrodvars(electrodes) # add capacitance to the electrode variables
# print(electrodes.get_vars("DC1"))
# print(electrodes.get_vars("RF2"))

evarsss = evars.get_electrodvars_w_oddities(2)
sim = Simulation("Simp58_101", evarsss)
rel_import = sim.get_principal_freq_at_min(
        getall=True, look_around=5, fitdeg=4, return_coefs=True
    )[4]

# make each term in rel_import be rounded to 3
rel_import = [str(round(x, 3)) for x in rel_import]
print(rel_import)


## Plot
# testsim2 = Simulation("Simp58_101", electrodes)
# print(testsim2.get_principal_freq_at_min(getall=False, look_around=50, fitdeg=4)[0])
# testsim2.plot_2d_color_contour_Vraw(0, "x")


# fig = experiment_funcs.recreate_old_data(377, 25500000 * 2 * math.pi, -0.275, 2, push_stop=1.5, step_size=0.1)



# for i in range(100):
#     if i%2 == 0:
#         testsim1.change_electrode_variables(evars.get_electrodvars_w_twist(377, 25500000 * 2 * math.pi, -0.275, 2.5))
#     else:
#         testsim1.change_electrode_variables(evars.get_electrodvars_w_twist(277, 28000000 * 2 * math.pi, -0.275, 2))
#     testsim1.get_principal_freq_at_min(look_around=3)

# fig = experiment_funcs.recreate_old_data(377, 25500000 * 2 * math.pi, -0.275, 2, push_stop=1.5, step_size=0.1)

# print(testsim1.get_principal_freq_at_min(getmintoo=False, fitdeg=2, look_around=5))
# print(testsim1.get_principal_freq_at_min(getmintoo=False, fitdeg=4, look_around=5))
# print(testsim1.get_principal_freq_at_min(getmintoo=False, fitdeg=4, look_around=50))

# experiment_funcs.test_against_expected_many(
#     100, 450,
#     1500000 * 2 * math.pi, 40000000 * 2 * math.pi,
#     breakpoints= 3,
#     simulation="Simp58_101"
# )


tstop = time.time()

plt.show()


# for fit in [2,4]:
#     for lookaround in [3, 5, 10, 15, 20, 30, 40, 50, 60]:
#         freqs = testsim1.get_freqs_at_point_withR3_fit(0,0,0, look_around=lookaround, polyfit=fit)[0]
#         print("Fit, lookaround: " + str(fit) + ", " + str(lookaround) + "--> Val: " + str((abs(freqs[2] - freqs[1]))))

# [5,10,15,20,30,40,50,60,90]

print("Time taken: " + str(tstop - tstart))
