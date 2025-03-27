'''
Run code here
'''

import math
import time
from matplotlib import pyplot as plt
from simulation import Simulation
import experiment_funcs
import electrode_vars as evars



tstart = time.time()

electrodes = evars.get_electrodvars_w_twist(377, 25500000 * 2 * math.pi, -0.275, 2)
testsim1 = Simulation("Simp58_101", electrodes)

# for i in range(100):
#     if i%2 == 0:
#         testsim1.change_electrode_variables(evars.get_electrodvars_w_twist(377, 25500000 * 2 * math.pi, -0.275, 2.5))
#     else:
#         testsim1.change_electrode_variables(evars.get_electrodvars_w_twist(277, 28000000 * 2 * math.pi, -0.275, 2))
#     testsim1.get_principal_freq_at_min(look_around=3)

# fig = experiment_funcs.recreate_old_data(377, 25500000 * 2 * math.pi, -0.275, 2, push_stop=1.5, step_size=0.3)

print(testsim1.get_principal_freq_at_min(getmintoo=False, fitdeg=2, look_around=5))
print(testsim1.get_principal_freq_at_min(getmintoo=False, fitdeg=4, look_around=5))
print(testsim1.get_principal_freq_at_min(getmintoo=False, fitdeg=4, look_around=50))


tstop = time.time()

plt.show()


# for fit in [2,4]:
#     for lookaround in [3, 5, 10, 15, 20, 30, 40, 50, 60]:
#         freqs = testsim1.get_freqs_at_point_withR3_fit(0,0,0, look_around=lookaround, polyfit=fit)[0]
#         print("Fit, lookaround: " + str(fit) + ", " + str(lookaround) + "--> Val: " + str((abs(freqs[2] - freqs[1]))))

# [5,10,15,20,30,40,50,60,90]

print("Time taken: " + str(tstop - tstart))
