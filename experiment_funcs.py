'''
This file contains functions to run experments using simulation
'''

from simulation import Simulation
import electrode_vars as evars
import matplotlib.pyplot as plt



def recreate_old_data(rfamp, rffreq, twist, endcaps, push_stop=1, step_size=0.1):

    '''
    This function recreates the old data from the experiment by varying the push
    along the x-axis and measuring the frequencies of the trap. It returns a plot
    of the frequencies vs the push value. The push value is varied from 0 to push_stop
    in increments of step_size. The frequencies are measured for both positive and
    negative push values.
    '''
    x_pos = []
    Radial1 = []
    Radial2 = []
    Axial = []
    x_push = 0
    test_sim = Simulation("Simp58_101")
    while True:
        if x_push >= push_stop:
            print("Push too large, stopping.")
            break
        test_sim.change_electrode_variables(
            evars.get_electrodvars_w_twist_and_push(
                rfamp, rffreq, twist, endcaps, pushx=x_push
            )
        )

        freqs, eigendir, minreal, min_snap = test_sim.get_principal_freq_at_min(
            getmintoo=True
        )

        x_pos.append(minreal[0])
        Radial1.append(freqs[1])
        Radial2.append(freqs[2])
        Axial.append(freqs[0])

        test_sim.change_electrode_variables(
            evars.get_electrodvars_w_twist_and_push(
                rfamp, rffreq, twist, endcaps, pushx=-x_push
            )
        )

        freqs, eigendir, minreal, min_snap = test_sim.get_principal_freq_at_min(
            getmintoo=True
        )

        x_pos.append(minreal[0])
        Radial1.append(freqs[1])
        Radial2.append(freqs[2])
        Axial.append(freqs[0])

        x_push += step_size
        print(x_push)

    # Plot x_pos vs Radial1 and Radial2 and Axial but have Radial 1 and 2 share an axis and Axial on the other
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.scatter(x_pos, Radial1, label="Radial", color="red")
    ax1.scatter(x_pos, Radial2, label="Radial", color="red")
    ax2.scatter(x_pos, Axial, label="Axial", color="green")
    ax1.set_xlabel("x_pos")
    ax1.set_ylabel("Radial Frequency (Hz)")
    ax2.set_ylabel("Axial Frequency (Hz)")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax1.set_ylim(ax1.get_ylim()[0] * 0.995, ax1.get_ylim()[1] * 1.005)
    ax2.set_ylim(ax2.get_ylim()[0] * 0.98, ax2.get_ylim()[1] * 1.02)
    return fig
