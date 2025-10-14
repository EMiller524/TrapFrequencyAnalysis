"""
This file contains functions to run experiments using simulation
"""

import math

import numpy as np
from simulation import Simulation
import electrode_vars_old as evars
import matplotlib.pyplot as plt
import constants


def recreate_old_data(rfamp, rffreq, twist, endcaps, push_stop=1, step_size=0.1):
    """
    This function recreates the old data from the experiment by varying the push
    along the x-axis and measuring the frequencies of the trap. It returns a plot
    of the frequencies vs the push value. The push value is varied from 0 to push_stop
    in increments of step_size. The frequencies are measured for both positive and
    negative push values.
    """
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

        freqs, eigendir, minreal, min_snap, coeffs = (
            test_sim.get_principal_freq_at_min_old(getall=True)
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

        freqs, eigendir, minreal, min_snap, coeffs = (
            test_sim.get_principal_freq_at_min_old(getall=True)
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


def test_against_expected_once(rfamp, rffreq, simulation=Simulation):
    """
    Calculate the percentage difference between the expected and actual frequencies.

    Parameters:
    rfamp (float): The amplitude of the radio frequency (RF) signal.
    rffreq (float): The frequency of the RF signal.
    simulation (Simulation): An instance of the Simulation class.

    Returns:
    float: The percentage difference between the expected and actual frequencies.
    """

    expected = (
        (1 / math.sqrt(2))
        * (constants.ion_charge / constants.ion_mass)
        * (
            rfamp / (rffreq * (2 * math.pi))
        )  # RF frequency (units are only for god to know)
        * (1 / ((constants.ion_electrode_dis) ** 2))
    )
    simulation.change_electrode_variables(
        evars.get_electrodvars_w_twist(rfamp, rffreq, 0, 0)
    )
    freqs = simulation.get_principal_freq_at_min_old(
        getall=False, fitdeg=4, look_around=5
    )[0]
    actual = (freqs[1] + freqs[2]) / 2
    return 100 * abs(expected - actual) / expected


def test_against_expected_many(
    rfamp_min, rfamp_max, rffreq_min, rffreq_max, breakpoints=5, simulation="Hyper_2"
):
    """
    Perform multiple tests against expected values for a range of RF amplitudes and frequencies.

    Parameters:
        rfamp_min (float): The minimum RF amplitude.
        rfamp_max (float): The maximum RF amplitude.
        rffreq_min (float): The minimum RF frequency.
        rffreq_max (float): The maximum RF frequency.
        breakpoints (int, optional): The number of breakpoints to divide the range into. Defaults to 5.
        simulation (str, optional): The simulation type. Defaults to "Hyper_2".

    Returns:
        None (prints the average error and standard deviation of errors for all tests).
    """

    sim = Simulation(simulation)

    # make a list of possible rfamps and rffreqs
    rfamps = [
        rfamp_min + (rfamp_max - rfamp_min) * i / (breakpoints - 1)
        for i in range(breakpoints)
    ]
    rffreqs = [
        rffreq_min + (rffreq_max - rffreq_min) * i / (breakpoints - 1)
        for i in range(breakpoints)
    ]

    errors = []
    for freq in rffreqs:
        for amp in rfamps:
            err = test_against_expected_once(amp, freq, sim)
            errors.append((amp, freq, err))
            # print(f"RF Amp: {amp}, RF Freq: {freq} --> Error: {err:.2f}%")

    # print avg error and the std of the errors
    avg_error = sum(err[2] for err in errors) / len(errors)
    std_dev_in_err = np.std([err[2] for err in errors])
    print(f"Avg Error: {avg_error:.4f}%, Std Dev in Error: {std_dev_in_err:.4f}%")
