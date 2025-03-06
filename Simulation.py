from functools import reduce
import math
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import dataextractor
import consts
import Electrode
import time
import matplotlib.cm as cm
import concurrent.futures
from SimulationMix.simulation_ploting import sim_ploting
from SimulationMix.simulation_hessianfitting import sim_hessian
from SimulationMix.simulation_normalfitting import sim_normalfitting

def init_electrode(electrode_name, dataset, electrode_vars):
    return electrode_name, Electrode.Electrode(
        electrode_name, dataset, electrode_vars.get_vars(electrode_name)
    )


class Simulation(sim_ploting, sim_hessian, sim_normalfitting):
    def __init__(self, dataset, variables=consts.Electrode_vars()):
        """
        Initialize the Electrode object.
        """
        print("initializing simulation")
        self.dataset = dataset
        self.file_path = "C:\\GitHub\\TrapFrequencyAnalysis\\Data\\" + dataset + "\\"

        # make a dictionary of electrodes with each name in consts.electrode_names as the key and an electrode class as the value
        self.electrode_vars = variables

        self.electrodes = {}
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    init_electrode, electrode, self.dataset, self.electrode_vars
                )
                for electrode in consts.electrode_names
            ]
            for future in concurrent.futures.as_completed(futures):
                electrode_name, electrode_instance = future.result()
                self.electrodes[electrode_name] = electrode_instance

        self.valid_points = self.get_valid_points()

        self.total_voltage_df = None
        self.get_total_voltage_at_all_points()

        print("simulation initialized")

    def get_variables(self, electrode):
        return self.electrode_vars.get_vars(electrode)

    def get_valid_points(self):
        non_empty_dfs = []
        for electrode in self.electrodes:
            if self.electrodes[electrode].get_dataframe() is not None:
                non_empty_dfs.append(self.electrodes[electrode].get_dataframe())

        point_sets = [set(zip(df["x"], df["y"], df["z"])) for df in non_empty_dfs]

        common_points = set.intersection(*point_sets)

        return np.array(sorted(common_points))

    def get_total_voltage_at_all_points(self):
        # get the data for each electrode
        dfs = []
        for electrode in consts.electrode_names:
            if self.electrodes[electrode].get_dataframe() is None:
                print(f"no data for {electrode}")
            else:
                dfs.append(self.electrodes[electrode].get_dataframe())

        # Step 1: Find the intersection of all unique (x, y, z) combinations
        common_keys = reduce(
            lambda left, right: pd.merge(left, right, on=["x", "y", "z"]),
            [df[["x", "y", "z"]] for df in dfs],
        )

        # Step 2: Concatenate all dataframes
        merged_df = pd.concat(dfs, ignore_index=True)

        # Step 3: Filter merged_df to only keep common (x, y, z) keys
        filtered_df = merged_df.merge(common_keys, on=["x", "y", "z"])

        # Step 4: Group by (x, y, z) and sum CalcV
        master_df = filtered_df.groupby(["x", "y", "z"], as_index=False)["CalcV"].sum()

        self.total_voltage_df = master_df

    def get_electrode(self, name):
        return self.electrodes[name]


elec_vars = consts.Electrode_vars()
elec_vars.set_vars("RF12", [377, 28000000 * 2 * math.pi, 0, 0])
# elec_vars.set_vars("RF2", [377, 28000000 * 2 * math.pi, 0, 0])
# elec_vars.set_vars("DC1", [0, 0, 0, 0])
# elec_vars.set_vars("DC5", [0, 0, 0, 0])
# elec_vars.set_vars("DC6", [0, 0, 0, 0])
# elec_vars.set_vars("DC10", [0, 0, 0, 0])

test_sim = Simulation("Simplified2", elec_vars)

print(test_sim.get_frequencys_at_point_xyz(0, 0, 0))

test_sim.get_full_report("Simp58(000)_377_28_DC_Grounded_NotCaught_RF_test")

# fig1 = test_sim.plot_potential_in_xyz_directions(0, 0, 0)
# fig2 = test_sim.plot_potential_in_xyz_directions(0, 0, 0, 0.0003, 0.00005, 0.00005)

# fig3 = test_sim.plot_freq_in_xyz_directions(0, 0, 0)
# fig4 = test_sim.plot_freq_in_xyz_directions(0, 0, 0, 0.0003, 0.00005, 0.00005)
# plt.show()


# Other todo
# Get the data again from comsol (grounded vs not??, added vs not??) (Yes)
# Clean up code, make the "catch RF" more robust
# Make a ploting class?
# Find ways to make the code run faster!!!
# find out if the (+) and (-) x^4 coeffs on the quartic fit is an error... (Ughhhhhhhhh)

# Hessian bullshit
# freq = test_sim.get_frequencys_at_point_hess(0, .00005, 0.00002)
# print('For point 0, 0, 0')
# for i in range(3):
#     print("Frequency in direction " + str(freq[i][1]) + " is " + str(freq[i][0]) + " Hz")
