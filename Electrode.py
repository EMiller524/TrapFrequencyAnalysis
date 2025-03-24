import math
import time
import pandas as pd
import dataextractor
import os
import consts
import numexpr as ne


class Electrode:

    def __init__(self, name, dataset, variables=[0, 0, 0, 0]):
        """
        Initialize the Electrode object.

        Parameters:
        - name (str): The name of the electrode.
        - voltage (float): The applied voltage.
        - data (pd.DataFrame, optional): A pandas DataFrame containing relevant electrode data.
        """
        start = time.time()
        # print("electrode init started")
        self.name = name
        self.file_path = (
            "C:\\GitHub\\TrapFrequencyAnalysis\\Data\\" + dataset + "\\" + self.name
        )
        self.Amplitude = variables[0]
        self.Frequency = variables[1]
        self.Phase = variables[2]
        self.Offset = variables[3]

        self.points = ()
        self.data = None

        # if the etracted data is already saved then define it as self.data
        if os.path.exists(self.file_path + "_extracted.csv"):
            self.data = pd.read_pickle(self.file_path + "_extracted.csv")

        elif os.path.exists(self.file_path + "_Raw.txt"):
            print("extracting dataaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
            dataextractor.extract_raw_trap_sim_data(self.file_path + "_Raw.txt")
            self.data = pd.read_pickle(self.file_path + "_extracted.csv")

        catch = False
        # Catch RF12 and sum the ExEyEz's from Rf1 and Rf2
        if self.name == "RF12" and catch:
            print("")
            print("Summing Rf1 and Rf2 Efeilds for RF12")
            print("")
            rf1 = Electrode("RF1", dataset, variables)
            rf2 = Electrode("RF2", dataset, variables)
            self.data = rf1.get_dataframe()
            self.data["Ex"] += rf2.get_dataframe()["Ex"]
            self.data["Ey"] += rf2.get_dataframe()["Ey"]
            self.data["Ez"] += rf2.get_dataframe()["Ez"]
            self.data["V"] += rf2.get_dataframe()["V"]

        # self.get_feild_mag_at_all_points()
        self.get_potential_at_all_points()

        end = time.time()
        # print("electrode init done" + str(end - start))

    def get_dataframe(self):
        return self.data

    def change_varaibles(self, variables):
        self.Amplitude = variables[0]
        self.Frequency = variables[1]
        self.Phase = variables[2]
        self.Offset = variables[3]

        self.get_feild_mag_at_all_points()
        self.get_potential_at_all_points()

    def determine_points(self):
        if self.data is None:
            return
        self.points = dataextractor.get_set_of_points(self.data)

    def get_potential_at_point(self, x, y, z):
        # search the dataframe for the point and return the "potential" value
        if self.data is None:
            return None

        return self.data.loc[
            (self.data["x"] == x) & (self.data["y"] == y) & (self.data["z"] == z), "CalcV"
        ].values[0]

    # for row in data add in a new column with the calculated potential
    def get_feild_mag_at_all_points(self):
        if self.data is None:
            return
        self.data["EMag"] = self.data.apply(
            lambda row: math.sqrt(row["Ex"]**2 + row["Ey"]**2 + row["Ez"]**2), axis=1
        )

    def get_potential_at_all_points(self):
        # print("getting potential for all points")
        amp = self.Amplitude
        freq = self.Frequency
        offset = self.Offset
        Q = consts.ion_charge
        M = consts.ion_mass

        if self.data is None:
            return

        # if voltage is constant
        # if self.Frequency == 0:
        #     self.data["CalcV"] = self.data.apply(
        #         lambda row: amp * row["V"] + offset,
        #         axis=1,
        #     )
        if self.Frequency == 0:
            self.data["CalcV"] = ne.evaluate(
                "amp * V + offset",
                local_dict={
                    "amp": amp,
                    "V": self.data["V"],
                    "offset": offset,
                },
            )

        # if voltage needs to be time averaged
        else:
            # # Vectorized calculation
            # self.data["CalcV"] = (amp**2 * Q * (self.data["Ex"]**2 + self.data["Ey"]**2 + self.data["Ez"]**2)) / (4 * M * freq**2)

            # other
            self.data["CalcV"] = ne.evaluate(
                "(amp**2 * Q * (Ex**2 + Ey**2 + Ez**2)) / (4 * M * freq**2)",
                local_dict={
                    "amp": amp,
                    "Q": Q,
                    "Ex": self.data["Ex"],
                    "Ey": self.data["Ey"],
                    "Ez": self.data["Ez"],
                    "M": M,
                    "freq": freq,
                },
            )

        #     self.data["CalcV"] = self.data.apply(
        #     lambda row: ((amp * amp * Q * (row["Ex"]**2 + row["Ey"]**2 + row["Ez"]**2)) /
        #                  (4* M * (freq**2))), axis=1
        #     # Pseudo potential Eq #####################################################
        # )

# rf12 = Electrode("RF12", "Simplified1")
# rf12.change_varaibles([0, 28000000 * 2 * math.pi, 0, 0])

# df = rf12.get_dataframe()
# print("hi")
# # save the dataframe to a csv file
# # df.to_csv("C:\\GitHub\\TrapFrequencyAnalysis\\Data\\Simplified1\\RF12_testdf.csv")

# print("pot:" + str(rf12.get_potential_at_point(0,0,0)))

# print()
