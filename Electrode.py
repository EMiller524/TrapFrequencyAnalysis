import math
import pandas as pd
import dataextractor
import os
import consts


class Electrode:

    def __init__(self, name, dataset, variables = [0,0,0,0]):
        """
        Initialize the Electrode object.

        Parameters:
        - name (str): The name of the electrode.
        - voltage (float): The applied voltage.
        - data (pd.DataFrame, optional): A pandas DataFrame containing relevant electrode data.
        """
        self.name = name
        self.file_path = "C:\\GitHub\\TrapFrequencyAnalysis\\Data\\" + dataset + "\\" + self.name
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
            dataextractor.extract_raw_trap_sim_data(self.file_path + "_Raw.txt")
            self.data = pd.read_pickle(self.file_path + "_extracted.csv")
            
        self.determine_points()

    def get_dataframe(self):
        return self.data

    def set_varaibles(self, variables):
        self.Amplitude = variables[0]
        self.Frequency = variables[1]
        self.Phase = variables[2]
        self.Offset = variables[3]
        
    def determine_points(self):
        if self.data is None:
            return
        dataextractor.get_set_of_points(self.data)
        
    def get_poionts(self):
        return self.points

    def get_all_at_point(self,x,y,z):
        """
        Get all the electric field values at a specified point.

        Parameters:
        - x (float): The x-coordinate of the point.
        - y (float): The y-coordinate of the point.
        - z (float): The z-coordinate of the point.

        Returns:
        - tuple: A tuple containing the electric field values at the specified point.
        """
        if self.data is None:
            return [0,0,0,0]
        return dataextractor.get_all_from_point(self.data, x, y, z)

    def get_Vraw_at_point(self,x,y,z):
        """
        Get the voltage at a specified point.

        Parameters:
        - x (float): The x-coordinate of the point.
        - y (float): The y-coordinate of the point.
        - z (float): The z-coordinate of the point.

        Returns:
        - float: The voltage at the specified point.
        """
        return self.get_all_at_point(x, y, z)[0]

    def get_Exraw_at_point(self,x,y,z):
        """
        Get the electric field at a specified point.

        Parameters:
        - x (float): The x-coordinate of the point.
        - y (float): The y-coordinate of the point.
        - z (float): The z-coordinate of the point.

        Returns:
        - float: The electric field at the specified point.
        """
        return self.get_all_at_point(x, y, z)[1]

    def get_Eyraw_at_point(self,x,y,z):
        """
        Get the electric field at a specified point.

        Parameters:
        - x (float): The x-coordinate of the point.
        - y (float): The y-coordinate of the point.
        - z (float): The z-coordinate of the point.

        Returns:
        - float: The electric field at the specified point.
        """
        return self.get_all_at_point(x, y, z)[2]

    def get_Ezraw_at_point(self,x,y,z):
        """
        Get the electric field at a specified point.

        Parameters:
        - x (float): The x-coordinate of the point.
        - y (float): The y-coordinate of the point.
        - z (float): The z-coordinate of the point.

        Returns:
        - float: The electric field at the specified point.
        """
        return self.get_all_at_point(x, y, z)[3]

    def get_feildmag_at_point(self,x,y,z):
        """
        Get the electric field at a specified point.

        Parameters:
        - x (float): The x-coordinate of the point.
        - y (float): The y-coordinate of the point.
        - z (float): The z-coordinate of the point.

        Returns:
        - float: The electric field at the specified point.
        """
        allpoints = self.get_all_at_point(x,y,z)

        return math.sqrt(allpoints[1]**2 + allpoints[2]**2 + allpoints[3]**2)

    def get_potential_at_point_using_var(self, x, y, z):
        # if voltage is constant
        if self.Frequency == 0:
            return self.Amplitude * self.get_Vraw_at_point(x, y, z) + self.Offset

        # if voltage needs to be time averaged
        else:
            return (self.Amplitude * consts.ion_charge * (self.get_feildmag_at_point(x,y,z))**2)/(4 * consts.ion_mass * (self.Frequency)**2)


# rf12 = Electrode("DC1", "Simplified1")
# print(rf12.get_potential_at_point_using_var(0,0,0))
