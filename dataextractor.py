# This file will contain all nessecary fucntions to take in a raw data file and extract the data from it

import pandas as pd
import numpy as np
import os
import re
import datetime
import time
import csv

# Funtion will take in a data file with the bellow data and return a pandas dataframe with the data
# Data in the raw data, x, y, z, V, Ex, Ey, Ez


def extract_raw_trap_sim_data(file_path):
    """
    Extract raw data from a text file with the following columns: x, y, z, V, Ex, Ey, Ez.

    Parameters:
    file_path (str): The path to the text file.

    Returns:
    pd.DataFrame: DataFrame containing the extracted data.
    """
    # extract the last bit of the file path to use as the name of the dataframe
    blade_name = os.path.basename(file_path).split(".")[0].split("_")[0]
    
    #extract the simulation from the file path, meaing the "simplified1" or "simplified2" part of the file path
    simulation = file_path.split("\\")[4]
    
    print("Extracting data from " + blade_name + " in " + simulation + " simulation")
    
    
    # Read the file, skipping the metadata lines
    df = pd.read_csv(file_path, sep="\s+", skiprows=9)

    # Assign meaningful column names based on the file's description
    df.columns = [
        "x",
        "y",
        "z",
        "V",
        "Ex",
        "Ey",
        "Ez",
    ]
    
    #Now we iterated through x,y,z and divide eveery value by 1000 to convert it from mm to m
    for column in ["x", "y", "z"]:
        df[column] = df[column] / 1000
    
    
    # #Now convert the Ex,Ey,Ez to standard SI units
    # for column in ["Ex", "Ey", "Ez"]:
    #     df[column] = df[column]
        
    # #and now for V
    # df["V"] = df["V"]
    
    
    # Now we will iterate through all the columns and round the values to a specified number of decimal places
    for column in df.columns:
        df[column] = df[column].round(9)

    # now we will find the dimensions of the data, meaning how many distinct x, y, z values are there and how sperated each axis's sampling is
    # we will return the dimensions as a tuple
    x_dimension = len(df["x"].unique())
    y_dimension = len(df["y"].unique())
    z_dimension = len(df["z"].unique())

    x_spacing = round(float(df["x"].unique()[1] - df["x"].unique()[0]), 6)
    y_spacing = round(float(df["y"].unique()[1] - df["y"].unique()[0]), 6)
    z_spacing = round(float(df["z"].unique()[1] - df["z"].unique()[0]), 6)

    dimension = (("x_dim", x_dimension), ("x_spacing", x_spacing), 
                 ("y_dim", y_dimension), ("y_spacing", y_spacing),
                 ("z_dim", z_dimension), ("z_spacing", z_spacing))

    # add the dimension as a tupple to the dataframe under the name "dim"
    df.attrs["dim"] = dimension

    df.to_pickle(
       "C:\\GitHub\\TrapFrequencyAnalysis\\Data\\" + simulation + "\\" + blade_name + "_extracted.csv")

    return df


# define a fucntion that takes in a dataframe and a point and returns the electric field at that point
def get_val_from_point(dataframe, x, y, z, val):
    """
    Get the electric field value (V) at a specific point (x, y, z) from the dataframe.

    Args:
        dataframe (pd.DataFrame): The dataframe containing the data.
        x (float): The x-coordinate of the point.
        y (float): The y-coordinate of the point.
        z (float): The z-coordinate of the point.

    Returns:
        float: The electric field value (V) at the specified point, or a large negative value if the point is not found.
    """
    # Filter the dataframe to find the exact point
    filtered_df = dataframe[
        (dataframe["x"] == x) & (dataframe["y"] == y) & (dataframe["z"] == z)
    ]

    if not filtered_df.empty:
        # Return the V value if the point is found
        return filtered_df.iloc[0][val]
    else:
        # Return a large negative value if the point is not found
        return -1e6

def get_V_from_point(dataframe, x, y, z):
    return get_val_from_point(dataframe, x, y, z, "V")

def get_Ex_from_point(dataframe, x, y, z):
    return get_val_from_point(dataframe, x, y, z, "Ex")

def get_Ey_from_point(dataframe, x, y, z):
    return get_val_from_point(dataframe, x, y, z, "Ey")

def get_Ez_from_point(dataframe, x, y, z):
    return get_val_from_point(dataframe, x, y, z, "Ez")

def get_all_from_point(dataframe, x, y, z):
    # Use the query method for efficient filtering
    print("geting points from data frame")
    filtered_df = dataframe.query("x == @x and y == @y and z == @z")

    if not filtered_df.empty:
        # Directly access the values using iat for better performance
        return [
            filtered_df.iat[0, dataframe.columns.get_loc("V")],
            filtered_df.iat[0, dataframe.columns.get_loc("Ex")],
            filtered_df.iat[0, dataframe.columns.get_loc("Ey")],
            filtered_df.iat[0, dataframe.columns.get_loc("Ez")]
        ]
    else:
        # Return a large negative value if the point is not found
        return -1e6

    # return [
    #     get_V_from_point(dataframe, x, y, z),
    #     get_Ex_from_point(dataframe, x, y, z),
    #     get_Ey_from_point(dataframe, x, y, z),
    #     get_Ez_from_point(dataframe, x, y, z),
    # ]

def get_set_of_points(dataframe):
    return set(zip(dataframe["x"], dataframe["y"], dataframe["z"]))


# dataframe = pd.read_pickle(
#     "C:\GitHub\TrapFrequencyAnalysis\Data\Simplified1\RF12_extracted.csv"
# )

# print(get_set_of_points(dataframe))

# extract_raw_trap_sim_data("C:\\GitHub\\TrapFrequencyAnalysis\\Data\\Simplified1\\RF12_Raw.txt")