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


def extract_data(file_path, delimiter="\t", header=0):
    """
    Extract data from a text file with a specified delimiter and header.

    Parameters:
    file_path (str): The path to the text file.
    delimiter (str): The delimiter used in the text file (default is tab).
    header (int): Row number to use as the column names (default is 0).

    Returns:
    pd.DataFrame: DataFrame containing the extracted data.
    """
    try:
        df = pd.read_csv(file_path, delimiter=delimiter, header=header)
        return df
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    


def extract_raw_trap_sim_data(file_path):
    """
    Extract raw data from a text file with the following columns: x, y, z, V, Ex, Ey, Ez.

    Parameters:
    file_path (str): The path to the text file.

    Returns:
    pd.DataFrame: DataFrame containing the extracted data.
    """
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

    # Now we will iterate through all the columns and round the values to a specified number of decimal places
    for column in df.columns:
        df[column] = df[column].round(6)

    # now we will find the dimensions of the data, meaning how many distinct x, y, z values are there and how sperated each axis's sampling is
    # we will return the dimensions as a tuple
    x_dimension = len(df["x"].unique())
    y_dimension = len(df["y"].unique())
    z_dimension = len(df["z"].unique())

    x_spacing = round(float(df["x"].unique()[1] - df["x"].unique()[0]), 6)
    y_spacing = round(float(df["y"].unique()[1] - df["y"].unique()[0]), 6)
    z_spacing = round(float(df["z"].unique()[1] - df["z"].unique()[0]), 6)

    dimension = (x_dimension, x_spacing, y_dimension, y_spacing, z_dimension, z_spacing)

    return df, dimension


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
    return (
        get_V_from_point(dataframe, x, y, z),
        get_Ex_from_point(dataframe, x, y, z),
        get_Ey_from_point(dataframe, x, y, z),
        get_Ez_from_point(dataframe, x, y, z),
    )


dataframe, dim = extract_raw_trap_sim_data(
    "C:\GitHub\TrapFrequencyAnalysis\Data\Datatest1.txt"
)

values = dataframe["V"].tolist()
print(values[:10])
print(dim)

v_value = get_all_from_point(dataframe, -.99, -.1, -.1)
print(v_value)

dataframe.to_csv(
    "C:\\GitHub\\TrapFrequencyAnalysis\\Data\\exported_data.csv", index=False
)
