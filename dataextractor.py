# This file will contain all nessecary fucntions to take in a raw data file and extract the data from it

import pandas as pd
import numpy as np
import os
import re
import datetime
import time
import csv
import consts

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
        df[column] = df[column].round(12)

    # now we will find the dimensions of the data, meaning how many distinct x, y, z values are there and how sperated each axis's sampling is
    # we will return the dimensions as a tuple
    x_dimension = len(df["x"].unique())
    y_dimension = len(df["y"].unique())
    z_dimension = len(df["z"].unique())

    x_spacing = round(float(df["x"].unique()[1] - df["x"].unique()[0]), 8)
    y_spacing = round(float(df["y"].unique()[1] - df["y"].unique()[0]), 8)
    z_spacing = round(float(df["z"].unique()[1] - df["z"].unique()[0]), 8)

    dimension = (("x_dim", x_dimension), ("x_spacing", x_spacing), 
                 ("y_dim", y_dimension), ("y_spacing", y_spacing),
                 ("z_dim", z_dimension), ("z_spacing", z_spacing))

    # add the dimension as a tupple to the dataframe under the name "dim"
    df.attrs["dim"] = dimension

    df.to_pickle(
       "C:\\GitHub\\TrapFrequencyAnalysis\\Data\\" + simulation + "\\" + blade_name + "_extracted.csv")

    return df

def make_simulation_dataframe(folder_path):
    # TODO
    """
    Create a dataframe from all the extracted data files in a given sim.

    Parameters:
    folder_path (str): The path to the folder containing the extracted data files.

    Returns:
    pd.DataFrame: DataFrame containing the combined data from all files.
    also saves the df as a pickle thing
    """

    # for each txt file in folder_path check if the corresponding csv file exists, if it does skip this txt file
    # if it does not exist, extract the data from the txt file and save it as a csv file using the extract_raw_trap_sim_data function
    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            csv_file = os.path.join(folder_path, file.replace("_Raw.txt", "_extracted.csv"))
            if not os.path.exists(csv_file):
                extract_raw_trap_sim_data(os.path.join(folder_path, file))

    # Now we shall merge these dataframes into a single dataframe
    # This data frame columns will be ["x", "y", "z", and then for each electrode in the simulation, "V", "Ex", "Ey", "Ez" for each electrode] ex: RF1_V, RF1_Ex, RF1_Ey, RF1_Ez, etc.
    # The data frame will also have a column for TotalV
    # We will also give the dataframe an atrabute called electrode_vars, which will be None for now

    # init the dataframe
    df = pd.DataFrame()
    # get the list of all csv files in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith("_extracted.csv")]
    # print(f"Found {len(csv_files)} csv files in {folder_path}")
    # print(f"csv files: {csv_files}")
    # for each csv file, read it and append it to the dataframe
    names_of_electodes = []  # to keep track of electrode names for later use
    for csv_file in csv_files:
        # Extract the electrode name from the file name
        electrode_name = os.path.basename(csv_file).split("_")[
            0
        ]  # Adjust the split logic based on your file naming convention

        # read the csv file
        file_path = os.path.join(folder_path, csv_file)
        temp_df = pd.read_pickle(file_path)

        # Rename the columns for V, Ex, Ey, Ez
        temp_df.rename(
            columns={
                "V": f"{electrode_name}_V",
                "Ex": f"{electrode_name}_Ex",
                "Ey": f"{electrode_name}_Ey",
                "Ez": f"{electrode_name}_Ez",
            },
            inplace=True,
        )

        names_of_electodes.append(electrode_name)  # add the electrode name to the list

        # Merge it with the main dataframe on x, y, z columns using inner join
        if df.empty:
            df = temp_df
        else:
            df = pd.merge(df, temp_df, on=["x", "y", "z"], how="inner")

    df["TotalV"] = np.nan

    # Add custom attributes using the attrs property
    df.attrs['electrode_names'] = names_of_electodes
    df.attrs['electrode_vars'] = consts.Electrode_vars()
    df.attrs['name'] = os.path.basename(folder_path)

    # Save the combined dataframe as a pickle file
    df.to_pickle(os.path.join(folder_path, "combined_dataframe.csv"))
    
    # print the number of points in the dataframe
    print(f"Number of points in the dataframe: {len(df)}")
    print(f"Dataframe shape: {df.shape}")

    return df

## Just for testing ##
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

# print("HI")
# make_simulation_dataframe("C:\GitHub\TrapFrequencyAnalysis\Data\Simp58_101_copy")