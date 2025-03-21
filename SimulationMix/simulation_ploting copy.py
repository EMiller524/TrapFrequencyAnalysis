import numpy as np 
import pandas as pd
import consts
#TODO
# Will replace the need for elctrode. Instead a main data frame will be used for a simulation
# This class will init that dataframe and include the nessecary extraction if nessecary. 
#       (Working with an overhaul of dataextractor)
# The idea is to have a saved df that has everything possible that is independent of electrode variables
# And then this subclass will take that df and alter it based on the electrode varaibles

# Ok so for pseudoV there are only so many combinations of electrodes, 
    # maybe all just get calculated in dataextractor??
# For now we work under the assumption that only RF12 can have RF

class sim_df_maneger:
    
    def init_data_frame(self, electrode_vars = None):
        if electrode_vars is None:
            electrode_vars = consts.Electrode_vars()
        return self.update_data_frame(electrode_vars)
    
    def update_data_frame(self, electrode_vars):
        return

