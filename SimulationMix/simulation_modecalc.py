"""
This file will contain the class Mode_Calculations
"""

import math
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import r2_score, mean_squared_error
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import RBFInterpolator
import constants
from scipy.optimize import minimize, BFGS, basinhopping


class Mode_Calculations:
    """
    This class will be inherited by the simulation class and will contain all fucntions used to find modes and their coupling.
    """
        
    def get_mode_eigenvec_and_val(self):
        """
        This function will return the eigenvalues and eigenvectors of the hessian matrix, in the desired order.
        """
        pass
    
    def get_mode_couplings(self):
        pass
    
    
