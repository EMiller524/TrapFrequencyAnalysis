import constants as c

def get_potential_nofreq(electrode, potential):
    return c.electrode[0] * potential

def get_potential_freq(electrode, potential):
    return c.electrode[0] * potential * np.sin(2 * np.pi * c.electrode[1] * 10**3 * t)  + c.electrode[2]


#Takes in a point and a dataframe and returns a 12 elemetn tuple with each elemetn being a ordered pair of electrode, potential
