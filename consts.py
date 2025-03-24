# Description: This file contains the constants used in the simulation
ion_mass = 2.885 * (10 **(-25)) #kg
ion_charge = 1.60217662  * (10 **(-19)) #C

electrode_names = (
    "DC1",
    "DC2",
    "DC3",
    "DC4",
    "DC5",
    "DC6",
    "DC7",
    "DC8",
    "DC9",
    "DC10",
    "RF1",
    "RF2",
    "RF12"
)

class Electrode_vars:
    def __init__(self, DC1 = [0,0,0,0],
                        DC2 = [0,0,0,0],
                        DC3 = [0,0,0,0],
                        DC4 = [0,0,0,0],
                        DC5 = [0,0,0,0],
                        DC6 = [0,0,0,0],
                        DC7 = [0,0,0,0],
                        DC8 = [0,0,0,0],
                        DC9 = [0,0,0,0],
                        DC10 = [0,0,0,0],
                        RF1= [0,0,0,0],
                        RF2 = [0,0,0,0],
                        RF12 = [0,0,0,0]):
        self.vars = {}
        for electrode in electrode_names:
            self.vars[electrode] = eval(electrode)
        
    def set_vars(self, electrode, var):
        self.vars[electrode] = var
    
    def get_vars(self, electrode):
        return self.vars[electrode]
    
    def get_amplitude(self, electrode):
        return self.vars[electrode][0]
    
    def get_frequency(self, electrode):
        return self.vars[electrode][1]
    
    def get_offset(self, electrode):
        return self.vars[electrode][2]
    
    def get_phase(self, electrode):
        return self.vars[electrode][3]

def get_electrodvars_w_twist(rfamp, rffreq, twist, endcaps):
    RF1 = [twist, 0, 0, 0]
    RF2 = [twist, 0, 0, 0]
    RF12 = [rfamp, rffreq, 0, 0]

    DC1 = [endcaps - twist, 0, 0, 0]
    DC2 = [(-1) * twist, 0, 0, 0]
    DC3 = [(-1) * twist, 0, 0, 0]
    DC4 = [(-1) * twist, 0, 0, 0]
    DC5 = [endcaps - twist, 0, 0, 0]
    DC6 = [endcaps - twist, 0, 0, 0]
    DC7 = [(-1) * twist, 0, 0, 0]
    DC8 = [(-1) * twist, 0, 0, 0]
    DC9 = [(-1) * twist, 0, 0, 0]
    DC10 = [endcaps - twist, 0, 0, 0]

    return Electrode_vars(DC1=DC1,
                          DC2=DC2,
                          DC3=DC3,
                          DC4=DC4,
                          DC5=DC5,
                          DC6=DC6,
                          DC7=DC7,
                          DC8=DC8,
                          DC9=DC9,
                          DC10=DC10,
                          RF1=RF1,
                          RF2=RF2,
                          RF12=RF12)

def get_electrodvars_w_twist_and_push(rfamp, rffreq, twist = 0, endcaps = 0, pushx = 0, pushy = 0, pushz = 0):
    regular_twist = get_electrodvars_w_twist(rfamp, rffreq, twist, endcaps)

    # pushx
    regular_twist.vars["DC1"][0] -= pushx
    regular_twist.vars["DC10"][0] -= pushx
    regular_twist.vars["DC6"][0] += pushx
    regular_twist.vars["DC5"][0] += pushx

    # pushy and pushz
    regular_twist.vars["DC1"][0] += (pushy - pushz)
    regular_twist.vars["DC2"][0] += (pushy - pushz)
    regular_twist.vars["DC3"][0] += (pushy - pushz)
    regular_twist.vars["DC4"][0] += (pushy - pushz)
    regular_twist.vars["DC5"][0] += (pushy - pushz)
    regular_twist.vars["DC6"][0] += (-pushy + pushz)
    regular_twist.vars["DC7"][0] += (-pushy + pushz)
    regular_twist.vars["DC8"][0] += (-pushy + pushz)
    regular_twist.vars["DC9"][0] += (-pushy + pushz)
    regular_twist.vars["DC10"][0] += (-pushy + pushz)
    regular_twist.vars["RF1"][0] += (pushy + pushz)
    regular_twist.vars["RF2"][0] += (-pushy - pushz)


    return regular_twist
