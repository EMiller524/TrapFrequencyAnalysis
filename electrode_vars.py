import constants

class Electrode_vars:
    def __init__(
        self,
        DC1=[0, 0.01, 0, 0],
        DC2=[0, 0.01, 0, 0],
        DC3=[0, 0.01, 0, 0],
        DC4=[0, 0.01, 0, 0],
        DC5=[0, 0.01, 0, 0],
        DC6=[0, 0.01, 0, 0],
        DC7=[0, 0.01, 0, 0],
        DC8=[0, 0.01, 0, 0],
        DC9=[0, 0.01, 0, 0],
        DC10=[0, 0.01, 0, 0],
        RF1=[0, 0.01, 0, 0],
        RF2=[0, 0.01, 0, 0],
    ):
        self.vars = {}
        for electrode in constants.electrode_names:
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
    RF1 = [rfamp, rffreq, twist, 0]
    RF2 = [rfamp, rffreq, twist, 0]

    DC1 = [0, 0, endcaps - twist, 0]
    DC2 = [0, 0, -twist, 0]
    DC3 = [0, 0, -twist, 0]
    DC4 = [0, 0, -twist, 0]
    DC5 = [0, 0, endcaps - twist, 0]
    DC6 = [0, 0, endcaps - twist, 0]
    DC7 = [0, 0, -twist, 0]
    DC8 = [0, 0, -twist, 0]
    DC9 = [0, 0, -twist, 0]
    DC10 = [0, 0, endcaps - twist, 0]

    return Electrode_vars(
        DC1=DC1,
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
    )


def get_electrodvars_w_twist_and_push(
    rfamp, rffreq, twist=0, endcaps=0, pushx=0, pushy=0, pushz=0
):
    regular_twist = get_electrodvars_w_twist(rfamp, rffreq, twist, endcaps)

    # pushx
    regular_twist.vars["DC1"][2] -= pushx
    regular_twist.vars["DC10"][2] -= pushx
    regular_twist.vars["DC6"][2] += pushx
    regular_twist.vars["DC5"][2] += pushx

    # pushy and pushz
    regular_twist.vars["DC1"][2] += pushy - pushz
    regular_twist.vars["DC2"][2] += pushy - pushz
    regular_twist.vars["DC3"][2] += pushy - pushz
    regular_twist.vars["DC4"][2] += pushy - pushz
    regular_twist.vars["DC5"][2] += pushy - pushz
    regular_twist.vars["DC6"][2] += -pushy + pushz
    regular_twist.vars["DC7"][2] += -pushy + pushz
    regular_twist.vars["DC8"][2] += -pushy + pushz
    regular_twist.vars["DC9"][2] += -pushy + pushz
    regular_twist.vars["DC10"][2] += -pushy + pushz
    regular_twist.vars["RF1"][2] += pushy + pushz
    regular_twist.vars["RF2"][2] += -pushy - pushz

    return regular_twist


def get_electrodvars_w_list(list_of_vars):
    return Electrode_vars(DC1 = list_of_vars[0],
                          DC2 = list_of_vars[1],
                          DC3 = list_of_vars[2],
                          DC4 = list_of_vars[3],
                          DC5 = list_of_vars[4],
                          DC6 = list_of_vars[5],
                          DC7 = list_of_vars[6],
                          DC8 = list_of_vars[7],
                          DC9 = list_of_vars[8],
                          DC10 = list_of_vars[9],
                          RF1 = list_of_vars[10],
                          RF2 = list_of_vars[11])
