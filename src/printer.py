import numpy as np

def print_config(config_dict):
    string = "   Configuration input data:\n\n"
    for key, val in config_dict.items():
        if not isinstance(val,str):
            val = str(val)
        string += f"    {key:>12}: {val:<12} \n"
    print(string)

def print_initial_conditions(x_init,v_init,a_init):
    string = "\n   Initial conditions:\n\n"
    string += "          x0      |       v0       |       a0\n"
    string += "    -----------------------------------------------\n"
    for x,v,a in zip(x_init,v_init,a_init):
        string += "     {:5.9f}  |  {: 5.9f}  |  {: 5.9f}\n".format(x,v,a)
    string += "    -----------------------------------------------"
    print(string)

def print_energies(ekin, epot, step, time):
    string = "\n==============================================================\n"
    string += " Computed energies at step " 
    string += "{} (t = {:10.5f} [fs]):\n".format(step,time)
    string += "==============================================================\n\n"
    string += "         Potential  |    Kinetic   |    Total\n"
    
    P_actual = np.format_float_positional(epot[-1], unique=False, precision=8)
    K_actual = np.format_float_positional(ekin[-1], unique=False, precision=8)
    T_actual = np.format_float_positional((epot[-1] + ekin[-1]), unique=False, precision=8)

    #if step == 0:
    if len(ekin) == 1:    
        string += "t-dt:   0.00000000  |  0.00000000  |  0.00000000\n"
    else:
        P_old = np.format_float_positional(epot[-2], unique=False, precision=8) 
        K_old = np.format_float_positional(ekin[-2], unique=False, precision=8)
        T_old = np.format_float_positional((epot[-2] + ekin[-2]), unique=False, precision=8)
        string += "t-dt:  " + str(P_old) + "  |  "
        string += str(K_old) + "  |  "
        string += str(T_old) + "\n"

    string += "t:     " + str(P_actual) + "  |  "
    string += str(K_actual) + "  |  "
    string += str(T_actual)
    print(string)
    print(" ")
