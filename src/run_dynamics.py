import os
import sys
import ast
import yaml
import time
import psutil
import numpy as np

from printer import *
from models import SpinBoson as sbh
from models import FitPES as fpes
from integrators import Integrator
from trajectory import Trajectory

pid = os.getpid()

#%% Auxiliary constants
ang2bohr = 1.889725989

#%% Auxiliary functions
def read_sbh_inputs(input_file):
    data = np.loadtxt(input_file)
    erro_msg = "Wrong shape! Check the " + input_file + "input file." 

    try:
        ncols = data.shape[1]
    except:    
        print(erro_msg)

    if ncols == 3:
        f, m, g = (data[:,i] for i in range(3))
        return f, m, g
    elif ncols == 2:
        coords, veloc = (data[:,i] for i in range(2))
        return coords, veloc
    else:
        print(erro_msg)    

def timeunits(t, conv='fs2au'):
    constant = 2.418884326509 * 10**(-2)
    if conv == 'fs2au':
        return t / constant
    elif conv == 'au2fs':
        return t * constant
    else:
        print("Time units not recognized.")

def kinetic(mass, veloc):
#    mass = mass * 1822.888515
    Ekin = 0.5 * mass @ veloc**2
    return Ekin

def memory_usage(pip):
    '''This function is used to monitor the memory usage during the 
    simulation run. If the memory exceed a given threshold, the partial
    results are stored as npz and the variables are cleaned to save memory.'''
    py = psutil.Process(pid)
    mem_usage = py.memory_info()[0]*(9.53674*10**(-7))  # memory use in MB
    mem_perc = py.memory_percent()
    print("Memory usage is {:.2f} MB. Percentage: {:.2f}%".format(mem_usage, mem_perc))
    return mem_perc

def check_energy_conserv(Ekin, Epot, step, verbose=False):
    Etot = Ekin + Epot
    DEtot = Etot - Etot[0]
    DEtot_mean_center = DEtot - np.mean(DEtot)
    DEtot_std = np.std(DEtot_mean_center, ddof=1)
    DEtot_med = np.median(DEtot_mean_center)
    if verbose:
        print("----------------------------------------------------")
        print("Checking energy conservation\n")
        print("Standard deviation of Etot: {:.5E} ha".format(DEtot_std))
        print("Median of centered Etot: {:.5E} ha".format(DEtot_med))
        print("Etot deviation in the current step: {:.5E} ha".format(DEtot[-1]))
        print("----------------------------------------------------\n")
    return DEtot_std

#%% Defining SB model parameters from the 'config.yml' file
def create_sbh_model(conf_dict):
    omega, M, g = read_sbh_inputs(conf_dict['par_file'])
    num_modes = omega.size
    eps0 = conf_dict['eps0']
    nu0 = conf_dict['nu0']

    if ('state' not in conf_dict.keys()) or (conf_dict['state'] == None):
        state = 1
    else:
        state = conf_dict['state']

    model = sbh(M, omega, g, eps0, nu0)

    # Readining initial coordinates and velocities for dynamics
    if ('init_file' not in conf_dict.keys()) or (conf_dict['init_file'] == None):
        v0 = np.random.uniform(-0.005,0.005,(num_modes,))
        x0 = np.random.uniform(0,3,(num_modes,))
    else:
        x0, v0 = read_sbh_inputs(conf_dict['init_file'])    

    # Calculate the initial acceleration from model potential
    a0 = model.acceleration(x0)
    
    init_conds = (x0, v0, a0)

    return model, state, init_conds

#%% Defining molecular PES model parameters from the 'config.yml' file
def create_pes_model(conf_dict):

    model = fpes()

    # Read initial geometry (Ang) and velocities from standard xyz files   
    x0 = np.loadtxt("geom.xyz", skiprows=2, usecols=np.arange(1, 4))
    # Convert coordinates from Angstrom to Bohr
    x0 = x0.flatten() * ang2bohr 
    v0 = np.loadtxt("init_veloc.xyz", skiprows=2, usecols=np.arange(1, 4))
    v0 = v0.flatten()

    a0 = model.acceleration(x0)

    init_conds = (x0, v0, a0)

    return model, init_conds 

#%% Reading configuration file
with open('config.yml', 'r') as c:
    config = yaml.safe_load(c)

# For strings that yaml doesn't parse (e.g. None)
for key, val in config.items():
    if type(val) is str:
        try:
            config[key] = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            pass

if config['model'].lower() == 'sbh':
    model, state, init_cond = create_sbh_model(config)    
elif config['model'].lower() == 'fitpes':
    model, init_cond = create_pes_model(config)
else:
    sys.exit("Model type not found. Check your config.yml file.")

x0, v0, a0 = init_cond
M = model.mass
num_particles = M.size

#%% Parameters for the dynamics simulation
tmax = float(config['tmax'])
dt = float(config['dt'])
dtau = timeunits(dt,conv='fs2au')

integ = Integrator(dtau, x0, v0, a0, model)

integ_options = {0: "Euler", 1: "Verlet",
                 2: "Ruth-3", 3: "Ruth-4",
                 4: "Yoshida-6", 5: "Yoshida-8",
                 6: "RungeKutta4"}

if config['integrator'] not in integ_options.keys():
    sys.exit("Invalid option. Check your config.yml input file.")
else:
    method = integ_options[config['integrator']]

if ('save_traj' in config.keys()): 
    if (config['save_traj'] is not None):
        out_traj_file = config['save_traj']
    else:
        out_traj_file = "trajectory.npz"
    save_npz = True        

#%% Setting variables to start the dynamics

t = 0.000
step = 0

traj = Trajectory(num_particles)
traj.create()

print("************************************************************")
print("Starting dynamics simulation for the ", model)
print("************************************************************")
print(" ")

print_config(config)

print("   Numerical integration method:", method)

print_initial_conditions(x0,v0,a0)

start = time.time()

#%% Here start the main loop
while t <= tmax:
    
    x = x0.reshape(-1,1)
    v = v0.reshape(-1,1)
    a = a0.reshape(-1,1)

    if isinstance(model, sbh):
        epot = model.potential(x,state)
    elif isinstance(model, fpes):
        epot = model.potential

    ekin = kinetic(M,v)

    traj.update(step,t,x,v,a,epot,ekin)
    
    ekin_vec = traj.data['Ekin'][:step+1]
    epot_vec = traj.data['Epot'][:step+1]    
    
    print_energies(ekin_vec, epot_vec, step, t)
    
    if step >= 1:
        check_energy_conserv(ekin_vec, epot_vec, step, verbose=True)
        
    if "-" in method:
        m, order = method.split("-")
        results = getattr(integ, m)(int(order))
    else:
        results = getattr(integ, method)()

    mem_percentage = memory_usage(pid)
    
#    s = np.random.uniform(low=0.2,high=1.0)
    t = t + dt 
    step += 1

    if mem_percentage >= 75.0:
        partial_out = "traj_step" + str(step) + ".npz"
        traj.save(step,partial_out)
        last_step_saved = step
        print("\nCleaning memory...")
        traj.create()

end = time.time()
hours, rem = divmod(end-start, 3600)
minutes, seconds = divmod(rem, 60)

if save_npz:
    traj.save(step,out_traj_file) 

print(" ")
print("------------------------------------------------------")
print("Total simulation time - {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
print("------------------------------------------------------")
print(" ")

print("Finished!")
