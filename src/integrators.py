import time
import copy
import functools
import numpy as np

def timeit(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timeit(*args, **kargs):
        start_time = time.perf_counter()
        value = func(*args, **kargs)
        end_time = time.perf_counter()
        run_time = (end_time - start_time) * (10**3) # convert to miliseconds
        print(f"Runtime for {func.__name__!r} integrator: {run_time:.4f} ms")
        return value
    return wrapper_timeit

class Integrator:
    
    def __repr__(self):
         return "integrator"

    def __init__(self,dt,pos,vel,acc,model=None):
        self.pos = pos
        self.vel = vel
        self.acc = acc
        self.dt = dt
        # This is an object that contains functions related
        # to the analytical model of the physical system
        self.model = model 

    def _update_accel(self):
        return self.model.acceleration(self.pos)

    def _update_accel_rk4(self,x):
        return self.model.acceleration(x)

    @timeit
    def Euler(self):
        # dx/dt = F(x,t) -> x(t+dt) = x(t) + dt*F(x,t)
        self.acc = self._update_accel()
        self.pos += self.dt*self.vel
        self.vel += self.dt*self.acc
        return self.pos, self.vel, self.acc

    @timeit
    def Verlet(self):
        self.vel += self.acc * (self.dt/2)
        self.pos += self.dt*self.vel
        self.acc = self._update_accel()
        self.vel += self.acc * (self.dt/2)
        return self.pos, self.vel, self.acc

    @timeit
    def Ruth(self,order):
        if order == 3:
            coeffs = np.array([[2.0/3.0, -2.0/3.0, 1.0], 
                               [7.0/24.0, 0.75, -1.0/24.0]])
        elif order == 4:
            c = np.math.pow(2.0, 1.0/3.0)
            coeffs = np.array([[0.5, 0.5*(1.0-c), 0.5*(1.0-c), 0.5],
                               [0.0, 1.0,-c, 1.0]]) / (2.0 - c)                       
        
        for ai,bi in coeffs.T:
            self.vel += bi * self._update_accel() * self.dt
            self.pos += ai * self.vel * self.dt

        self.acc = self._update_accel()
        return self.pos, self.vel, self.acc

    @timeit
    def RungeKutta4(self):
        
        l1 = self.vel * self.dt
        k1 = self.acc * self.dt

        l2 = (self.vel + 0.5*k1) * self.dt
        k2 = self._update_accel_rk4(self.pos+0.5*l1) * self.dt

        l3 = (self.vel + 0.5*k2) * self.dt
        k3 = self._update_accel_rk4(self.pos+0.5*l2) * self.dt

        l4 = (self.vel + k3) * self.dt
        k4 = self._update_accel_rk4(self.pos+l3) * self.dt
        
        self.pos += (l1 + 2*l2 + 2*l3 + l4)/6
        self.vel += (k1 + 2*k2 + 2*k3 + k4)/6
        self.acc = self._update_accel_rk4(self.pos)

        return self.pos, self.vel, self.acc

    @timeit    
    def Yoshida(self,order):
        if order == 6:
            # velocity coeffs
            d = np.array([0.784513610477,0.235573213359,-1.17767998417887,
                          1.3151863206857402,-1.17767998417887,0.235573213359,
                          0.784513610477])
            # position coeffs
            c = np.array([0.3922568052385,0.510043411918,-0.471053385409935,
                          0.068753168253435026,0.068753168253435026,
                         -0.471053385409935,0.510043411918,0.3922568052385])
        elif order == 8:
            # velocity coeffs
            d = np.array([1.48819229202922,-2.33864815101035,2.89105148970595,
                         -2.89688250328827,0.00378039588360192,2.89195744315849,
                         -0.00169248587770116,-3.075516961201882,-0.00169248587770116,
                          2.89195744315849,0.00378039588360192,-2.89688250328827,
                          2.89105148970595,-2.33864815101035,1.48819229202922])
            # position coeffs
            c = np.array([0.74409614601461,-0.425227929490565,0.2762016693478,
                         -0.0029155067911599275,-1.4465510537023341,1.4478689195210459,
                          1.4451324786403945,-1.5386047235397915,-1.5386047235397915,
                          1.4451324786403945,1.4478689195210459,-1.4465510537023341,
                         -0.0029155067911599275,0.2762016693478,-0.425227929490565,
                          0.74409614601461])

        for i in range(len(d)):
            self.pos += c[i] * self.dt * self.vel
            self.vel += d[i] * self.dt * self._update_accel()
        
        self.pos += c[-1] * self.dt * self.vel
        return self.pos, self.vel, self.acc
