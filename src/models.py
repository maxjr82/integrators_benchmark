import os
import numpy as np

class FitPES:

    def __repr__(self):
         return "Molecular PES from POTLIB"

    # Defining slots to optimize performance (RAM):
    __slots__ = ['mass', 'labels', 'num_atoms', 'gradient', 'potential']

    def __init__(self):
        # read labels from xyz file and convert into atomic masses
        self.labels, self.mass = self._label2mass
        self.num_atoms = len(self.labels)
        # gradients and potential energy will be updated
        # every time the acceleration function is called
        self.gradient = None
        self.potential = None

    def _bohr2ang(self,val):
        return val * 0.529177249

    @property
    def _label2mass(self):
        converter = {'H': 1.00782504, 'C': 12.00000000, 
                     'N': 14.00307401, 'O': 15.99491464}
        labels, _ = self._read_xyz
        masses = np.vectorize(converter.get)(labels)
        masses *= 1822.888515
        masses = masses.flatten()
        masses = np.repeat(masses, 3)
        return labels, masses
        
    @property
    def _read_xyz(self):
        labels = list()
        coords = list()
        with open('geom.xyz', 'r') as xyz:
            for line in xyz.readlines():
                if len(line.split()) > 1:
                    labels.append(line.split()[0])
                    coords.append(line.split()[1:4])
        labels = np.asarray(labels)
        labels = labels.reshape(-1,1)            
        coords = np.asarray(coords, dtype=np.float64)
        return labels, coords

    @property
    def _read_out(self):
        results = list()
        with open('geom.out', 'r') as out:
            for line in out.readlines():
                if len(line.split()) == 2:
                    results.append(line.split()[1])
        results = np.asarray(results, dtype=np.float64)
        return results

    # Here we write the current molecular geometry into 
    # the geom.xyz file to be read by the POTLIB program
    def _write_xyz(self,x):
        x = self._bohr2ang(x)
        coords = x.reshape(-1,3)
        null_grad = np.zeros((self.num_atoms,3), dtype=np.float64)
        xyz = np.concatenate((self.labels,coords,null_grad), axis=1)
        geom = str(self.num_atoms) + '\n' + ' 0.00000000' + '\n'
        geom += '\n'.join('  '.join('%s' %x for x in y) for y in xyz) + '\n'
        geom = geom.replace("'", "")
        with open('geom.xyz', 'w') as f:
            f.write(geom)
 
    def _run_potlib(self,x):
        '''This function calls the external program of POTLIB to
           compute the gradient and potential energy for a given
           molecular geometry x. 
           Arg: x must be provided as a 1D numpy array.'''
        self._write_xyz(x)
        os.system("./potlib.x geom.xyz")
        results = self._read_out
        self.potential = results[0]
        self.gradient = results[1:]

    def acceleration(self,x):
        '''Calculate the classical acceleration given the coordinates 
           vector x provided in atomic units. After running this 
           function the gradient and potential class variables will be
           updated.
           Arg: x as 1D numpy array (units = Bohr)'''
        self._run_potlib(x)
        acc = -(1/self.mass) * self.gradient
        return acc
            
class SpinBoson:
    
    def __repr__(self):
         return "Spin-Boson model"

    # Defining slots to optimize performance (RAM):
    __slots__ = ['mass', 'omg', 'g', 'eps', 'nu']

    def __init__(self, mass, omg, g, eps0, nu0):
        self.mass = self._amu2proton(mass)
        self.omg = self._cm2au(omg)
        self.g = g
        self.eps = self._cm2au(eps0) # two level splitting (float)
        self.nu = self._cm2au(nu0)   # coupling constant (float)
    
    def _amu2proton(self,val):
        return val * 1822.888515

    def _cm2au(self,val):
        return val * 4.55633539e-6

    def eta(self,x):
        eta_val = float(self.g @ x + self.eps)
        return eta_val

    def potential(self,x,state=1):
        t1 = (self.mass * self.omg**2) @ x**2
        t2 = (-1)**state * np.sqrt(self.eta(x)**2 + self.nu**2)
        V = 0.5 * t1 + t2
        return V

    def gradient(self,x,state=1):
        t1 = self.mass * self.omg**2 * x
        t2 = (-1)**state * self.g * self.eta(x) * (self.eta(x)**2 + self.nu**2)**(-1/2)
        dV = t1 + t2        
        return dV

    def acceleration(self,x,st=1):
        return -(1/self.mass) * self.gradient(x,state=st)

    def _diabatic(self,x):
        t = 0.5 * (self.mass * self.omg**2) @ x**2
        d11, d22 = t + self.eta(x), t - self.eta(x)
        d12 = d21 = self.nu
        mat = np.array([[d11,d12],[d21,d22]])
        return mat

class HarmonicOscillator:

    def __repr__(self):
         return "Classical Harmonic Oscillator"

    def __init__(self, m, k, c=0.0):
        self.m = m
        self.k = k
        # Damping constant
        self.c = c

    def potential(self, x):
        Epot = 0.5 * self.k * x**2
        return Epot

    def acceleration(self,x):
        acc = -(self.k/self.m) * x
        return acc
