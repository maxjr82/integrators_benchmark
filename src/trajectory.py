import numpy as np

class Trajectory:
    
    def __init__(self,size_vecs):
        self.size_vecs = size_vecs
        self.capacity = 1000
        self.variables = {'X', 'Vel', 'Acc', 'Epot', 'Ekin', 'time'}
        self.data = dict([(key, None) for key in self.variables])

    def create(self):
        self.data['X'] = np.empty((self.capacity,self.size_vecs), dtype=np.float64)
        self.data['Vel'] = np.empty((self.capacity,self.size_vecs), dtype=np.float64)
        self.data['Acc'] = np.empty((self.capacity,self.size_vecs), dtype=np.float64)
        self.data['Epot'] = np.empty((self.capacity,), dtype=np.float64)
        self.data['Ekin'] = np.empty((self.capacity,), dtype=np.float64)
        self.data['time'] = np.empty((self.capacity,), dtype=np.float64)

    @property
    def _resize_data(self):
        self.data['X'] = np.resize(self.data['X'],(self.capacity,self.size_vecs))
        self.data['Vel'] = np.resize(self.data['Vel'],(self.capacity,self.size_vecs))
        self.data['Acc'] = np.resize(self.data['Acc'],(self.capacity,self.size_vecs))
        self.data['Epot'] = np.resize(self.data['Epot'],(self.capacity,))
        self.data['Ekin'] = np.resize(self.data['Ekin'],(self.capacity,))
        self.data['time'] = np.resize(self.data['time'],(self.capacity,))

    def update(self,step,t,pos,vel,acc,epot,ekin):
        if step == self.capacity:
            self.capacity += 2000
            self._resize_data

        self.data['X'][step] = pos.T
        self.data['Vel'][step] = vel.T
        self.data['Acc'][step] = acc.T
        # Update energy values calculated in the loop
        self.data['Epot'][step] = epot
        self.data['Ekin'][step] = ekin
        self.data['time'][step] = t

    def save(self,step,out_name):
        print(" ")
        print("Saving trajectory results in", out_name)
        for k in self.data.keys():
            self.data[k] = self.data[k][:step]
        np.savez(out_name,**self.data)
