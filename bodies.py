import pdb
import numpy as np

# internal imports
import initial_conditions
from visualization import axisEqual3D

class system():
    def __init__(self, N, c, integrator, ics):
        """
        Class to represent the state of an N-body system at some time t.

        Parameters
        ----------
        N : int
            number of particles to populate by the initial conditions model
        c : float 
            dimensionless Plummer softening scale, with 0 <= c < 1
        integrator : string
            Integrator to use. Options are:
            - "KDK": uses the kick-drift-kick leapfrog integrator
            - "euler": uses a standard Euler integrator
            - Maybe more one day        
        ics : string
            The model to use for the initial conditiosn. Options are:
            - 'cold_spherical_collapse': particles will represent a discrete realization of a 
              spherically symmetric region with uniform denisty, and zero kinetic energy.
            - Maybe more some day
            Defaults to 'cold_spherical_collapse'.
        
        Attributes
        ----------
        self.N
            Number of particles in the system
        c 
            dimensionless Plummer softening scale
        self.x
            Normalized particle positions at current time as a 3xN matrix
        self.v
            Normalized particle velocitioes at current time as a 3xN matrix
        self.t
            The current dimensionless time
        """
        self.N = N
        self.c = c
        self.x = None
        self.v = None
        self.g = None
        self.u = None
        self.k = None
        self.t = 0
        self.integrator = integrator
        self.prepICs(self.N, ics)


    def prepICs(self, N, ic_type):
        """
        Prepare the initial particle and velocity vectors by calling the desired
        intial conditions function
        """

        models = {'cold_spherical_collapse' : initial_conditions.cold_spherical_collapse}
        try: ic_model = models[ic_type]
        except NameError: 
            raise NameError('Initial condition type {} not implemented'.format(ic_type))
        
        self.x, self.v = ic_model(N)
        self.update_g()
        self.update_E()


    def update_g(self):
        """
        Compute the accelerations per particle. 
        """
        if(self.g is None):
            self.g = np.zeros((3, self.N))
        
        mask = ~np.eye(self.N, dtype='bool')
        r = np.linalg.norm(self.x, axis=0)
        delta_r = np.subtract.outer(r, r)[mask].reshape(self.N, -1)
        for i in range(3):
            pair_dist = np.subtract.outer(\
                        self.x[i], self.x[i])[mask]\
                        .reshape(self.N, -1)
            self.g[i, :] = -np.sum(\
                           pair_dist/\
                           (delta_r**2 + self.c**2)**(3/2),
                           axis=1)
    
    def update_E(self):
        """
        computes kinetic energy
        """
        r = np.linalg.norm(self.x, axis=0)
        delta_r = np.subtract.outer(r, r)[np.triu_indices(self.N, k=1)]
        self.u = np.sum(1/np.sqrt((delta_r)**2 + self.c**2))
        self.k = np.sum((0.5) * np.linalg.norm(self.v, axis=0)**2)
        self.E = self.u + self.k
    

    def step(self, dt):
        """
        Advance the particles in time by a timestep of size delta_t, by the integrator 
        defined in the constructor

        Parameters
        ----------
        dt : float
            The dimensionless time step to advance the system by.
        """
        
        x0, v0, g0 = self.x, self.v, self.g
        
        # Kick-Drift-Kick
        if(self.integrator == 'KDK'):
            self.v = v0 + g0*(dt/2)           # kick
            self.x = x0 + self.v*dt           # drift
            self.update_g()                   # G
            self.v = self.v + self.g*(dt/2)   # kick
        
        # Euler
        elif(self.integrator == 'euler'):
            self.x = x0 + v0*dt
            self.v = v0 + g0*dt
        else:
            raise NameError('Integrator {} not implemented'.format(ic_type))
        self.t = self.t + dt
        self.update_g()
        self.update_E()
