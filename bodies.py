import pdb
import numpy as np

# internal imports
import initial_conditions
from visualization import axisEqual3D

class system():
    def __init__(self, N, c, ics, bcs):
        """
        Class to represent the state of an N-body system at some time t.

        Parameters
        ----------
        N : int
            number of particles to populate by the initial conditions model
        c : float 
            dimensionless Plummer softening scale, with 0 <= c < 1
        ics : string
            The model to use for the initial conditiosn. Options are:
            - 'spherical_collapse': particles will represent a discrete realization of a 
              spherically symmetric region with uniform denisty.
            - Maybe more some day
            Defaults to 'spherical_collapse'.
        bcs : string
            Type of boundary conditions to use. Options are:
            - 'open': there is no defined "box", are particles that reach escape velcoity 
              are always retained without bound on their separation from the center of the system
            - Maybe more some day
            Defaults to 'open'.
        
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
        self.t = 0
        self.prepICs(self.N, ics)
        self.calc_g() 


    def prepICs(self, N, ic_type):
        """
        Prepare the initial particle and velocity vectors by calling the desired
        intial conditions function
        """

        models = {'spherical_collapse':initial_conditions.spherical_collapse}
        try: ic_model = models[ic_type]
        except NameError: 
            raise NameError('Initial condition type {} not implemented'.format(ic_type))
        
        self.x, self.v = ic_model(N)


    def calc_g(self):
        """
        Compute the accelerations per particle. 
        """
        if(self.g is None):
            self.g = np.zeros((3, self.N))
        for i in range(3):
            mask = ~np.eye(self.N, dtype='bool')
            pair_dist = np.subtract.outer(self.x[i], self.x[i])[mask].reshape(self.N, -1)
            self.g[i, :] = -np.sum(pair_dist/(pair_dist**3 + self.c**3), axis=1) 
 

    def step(self, dt, integrator):
        """
        Advance the particles in time by a timestep of size delta_t

        Parameters
        ----------
        dt : float
            The dimensionless time step to advance the system by.
        integrator : string
            Integrator to use. Options are:
            - "KDK": uses the kick-drift-kick leapfrog integrator
            - "euler": uses a standard Euler integrator
            - Maybe more one day        
        """
        
        x0, v0, g0 = self.x, self.v, self.g
        
        # Kick-Drift-Kick
        if(integrator == 'KDK'):
            self.v = v0 + g0*(dt/2)           # kick
            self.x = x0 + self.v*dt           # drift
            self.calc_g()                     # G
            self.v = self.v + self.g*(dt/2)   # kick
        
        # Euler
        elif(integrator == 'euler'):
            self.x = x0 + v0*dt
            self.v = v0 + g0*dt
        else:
            raise NameError('Integrator {} not implemented'.format(ic_type))
        self.t = self.t + dt
        self.calc_g() 
