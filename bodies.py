import pdb
import numpy as np

# internal imports
import initial_conditions

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
        self.x
            Normalized particle positions at current time as a 3xN matrix (row per x,y,z)
        self.v
            Normalized particle velocitioes at current time as a 3xN matrix (row per vx,vy,vz)
        self.t
            The current dimensionless time
        self.u
            The current potential energy of the system
        self.k
            The current kinetic energy of the system
        """
        
        self.N = N
        self.c = c
        self.x = None
        self.v = None
        self.g = None
        self.u = 0
        self.k = 0
        self.t = 0
        self.integrator = integrator

        # generate the initial particle realization
        self.prepICs(self.N, ics)


    def prepICs(self, N, ic_type):
        """
        Prepare the initial particle and velocity vectors by calling the desired
        intial conditions function
        """
        
        # choose initial condition model
        # (currently only one option, probably only ever will be one option, but I pull 
        # the callable from a dictionary anyway because I'm ~pYtHoNiC~)
        models = {'cold_spherical_collapse' : initial_conditions.cold_spherical_collapse}
        try: ic_model = models[ic_type]
        except NameError: 
            raise NameError('Initial condition type {} not implemented'.format(ic_type))
        
        # get initial x and v, set accelerations and energy
        self.x, self.v = ic_model(N)
        self.update_g()
        self.update_E()


    def update_g(self):
        """
        Compute the accelerations per particle. 
        """

        if(self.g is None):
            self.g = np.zeros((3, self.N))
        
        # compute the per-pair particle separation magnitude, and delete the diagonal;
        # notice that I am computing each pair separation twice-- really all of the 
        # information is stored in the upper triangle of the matrix. But, this keeps 
        # the subtraction simple below, and is plenty fast to support the values of N
        # for which this code is intended to run
        mask = ~np.eye(self.N, dtype='bool')
        r = np.linalg.norm(self.x, axis=0)
        delta_r = np.subtract.outer(r, r)[mask].reshape(self.N, -1)
       
        # compute the per-pair particle separation in x,y,z, delete the diagonal;
        # the acceleration in this dimension is then the pair separtion, over the separation
        # magnitude plus the softening scale
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
        Compute kinetic and potential energy of the system
        """
        # compute the per-pair particle separation magnitude, delete the lower triangle
        # (including the diagonal), and flatten. This is because each pair should only 
        # contribute to the potential energy once
        r = np.linalg.norm(self.x, axis=0)
        delta_r = np.subtract.outer(r, r)[np.triu_indices(self.N, k=1)]

        # Potential contributed per-pair is the usual Newtonian 1/r, but with the softening
        # scale added in in quadrature to the denominator
        self.u = np.sum(1/np.sqrt(delta_r**2 + self.c**2))
        self.k = np.sum((0.5) * np.linalg.norm(self.v, axis=0)**2)
    

    def step(self, dt):
        """
        Advance the particles in time by a timestep of size delta_t, by the integrator 
        defined in the constructor. For information on the numerical techniques applied 
        here, see this review by Klypin:
        http://www.skiesanduniverses.org/resources/KlypinNbody.pdf

        Parameters
        ----------
        dt : float
            The dimensionless time step to advance the system by.
        """
        
        x0, v0, g0 = self.x, self.v, self.g
        
        # Kick-Drift-Kick symplectic leapfrog integrator
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
