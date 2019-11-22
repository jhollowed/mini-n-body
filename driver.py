import pdb
import yaml
import numpy as np

# internal modules
import bodies
import visualization
import initial_conditions

class simulation:
    def __init__(self, param_file, initial_conditions, boundary_condition):
        """
        This class prepares an environment for running an n-body simulation, setting the input
        parameters, initial conditions, creating the n-body system, and advancing the time stepper.

        Parameters
        ----------
        param_file : string
            Path to YAML file containing model parameters. Present in the file must
            be values for the parameters in the Peebles 1970 model. These are: 
            - N: the number of particles
            - epsilon: the dimensionless Plummer softening length
            - (optional) alpha: the physical length scale to convert dimensionless distances
            - (optional) Beta: the physical time scale to convert dimensionles times
            If the last two parameters are not provieded, they will be to 1. The final parameter
            from Peebles is inferred as gamma = alpha/beta.
        initial_condition : string
            The model to use for the initial conditiosn. Options are:
            - 'spherical_collapse': particles will represent a discrete realization of a 
              spherically symmetric region with uniform denisty.
            - Maybe more some day
            Defaults to 'spherical_collapse'.
        boundary_condition : string
            Type of boundary conditions to use. Options are:
            - 'open': there is no defined "box", are particles that reach escape velcoity 
              are always retained without bound on their separation from the center of the system
            - Maybe more some day
            Defaults to 'open'.

        Attributes
        ----------
        self.params : numpy array
            Array storing model parameters; should be populated using loadParams().
        self.system : bodies.system instance object
            Maintains the state of the system at a given time, and performs timestepping.
        """
        self.params = None
        self._loadParams(param_file)
        self._N = self.params['N']
        self._c = self.params['epsilon']
        self._x = None
        self._v = None
        self._t = None
        self.system = bodies.system(self._N, self._c, initial_conditions, boundary_condition)


    def _loadParams(self, param_file):
        """
        Loads model parameters from an input YAML file to a dictionary class attribute.
        params : string
            Path to YAML file containing model parameters. Present in the file must be values 
            for the parameters in the Peebles 1970 model. See class constructor for details.
        """
        stream = open(param_file, 'r')
        self.params = yaml.load(stream, yaml.SafeLoader)


    def run(self, T, tsteps):
        """
        Runs the simulation for the desired number of time steps, spaced uniformly from 0 to T. 
        This works by repeatedly calling bodies.system.step(). On each step, the new particle 
        positions are written to the next entry in the third dimension of a 3xNx(tsteps) numpy array, 
        and likewise for the velocities.

        Parameters
        ----------
        T : float
            Maximum (dimensionless) time to achieve
        tsteps : int
            The number of timesteps to take.
        """
        self._x = np.zeros((tsteps, 3, self._N))
        self._v = np.zeros((tsteps, 3, self._N))
        self._t = np.linspace(0, T, tsteps)
        dt = np.diff(self._t)[0]
        
        for i in range(tsteps):
            self.system.step(dt, integrator='euler')
            self._x[i,:,:] = self.system.x
            self._v[i,:,:] = self.system.v
    

    def visualize(self, out_dir, style='movie', show=True):
        """
        Visualizes the result of running the simulation in a variety of forms.

        Parameters
        ----------
        style : string
            What kind of plot to make, with the following options:
        out_dir : string
            Where to save the resulting image and/or gif files.
        show : bool
            Whether or not to display the plots rather than just saving to file.
        """
        if(style == 'movie'):
            visualization.animate_collapse(self._x, out_dir)


    def write_out(self, out_dir, scale = False):
        """
        Writes out the result of running the simulation with run(). That is, the three dimensional
        matrices self.x and self.v are each written to .npy files.
        
        Parameters
        ----------
        out_dir : string
            Full path to desired output location.
        scale : bool
            Whether or not to otuput a second copy of the data which is scaled by the 
            physical values that were (optionally) given in the parameter file. If not given,
            and scale = True, this function will raise an error.
        """
        np.save('{}/positions.npy'.format(out_dir), self._x)
        np.save('{}/velocities.npy'.format(out_dir), self._v)
        np.save('{}/times.npy'.format(out_dir), self._t)
        if(scale):
            alpha = self.params['alpha']
            beta = self.params['Beta']
            gamma = alpha/beta
            np.save('{}/positions_phys.npy'.format(out_dir), self._x * alpha)
            np.save('{}/velocities_phys.npy'.format(out_dir), self._v * gamma)
            np.save('{}/times_phys.npy'.format(out_dir), self._t * beta)

# ===========================================================================================

def Peebles1970():
    """
    Runs a simulation using the parameters from Peebles 1970
    """
    params = 'model_parameters/peebles1970_standard.yaml'
    sim = simulation(params, 'spherical_collapse', 'open')
    sim.run(0.3, 100)
    sim.visualize(out_dir = './img_out')


    
