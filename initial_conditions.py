import pdb
import numpy as np

def cold_spherical_collapse(N):
    """
    Generate an initial realization of particles, which represent the discrete sampling
    of a uniform denisty sphere with zero kinetic energy. As in Peebles 1970, this is a 
    simplified model for a CDM halo which has just decoupled from the expansion, and is 
    starting an infall phase. The positions returned will all be dimensionless and 
    normalized, i.e. the radius of the sphere is r = 1.

    Parameters
    ----------
    N : int
        number of particles to place
    
    Return
    ------
    List of two matrices, one being a 3xN matrix of the particle positions, the other being a 
    3xN matrix of particle velocities (zeros)
    """
    
    # uniform spherical
    u = np.random.uniform(size=N)
    v = np.random.uniform(size=N)
    phi = u * 2*np.pi
    theta = np.arccos(2*v - 1.0)
    r = (np.random.uniform(size=N))**(1/3)
    
    # to cartesian
    xx = r * np.sin(theta) * np.cos(phi) 
    yy = r * np.sin(theta) * np.sin(phi) 
    zz = r * np.cos(theta)
    x = np.array([xx, yy, zz])

    # zero initial velocity
    v = np.zeros((3, N))

    return [x, v]
