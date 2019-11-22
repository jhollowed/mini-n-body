import pdb
import numpy as np

def spherical_collapse(N):
    """
    Generate an initial realization of particles, which represent the discrete sampling
    of a uniform denisty sphere with zero velocities. As in Peebles 1970, this is a 
    simplified model for a CDM halo which has just decoupled from the expansion and is 
    starting an infall phase. The positions returned will all be dimensionless and 
    normalized, i.e. the outermost particles from the sphere center are at 
    (x^2+y^2+z^2)^(1/2) = 1.

    Parameters
    ----------
    N : int
        number of particles to place
    
    Return
    ------
    List of two matrices, one being a 3xN matrix of the particle positions, the other being a 
    3xN matrix of particle velocities
    """
    
    u = np.random.uniform(size=N)
    v = np.random.uniform(size=N)
    theta = u * 2*np.pi
    phi = np.arccos(2*v - 1.0)
    r = np.random.uniform(size=N)**(1/3)
    xx = r * np.sin(phi) * np.cos(theta) 
    yy = r * np.sin(phi) * np.sin(theta) 
    zz = r * np.cos(phi)
    x = np.array([xx, yy, zz])
    v = np.zeros((3, N))

    return [x, v]
