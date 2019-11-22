import glob
import imageio
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

def animate_collapse(x, out_dir, trails=True):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    for t in range(len(x)):
        ax.clear()
        ax.plot(x[t,0,:], x[t,1,:], x[t,2,:], '.k')
        
        if(trails == True):
            trail_length = min(t, 6)
            for i in range(trail_length):
                ax.plot(x[t-i,0,:], x[t-i,1,:], x[t-i,2,:], '.k', alpha=(i/trail_length)*0.5)

        plt.savefig('{}/out_{}.jpg'.format(out_dir, t), dpi=300)

    images = []
    for filename in glob.glob('{}/*.jpg'.format(out_dir)):
        images.append(imageio.imread(filename))
    imageio.mimsave('{}/out.gif'.format(out_dir), images)
