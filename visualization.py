import pdb
import glob
import imageio
import numpy as np
from matplotlib import rc 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
rc('text', usetex=True)

def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

def animate_collapse(x, ku, T, out_dir, trails=True):

    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122)

    print('Plotting...')
    for t in range(len(x)):
        if(t%50==0):print('frame {}/{}'.format(t, len(x)))
        
        ax.clear()
        ax.plot(x[t,0,:], x[t,1,:], x[t,2,:], '.k', mew=0, ms=4)
        ax2.plot(T[0:t], ku[0:t], '-r', lw=2)
        
        if(trails == True):
            trail_length = min(t, 8)
            for i in range(trail_length):
                ax.plot(x[t-i,0,:], x[t-i,1,:], x[t-i,2,:], '.k', 
                        alpha=(((trail_length-i)/trail_length)*0.5), 
                        mew = 0, ms=4)
        ax.elev = 20
        ax.azim = t/len(x) * 180
        lim = 2
        ax.set_xlim([-lim, lim])
        ax.set_ylim([-lim, lim])
        ax.set_zlim([-lim, lim])
        ax2.set_xlim([0, max(T)])
        ax2.set_ylim([0, np.mean(ku) + np.std(ku)*2])
        ax2.set_xlabel(r'$\mathrm{Model\>time}$', fontsize=16)
        ax2.set_ylabel(r'$T/U$', fontsize=16)
        plt.tight_layout()
        plt.savefig('{}/out_{}.jpg'.format(out_dir, t), dpi=300)
    print('Compiling to gif')
    makeGif(out_dir)

def makeGif(out_dir):
    images = []
    files = np.array(glob.glob('{}/*.jpg'.format(out_dir)))
    sort = np.argsort([int(f.split('_')[-1].split('.')[0]) for f in files])
    for filename in files[sort]:
        images.append(imageio.imread(filename))
    imageio.mimsave('{}/out.gif'.format(out_dir), images)

