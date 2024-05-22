import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plotRicattiGains(ddp,fig=None,show=False):
    M = np.array([K.T for K in ddp.K])
    if fig is None: fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x,y,z = np.meshgrid(*[range(n) for n in M.shape])
    ax.scatter(x,y,z, c=M.flat)
    ax.set_xlabel(f'Time (0..{len(ddp.K)})')
    ax.set_ylabel(f'X [0..{len(ddp.xs[0])}[')
    ax.set_zlabel(f'U [0..{len(ddp.us[0])}[')
    if show: plt.show()

