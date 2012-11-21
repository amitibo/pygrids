"""
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import grids
import amitibo
import time
import mayavi.mlab as mlab
import matplotlib.pyplot as plt


atmosphere_params = amitibo.attrClass(
    cartesian_grids=(
        slice(0, 300., 10), # Y
        slice(0, 300., 10), # X
        slice(0, 300., 10)  # H
        ),
)

camera_center = (150.0, 150.0, .2)
phi = np.pi/8
theta = np.pi/4
Y, X, H = np.mgrid[atmosphere_params.cartesian_grids]
sensor_res = 128
pixel_fov = 0.03


def point():
    """Main doc """
    
    t0 = time.time()
    
    H_dist = grids.point2grids(camera_center, Y, X, H)
    
    print time.time() - t0

    #x = np.ones(Y.size).reshape((-1, 1))
    x = np.exp(-H/10)
    x[H<100] = 0
    y = H_dist * x.reshape((-1, 1))

    amitibo.viz3D(Y, X, H, x.reshape(Y.shape), interpolation='nearest_neighbour')
    amitibo.viz3D(Y, X, H, y.reshape(Y.shape), interpolation='nearest_neighbour')
    mlab.show()
    

def direction():
    """Main doc """
    
    t0 = time.time()
    
    H_dist = grids.direction2grids(phi, theta, Y, X, H)
    
    print time.time() - t0

    x = (Y<100).reshape((-1, 1)).astype(np.float)
    #x = np.ones(Y.size).reshape((-1, 1))
    y = H_dist * x
    
    amitibo.viz3D(Y, X, H, y.reshape(Y.shape))
    mlab.show()
    

def integrate():
    
    t0 = time.time()
    
    H_int = grids.integrateGrids(camera_center, Y, X, H, sensor_res, pixel_fov)
    
    print time.time() - t0

    x = np.ones(Y.shape)
    #x = np.exp(-H/10)
    x[X<150] = 0
    y = H_int * x.reshape((-1, 1))


    plt.imshow(y.reshape((sensor_res, sensor_res)))
    plt.colorbar()
    plt.show()
        

if __name__ == '__main__':
    #point()
    #direction()
    integrate()

      