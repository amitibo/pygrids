"""
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import grids
import amitibo
import time
import mayavi.mlab as mlab


atmosphere_params = amitibo.attrClass(
    cartesian_grids=(
        slice(0, 400., 20), # Y
        slice(0, 400., 20), # X
        slice(0, 400., 20)  # H
        ),
)

camera_center = (200, 200, 0.1)
phi = np.pi/4
theta = np.pi/4
Y, X, H = np.mgrid[atmosphere_params.cartesian_grids]


def point():
    """Main doc """
    
    t0 = time.time()
    
    H_dist = grids.point2grids(camera_center, Y, X, H)
    
    print time.time() - t0

    x = np.ones(Y.size).reshape((-1, 1))
    y = H_dist * x
    
    amitibo.viz3D(Y, X, H, y.reshape(Y.shape))
    mlab.show()
    

def direction():
    """Main doc """
    
    t0 = time.time()
    
    H_dist = grids.direction2grids(phi, theta, Y, X, H)
    
    print time.time() - t0

    x = np.ones(Y.size).reshape((-1, 1))
    y = H_dist * x
    
    amitibo.viz3D(Y, X, H, y.reshape(Y.shape))
    mlab.show()
    

if __name__ == '__main__':
    #point()
    direction()

      