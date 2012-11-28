"""
"""

from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import grids
import amitibo
import time
import scipy.io as sio
import amitibo


atmosphere_params = amitibo.attrClass(
    cartesian_grids=(
        slice(0, 50, 1.0), # Y
        slice(0, 50, 1.0), # X
        slice(0, 10, 0.1)   # H
        ),
    earth_radius=4000,
    air_typical_h=8,
    aerosols_typical_h=1.2
)

camera_params = amitibo.attrClass(
    image_res=128,
    subgrid_res=(10, 10, 1),
    grid_noise=0.01
)

camera_position = np.array((25., 25., 0.)) + 0.1*np.random.rand(3)

phi = 0
theta = -np.pi/4
Y, X, H = np.mgrid[atmosphere_params.cartesian_grids]


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

    x = (Y<150).reshape((-1, 1)).astype(np.float)
    #x = np.ones(Y.size).reshape((-1, 1))
    y = H_dist * x
    
    amitibo.viz3D(Y, X, H, y.reshape(Y.shape))
    mlab.show()
    

def integrate():
    
    """Compare two different resolutions"""
    
    H_int1 = grids.integrateGrids(
        camera_position,
        Y,
        X,
        H,
        camera_params.image_res,
        subgrid_res=camera_params.subgrid_res,
        noise=camera_params.grid_noise
    )
    
    x = np.ones(Y.shape)
    y1 = H_int1 * x.reshape((-1, 1))

    sio.savemat(
        'img.mat',
        {
            'y1': y1.reshape((camera_params.image_res, camera_params.image_res)),
            'H1': H_int1,
        }
    )
    
    plt.gray()
    plt.imshow(y1.reshape((camera_params.image_res, camera_params.image_res)))
    plt.colorbar()
    plt.show()
          

if __name__ == '__main__':
    #point()
    #direction()
    integrate()

      