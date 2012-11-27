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
import scipy.io as sio


atmosphere_params = amitibo.attrClass(
    cartesian_grids=(
        slice(0, 400., 4.0), # Y
        slice(0, 400., 4.), # X
        slice(0, 10., 0.1)  # H
        ),
)

camera_center = (202., 202., .05)
phi = 0
theta = -np.pi/4
Y, X, H = np.mgrid[atmosphere_params.cartesian_grids]
sensor_res = 128
pixel_fov = np.tan(np.arccos((sensor_res**2 - 1)/sensor_res**2)) * 10


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
    
    t0 = time.time()
    
    H_int = grids.integrateGrids(camera_center, Y, X, H, sensor_res, pixel_fov)
    
    print time.time() - t0

    #x = np.ones(Y.shape)
    x = np.exp(-H/10)
    x[X>160] = 0
    x[X<140] = 0
    x[Y>160] = 0
    x[Y<140] = 0
    x[H>10] = 0
    y = H_int * x.reshape((-1, 1))


    amitibo.viz3D(Y, X, H, x.reshape(Y.shape))
    mlab.show()
    plt.imshow(y.reshape((sensor_res, sensor_res)))
    plt.colorbar()
    plt.show()
        

def integrate2():
    
    t0 = time.time()
    
    H_int = grids.integrateGrids(camera_center, Y, X, H, sensor_res, subgrid_res=(10, 10, 10))
    
    print time.time() - t0

    x = np.ones(Y.shape)
    #x = np.exp(-H/10)
    #x[X<200] = 0
    y = H_int * x.reshape((-1, 1))


    #amitibo.viz3D(Y, X, H, x.reshape(Y.shape))
    #mlab.show()
    sio.savemat('img3.mat', {'y': y.reshape((sensor_res, sensor_res)), 'H':H_int})
    plt.imshow(y.reshape((sensor_res, sensor_res)))
    plt.colorbar()
    plt.show()
        

def integrate3():
    
    t0 = time.time()
    
    H_int = grids.integrateGrids(camera_center, Y, X, H, sensor_res, subgrid_res=(40, 40, 1))
    
    print time.time() - t0

    x = np.zeros((sensor_res, sensor_res))
    x[:, range(0, sensor_res, 2)] = 1
    x[range(0, sensor_res, 2), :] = 1    
    #x[int(sensor_res/2), range(0, sensor_res, 4)] = 1
    y = H_int.T * x.reshape((-1, 1))

    amitibo.viz3D(Y, X, H, y.reshape(Y.shape))
    mlab.show()
        

def integrate4():
    
    """Compare two different resolutions"""
    
    H_int1 = grids.integrateGrids(camera_center, Y, X, H, sensor_res, subgrid_res=(1, 1, 1), noise=0)
    H_int2 = grids.integrateGrids(camera_center, Y, X, H, sensor_res, subgrid_res=(1, 1, 1), noise=0.05)
    H_int3 = grids.integrateGrids(camera_center, Y, X, H, sensor_res, subgrid_res=(10, 10, 10), noise=0)
    
    x = np.ones(Y.shape)
    y1 = H_int1 * x.reshape((-1, 1))
    y2 = H_int2 * x.reshape((-1, 1))
    y3 = H_int3 * x.reshape((-1, 1))

    sio.savemat(
        'img4_2_newR.mat',
        {
            'y1': y1.reshape((sensor_res, sensor_res)),
            'y2': y2.reshape((sensor_res, sensor_res)),
            'y3': y3.reshape((sensor_res, sensor_res)),
            'H1': H_int1,
            'H2': H_int2,
            'H3': H_int3
        }
    )
    
    plt.gray()
    plt.imshow(y1.reshape((sensor_res, sensor_res)))
    plt.colorbar()
    plt.figure()
    plt.imshow(y2.reshape((sensor_res, sensor_res)))
    plt.colorbar()
    plt.figure()
    plt.imshow(y3.reshape((sensor_res, sensor_res)))
    plt.colorbar()
    plt.figure()
    plt.imshow(np.abs(y1-y3).reshape((sensor_res, sensor_res)))
    plt.colorbar()    
    plt.figure()
    plt.imshow(np.abs(y2-y3).reshape((sensor_res, sensor_res)))
    plt.colorbar()    
    plt.show()
          

def integrate5():
    
    """Compare two different resolutions"""
    
    H_int1 = grids.integrateGrids(camera_center, Y, X, H, sensor_res, subgrid_res=(1, 1, 1), noise=0)
    H_int2 = grids.integrateGrids(camera_center, Y, X, H, sensor_res, subgrid_res=(1, 1, 1), noise=0.05)
    H_int3 = grids.integrateGrids(camera_center, Y, X, H, sensor_res, subgrid_res=(40, 40, 1), noise=0)
    
    x = np.ones(Y.shape)
    y1 = H_int1 * x.reshape((-1, 1))
    y2 = H_int2 * x.reshape((-1, 1))
    y3 = H_int3 * x.reshape((-1, 1))

    sio.savemat(
        'img5.mat',
        {
            'y1': y1.reshape((sensor_res, sensor_res)),
            'y2': y2.reshape((sensor_res, sensor_res)),
            'y3': y3.reshape((sensor_res, sensor_res)),
            'H1': H_int1,
            'H2': H_int2,
            'H3': H_int3
        }
    )
    
    plt.gray()
    plt.imshow(y1.reshape((sensor_res, sensor_res)))
    plt.colorbar()
    plt.figure()
    plt.imshow(y2.reshape((sensor_res, sensor_res)))
    plt.colorbar()
    plt.figure()
    plt.imshow(y3.reshape((sensor_res, sensor_res)))
    plt.colorbar()
    plt.figure()
    plt.imshow(np.abs(y1-y3).reshape((sensor_res, sensor_res)))
    plt.colorbar()    
    plt.figure()
    plt.imshow(np.abs(y2-y3).reshape((sensor_res, sensor_res)))
    plt.colorbar()    
    plt.show()
          

def integrate6():
    
    """Compare two different resolutions"""
    
    print 'H1'
    H_int1 = grids.integrateGrids(camera_center, Y, X, H, sensor_res, subgrid_res=(1, 1, 1), noise=0.005)
    print 'H2'
    H_int2 = grids.integrateGrids(camera_center, Y, X, H, sensor_res, subgrid_res=(10, 10, 10), noise=0)
    
    x = np.ones(Y.shape)
    y1 = H_int1 * x.reshape((-1, 1))
    y2 = H_int2 * x.reshape((-1, 1))

    sio.savemat(
        'img6.mat',
        {
            'y1': y1.reshape((sensor_res, sensor_res)),
            'y2': y2.reshape((sensor_res, sensor_res)),
            'H1': H_int1,
            'H2': H_int2
        }
    )
    
    plt.gray()
    plt.imshow(y1.reshape((sensor_res, sensor_res)))
    plt.colorbar()
    plt.figure()
    plt.imshow(y2.reshape((sensor_res, sensor_res)))
    plt.colorbar()
    plt.figure()
    plt.imshow(np.abs(y1-y2).reshape((sensor_res, sensor_res)))
    plt.colorbar()    
    plt.show()
          

if __name__ == '__main__':
    #point()
    #direction()
    integrate5()

      