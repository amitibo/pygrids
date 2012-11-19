"""
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import grids
import amitibo
import time
import mayavi.mlab as mlab


def main():
    """Main doc """
    
    params = amitibo.attrClass(
        cartesian_grids=(
            slice(0, 10., 1), # Y
            slice(0, 10., 1), # X
            slice(0, 10., 1)  # H
            ),
    )
    
    camera_center = (4.5, 0.5, 0.5)

    Y, X, H = np.mgrid[params.cartesian_grids]

    t0 = time.time()
    
    H_dist = grids.point2grids(camera_center, Y, X, H)
    
    print time.time() - t0

    x = np.ones(Y.size).reshape((-1, 1))
    y = H_dist * x
    
    amitibo.viz3D(Y, X, H, y.reshape(Y.shape))
    mlab.show()
    

if __name__ == '__main__':
    main()

      