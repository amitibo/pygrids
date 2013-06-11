from __future__ import division
import numpy as np
import scipy.sparse as sps
import itertools
from .base import *

__all__ = ['cameraTransformMatrix', 'fisheyeTransformMatrix', 'linearCameraTransformMatrix']


def cameraTransformMatrix(PHI, THETA, focal_ratio=0.5, image_res=256, theta_compensation=False):
    """
    Calculate a sparse matrix representation of camera projection transform.
    
    Parameters
    ----------
    PHI, THETA : 3D arrays
        \phi and \theta angle grids.
    
    focal_ratio : float, optional (default=0.5)
        Ratio between the focal length of the camera and the size of the sensor.
    
    image_res : int, optional (default=256)
        Resolution of the camera image (both dimensions)
        
    theta_compensation : bool, optional (default=False)
        Compensate for angle between ray and pixel
        
    Returns
    -------
    H : sparse matrix
        Sparse matrix, in csr format, representing the transform.
"""

    import numpy as np
    import amitibo
    
    Y, X = np.mgrid[-1:1:complex(0, image_res), -1:1:complex(0, image_res)]
    PHI_ = np.arctan2(Y, X) + np.pi
    R_ = np.sqrt(X**2 + Y**2 + focal_ratio**2)
    THETA_ = np.arccos(focal_ratio / (R_ + amitibo.eps(R_)))

    #
    # Calculate the transform
    #
    H = calcTransformMatrix((PHI, THETA), (PHI_, THETA_))

    #
    # Account for cos(\theta)
    #
    if theta_compensation:
        H = spdiag(np.cos(THETA_)) * H
    
    return H


def fisheyeTransformMatrix(PHI, THETA, image_res=256, theta_compensation=False):
    """
    Calculate a sparse matrix representation of fisheye projection transform.
    
    Parameters
    ----------
    PHI, THETA : 3D arrays
        \phi and \theta angle grids.
    
    image_res : int, optional (default=256)
        Resolution of the camera image (both dimensions)
        
    Returns
    -------
    H : sparse matrix
        Sparse matrix, in csr format, representing the transform.
"""

    import numpy as np
    import amitibo
    
    Y, X = np.mgrid[-1:1:complex(0, image_res), -1:1:complex(0, image_res)]
    PHI_ = np.arctan2(Y, X) + np.pi
    R_ = np.sqrt(X**2 + Y**2)
    R_[R_ > 1] = 1
    THETA_ = np.arccos(1-R_)

    #
    # Calculate the transform
    #
    H = calcTransformMatrix((PHI, THETA), (PHI_, THETA_))

    H = spdiag(np.cos(THETA_)) * H
    
    #
    # Account for cos(\theta)
    #
    if theta_compensation:
        H = spdiag(np.cos(THETA_)**2) * H
    
    return H


def linearCameraTransformMatrix(PHI, THETA, image_res=256, theta_compensation=False):
    """
    Calculate a sparse matrix representation of a linear camera projection transform.
    
    Parameters
    ----------
    PHI, THETA : 3D arrays
        \phi and \theta angle grids.
    
    image_res : int, optional (default=256)
        Resolution of the camera image (both dimensions)
        
    theta_compensation : bool, optional (default=False)
        Compensate for angle between ray and pixel
        
    Returns
    -------
    H : sparse matrix
        Sparse matrix, in csr format, representing the transform.
"""

    import numpy as np
    import amitibo
    
    Y, X = np.mgrid[-1:1:complex(0, image_res), -1:1:complex(0, image_res)]
    PHI_ = np.arctan2(Y, X) + np.pi
    R_ = np.sqrt(X**2 + Y**2)
    THETA_ = R_ * np.pi / 2

    #
    # Calculate the transform
    #
    H = calcTransformMatrix((PHI, THETA), (PHI_, THETA_))

    #
    # Account for cos(\theta)
    #
    if theta_compensation:
        H = spdiag(np.cos(THETA_)) * H
    
    return H



