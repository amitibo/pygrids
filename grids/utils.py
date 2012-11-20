"""
"""

from __future__ import division
import numpy as np
import scipy.sparse as sps
import itertools


def processGrids(grids):
    """Calculate open grids and centered grids"""

    open_grids = []
    centered_grids = []
    
    for dim, grid in enumerate(grids):
        sli = [0] * len(grid.shape)
        sli[dim] = Ellipsis
        open_grid = grid[sli].ravel()
        open_grid = np.hstack((open_grid, 2*open_grid[-1]-open_grid[-2]))
        open_grids.append(np.ascontiguousarray(open_grid))
        
        vec_shape = [1] * len(grid.shape)
        vec_shape[dim] = -1
        centered_grid = grid + (open_grid[1:] - open_grid[:-1]).reshape(vec_shape) / 2
        centered_grids.append(np.ascontiguousarray(centered_grid))
        
    return centered_grids, open_grids


def limitDGrids(DGrid, Grid, lower_limit, upper_limit):
    LI = (Grid + DGrid) < lower_limit
    if np.any(LI):
        ratio = (lower_limit - Grid[LI]) / DGrid[LI]
        DGrid[LI] = DGrid[LI] * ratio
        
    LI = (Grid + DGrid) > upper_limit
    if np.any(LI):
        ratio = (upper_limit - Grid[LI]) / DGrid[LI]
        DGrid[LI] = DGrid[LI] * ratio
    
    return DGrid
    

def integrateGrids(camera_center, Y, X, Z, image_res, pixel_fov):
    #
    # Calculate open and centered grids
    #
    (Y, X, Z), (Y_open, X_open, Z_open) = processGrids((Y, X, Z))
    Y = Y - camera_center[0]
    X = X - camera_center[1]
    Z = Z - camera_center[2]
    R = np.sqrt(Y**2 + X**2 + Z**2)
    Y = Y/R
    X = X/R
    Z = Z/R
    R_voxel = np.sqrt(
        (Y_open[1:]-Y_open[:-1]).reshape((-1, 1, 1))**2 +
        (X_open[1:]-X_open[:-1]).reshape((1, -1, 1))**2 +
        (Z_open[1:]-Z_open[:-1]).reshape((1, 1, -1))**2
    )
    
    #
    # Calculate the relation between sensor pixel and line of sight (LOS) vector
    #
    Y_sensor, X_sensor = np.mgrid[-1:1:complex(0, image_res), -1:1:complex(0, image_res)]
    PHI_los = np.arctan2(Y_sensor, X_sensor) + np.pi
    R_los = np.sqrt(X_sensor**2 + Y_sensor**2)
    THETA_los = R_los * np.pi / 2
    Y_los = np.sin(PHI_los) * np.sin(THETA_los)
    X_los = np.cos(PHI_los) * np.sin(THETA_los)
    Z_los = np.cos(THETA_los)
    VIZ_los = R_los <= 1
    
    #
    # Iterate on all sensor pixels
    #
    data = []
    indices = []
    indptr = [0]
    for y_los, x_los, z_los, viz_los in itertools.izip(Y_los.flat, X_los.flat, Z_los.flat, VIZ_los.flat):
        #
        # Check if pixel is visible
        #
        if not viz_los:
            indptr.append(indptr[-1])
            continue
        
        #
        # Calculate the angle between the voxel and the pixel ray
        #
        cos_THETA = (Y*y_los + X*x_los + Z*z_los)
        sin_THETA = np.sqrt(1 - cos_THETA**2)
        
        #
        # Calculate the distance between all voxels to the ray.
        #
        D = np.abs(R * sin_THETA)
        R_cone = np.abs(R * cos_THETA) * pixel_fov
        
        #
        # Check intersecting Indices
        #
        II = (R_cone + R_voxel) > D
        nnz_inds = np.flatnonzero(II)
        indices.append(nnz_inds)
        data.append(np.ones(nnz_inds.size) / nnz_inds.size)
        indptr.append(indptr[-1]+nnz_inds.size)

    #
    # Form the sparse transform matrix
    #
    data = np.hstack(data)
    indices = np.hstack(indices)
    
    H_int = sps.csr_matrix(
        (data, indices, indptr),
        shape=(image_res*image_res, Y.size)
    )
    
    return H_int
