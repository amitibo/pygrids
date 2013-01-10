"""
"""

from __future__ import division
import numpy as np
import scipy.sparse as sps
import itertools
import amitibo
import time

eps = np.finfo(np.float).eps
eps32 = np.finfo(np.float32).eps


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
    ratio = np.ones_like(DGrid)
    
    Ll = (Grid + DGrid) < lower_limit
    if np.any(Ll):
        ratio[Ll] = (lower_limit - Grid[Ll]) / DGrid[Ll]
        
    Lh = (Grid + DGrid) > upper_limit
    if np.any(Lh):
        ratio[Lh] = (upper_limit - Grid[Lh]) / DGrid[Lh]
    
    return ratio, Ll + Lh
    

def count_unique(keys, dists):
    """count frequency of unique values in array. non positive values are ignored."""
    
    nnzs = keys>0
    filtered_keys = keys[nnzs]
    filtered_dists = dists[nnzs]
    if filtered_keys.size == 0:
        return filtered_keys, filtered_dists
    
    uniq_keys, inv_indices = np.unique(filtered_keys, return_inverse=True)
    uniq_dists = np.empty(uniq_keys.size)
    
    for i in range(uniq_keys.size):
        d = filtered_dists[inv_indices == i]
        uniq_dists[i] = d.sum()
        
    return uniq_keys, uniq_dists


def integrateGrids(camera_center, Y, X, Z, image_res, subgrid_res=(10, 10, 10), noise=0):
    
    dummy, (Y_open, X_open, Z_open) = processGrids((Y, X, Z))
    
    del dummy
    
    Y = Y - camera_center[0]
    X = X - camera_center[1]
    Z = Z - camera_center[2]
    
    Y1, X1, Z1 = np.mgrid[0:Y.shape[0], 0:Y.shape[1], 0:Y.shape[2]]
    Y1 -= np.searchsorted(Y_open, camera_center[0])
    X1 -= np.searchsorted(X_open, camera_center[1])
    Z1 -= np.searchsorted(Z_open, camera_center[2])
    R = np.sqrt(Y1**2 + X1**2 + Z1**2)
    
    del Y1, X1, Z1
    
    #
    # Calculate the subpixel density
    #
    P = (100/(R+eps)).astype(np.int)
    sub_resY, sub_resX, sub_resZ = subgrid_res
    Py = P * sub_resY
    Px = P * sub_resX
    Pz = P * sub_resZ
    
    Py[Py>100] = 50
    Py[Py<2] = 2
    Px[Px>100] = 50
    Px[Px<2] = 2
    Pz[Pz>100] = 50
    Pz[Pz<2] = 2

    
    #
    # Calculate sub grids
    #
    deltay = Y_open[1:]-Y_open[:-1]
    deltax = X_open[1:]-X_open[:-1]
    deltaz = Z_open[1:]-Z_open[:-1]
    
    #
    # Loop on all voxels
    #
    data = []
    indices = []
    indptr = [0]
    i = 0
    for y, x, z, py, px, pz, (dy, dx, dz) in \
        itertools.izip(Y.ravel(), X.ravel(), Z.ravel(), Py.ravel(), Px.ravel(), Pz.ravel(), itertools.product(deltay, deltax, deltaz)):

        #
        # Create the sub grids
        #
        dY, dX, dZ = np.mgrid[0:1:1/py, 0:1:1/px, 0:1:1/pz]
        
        #
        # Advance to next sub grid position
        #
        subY = y + (dY + (np.random.rand(*dY.shape)-0.5) * noise) * dy
        subX = x + (dX + (np.random.rand(*dY.shape)-0.5) * noise) * dx
        subZ = z + (dZ + (np.random.rand(*dY.shape)-0.5) * noise) * dz
        
        #
        # Project the grids to the image space
        #
        subr2 = subY**2 + subX**2
        subR2 = subr2 + subZ**2
        subR2[subR2<0.05] = 0.05
        THETA = np.arctan2(np.sqrt(subr2), subZ)
        PHI = np.arctan2(subY, subX)
        
        #
        # Note:
        # Non valid values are set to NaN. When converting to np.int, the NaN values
        # become zero (when turned to in16). These are ignored in the count_unique
        # function.
        #
        R_sensor = THETA / np.pi * 2
        R_sensor[R_sensor>1] = np.nan
        Y_sensor = R_sensor * np.sin(PHI)
        X_sensor = R_sensor * np.cos(PHI)
        
        #
        # Calculate index of target image pixel and distance to camera
        #
        Y_index = ((Y_sensor + 1)*image_res/2).astype(np.int16)
        X_index = ((X_sensor + 1)*image_res/2).astype(np.int16)
        sub_indices = (Y_index * image_res + X_index).ravel()
        sub_dists = 1/subR2.ravel()
        
        inds, dists = count_unique(sub_indices, sub_dists)
        data.append(dists / subY.size)
        indices.append(inds)
        indptr.append(indptr[-1]+inds.size)

    print 'end first stage %s' % time.asctime()
    
    
    #
    # Create sparse matrix
    #
    data = np.hstack(data)
    indices = np.hstack(indices)

    H_int = sps.csr_matrix(
        (data, indices, indptr),
        shape=(Y.size, image_res*image_res)
    )
    
    print 'end second stage %s' % time.asctime()

    #
    # Transpose matrix
    #
    H_int = H_int.transpose()
    
    #
    # TODO: Weight by distance
    #
    return H_int

    
if __name__ == '__main__':
    pass
    