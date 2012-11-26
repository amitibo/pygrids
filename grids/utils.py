"""
"""

from __future__ import division
import numpy as np
import scipy.sparse as sps
import itertools
import amitibo


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
    

def integrateGrids_old(camera_center, Y, X, Z, image_res, pixel_fov):
    #
    # Calculate open and centered grids
    #
    (Y, X, Z), (Y_open, X_open, Z_open) = processGrids((Y, X, Z))
    
    #
    # Calculate the unit vector direction for each voxel
    #
    Y = Y - camera_center[0]
    X = X - camera_center[1]
    Z = Z - camera_center[2]
    R = np.sqrt(Y**2 + X**2 + Z**2)
    Y = Y/R
    X = X/R
    Z = Z/R
    
    #
    # Calculate the radius of each voxel
    #
    R_voxel = np.sqrt(
        (Y_open[1:]-Y_open[:-1]).reshape((-1, 1, 1))**2 +
        (X_open[1:]-X_open[:-1]).reshape((1, -1, 1))**2 +
        (Z_open[1:]-Z_open[:-1]).reshape((1, 1, -1))**2
    )
    
    #
    # Calculate the relation between sensor pixel and line of sight (LOS) unit vector
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
        D_voxel_ray = np.abs(R * sin_THETA)
        R_cone = R * cos_THETA * pixel_fov
        
        #
        # Find voxels intersecting the cone
        #
        II = ((R_cone + R_voxel) > D_voxel_ray) * (cos_THETA > 0)
        nnz_inds = np.flatnonzero(II)
        indices.append(nnz_inds)
        data.append(np.ones(nnz_inds.size)/ R[II]**2)
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


def count_unique(keys):
    filtered_keys = keys[keys>0]
    if filtered_keys.size == 0:
        return filtered_keys, filtered_keys
    uniq_keys = np.unique(filtered_keys)
    bins = uniq_keys.searchsorted(filtered_keys)
    return uniq_keys, np.bincount(bins)


def integrateGrids(camera_center, Y, X, Z, image_res, subgrid_res=None):
    
    dummy, (Y_open, X_open, Z_open) = processGrids((Y, X, Z))
    
    del dummy
    
    Y = Y - camera_center[0]
    X = X - camera_center[1]
    Z = Z - camera_center[2]
    
    #
    # Calculate sub grids
    #
    if subgrid_res is None:
        subgrid_res = (10, 10, 10)
    resY, resX, resZ = subgrid_res
    subgrid_size = resY*resX*resZ
    
    dY = (Y_open[1:]-Y_open[:-1]).reshape((-1, 1, 1))/resY
    dX = (X_open[1:]-X_open[:-1]).reshape((1, -1, 1))/resX
    dZ = (Z_open[1:]-Z_open[:-1]).reshape((1, 1, -1))/resZ
    
    #
    # Loop on all sub grids
    #
    I = np.empty((Y.size, subgrid_size), dtype=np.int)
    for j, (dy, dx, dz) in enumerate(itertools.product(range(resY), range(resX), range(resZ))):
        #
        # Advance to next sub grid position
        #
        subY = Y + dY*dy
        subX = X + dX*dx
        subZ = Z + dZ*dz
        
        #
        # Project the grids to the image space
        #
        subr2 = subY**2 + subX**2
        subR2 = subr2 + subZ**2
        THETA = np.arctan2(np.sqrt(subr2), subZ)
        PHI = np.arctan2(subY, subX)
        
        R_sensor = THETA / np.pi * 2
        R_sensor[R_sensor>1] = np.nan
        Y_sensor = R_sensor * np.sin(PHI)
        X_sensor = R_sensor * np.cos(PHI)
        
        #
        # Calculate index of image pixel
        #
        Y_index = ((Y_sensor + 1)*image_res/2).astype(np.int)
        X_index = ((X_sensor + 1)*image_res/2).astype(np.int)
        I[:, j] = (Y_index * image_res + X_index).ravel()
        
    #
    # Calculate the unique indices
    #
    data = []
    indices = []
    indptr = [0]
    for i in xrange(I.shape[0]):
        inds, counts = count_unique(I[i, :])
        data.append(counts / subgrid_size)
        indices.append(inds)
        indptr.append(indptr[-1]+inds.size)
        
        #data.append(np.unique(I[i, :]))
        
    #
    # Calculate repetitions
    #
    
    #
    # Create sparse matrix
    #
    data = np.hstack(data)
    indices = np.hstack(indices)

    H_int = sps.csr_matrix(
        (data, indices, indptr),
        shape=(Y.size, image_res*image_res)
    )
    
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
    