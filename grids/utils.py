"""
"""

from __future__ import division
import numpy as np
import scipy.sparse as sps
import itertools
import amitibo

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


def integrateGrids(camera_center, Y, X, Z, image_res, subgrid_noise=0, subgrid_min=(2, 2, 2), subgrid_max=(50, 50, 50), min_radius=0.05):
    
    dummy, (Y_open, X_open, Z_open) = processGrids((Y, X, Z))
    
    del dummy
    
    #
    # Center the grids
    #
    Y = Y - camera_center[0]
    X = X - camera_center[1]
    Z = Z - camera_center[2]
    
    #
    # Calculate the distace from each camera to each voxel (in voxels)
    #
    Y1, X1, Z1 = np.mgrid[0:Y.shape[0], 0:Y.shape[1], 0:Y.shape[2]]
    Y1 -= np.searchsorted(Y_open, camera_center[0])
    X1 -= np.searchsorted(X_open, camera_center[1])
    Z1 -= np.searchsorted(Z_open, camera_center[2])
    P = 1/(np.sqrt(Y1**2 + X1**2 + Z1**2) + eps)
    P[P>1] = 1
    
    del Y1, X1, Z1
    
    #
    # Calculate the subgrid density
    #
    Py = (P * subgrid_max[0]).astype(np.int)
    Px = (P * subgrid_max[1]).astype(np.int)
    Pz = (P * subgrid_max[2]).astype(np.int)
    
    Py[Py<subgrid_min[0]] = subgrid_min[0]
    Px[Px<subgrid_min[1]] = subgrid_min[1]
    Pz[Pz<subgrid_min[2]] = subgrid_min[2]

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
        # y, x, z    - The current voxel coordinates (relative to the camera)
        # py, px, pz - The resolution of the subgrid at the currect voxel.
        # dy, dx, dz - The size of the current voxel
        # 
        
        #
        # Create the sub grids of the current voxel
        # dY, dX, dZ - Relative coordinates of the subgrid
        #
        dY, dX, dZ = np.mgrid[0:py, 0:px, 0:pz]
        
        #
        # Calculate the subgrid coordinates with noise
        # A uniformly distributed noise in the size of the subgrid multiplied by subgrid_noise
        #
        subY = y + (dY + (np.random.rand(*dY.shape)-0.5) * subgrid_noise) / py * dy
        subX = x + (dX + (np.random.rand(*dY.shape)-0.5) * subgrid_noise) / px * dx
        subZ = z + (dZ + (np.random.rand(*dY.shape)-0.5) * subgrid_noise) / pz * dz
        
        #
        # Project the grids to the image space
        #
        subr2 = subY**2 + subX**2
        subR2 = subr2 + subZ**2
        subR2[subR2<min_radius] = min_radius
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


def rayCasting(Y, X, Z, img_res, camera_center, samples_num):
    
    #
    # Center the grids
    #
    Y = Y - camera_center[0]
    X = X - camera_center[1]
    Z = Z - camera_center[2]
    
    dummy, (Y_atmo, X_atmo, Z_atmo) = processGrids((Y, X, Z))
    
    del dummy
    
    #
    # Convert image pixels to ray direction
    #
    Y_img, step = np.linspace(-1.0, 1.0, img_res, endpoint=False, retstep=True)
    X_img = np.linspace(-1.0, 1.0, img_res, endpoint=False)
    X_img, Y_img = np.meshgrid(X_img+step/2, Y_img+step/2)
    R_img = np.sqrt(X_img**2 + Y_img**2)
    THETA_ray = R_img * np.pi /2
    PHI_ray = np.arctan2(Y_img, X_img)
    DY_ray = Y_img / R_img
    DX_ray = X_img / R_img
    DZ_ray = np.cos(THETA_ray)
    
    #
    # Calculate ray samples
    #
    R_max = np.max(np.sqrt(Y**2 + X**2 + Z**2))
    R_samples = np.linspace(0.0, R_max, samples_num)[1:]
    
    #
    # Loop on all rays
    #
    data = []
    indices = []
    indptr = [0]
    cnt = 0
    xmlab = []
    ymlab = []
    zmlab = []
    vmlab = []
    filt_weights = []
    filt_indices = []
    for r, dy, dx, dz in itertools.izip(R_img.ravel(), DY_ray.ravel(), DX_ray.ravel(), DZ_ray.ravel()):
        print cnt
        cnt += 1
        
        if r > 1:
            indptr.append(indptr[-1])
            continue
        
        #
        # 'Dither' the ray
        #
        
        #
        # Convert the ray samples to volume indices
        #
        Y_ray = R_samples * dy
        X_ray = R_samples * dx
        Z_ray = R_samples * dz
        
        #
        # Calculate the atmosphere indices
        #
        Y_indices = np.searchsorted(Y_atmo, Y_ray)
        X_indices = np.searchsorted(X_atmo, X_ray)
        Z_indices = np.searchsorted(Z_atmo, Z_ray)
        
        #
        # Calculate unique indices
        #
        Y_filter = (Y_indices > 0) * (Y_indices < Y_atmo.size)
        X_filter = (X_indices > 0) * (X_indices < X_atmo.size)
        Z_filter = (Z_indices > 0) * (Z_indices < Z_atmo.size)
        
        Y_indices = Y_indices[Y_filter*X_filter*Z_filter]-1
        X_indices = X_indices[Y_filter*X_filter*Z_filter]-1
        Z_indices = Z_indices[Y_filter*X_filter*Z_filter]-1

        if cnt % 23 == 0:
            xmlab.append(X_indices)
            ymlab.append(Y_indices)
            zmlab.append(Z_indices)
            
        inds_ray = (Y_indices*Y.shape[1] + X_indices)*Y.shape[2] + Z_indices
        
        #
        # Calculate weights
        #
        uniq_indices, inv_indices = np.unique(inds_ray, return_index=True)

        weights = []
        for i, ind in enumerate(uniq_indices):
            weights.append((inv_indices == i).sum())
        
        #
        # Sum up the indices and weights
        #
        data.append(weights)
        indices.append(uniq_indices)
        indptr.append(indptr[-1]+uniq_indices.size)

        if cnt % 20 == 0:
            filt_indices.append(uniq_indices)
            filt_weights.append(weights)
            
    xmlab = np.concatenate(xmlab)
    ymlab = np.concatenate(ymlab)
    zmlab = np.concatenate(zmlab)
    filt_indices = np.concatenate(filt_indices)
    filt_weights = np.concatenate(filt_weights)
    
    #
    # Create sparse matrix
    #
    data = np.hstack(data)
    indices = np.hstack(indices)

    H_int = sps.csr_matrix(
        (data, indices, indptr),
        shape=(img_res*img_res, Y.size)
    )
    
    return H_int


if __name__ == '__main__':
    
    import matplotlib.pyplot as plt

    Y, X, Z = np.mgrid[0:50000:1000, 0:50000:1000, 0:10000:100]
    img_res = 128
    camera_center = (25001, 25001, 1)
    samples_num = 1000
    
    atmo = np.zeros_like(Y)
    R = (Y-50000/3)**2/16 + (X-50000*2/3)**2/16 + (Z-10000/4)**2*8
    atmo[R<4000**2] = 1

    H_int = rayCasting(Y, X, Z, img_res, camera_center, samples_num)

    img = (H_int * atmo.reshape((-1, 1))).reshape((img_res, img_res))
    
    plt.imshow(img)
    plt.show()