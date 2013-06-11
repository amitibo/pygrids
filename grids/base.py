from __future__ import division
import numpy as np
import scipy.sparse as sps
import itertools
from .base import *

__all__ = [
    'eps',
    'eps32',
    'processGrids',
    'count_unique',
    'calcTransformMatrix',
    'coords2Indices',
    'limitDGrids'
]

eps = np.finfo(np.float).eps
eps32 = np.finfo(np.float32).eps


def spdiag(X):
    """
    Return a sparse diagonal matrix. The elements of the diagonal are made of 
    the elements of the vector X.

    Parameters
    ----------
    X : array
        1D array to be placed on the diagonal.
        
    Returns
    -------
    H : sparse matrix
        Sparse diagonal matrix, in dia format.
"""

    import scipy.sparse as sps

    return sps.dia_matrix((X.ravel(), 0), (X.size, X.size)).tocsr()


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


def calcTransformMatrix(src_grids, dst_coords):
    """
    Calculate a sparse transformation matrix. The transform
    is represented as a mapping from the src_coords to the dst_coords.
    
    Parameters
    ----------
    src_grids : list of arrays
        Array of source grids.
        
    dst_coords : list of arrays
        Array of destination grids as points in the source grids.
        
    Returns
    -------
    H : parse matrix
        Sparse matrix, in csr format, representing the transform.
"""
    
    import numpy as np
    import scipy.sparse as sps
    import itertools

    #
    # Shape of grid
    #
    src_shape = src_grids[0].shape
    src_size = np.prod(np.array(src_shape))
    dst_shape = dst_coords[0].shape
    dst_size = np.prod(np.array(dst_shape))
    dims = len(src_shape)
    
    #
    # Calculate grid indices of coords.
    #
    indices, src_grids_slim = coords2Indices(src_grids, dst_coords)

    #
    # Filter out coords outside of the grids.
    #
    nnz = np.ones(indices[0].shape, dtype=np.bool_)
    for ind, dim in zip(indices, src_shape):
        nnz *= (ind > 0) * (ind < dim)

    dst_indices = np.arange(dst_size)[nnz]
    nnz_indices = []
    nnz_coords = []
    for ind, coord in zip(indices, dst_coords):
        nnz_indices.append(ind[nnz])
        nnz_coords.append(coord.ravel()[nnz])
    
    #
    # Calculate the transform matrix.
    #
    diffs = []
    indices = []
    for grid, coord, ind in zip(src_grids_slim, nnz_coords, nnz_indices):
        diffs.append([grid[ind] - coord, coord - grid[ind-1]])
        indices.append([ind-1, ind])

    diffs = np.array(diffs)
    diffs /= np.sum(diffs, axis=1).reshape((dims, 1, -1))
    indices = np.array(indices)

    dims_range = np.arange(dims)
    strides = np.array(src_grids[0].strides).reshape((-1, 1))
    strides /= strides[-1]
    I, J, VALUES = [], [], []
    for sli in itertools.product(*[[0, 1]]*dims):
        i = np.array(sli)
        c = indices[dims_range, sli, Ellipsis]
        v = diffs[dims_range, sli, Ellipsis]
        I.append(dst_indices)
        J.append(np.sum(c*strides, axis=0))
        VALUES.append(np.prod(v, axis=0))
        
    H = sps.coo_matrix(
        (np.array(VALUES).ravel(), np.array((np.array(I).ravel(), np.array(J).ravel()))),
        shape=(dst_size, src_size)
        ).tocsr()

    return H


def coords2Indices(grids, coords):
    """
    """

    import numpy as np

    inds = []
    slim_grids = []
    for dim, (grid, coord) in enumerate(zip(grids, coords)):
        sli = [0] * len(grid.shape)
        sli[dim] = Ellipsis
        grid = grid[sli]
        slim_grids.append(grid)
        inds.append(np.searchsorted(grid, coord.ravel()))

    return inds, slim_grids


def limitDGrids(DGrid, Grid, lower_limit, upper_limit):
    ratio = np.ones_like(DGrid)
    
    Ll = (Grid + DGrid) < lower_limit
    if np.any(Ll):
        ratio[Ll] = (lower_limit - Grid[Ll]) / DGrid[Ll]
        
    Lh = (Grid + DGrid) > upper_limit
    if np.any(Lh):
        ratio[Lh] = (upper_limit - Grid[Lh]) / DGrid[Lh]
    
    return ratio, Ll + Lh
    


