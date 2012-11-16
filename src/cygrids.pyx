"""
"""

from __future__ import division
import scipy.sparse as sps
import itertools
import numpy as np
cimport numpy as np
import cython

DTYPEd = np.double
ctypedef np.double_t DTYPEd_t
DTYPEi32 = np.int32
ctypedef np.int32_t DTYPEi32_t
DTYPEi = np.int
ctypedef np.int_t DTYPEi_t


def slimGrids(grids):
    """Calculate open grids from full grids"""

    slim_grids = []
    for dim, grid in enumerate(grids):
        sli = [0] * len(grid.shape)
        sli[dim] = Ellipsis
        grid = grid[sli]
        slim_grids.append(grid)

    return slim_grids

        
def calcCrossings(Y, X, Z, p0, p1):
    #
    # Collect the inter indices (grid crossings)
    #
    cdef int i
    cdef int j
    indices = []
    start_indices = []
    for i, coords in enumerate((Y, X, Z)):
        i0, i1 = np.searchsorted(coords, [p0[i], p1[i]])
        start_indices.append(i0-1)
        if i0 == i1:
            indices.append(np.ones(0, dtype=np.int))
        elif i0 < i1:
            indices.append(np.arange(i0, i1, dtype=np.int))
        else:
            indices.append(np.arange(i0-1, i1-1, -1, dtype=np.int))

    start_indices = np.array(start_indices, dtype=np.int32).reshape((-1, 1))
    
    #
    # Calculate inter points (grid crossings)
    #
    d = p1 - p0
    new_points = []
    sort_index = 0
    for i, (inds, coords) in enumerate(zip(indices, (Y, X, Z))):
        if inds.size == 0:
            continue
        sort_index = i
        new_points.append(d/d[i] * (coords[inds]-p0[i]) + p0)
    
    #
    # Check whether the start and end points are in the same voxel
    #
    if new_points == []:
        indices = np.ravel_multi_index(start_indices, dims=(Y.size, X.size, Z.size))
        r = np.sqrt(np.sum(d * d))
        return r, indices
    
    #
    # Sort points according to their order
    #
    new_points = np.hstack(new_points)
    order = np.argsort(new_points[sort_index, :])
    if p0[sort_index] > p1[sort_index]:
        order = order[::-1]
    new_points = np.hstack((p0, new_points[:, order], p1))

    #
    # calculate the indices of the voxels
    #
    lengths = [inds.size for inds in indices]
    cdef np.ndarray[DTYPEi_t, ndim=2] new_indices = -np.ones((3, np.sum(lengths)+1), dtype=np.int)
    cdef int start
    cdef int length
    cdef np.ndarray[DTYPEi_t, ndim=1] np_inds
    for i in range(3):
        start = 1
        length = lengths[i]
        np_inds = indices[i]
        new_indices[i, 0] = start_indices[i, 0]
        for j in range(length):
            new_indices[i, start + j] = np_inds[j]
        start += length
            
    new_indices[:, 1:] = new_indices[:, order]
    
    length = new_indices.shape[1]
    for i in range(3):
        for j in range(1, length):
            if new_indices[i, j] == -1:
                new_indices[i, j] = new_indices[i, j-1]
                
    #
    # Calculate path segments length
    #
    r = new_points[:, 1:] - new_points[:, :-1]
    r = np.sqrt(np.sum(r*r, axis=0))
    
    #
    # Translate the voxel coordinates to indices
    #
    indices = np.ravel_multi_index(new_indices, dims=(Y.size, X.size, Z.size))
    
    #
    # Order the indices
    #
    if indices[0] > indices[-1]:
        indices = indices[::-1]
        r = r[::-1]
        
    return r, indices


@cython.boundscheck(False)
def point2grids(point, Y, X, Z):
    
    Y_slim, X_slim, Z_slim = slimGrids((Y, X, Z))

    data = []
    indices = []
    indptr = [0]

    cdef np.ndarray[DTYPEd_t, ndim=1] np_Y = np.array(Y, dtype=DTYPEd, order='C', copy=False).ravel()
    cdef np.ndarray[DTYPEd_t, ndim=1] np_X = np.array(X, dtype=DTYPEd, order='C', copy=False).ravel()
    cdef np.ndarray[DTYPEd_t, ndim=1] np_Z = np.array(Z, dtype=DTYPEd, order='C', copy=False).ravel()

    p1 = np.array(point).reshape((-1, 1))
    cdef np.ndarray[DTYPEd_t, ndim=2] np_p2 = np.zeros((3, 1), dtype=DTYPEd, order='C')
    
    cdef int grid_size = Y.size
    cdef int i = 0
    for i in range(grid_size):
        np_p2[0, 0] = np_Y[i] + 0.5
        np_p2[1, 0] = np_X[i] + 0.5
        np_p2[2, 0] = np_Z[i] + 0.5
        r, ind = calcCrossings(Y_slim, X_slim, Z_slim, p1, np_p2)
        #if (np.linalg.norm(p2-p1) - np.sum(r)) > 10.0**-6:
            #print np.linalg.norm(p2-p1), np.sum(r), i, j, k p1, p2
            #raise Exception('Bad distances')
        data.append(r)
        indices.append(ind)
        indptr.append(indptr[-1]+r.size)

    data = np.hstack(data)
    indices = np.hstack(indices)
    
    H_dist = sps.csr_matrix(
        (data, indices, indptr),
        shape=(Y.size, Y.size)
    )
    
    return H_dist