"""
"""

from __future__ import division
import numpy as np
import scipy.sparse as sps
import itertools


def slimGrids(grids):
    """Calculate open grids from full grids"""

    slim_grids = []
    for dim, grid in enumerate(grids):
        sli = [0] * len(grid.shape)
        sli[dim] = Ellipsis
        grid = grid[sli]
        slim_grids.append(grid)

    return slim_grids

        
class GridsEnumerate(object):
    """
    """
    def __init__(self, Y, X, H):
        self.Y_iter = np.asarray(Y).flat
        self.X_iter = np.asarray(X).flat
        self.H_iter = np.asarray(H).flat

    def next(self):
        """
        Standard iterator method, returns the index tuple and array value.

        Returns
        -------
        coords : tuple of ints
            The indices of the current iteration.
        vals : scalar
            The array element of the current iteration.

        """
        return self.Y_iter.coords, (self.Y_iter.next(), self.X_iter.next(), self.H_iter.next())

    def __iter__(self):
        return self


def calcCrossings(Y, X, Z, p0, p1):
    
    #
    # Collect the inter indices (grid crossings)
    #
    indices = []
    start_indices = []
    for i, coords in enumerate((Y, X, Z)):
        i0, i1 = np.searchsorted(coords, [p0[i], p1[i]])
        start_indices.append(i0-1)
        if i0 == i1:
            indices.append(np.ones(0, dtype=np.int32))
        elif i0 < i1:
            indices.append(np.arange(i0, i1).reshape((1, -1)))
        else:
            indices.append(np.arange(i0-1, i1-1, -1).reshape((1, -1)))

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

    lengths = [0] + [inds.size for inds in indices]
    new_indices = -np.ones((3, np.sum(lengths)), dtype=np.int32)
    for i in range(3):
        new_indices[i, lengths[i]:lengths[i]+lengths[i+1]] = indices[i]
    new_indices = np.hstack((start_indices, new_indices[:, order]))
    for i in range(3):
        for j in range(new_indices.shape[1]):
            if new_indices[i, j] == -1:
                new_indices[i, j] = new_indices[i, j-1]
                
    #
    # Calculate distance between points
    #
    r = new_points[:, 1:] - new_points[:, :-1]
    r = np.sqrt(np.sum(r*r, axis=0))
    
    indices = np.ravel_multi_index(new_indices, dims=(Y.size, X.size, Z.size))
    
    if indices[0] > indices[-1]:
        indices = indices[::-1]
        r = r[::-1]
        
    return r, indices


def point2grids(point, Y, X, H):
    
    Y_slim, X_slim, H_slim = slimGrids((Y, X, H))
    p1 = np.array(point).reshape((-1, 1))

    data = []
    indices = []
    indptr = [0]

    for coords, p2 in GridsEnumerate(Y, X, H):
        p2 = np.array(p2).reshape((-1, 1)) + 0.5
        r, ind = calcCrossings(Y_slim, X_slim, H_slim, p1, p2)
        if (np.linalg.norm(p2-p1) - np.sum(r)) > 10**-6:
            print coords, p1, p2
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