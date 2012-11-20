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
    

