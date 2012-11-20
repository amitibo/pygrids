"""
"""

import scipy.sparse as sps
import numpy as np
cimport numpy as np
import cython
from cpython cimport bool
from libc.math cimport sqrt
from .utils import processGrids, limitDGrids

DTYPEd = np.double
ctypedef np.double_t DTYPEd_t
DTYPEi32 = np.int32
ctypedef np.int32_t DTYPEi32_t
DTYPEi = np.int
ctypedef np.int_t DTYPEi_t


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline np.intp_t local_argsearch_left(double [:] grid, double key):

    cdef np.intp_t imin = 0
    cdef np.intp_t imax = grid.size
    cdef np.intp_t imid
    
    while imin < imax:
        imid = imin + ((imax - imin) >> 1)
        
        if grid[imid] < key:
            imin = imid + 1
        else:
            imax = imid

    return imin
    

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline bool interpolatePoints(
    double[:] grid,
    np.intp_t i0,
    np.intp_t i1,
    np.intp_t si,
    np.intp_t dim,
    double[:] p0,
    double[:] d,
    int[:, ::1] I,
    double[:, ::1] P
    ):
    
    cdef np.intp_t i, j, k
    cdef np.intp_t dt
    
    if i0 == i1:
        return False

    if i0 > i1:
        i0 -= 1
        i1 -= 1
        dt = -1
    else:
        dt = 1
        
    #
    # Loop on the dimension
    #
    for j in xrange(3):
        #
        # Loop on all the intersections of a dimension
        #
        i = si
        for k in xrange(i0, i1, dt):
            #
            # Calculate the indices of the points
            #
            if j == dim:
                I[j, i] = k

            #
            # Interpolate the value at the point
            #
            P[j, i] = d[j]/d[dim] * (grid[k]-p0[dim]) + p0[j]

            i += 1

    return True


@cython.boundscheck(False)
@cython.wraparound(False)
cdef calcCrossings(
    double[:] Y,
    double[:] X,
    double[:] Z,
    double[:] p0,
    double[:] p1
    ):
    #
    # Collect the inter indices (grid crossings)
    #
    cdef np.intp_t i, j, k, sort_index
    cdef np.intp_t points_num
    cdef np.intp_t x_i0, x_i1, y_i0, y_i1, z_i0, z_i1
    cdef np.intp_t dimx = X.size - 1
    cdef np.intp_t dimz = Z.size - 1
    cdef double tmpd
    cdef int tmpi
    
    #
    # 5 - just searchsorted
    # 10 - all loop
    #
    cdef int[:] start_indices = np.empty(3, dtype=int,)
    
    y_i0 = local_argsearch_left(Y, p0[0])
    y_i1 = local_argsearch_left(Y, p1[0])
    x_i0 = local_argsearch_left(X, p0[1])
    x_i1 = local_argsearch_left(X, p1[1])
    z_i0 = local_argsearch_left(Z, p0[2])
    z_i1 = local_argsearch_left(Z, p1[2])
    start_indices[0] = y_i0-1
    start_indices[1] = x_i0-1
    start_indices[2] = z_i1-1
    
    #
    # Calculate inter points (grid crossings)
    #
    cdef double[:] d = np.empty(3)
    d[0] = p1[0] - p0[0]
    d[1] = p1[1] - p0[1]
    d[2] = p1[2] - p0[2]
    points_num = abs(y_i1 - y_i0) + abs(x_i1 - x_i0) + abs(z_i1 - z_i0)

    #
    # Check whether the start and end points are in the same voxel
    #
    np_r = np.empty(points_num+1)
    np_indices = np.empty(points_num+1, dtype=DTYPEi)
    cdef double[:] r = np_r
    cdef int[:] indices = np_indices
    
    if points_num == 0:
        tmpd = 0
        for i in range(3):
            tmpd += (d[i] - d[i])**2
        r[0] = sqrt(tmpd)
        indices[0] = start_indices[0]*dimx + start_indices[1]*dimz + start_indices[2]
        return np_r, np_indices
    
    np_I = -np.ones((3, points_num), dtype=DTYPEi)
    np_P = np.empty((3, points_num))
    cdef int[:, ::1] I = np_I
    cdef double[:, ::1] P = np_P
    
    if interpolatePoints(Y, y_i0, y_i1, 0, 0, p0, d, I, P):
        sort_index = 0
    if interpolatePoints(X, x_i0, x_i1, abs(y_i1 - y_i0), 1, p0, d, I, P):
        sort_index = 1
    if interpolatePoints(Z, z_i0, z_i1, abs(y_i1 - y_i0)+abs(x_i1 - x_i0), 2, p0, d, I, P):
        sort_index = 2

    #
    # Sort points according to their order
    #
    cdef int[:] order = np.argsort(P[sort_index, :]).astype(int)
    if p0[sort_index] > p1[sort_index]:
        order = order[::-1]

    np_SI = np.empty((3, points_num+1), dtype=DTYPEi)
    np_SP = np.empty((3, points_num+2))
    cdef int[:, ::1] SI = np_SI
    cdef double[:, ::1] SP = np_SP
    
    for i in range(3):        
        SI[i, 0] = start_indices[i]
        SP[i, 0] = p0[i]
        SP[i, points_num+1] = p1[i]
        for j in range(points_num):
            SI[i, j+1] = I[i, order[j]]
            SP[i, j+1] = P[i, order[j]]

    #
    # Fillup the missing indices in the different dimensions
    #
    for i in range(3):
        for j in range(points_num+1):
            if SI[i, j] == -1:
                SI[i, j] = SI[i, j-1]
    
    #
    # Calculate path segments length
    #
    for j in range(points_num+1):
        tmpd = 0
        for i in range(3):
            tmpd += (SP[i, j+1] - SP[i, j])**2
        r[j] = sqrt(tmpd)
        indices[j] = SI[0, j]*dimx + SI[1, j]*dimz + SI[2, j]
    
    #
    # Order the indices
    #
    if indices[0] > indices[points_num]:
        for j in range(points_num+1):
            tmpd = r[j]
            r[j] = r[points_num-j]
            r[points_num-j] = tmpd
            tmpi = indices[j]
            indices[j] = indices[points_num-j]
            indices[points_num-j] = tmpi
            
    return np_r, np_indices


@cython.boundscheck(False)
def point2grids(point, Y, X, Z):
    
    #
    # Calculate open and centered grids
    #
    (Y, X, Z), (Y_open, X_open, Z_open) = processGrids((Y, X, Z))

    cdef DTYPEd_t [:] p_Y = Y.ravel()
    cdef DTYPEd_t [:] p_X = X.ravel()
    cdef DTYPEd_t [:] p_Z = Z.ravel()

    p1 = np.array(point, order='C').ravel()
    cdef double[:] p2 = np.empty(3)

    data = []
    indices = []
    indptr = [0]
    cdef int grid_size = Y.size
    cdef int i = 0
    for i in xrange(grid_size):
        #
        # Process next voxel
        #
        p2[0] = p_Y[i]
        p2[1] = p_X[i]
        p2[2] = p_Z[i]
        
        #
        # Calculate crossings for line between p1 and p2
        #
        r, ind = calcCrossings(Y_open, X_open, Z_open, p1, p2)
        
        #
        # Accomulate the crossings for the sparse matrix
        #
        data.append(r)
        indices.append(ind)
        indptr.append(indptr[-1]+r.size)

    #
    # Create the sparse matrix
    #
    data = np.hstack(data)
    indices = np.hstack(indices)
    
    H_dist = sps.csr_matrix(
        (data, indices, indptr),
        shape=(Y.size, Y.size)
    )
    
    return H_dist
    

@cython.boundscheck(False)
def direction2grids(phi, theta, Y, X, Z):
    
    #
    # Calculate open and centered grids
    #
    (Y, X, Z), (Y_open, X_open, Z_open) = processGrids((Y, X, Z))

    cdef DTYPEd_t [:] p_Y = Y.ravel()
    cdef DTYPEd_t [:] p_X = X.ravel()
    cdef DTYPEd_t [:] p_Z = Z.ravel()

    #
    # Calculate the intersection with the TOA (Top Of Atmosphere)
    #
    toa = np.max(Z_open)
    DZ = toa - Z
    DX = DZ * np.cos(phi) * np.tan(theta)
    DY = DZ * np.sin(phi) * np.tan(theta)

    #
    # Check crossing with any of the sides
    #
    DY = limitDGrids(DY, Y, np.min(Y_open), np.max(Y_open))
    DX = limitDGrids(DX, X, np.min(X_open), np.max(X_open))
    DZ = limitDGrids(DZ, Z, np.min(Z_open), np.max(Z_open))
    
    cdef DTYPEd_t [:] p_DY = DY.ravel()
    cdef DTYPEd_t [:] p_DX = DX.ravel()
    cdef DTYPEd_t [:] p_DZ = DZ.ravel()

    cdef double[:] p1 = np.empty(3)
    cdef double[:] p2 = np.empty(3)
    
    data = []
    indices = []
    indptr = [0]
    cdef int grid_size = Y.size
    cdef int i = 0
    for i in xrange(grid_size):
        #
        # Center of each voxel
        #
        p1[0] = p_Y[i]
        p1[1] = p_X[i]
        p1[2] = p_Z[i]
        
        #
        # Intersection of the ray with the TOA
        #
        p2[0] = p1[0] + p_DY[i]
        p2[1] = p1[1] + p_DX[i]
        p2[2] = p1[2] + p_DZ[i]
        
        #
        # Calculate crossings for line between p1 and p2
        #
        r, ind = calcCrossings(Y_open, X_open, Z_open, p1, p2)
        
        #
        # Accomulate the crossings for the sparse matrix
        #
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
