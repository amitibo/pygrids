"""
"""

from __future__ import division
import scipy.sparse as sps
import itertools
import numpy as np
cimport numpy as np
import cython
from cpython cimport bool
from libc.math cimport sqrt

DTYPEd = np.double
ctypedef np.double_t DTYPEd_t
DTYPEi32 = np.int32
ctypedef np.int32_t DTYPEi32_t
DTYPEi = np.int
ctypedef np.int_t DTYPEi_t


#static int
#local_argsearch_left(PyArrayObject *arr, PyArrayObject *key,
                     #PyArrayObject *sorter, PyArrayObject *ret)
#{
    #PyArray_CompareFunc *compare = PyArray_DESCR(key)->f->compare;
    #npy_intp nelts = PyArray_DIMS(arr)[PyArray_NDIM(arr) - 1];
    #npy_intp nkeys = PyArray_SIZE(key);
    #char *parr = PyArray_DATA(arr);
    #char *pkey = PyArray_DATA(key);
    #npy_intp *psorter = (npy_intp *)PyArray_DATA(sorter);
    #npy_intp *pret = (npy_intp *)PyArray_DATA(ret);
    #int elsize = PyArray_DESCR(arr)->elsize;
    #npy_intp i;

    #for (i = 0; i < nkeys; ++i) {
        #npy_intp imin = 0;
        #npy_intp imax = nelts;
        #while (imin < imax) {
            #npy_intp imid = imin + ((imax - imin) >> 1);
            #npy_intp indx = psorter[imid];

            #if (indx < 0 || indx >= nelts) {
                #return -1;
            #}
            #if (compare(parr + elsize*indx, pkey, key) < 0) {
                #imin = imid + 1;
            #}
            #else {
                #imax = imid;
            #}
        #}
        #*pret = imin;
        #pret += 1;
        #pkey += elsize;
    #}
    #return 0;
#}


def slimGrids(grids):
    """Calculate open grids from full grids"""

    slim_grids = []
    for dim, grid in enumerate(grids):
        sli = [0] * len(grid.shape)
        sli[dim] = Ellipsis
        grid = grid[sli]
        slim_grids.append(np.ascontiguousarray(grid.ravel()))

    return slim_grids


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


cdef calcCrossings(Y, X, Z, p0, p1):
    #
    # Collect the inter indices (grid crossings)
    #
    cdef np.intp_t i, j, k, sort_index
    cdef np.intp_t points_num
    cdef np.intp_t x_i0, x_i1, y_i0, y_i1, z_i0, z_i1
    cdef dimx = X.size
    cdef dimz = Z.size
    
    #
    # 5 - just searchsorted
    # 10 - all loop
    #
    cdef int[:] start_indices = np.empty(3, dtype=int,)
    y_i0, y_i1 = np.searchsorted(Y, [p0[0], p1[0]])
    start_indices[0] = y_i0-1
    x_i0, x_i1 = np.searchsorted(X, [p0[1], p1[1]])
    start_indices[1] = x_i0-1
    z_i0, z_i1 = np.searchsorted(Z, [p0[2], p1[2]])
    start_indices[2] = z_i1-1
    
    #
    # Calculate inter points (grid crossings)
    #
    # 6 sec
    d = p1 - p0
    points_num = abs(y_i1 - y_i0) + abs(x_i1 - x_i0) + abs(z_i1 - z_i0)

    #
    # Check whether the start and end points are in the same voxel
    #
    np_r = np.empty(points_num+1)
    np_indices = np.empty(points_num+1, dtype=DTYPEi)
    cdef double[:] r = np_r
    cdef int[:] indices = np_indices
    
    if points_num == 0:
        r[0] = sqrt(np.sum(d * d))
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

    print np_I
    print np_P
    
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

    print np_SI
    print np_SP
    
    #
    # 0.3 sec
    #
    for i in range(3):
        for j in range(points_num+1):
            if SI[i, j] == -1:
                SI[i, j] = SI[i, j-1]
    
    print np_SI
    
    #
    # Calculate path segments length
    #
    cdef double tmpd
    cdef int tmpi
    for j in range(points_num+1):
        tmpd = 0
        for i in range(3):
            tmpd += (SP[i, j+1] - SP[i, j])**2
        r[j] = sqrt(tmpd)
        indices[j] = SI[0, j]*dimx + SI[1, j]*dimz + SI[2, j]
    
    print np_r, np_indices
    
    #
    # Order the indices
    #
    # 0 sec
    if indices[0] > indices[points_num]:
        for j in range(points_num+1):
            tmpd = r[j]
            r[j] = r[points_num-j]
            r[points_num-j] = tmpd
            tmpi = indices[j]
            indices[j] = indices[points_num-j]
            indices[points_num-j] = tmpi
            
    return np_r, np_indices


def point2grids(point, Y, X, Z):
    
    Y_slim, X_slim, Z_slim = slimGrids((Y, X, Z))

    data = []
    indices = []
    indptr = [0]

    cdef DTYPEd_t [:] np_Y = np.array(Y, dtype=DTYPEd, order='C', copy=False).ravel()
    cdef DTYPEd_t [:] np_X = np.array(X, dtype=DTYPEd, order='C', copy=False).ravel()
    cdef DTYPEd_t [:] np_Z = np.array(Z, dtype=DTYPEd, order='C', copy=False).ravel()

    p1 = np.array(point, order='C').ravel()
    np_p2 = np.empty(3)
    cdef double[:] p2 = np_p2
    
    cdef int grid_size = Y.size
    cdef int i = 0
    for i in xrange(grid_size):
        p2[0] = np_Y[i] + 0.5
        p2[1] = np_X[i] + 0.5
        p2[2] = np_Z[i] + 0.5
        r, ind = calcCrossings(Y_slim, X_slim, Z_slim, p1, p2)
        print p1, np_p2
        print r, ind
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