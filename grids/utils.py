"""
"""

from __future__ import division
import numpy as np
import scipy.sparse as sps
import itertools
from .base import *

__all__ = ["integrateGrids", "rayCasting", "cartesian2sensor", "cumsumTransformMatrix", "integralTransformMatrix", "calcRotationMatrix"]


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


def rayCasting(camera_center, Y, X, Z, img_res, samples_num=1000, dither_noise=10, replicate=4):
    
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
    # The image is assumed the [-1, 1]x[-1, 1] square.
    #
    Y_img, step = np.linspace(-1.0, 1.0, img_res, endpoint=False, retstep=True)
    X_img = np.linspace(-1.0, 1.0, img_res, endpoint=False)
    X_img, Y_img = np.meshgrid(X_img, Y_img)

    #
    # Randomly replicate rays inside each pixel
    #
    X_img = np.tile(X_img[:, :, np.newaxis], [1, 1, replicate])
    Y_img = np.tile(Y_img[:, :, np.newaxis], [1, 1, replicate])
    X_img += np.random.rand(*X_img.shape)*step
    Y_img += np.random.rand(*Y_img.shape)*step
    
    #
    # Calculate rays angles
    # R_img is the radius from the center of the image (0, 0) to the
    # pixel. It is used for calculating th ray direction (PHI, THETA)
    # and for filtering pixels outside the image (radius > 1).
    #
    R_img = np.sqrt(X_img**2 + Y_img**2)
    THETA_ray = R_img * np.pi /2
    PHI_ray = np.arctan2(Y_img, X_img)
    DY_ray = np.sin(THETA_ray) * np.sin(PHI_ray)
    DX_ray = np.sin(THETA_ray) * np.cos(PHI_ray)
    DZ_ray = np.cos(THETA_ray)
    
    #
    # Calculate sample steps along ray
    #
    R_max = np.max(np.sqrt(Y**2 + X**2 + Z**2))
    R_samples, R_step = np.linspace(0.0, R_max, samples_num, retstep=True)
    R_samples = R_samples[1:].reshape((-1, 1))
    R_dither = np.random.rand(R_img.size) * R_step * dither_noise
        
    #
    # Loop on all rays
    #
    data = []
    indices = []
    indptr = [0]
    for r, dy, dx, dz, r_dither in itertools.izip(
        R_img.reshape((-1, replicate)),
        DY_ray.reshape((-1, replicate)),
        DX_ray.reshape((-1, replicate)),
        DZ_ray.reshape((-1, replicate)),
        R_dither.ravel(),
        ):

        if np.all(r > 1):
            indptr.append(indptr[-1])
            continue
        
        #
        # Filter steps where r > 1
        #
        dy = dy[r<=1]
        dx = dx[r<=1]
        dz = dz[r<=1]
        
        #
        # Convert the ray samples to volume indices
        #
        Y_ray = (r_dither+R_samples) * dy
        X_ray = (r_dither+R_samples) * dx
        Z_ray = (r_dither+R_samples) * dz
        
        #
        # Calculate the atmosphere indices
        #
        Y_indices = np.searchsorted(Y_atmo, Y_ray.ravel())
        X_indices = np.searchsorted(X_atmo, X_ray.ravel())
        Z_indices = np.searchsorted(Z_atmo, Z_ray.ravel())
        
        #
        # Calculate unique indices
        #
        Y_filter = (Y_indices > 0) * (Y_indices < Y_atmo.size)
        X_filter = (X_indices > 0) * (X_indices < X_atmo.size)
        Z_filter = (Z_indices > 0) * (Z_indices < Z_atmo.size)
        
        Y_indices = Y_indices[Y_filter*X_filter*Z_filter]-1
        X_indices = X_indices[Y_filter*X_filter*Z_filter]-1
        Z_indices = Z_indices[Y_filter*X_filter*Z_filter]-1

        inds_ray = (Y_indices*Y.shape[1] + X_indices)*Y.shape[2] + Z_indices
        
        #
        # Calculate weights
        #
        uniq_indices, inv_indices = np.unique(inds_ray, return_inverse=True)

        weights = []
        for i, ind in enumerate(uniq_indices):
            weights.append((inv_indices == i).sum())
        
        #
        # Sum up the indices and weights
        #
        data.append(weights)
        indices.append(uniq_indices)
        indptr.append(indptr[-1]+uniq_indices.size)

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


def cartesian2sensor(camera_center, Y, X, Z, img_res, radius_bins, samples_num=1000, dither_noise=10, replicate=4):
    
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
    # The image is assumed the [-1, 1]x[-1, 1] square.
    #
    Y_img, step = np.linspace(-1.0, 1.0, img_res, endpoint=False, retstep=True)
    X_img = np.linspace(-1.0, 1.0, img_res, endpoint=False)
    X_img, Y_img = np.meshgrid(X_img, Y_img)

    #
    # Randomly replicate rays inside each pixel
    #
    X_img = np.tile(X_img[:, :, np.newaxis], [1, 1, replicate])
    Y_img = np.tile(Y_img[:, :, np.newaxis], [1, 1, replicate])
    X_img += np.random.rand(*X_img.shape)*step
    Y_img += np.random.rand(*Y_img.shape)*step
    
    #
    # Calculate rays angles
    # R_img is the radius from the center of the image (0, 0) to the
    # pixel. It is used for calculating th ray direction (PHI, THETA)
    # and for filtering pixels outside the image (radius > 1).
    #
    R_img = np.sqrt(X_img**2 + Y_img**2)
    THETA_ray = R_img * np.pi /2
    PHI_ray = np.arctan2(Y_img, X_img)
    DY_ray = np.sin(THETA_ray) * np.sin(PHI_ray)
    DX_ray = np.sin(THETA_ray) * np.cos(PHI_ray)
    DZ_ray = np.cos(THETA_ray)
    
    #
    # Calculate sample steps along ray
    #
    R_max = np.max(np.sqrt(Y**2 + X**2 + Z**2))
    R_samples, R_step = np.linspace(0.0, R_max, samples_num, retstep=True)
    R_samples = R_samples[1:]
    R_dither = np.random.rand(R_img.size) * R_step * dither_noise
    
    #
    # Calculate radius bins
    #
    R_bins = np.logspace(np.log10(R_samples[0]), np.log10(R_samples[-1]+1), radius_bins+1)-1
    samples_bin = np.digitize(R_samples, R_bins)
    samples_array = []
    for i in range(1, radius_bins+1):
        samples_array.append(R_samples[samples_bin==i].reshape((-1, 1)))
    
    #
    # Loop on all rays
    #
    data = []
    indices = []
    indptr = [0]
    for samples in samples_array:
        for r, dy, dx, dz, r_dither in itertools.izip(
            R_img.reshape((-1, replicate)),
            DY_ray.reshape((-1, replicate)),
            DX_ray.reshape((-1, replicate)),
            DZ_ray.reshape((-1, replicate)),
            R_dither.ravel(),
            ):
    
            if np.all(r > 1):
                indptr.append(indptr[-1])
                continue
            
            #
            # Filter steps where r > 1
            #
            dy = dy[r<=1]
            dx = dx[r<=1]
            dz = dz[r<=1]
        
            #
            # Convert the ray samples to volume indices
            #
            Y_ray = (r_dither+samples) * dy
            X_ray = (r_dither+samples) * dx
            Z_ray = (r_dither+samples) * dz
            
            #
            # Calculate the atmosphere indices
            #
            Y_indices = np.searchsorted(Y_atmo, Y_ray.ravel())
            X_indices = np.searchsorted(X_atmo, X_ray.ravel())
            Z_indices = np.searchsorted(Z_atmo, Z_ray.ravel())
            
            #
            # Calculate unique indices
            #
            Y_filter = (Y_indices > 0) * (Y_indices < Y_atmo.size)
            X_filter = (X_indices > 0) * (X_indices < X_atmo.size)
            Z_filter = (Z_indices > 0) * (Z_indices < Z_atmo.size)
            
            Y_indices = Y_indices[Y_filter*X_filter*Z_filter]-1
            X_indices = X_indices[Y_filter*X_filter*Z_filter]-1
            Z_indices = Z_indices[Y_filter*X_filter*Z_filter]-1
    
            inds_ray = (Y_indices*Y.shape[1] + X_indices)*Y.shape[2] + Z_indices
            
            #
            # Calculate weights
            #
            uniq_indices, inv_indices = np.unique(inds_ray, return_inverse=True)
    
            weights = []
            for i, ind in enumerate(uniq_indices):
                weights.append((inv_indices == i).sum())
            
            #
            # Sum up the indices and weights
            #
            data.append(weights)
            indices.append(uniq_indices)
            indptr.append(indptr[-1]+uniq_indices.size)
    
    #
    # Create sparse matrix
    #
    data = np.hstack(data)
    indices = np.hstack(indices)

    H_int = sps.csr_matrix(
        (data, indices, indptr),
        shape=(img_res*img_res*radius_bins, Y.size)
    )
    
    #
    # Create the polar grid.
    # The image is assumed the [-1, 1]x[-1, 1] square.
    #
    Y_img, step = np.linspace(-1.0, 1.0, img_res, endpoint=False, retstep=True)
    X_img = np.linspace(-1.0, 1.0, img_res, endpoint=False)
    X_img, Y_img = np.meshgrid(X_img, Y_img)
    R_img = np.sqrt(X_img**2 + Y_img**2)
    THETA = R_img * np.pi /2
    PHI = np.arctan2(Y_img, X_img)
    THETA = np.tile(THETA[np.newaxis, :, :], [len(R_bins)-1, 1, 1])
    PHI = np.tile(PHI[np.newaxis, :, :], [len(R_bins)-1, 1, 1])
    R = np.tile(R_bins[:-1].reshape((-1, 1, 1)), [1, THETA.shape[1], THETA.shape[2]])
    
    return H_int, R, PHI, THETA


def polarTransformMatrix(X, Y, center, radius_res=None, angle_res=None):
    """(sparse) matrix representation of cartesian to polar transform.
    params:
        X, Y - 2D arrays that define the cartesian coordinates
        center - Center (in cartesian coords) of the polar coordinates.
        radius_res, angle_res - Resolution of polar coordinates.
 """

    import numpy as np

    if X.ndim == 1:
        X, Y = np.meshgrid(X, Y)

    if radius_res == None:
        radius_res = max(*X.shape)

    if angle_res == None:
        angle_res = radius_res

    #
    # Create the polar grid over which the target matrix (H) will sample.
    #
    max_R = np.max(np.sqrt((X-center[0])**2 + (Y-center[1])**2))
    T, R = np.meshgrid(np.linspace(0, np.pi, angle_res), np.linspace(0, max_R, radius_res))

    #
    # Calculate the indices of the polar grid in the Cartesian grid.
    #
    X_ = R * np.cos(T) + center[0]
    Y_ = R * np.sin(T) + center[1]

    #
    # Calculate the transform
    #
    H = calcTransformMatrix((Y, X), (Y_, X_))

    return H, T, R


def sphericalTransformMatrix(Y, X, Z, center, radius_res=None, phi_res=None, theta_res=None, THETA_portion=0.9):
    """(sparse) matrix representation of cartesian to spherical transform.
    
    Parameters
    ----------
    Y, X, Z : array,
        List of grids. The grids are expected to be of the form created by mgrid
        and in the same order of creation. This implies that the first member has
        its changing dimension as the first dimension the second member should
        have its second dimension changing etc. It also implies that the grid should
        change only in one dimension each.
    
    center : [float, float, float]
        Center (in cartesian coords) of the spherical coordinates.
    
    radius_res, phi_res, theta_res: int, optional (default=None)
        Resolution of spherical coordinates. If None, will use the maximal
        resolution of the cartesian coords.
        
    THETA_portion : [0.0-1.0], optional (default=0.9)
        The theta range will be 0-pi/2 * THETA_portion
        
    Returns
    -------
    H : sparse matrix in CSR format,
        Transform matrix the implements the spherical transform.
        
    R, PHI, THETA : arrays
        List of grids (created using mgrid) represent the Spherical coords.
 """

    import numpy as np

    if radius_res == None:
        radius_res = max(*X.shape)

    if phi_res == None:
        phi_res = radius_res
        theta_res = radius_res

    #
    # Create the polar grid over which the target matrix (H) will sample.
    #
    max_R = np.max(np.sqrt((Y-center[0])**2 + (X-center[1])**2 + (Z-center[2])**2))
    R, PHI, THETA = np.mgrid[0:max_R:complex(0, radius_res), 0:2*np.pi:complex(0, phi_res), 0:np.pi/2*THETA_portion:complex(0, theta_res)]

    #
    # Calculate the indices of the polar grid in the Cartesian grid.
    #
    X_ = R * np.sin(THETA) * np.cos(PHI) + center[0]
    Y_ = R * np.sin(THETA) * np.sin(PHI) + center[1]
    Z_ = R * np.cos(THETA) + center[2]

    #
    # Calculate the transform
    #
    H = calcTransformMatrix((Y, X, Z), (Y_, X_, Z_))

    return H, R, PHI, THETA


def rotationTransformMatrix(X, Y, angle, X_dst=None, Y_dst=None):
    """(sparse) matrix representation of rotation transform.
    params:
        X, Y - 2D arrays that define the cartesian coordinates
        angle - Angle of rotation [radians].
        dst_shape - Shape of the destination matrix (after rotation). Defaults
             to the shape of the full matrix after rotation (no cropping).
        X_rot, Y_rot - grid in the rotated coordinates (optional, calculated if not given). 
"""

    import numpy as np
    
    H_rot = np.array(
        [[np.cos(angle), -np.sin(angle), 0],
         [np.sin(angle), np.cos(angle), 0],
         [0, 0, 1]]
        )

    if X_dst == None:
        X_slim = X[0, :]
        Y_slim = Y[:, 0]
        x0_src = np.floor(np.min(X_slim)).astype(np.int)
        y0_src = np.floor(np.min(Y_slim)).astype(np.int)
        x1_src = np.ceil(np.max(X_slim)).astype(np.int)
        y1_src = np.ceil(np.max(Y_slim)).astype(np.int)
        
        coords = np.hstack((
            np.dot(H_rot, np.array([[x0_src], [y0_src], [1]])),
            np.dot(H_rot, np.array([[x0_src], [y1_src], [1]])),
            np.dot(H_rot, np.array([[x1_src], [y0_src], [1]])),
            np.dot(H_rot, np.array([[x1_src], [y1_src], [1]]))
            ))

        x0_dst, y0_dst, dump = np.floor(np.min(coords, axis=1)).astype(np.int)
        x1_dst, y1_dst, dump = np.ceil(np.max(coords, axis=1)).astype(np.int)

        dxy_dst = min(np.min(np.abs(X_slim[1:]-X_slim[:-1])), np.min(np.abs(Y_slim[1:]-Y_slim[:-1])))
        X_dst, Y_dst = np.meshgrid(
            np.linspace(x0_dst, x1_dst, int((x1_dst-x0_dst)/dxy_dst)+1),
            np.linspace(y0_dst, y1_dst, int((y1_dst-y0_dst)/dxy_dst)+1)
        )

    #
    # Calculate a rotated grid by applying the rotation.
    #
    XY_dst = np.vstack((X_dst.ravel(), Y_dst.ravel(), np.ones(X_dst.size)))
    XY_src_ = np.dot(np.linalg.inv(H_rot), XY_dst)

    X_indices = XY_src_[0, :].reshape(X_dst.shape)
    Y_indices = XY_src_[1, :].reshape(X_dst.shape)

    H = calcTransformMatrix((Y, X), (Y_indices, X_indices))

    return H, X_dst, Y_dst


def rotation3DTransformMatrix(Y, X, Z, rotation, Y_dst=None, X_dst=None, Z_dst=None):
    """Calculate a (sparse) matrix representation of rotation transform in 3D.
    
    Parameters
    ----------
    Y, X, Z : array,
        List of grids. The grids are expected to be of the form created by mgrid
        and in the same order of creation. This implies that the first member has
        its changing dimension as the first dimension the second member should
        have its second dimension changing etc. It also implies that the grid should
        change only in one dimension each.
        
    rotation : list of floats or rotation matrix
        Either a list of floats representating the rotation in Y, X, Z axes.
        The rotations are applied separately in this order. Alternatively, rotation
        can be a 4x4 rotation matrix
    
    Y_dst, X_dst, Z_dst : array, optional (default=None)
        List of grids. The grids are expected to be of the form created by mgrid
        and in the same order of creation. The transform is calculated into these
        grids. This enables croping of the target domain after the rotation transform.
        If none, the destination grids will be calculated to contain the full transformed
        source.
    
    Returns
    -------
    H : sparse matrix in CSR format,
        Transform matrix the implements the rotation transform.
        
    H_rot : array [4x4]
        The rotation transform as calculated from the input rotation parameter.
        
    Y_dst, X_dst, Z_dst : array,
        Target grid. Either the input Y_dst, X_dst, Z_dst or the calculated grid.
"""

    import numpy as np

    if isinstance(rotation, np.ndarray) and rotation.shape == (4, 4):
        H_rot = rotation
    else:
        H_rot = calcRotationMatrix(rotation)
        
    if X_dst == None:
        Y_dst, X_dst, Z_dst = _calcRotateGrid(Y, X, Z, H_rot)

    #
    # Calculate a rotated grid by applying the rotation.
    #
    XYZ_dst = np.vstack((X_dst.ravel(), Y_dst.ravel(), Z_dst.ravel(), np.ones(X_dst.size)))
    XYZ_src_ = np.dot(np.linalg.inv(H_rot), XYZ_dst)

    Y_indices = XYZ_src_[1, :].reshape(X_dst.shape)
    X_indices = XYZ_src_[0, :].reshape(X_dst.shape)
    Z_indices = XYZ_src_[2, :].reshape(X_dst.shape)

    H = calcTransformMatrix((Y, X, Z), (Y_indices, X_indices, Z_indices))

    return H, H_rot, Y_dst, X_dst, Z_dst


def calcRotationMatrix(rotation):
    
    import numpy as np
    
    #
    # Calculate the rotation transform
    #
    theta, phi, psi = rotation

    H_rotx = np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(theta), -np.sin(theta), 0],
            [0, np.sin(theta), np.cos(theta), 0],
            [0, 0, 0, 1]]
        )

    H_roty = np.array(
        [
            [np.cos(phi), 0, np.sin(phi), 0],
            [0, 1, 0, 0],
            [-np.sin(phi), 0, np.cos(phi), 0],
            [0, 0, 0, 1]]
        )

    H_rotz = np.array(
        [
            [np.cos(psi), -np.sin(psi), 0, 0],
            [np.sin(psi), np.cos(psi), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]
        )

    H_rot = np.dot(H_rotz, np.dot(H_roty, H_rotx))
    
    return H_rot


def _calcRotateGrid(Y, X, Z, H_rot):
    #
    # Calculate the target grid.
    # The calculation is based on calculating the minimal grid that contains
    # the transformed input grid.
    #
    Y_slim = Y[:, 0, 0]
    X_slim = X[0, :, 0]
    Z_slim = Z[0, 0, :]
    x0_src = np.floor(np.min(X_slim)).astype(np.int)
    y0_src = np.floor(np.min(Y_slim)).astype(np.int)
    z0_src = np.floor(np.min(Z_slim)).astype(np.int)
    x1_src = np.ceil(np.max(X_slim)).astype(np.int)
    y1_src = np.ceil(np.max(Y_slim)).astype(np.int)
    z1_src = np.ceil(np.max(Z_slim)).astype(np.int)

    src_coords = np.array(
        [
            [x0_src, x0_src, x1_src, x1_src, x0_src, x0_src, x1_src, x1_src],
            [y0_src, y1_src, y0_src, y1_src, y0_src, y1_src, y0_src, y1_src],
            [z0_src, z0_src, z0_src, z0_src, z1_src, z1_src, z1_src, z1_src],
            [1, 1, 1, 1, 1, 1, 1, 1]
        ]
    )
    dst_coords = np.dot(H_rot, src_coords)

    
    x0_dst, y0_dst, z0_dst, dump = np.floor(np.min(dst_coords, axis=1)).astype(np.int)
    x1_dst, y1_dst, z1_dst, dump = np.ceil(np.max(dst_coords, axis=1)).astype(np.int)

    #
    # Calculate the grid density.
    # Note:
    # This calculation is important as having a dense grid results in a huge transform
    # matrix even if it is sparse.
    #
    dy = Y_slim[1] - Y_slim[0]
    dx = X_slim[1] - X_slim[0]
    dz = Z_slim[1] - Z_slim[0]

    delta_src_coords = np.array(
        [
            [0, dx, 0, 0, -dx, 0, 0],
            [0, 0, dy, 0, 0, -dy, 0],
            [0, 0, 0, dz, 0, 0, -dz],
            [1, 1, 1, 1, 1, 1, 1]
        ]
    )
    delta_dst_coords = np.dot(H_rot, delta_src_coords)
    delta_dst_coords.sort(axis=1)
    delta_dst_coords = delta_dst_coords[:, 1:] - delta_dst_coords[:, :-1]
    delta_dst_coords[delta_dst_coords<=0] = 10000000
    
    dx, dy, dz, dump = np.min(delta_dst_coords, axis=1)
    x_samples = min(int((x1_dst-x0_dst)/dx), GRID_DIM_LIMIT)
    y_samples = min(int((y1_dst-y0_dst)/dy), GRID_DIM_LIMIT)
    z_samples = min(int((z1_dst-z0_dst)/dz), GRID_DIM_LIMIT)
    
    dim_ratio = x_samples * y_samples * z_samples / SPARSE_SIZE_LIMIT
    if  dim_ratio > 1:
        dim_reduction = dim_ratio ** (-1/3)
        
        x_samples = int(x_samples * dim_reduction)
        y_samples = int(y_samples * dim_reduction)
        z_samples = int(z_samples * dim_reduction)
        
    Y_dst, X_dst, Z_dst = np.mgrid[
        y0_dst:y1_dst:complex(0, y_samples),
        x0_dst:x1_dst:complex(0, x_samples),
        z0_dst:z1_dst:complex(0, z_samples),
    ]
    return Y_dst, X_dst, Z_dst


def gridDerivatives(grids, forward=True):
    """
    Calculate first order partial derivatives for a list of grids.
    
    Parameters
    ----------
    grids : list,
        List of grids. The grids are expected to be of the form created by mgrid
        and in the same order of creation. This implies that the first member has
        its changing dimension as the first dimension the second member should
        have its second dimension changing etc. It also implies that the grid should
        change only in one dimension each.
        
    forward : boolean, optional (default=True)
        Forward or backward derivatives.
        
    Returns
    -------
    derivatives : list,
        List of the corresponding derivatives, as 1D arrays.
    """

    import numpy as np
    
    derivatives = []
    for dim, grid in enumerate(grids):
        sli = [0] * len(grid.shape)
        sli[dim] = Ellipsis
        grid = grid[sli]
        derivative = np.abs(grid[1:] - grid[:-1])
        if forward:
            derivative = np.concatenate((derivative, (derivative[-1],)))
        else:
            derivative = np.concatenate(((derivative[0],), derivative))
        derivatives.append(derivative)

    return derivatives
    

def cumsumTransformMatrix(grids, axis=0, direction=1, masked_rows=None):
    """
    Calculate a (sparse) matrix representation of integration (cumsum) transform.
    
    Parameters
    ----------
    grids : list,
        List of grids. The length of grids should correspond to the
        dimensions of the grid.
        
    axis : int, optional (default=0)
        Axis along which the cumsum operation is preformed.
    
    direction : {1, -1}, optional (default=1)
        Direction of integration, 1 for integrating up the indices
        -1 for integrating down the indices.
       
    masked_rows: array, optional(default=None)
        If not None, leave only the rows that are non zero in the
        masked_rows array.
    
    Returns
    -------
    H : sparse matrix in CSR format,
        Transform matrix the implements the cumsum transform.
"""
        
    import numpy as np
    import scipy.sparse as sps

    grid_shape = grids[0].shape
    strides = np.array(grids[0].strides).reshape((-1, 1))
    strides /= strides[-1]

    derivatives = gridDerivatives(grids)

    inner_stride = strides[axis]
    if direction == 1:
        inner_stride = -inner_stride
        
    inner_size = np.prod(grid_shape[axis:])

    inner_H = sps.spdiags(
        np.ones((grid_shape[axis], inner_size))*derivatives[axis].reshape((-1, 1)),
        inner_stride*np.arange(grid_shape[axis]),
        inner_size,
        inner_size)
    
    if axis == 0:
        H = inner_H
    else:
        m = np.prod(grid_shape[:axis])
        H = sps.kron(sps.eye(m, m), inner_H)

    if masked_rows != None:
        H = H.tolil()
        indices = masked_rows.ravel() == 0
        for i in indices.nonzero()[0]:
            H.rows[i] = []
            H.data[i] = []
    
    return H.tocsr()


def integralTransformMatrix(grids, jacobian=None, axis=0, direction=1):
    """
    Calculate a (sparse) matrix representation of an integration transform.
    
    Parameters
    ----------
    grids : list
        List of grids. The grids are expected to be of the form created by mgrid
        and in the same order of creation.
    
    axis : int, optional (default=0)
        The axis by which the integration is performed.
        
    direction : {1, -1}, optional (default=1)
        Direction of integration
        direction - 1: integrate up the indices, -1: integrate down the indices.
        
    Returns
    -------
    H : sparse matrix
        Sparse matrix, in csr format, representing the transform.
"""

    import numpy as np
    import scipy.sparse as sps

    grid_shape = grids[0].shape
    strides = np.array(grids[0].strides)
    strides /= strides[-1]

    derivatives = gridDerivatives(grids)

    inner_stride = strides[axis]
    
    if direction != 1:
        direction  = -1
        
    inner_height = np.abs(inner_stride)
    inner_width = np.prod(grid_shape[axis:])

    inner_H = sps.spdiags(
        np.ones((grid_shape[axis], max(inner_height, inner_width)))*derivatives[axis].reshape((-1, 1))*direction,
        inner_stride*np.arange(grid_shape[axis]),
        inner_height,
        inner_width
    )
    
    if axis == 0:
        H = inner_H
    else:
        m = np.prod(grid_shape[:axis])
        H = sps.kron(sps.eye(m, m), inner_H)

    H = H.tocsr()
    
    if jacobian != None:
        H = H * spdiag(jacobian)
        
    return H.tocsr()


if __name__ == '__main__':
    
    test_cartesian2radial()