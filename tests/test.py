"""
"""

from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import grids
import amitibo
import time
import scipy.io as sio
import amitibo


atmosphere_params = amitibo.attrClass(
    cartesian_grids=(
        slice(0, 50, 1.0), # Y
        slice(0, 50, 1.0), # X
        slice(0, 10, 0.1)   # H
        ),
    earth_radius=4000,
    air_typical_h=8,
    aerosols_typical_h=1.2
)

camera_params = amitibo.attrClass(
    image_res=128,
    subgrid_res=(50, 50, 10),
    grid_noise=0.01
)

camera_position = np.array((25., 25., 0.)) + 0.1*np.random.rand(3)

phi = 0
theta = -np.pi/4
Y, X, H = np.mgrid[atmosphere_params.cartesian_grids]


def point():
    """Main doc """
    
    t0 = time.time()
    
    H_dist = grids.point2grids(camera_center, Y, X, H)
    
    print time.time() - t0

    #x = np.ones(Y.size).reshape((-1, 1))
    x = np.exp(-H/10)
    x[H<100] = 0
    y = H_dist * x.reshape((-1, 1))

    amitibo.viz3D(Y, X, H, x.reshape(Y.shape), interpolation='nearest_neighbour')
    amitibo.viz3D(Y, X, H, y.reshape(Y.shape), interpolation='nearest_neighbour')
    mlab.show()
    

def direction():
    """Main doc """
    
    t0 = time.time()
    
    H_dist = grids.direction2grids(phi, theta, Y, X, H)

    print time.time() - t0

    x = (Y<150).reshape((-1, 1)).astype(np.float)
    #x = np.ones(Y.size).reshape((-1, 1))
    y = H_dist * x
    
    amitibo.viz3D(Y, X, H, y.reshape(Y.shape))
    mlab.show()
    

def integrate():
    
    """Compare two different resolutions"""
    
    H_int1 = grids.integrateGrids(
        camera_position,
        Y,
        X,
        H,
        camera_params.image_res,
        subgrid_max=camera_params.subgrid_res,
        subgrid_noise=camera_params.grid_noise
    )
    
    x = np.ones(Y.shape)
    y1 = H_int1 * x.reshape((-1, 1))

    sio.savemat(
        'img.mat',
        {
            'y1': y1.reshape((camera_params.image_res, camera_params.image_res)),
            'H1': H_int1,
        }
    )
    
    plt.gray()
    plt.imshow(y1.reshape((camera_params.image_res, camera_params.image_res)))
    plt.colorbar()
    plt.show()
          

def test2D():
    import matplotlib.pyplot as plt
    import numpy as np
    import scipy.misc as sm
    import time

    ##############################################################
    # 2D data
    ##############################################################
    lena = sm.lena()
    lena = lena[:256, ...]
    lena_ = lena.reshape((-1, 1))    
    X, Y = np.meshgrid(np.arange(lena.shape[1]), np.arange(lena.shape[0]))

    #
    # Polar transform
    #
    t0 = time.time()
    Hpol = polarTransformMatrix(X, Y, (256, 2))[0]
    lena_pol = Hpol * lena_
    print time.time() - t0
    
    plt.figure()
    plt.imshow(lena_pol.reshape((512, 512)), interpolation='nearest')

    #
    # Rotation transform
    #
    Hrot1, X_rot, Y_rot = rotationTransformMatrix(X, Y, angle=-np.pi/3)
    Hrot2 = rotationTransformMatrix(X_rot, Y_rot, np.pi/3, X, Y)[0]
    lena_rot1 = Hrot1 * lena_
    lena_rot2 = Hrot2 * lena_rot1

    plt.figure()
    plt.subplot(121)
    plt.imshow(lena_rot1.reshape(X_rot.shape))
    plt.subplot(122)
    plt.imshow(lena_rot2.reshape(lena.shape))

    #
    # Cumsum transform
    #
    Hcs1 = cumsumTransformMatrix((Y, X), axis=0, direction=1)
    Hcs2 = cumsumTransformMatrix((Y, X), axis=1, direction=1)
    Hcs3 = cumsumTransformMatrix((Y, X), axis=0, direction=-1)
    Hcs4 = cumsumTransformMatrix((Y, X), axis=1, direction=-1)
    lena_cs1 = Hcs1 * lena_
    lena_cs2 = Hcs2 * lena_
    lena_cs3 = Hcs3 * lena_
    lena_cs4 = Hcs4 * lena_

    plt.figure()
    plt.subplot(221)
    plt.imshow(lena_cs1.reshape(lena.shape))
    plt.subplot(222)
    plt.imshow(lena_cs2.reshape(lena.shape))
    plt.subplot(223)
    plt.imshow(lena_cs3.reshape(lena.shape))
    plt.subplot(224)
    plt.imshow(lena_cs4.reshape(lena.shape))

    plt.show()
    
    # t0 = time.time()
    # Hpol = polarTransformMatrix(X, Y, (256, 2))[0]
    # t2 = time.time() - t0
    # print 'first calculation: %g, memoized: %g' % (t1, t2)


def test3D():
    import mayavi.mlab as mlab
    import time

    #
    # Test several of the above functions
    #
    ##############################################################
    # 3D data
    ##############################################################
    Y, X, Z = np.mgrid[-10:10:50j, -10:10:50j, -10:10:50j]
    V = np.sqrt(Y**2 + X**2 + Z**2)
    V_ = V.reshape((-1, 1))
    
    #
    # Spherical transform
    #
    t0 = time.time()
    Hsph, R, PHI, THETA = sphericalTransformMatrix(Y, X, Z, (0, 0, 0))
    Vsph = Hsph * V_
    print time.time() - t0
     
    #
    # Rotation transform
    #
    t0 = time.time()
    Hrot, rotation, Y_rot, X_rot, Z_rot = rotation3DTransformMatrix(Y, X, Z, (np.pi/4, np.pi/4, 0))
    Vrot = Hrot * V_
    Hrot2 = rotation3DTransformMatrix(Y_rot, X_rot, Z_rot, np.linalg.inv(rotation), Y, X, Z)[0]
    Vrot2 = Hrot2 * Vrot
    print time.time() - t0
     
    # #
    # # Cumsum transform
    # #
    # Hcs1 = cumsumTransformMatrix((Y, X, Z), axis=0, direction=-1)
    # Vcs1 = Hcs1 * V_

    # #
    # # Integral transform
    # #
    # Hit1 = integralTransformMatrix((Y, X, Z), axis=0, direction=-1)
    # Vit1 = Hit1 * V_

    #
    # 3D visualization
    #
    viz3D(Y, X, Z, V, title='V Rotated')
    
    viz3D(Y_rot, X_rot, Z_rot, Vrot.reshape(Y_rot.shape), title='V Rotated')
    
    viz3D(Y, X, Z, Vrot2.reshape(Y.shape), title='V Rotated Back')

    viz3D(R, PHI, THETA, Vsph.reshape(R.shape), title='V Spherical')
    
    # mlab.figure()
    # mlab.contour3d(Vcs1.reshape(V.shape), contours=[1, 2, 3], transparent=True)
    # mlab.outline()
    
    mlab.show()
    
    #
    # 2D visualization
    #
    # import matplotlib.pyplot as plt
    
    # plt.figure()
    # plt.imshow(Vit1.reshape(V.shape[:2]))
    # plt.show()


def testProjection():

    import scipy.misc as scm
    import matplotlib.pyplot as plt
    
    l = scm.lena()

    PHI, THETA = np.mgrid[0:2*np.pi:512j, 0:np.pi/2*0.9:512j]
    
    H = cameraTransformMatrix(PHI, THETA, focal_ratio=0.15)
    lp = H * l.reshape((-1, 1))

    plt.figure()
    plt.imshow(l)
    
    plt.figure()
    plt.imshow(lp.reshape((256, 256)))

    plt.show()
    

def test_rayCasting():
    import matplotlib.pyplot as plt

    Y, X, Z = np.mgrid[0:50000:1000, 0:50000:1000, 0:10000:100]
    img_res = 128
    camera_center = (25050, 25050, 50)
    samples_num = 1000
    
    atmo = np.zeros_like(Y)
    R = (Y-50000/3)**2/16 + (X-50000*2/3)**2/16 + (Z-10000/4)**2*8
    atmo[R<4000**2] = 1

    H_int = rayCasting(camera_center, Y, X, Z, img_res, samples_num)

    img = (H_int * atmo.reshape((-1, 1))).reshape((img_res, img_res))
    
    plt.imshow(img)
    plt.show()

    
def test_cartesian2radial():
    Y, X, Z = np.mgrid[0:50000:1000, 0:50000:1000, 0:10000:100]
    img_res = 128
    radius_bins = 20
    camera_center = (25050, 25050, 50)
    samples_num = 2000
    
    atmo = np.zeros_like(Y)
    R = (Y-50000/3)**2/16 + (X-50000*2/3)**2/16 + (Z-10000/4)**2*8
    atmo[R<4000**2] = 1

    H_int = cartesian2radial(camera_center, Y, X, Z, img_res, radius_bins, samples_num)

    atmo_radial = (H_int * atmo.reshape((-1, 1))).reshape((img_res, img_res, radius_bins))
    
    import amitibo
    import mayavi.mlab as mlab
    
    Y_rad, X_rad, Z_rad = np.mgrid[0:img_res:1, 0:img_res:1, 0:radius_bins:1]
    
    amitibo.viz3D(Y, X, Z, atmo)
    amitibo.viz3D(Y_rad, X_rad, Z_rad, atmo_radial)
    
    mlab.show()

    
if __name__ == '__main__':
    #point()
    #direction()
    integrate()

      