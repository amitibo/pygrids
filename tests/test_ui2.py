""" Example showing a dialog with multiple embedded scenes.

When using several embedded scenes with mlab, you should be very careful
always to pass the scene you want to use for plotting to the mlab
function used, elsewhere it uses the current scene. In this example,
failing to do so would result in only one scene being used, the last
one created.

The trick is to use the 'mayavi_scene' attribute of the MlabSceneModel,
and pass it as a keyword argument to the mlab functions.

For more examples on embedding mlab scenes in dialog, see also:
the examples :ref:`example_mlab_interactive_dialog`, and
:ref:`example_lorenz_ui`, as well as the section of the user manual
:ref:`embedding_mayavi_traits`.
"""
import numpy as np

from traits.api import HasTraits, Instance, Button, \
    on_trait_change, Range, Tuple, Array
from traitsui.api import View, Item, VSplit, HSplit

from mayavi import mlab
from mayavi.core.ui.api import MlabSceneModel, SceneEditor

import scipy.io as sio
sensor_res = 128


class MyDialog(HasTraits):

    scene1 = Instance(MlabSceneModel, ())
    scene2 = Instance(MlabSceneModel, ())

    # The parameters for the Lorenz system, defaults to the standard ones.
    x = Range(0, 50., 25, desc='pixel coord x', enter_set=True,
              auto_set=False)
    y = Range(0, 50., 25, desc='pixel coord y', enter_set=True,
              auto_set=False)

    z = Range(0, 10., 5, desc='pixel coord z', enter_set=True,
              auto_set=False)

    r = Range(1, 5., 1, desc='radius of ball', enter_set=True, auto_set=False)
    
    # Tuple of x, y, z arrays where the field is sampled.
    points = Tuple(Array, Array, Array)

    # The layout of the dialog created
    view = View(
        HSplit(
            VSplit(
                Item('scene1',
                     editor=SceneEditor(), height=250,
                     width=300),
                'x',
                'y',
                'z',
                'r',
                show_labels=False,
              ),
            VSplit(
                 Item('scene2',
                      editor=SceneEditor(), height=250,
                      width=300),
                 show_labels=False,
               ),
            ),
        resizable=True,
    )

    def __init__(self):
        # Do not forget to call the parent's __init__
        HasTraits.__init__(self)
        
        data = sio.loadmat('img_max50')
        self.H = data['H1']
        
    @on_trait_change('scene1.activated')
    def create_scene(self):
        mlab.clf(figure=self.scene1.mayavi_scene)
        y, x, z = self.points
        r = (y-self.y)**2 + (x-self.x)**2 + (z-self.z)**2
        b = np.ones_like(y)
        b[r>self.r**2] = 0
        
        self.src = mlab.pipeline.scalar_field(y, x, z, b, figure=self.scene1.mayavi_scene)
        ipw_x = mlab.pipeline.image_plane_widget(self.src, plane_orientation='x_axes')
        ipw_y = mlab.pipeline.image_plane_widget(self.src, plane_orientation='y_axes')
        ipw_z = mlab.pipeline.image_plane_widget(self.src, plane_orientation='z_axes')
        mlab.colorbar()
        mlab.axes()
        
    @on_trait_change('x, y, z, r')
    def update_volume(self):
        y, x, z = self.points
        r = (y-self.y)**2 + (x-self.x)**2 + (z-self.z)**2
        b = np.ones_like(y)
        b[r>self.r**2] = 0
        self.src.mlab_source.scalars = b
        
        img = (self.H * b.reshape((-1, 1))).reshape((sensor_res, sensor_res))
        mlab.clf(figure=self.scene2.mayavi_scene)
        mlab.imshow(img, colormap='gray', figure=self.scene2.mayavi_scene)
        mlab.colorbar()
        
    def _points_default(self):
        y, x, z = np.mgrid[0:50:1., 0:50:1., 0:10:0.1]
        return y, x, z


m = MyDialog()
m.configure_traits()
