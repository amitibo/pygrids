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
from traitsui.api import View, Item, VSplit

from mayavi import mlab
from mayavi.core.ui.api import MlabSceneModel, SceneEditor

import scipy.io as sio
sensor_res = 32


class MyDialog(HasTraits):

    scene = Instance(MlabSceneModel, ())

    # The parameters for the Lorenz system, defaults to the standard ones.
    x = Range(0, 32, 16, desc='pixel coord x', enter_set=True,
              auto_set=False)
    y = Range(0, 32, 16, desc='pixel coord y', enter_set=True,
              auto_set=False)

    # Tuple of x, y, z arrays where the field is sampled.
    points = Tuple(Array, Array, Array)

    # The layout of the dialog created
    view = View(
        VSplit(
            Item('scene',
                 editor=SceneEditor(), height=250,
                 width=300),
            'x',
            'y',
            show_labels=False,
          ),
        resizable=True,
    )

    def __init__(self):
        # Do not forget to call the parent's __init__
        HasTraits.__init__(self)
        
        data = sio.loadmat('img4_2')
        self.H = data['H3'].T
        
    @on_trait_change('scene.activated')
    def update_cone(self):
        mlab.clf(figure=self.scene.mayavi_scene)
        y, x, z = self.points
        img = np.zeros((sensor_res, sensor_res))
        img[self.y, self.x] = 1
        cone = self.H * img.reshape((-1, 1))
        self.src = mlab.pipeline.scalar_field(y, x, z, cone.reshape(y.shape), figure=self.scene.mayavi_scene)
        ipw_x = mlab.pipeline.image_plane_widget(self.src, plane_orientation='x_axes', figure=self.scene.mayavi_scene)
        ipw_y = mlab.pipeline.image_plane_widget(self.src, plane_orientation='y_axes', figure=self.scene.mayavi_scene)
        ipw_z = mlab.pipeline.image_plane_widget(self.src, plane_orientation='z_axes', figure=self.scene.mayavi_scene)
        mlab.colorbar()
        mlab.axes()
        
    @on_trait_change('x, y')
    def update_img(self):
        y, x, z = self.points
        img = np.zeros((sensor_res, sensor_res))
        img[self.y, self.x] = 1
        cone = self.H * img.reshape((-1, 1))
        self.src.mlab_source.scalars = cone.reshape(y.shape)
        
    def _points_default(self):
        y, x, z = np.mgrid[0:10:0.1, 0:10:0.1, 0:10:0.1]
        return y, x, z


m = MyDialog()
m.configure_traits()
