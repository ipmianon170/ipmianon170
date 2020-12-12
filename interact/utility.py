import pystrum
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
import voxelmorph as vxm
import pkg_resources

from ipywidgets import *

from .model import HyperMorphModel


class HyperMorphInteractiveWindow:
    """
    Interactive exploration of hypermorph model.
    
    Wraps creating an interactive figure and running the forward model given 
    a user hyperparameter slider input
    """

    def __init__(self, init_h=0.5):
        """
        given two models (hypermorph and warper)
        """
        self.shape = (160, 192)
        self.warp_model = vxm.networks.Transform(inshape=self.shape)
        self.moving = None
        self.moving_k = None
        self.fixed = None
        self.fixed_k = None
        self.bw_grid = None
        self.bw_grid_k = None

        # load images
        filename = pkg_resources.resource_filename('interact', 'images.npy')
        self.image_data = np.load(filename)

        # set image pairs
        self.set_image_pairs(0, 1)

        # load hypermorph weights
        self.load_hypermorph_model()

        # initialize figure
        self.init_figure(init_h)

    def set_image_pairs(self, mvg, fxd):
        """
        sets images to register
        """
        self.moving = self.image_data[mvg]
        self.moving_k = self.moving[np.newaxis, ..., np.newaxis]
        self.fixed = self.image_data[fxd]
        self.fixed_k = self.fixed[np.newaxis, ..., np.newaxis]
        self.bw_grid = pystrum.pynd.ndutils.bw_grid(self.shape, 7)
        self.bw_grid_k = self.bw_grid[np.newaxis, ..., np.newaxis]

    def load_hypermorph_model(self):
        """
        loads hypermorph model from h5 weights
        """
        filename = pkg_resources.resource_filename('interact', 'model_mse.h5')
        self.hypermorph_model = HyperMorphModel.load(filename).get_registration_model()

    def update(self, h=0.5):
        """
        interactive update based on hyperparam value
        """
        moved, warped_grid = self.hyperpred(h)
        self.im_ax.set_array(moved)
        self.gr_ax.set_array(warped_grid)
        self.fig.canvas.draw_idle()

    def hyperpred(self, hyperparam):
        """
        move image with hypermodel
        """
        # compute 
        h = np.array(hyperparam).reshape((1,1))
        moved, warp = self.hypermorph_model.predict([self.moving_k, self.fixed_k, h])
        mse = np.mean(np.square([self.fixed - moved.squeeze()]))
        warped_grid = self.warp_model.predict([self.bw_grid_k, warp])
        return moved.squeeze(), warped_grid.squeeze()

    def init_figure(self, init_h):
        # prepare figure
        self.fig = plt.figure(figsize=(12,4), facecolor=(1, 1, 1))
        plt.style.use('grayscale')

        ax = self.fig.add_subplot(1, 4, 1)
        ax.imshow(self.moving);
        ax.set_title('moving');
        ax.set_axis_off();

        ax = self.fig.add_subplot(1, 4, 2)
        ax.imshow(self.fixed);
        ax.set_title('fixed');
        ax.set_axis_off();

        moved, warped_grid = self.hyperpred(init_h)
        ax = self.fig.add_subplot(1, 4, 3)
        self.im_ax = ax.imshow(moved);
        ax.set_title('moved');
        ax.set_axis_off();

        ax = self.fig.add_subplot(1, 4, 4)
        self.gr_ax = ax.imshow(warped_grid);
        ax.set_title('warp');
        ax.set_axis_off();

        plt.tight_layout()
