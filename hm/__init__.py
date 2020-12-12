# suppress tensorflow logs
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from . import model
from . import utility

from .utility import HyperMorphInteractiveWindow
