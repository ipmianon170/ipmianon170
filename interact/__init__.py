# suppress tensorflow logs
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)


from . import model
from . import utility

from .utility import HyperMorphInteractiveWindow
