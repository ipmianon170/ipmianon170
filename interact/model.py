import numpy as np
import tensorflow as tf
import voxelmorph as vxm
import neurite as ne

import tensorflow.keras.layers as KL
import tensorflow.keras.initializers as KI

from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.keras import backend as K

from voxelmorph.tf.networks import LoadableModel, store_config_args



class ConvFromTensor(KL.Layer):
    """
    Decoupled convolutional layer with input tensors replacing the kernel and bias weights.
    """

    def __init__(self, rank, filters, kernel_size=3, strides=1, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.rank = rank
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, self.rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, self.rank, 'strides')
        self.padding = conv_utils.normalize_padding('same')
        self.data_format = conv_utils.normalize_data_format(None)
        self.dilation_rate = conv_utils.normalize_tuple(1, self.rank, 'dilation_rate')

    def build(self, input_shape):
        img_input_shape = tensor_shape.TensorShape([1] + input_shape[0].as_list()[1:])
        input_channels = int(img_input_shape[-1])
        kernel_shape = tensor_shape.TensorShape(self.kernel_size + (input_channels, self.filters))
        self._convolution_op = nn_ops.Convolution(
            img_input_shape,
            filter_shape=kernel_shape,
            dilation_rate=self.dilation_rate,
            strides=self.strides,
            padding='SAME',
            data_format=conv_utils.convert_data_format(self.data_format, self.rank + 2)
        )
        self.built = True

    def call(self, inputs):
        return tf.map_fn(self._single_batch_call, inputs, dtype=tf.float32)

    def _single_batch_call(self, inputs):
        x = tf.expand_dims(inputs[0], axis=0)
        filter_weights = inputs[1]
        bias_weights = inputs[2]
        outputs = self._convolution_op(x, filter_weights)
        outputs = nn.bias_add(outputs, bias_weights, data_format='NHWC')
        outputs = outputs[0, ...]
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = input_shape[0]
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i],
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.dilation_rate[i]
            )
            new_space.append(new_dim)
        return tensor_shape.TensorShape([input_shape[0]] + new_space + [self.filters])


class Conv3DFromTensor(ConvFromTensor):
    def __init__(self, *args, **kwargs):
        super().__init__(3, *args, **kwargs)


class Conv2DFromTensor(ConvFromTensor):
    def __init__(self, *args, **kwargs):
        super().__init__(2, *args, **kwargs)



def build_conv_block(x, nfeat, hyp, kernel_size=3, strides=1, name=None, activate=True, kernel_initializer='he_normal'):
    """
    Builds a HyperMorph convolutional block.
    """

    ndims = len(x.get_shape()) - 2

    def build_conv_tensor(shape, name):
        y = KL.Dense(np.prod(shape), name='%s_dense' % name)(hyp)
        y = KL.Reshape(shape, name='%s_reshape' % name)(y)
        y = KL.Activation('tanh', name='%s_activation' % name)(y)
        return y

    input_features = x.shape.as_list()[-1]
    kernel_shape = tuple([kernel_size] * ndims)

    weight = build_conv_tensor((*kernel_shape, input_features, nfeat), name + '_weight')
    bias = build_conv_tensor((nfeat,), name + '_bias')

    Conv = globals()['Conv%dDFromTensor' % ndims]
    y = Conv(nfeat, kernel_size=kernel_size, strides=strides, name=name)([x, weight, bias])

    if activate:
        name = name + '_activation' if name else None
        y = KL.LeakyReLU(0.2, name=name)(y)

    return y


def default_unet_features():
    """
    Default unet features.
    """
    nb_features = [
        [16, 32, 32, 32],
        [32, 32, 32, 32, 32, 16, 16]
    ]
    return nb_features


def build_unet(
    input_tensor,
    hyp_tensor=None,
    nb_features=None,
    nb_levels=None,
    max_pool=2,
    feat_mult=1,
    nb_conv_per_level=1,
    do_res=False,
    half_res=False,
    name='unet'):
    """
    Builds a HyperMorph unet.
    """

    if nb_features is None:
        nb_features = default_unet_features()

    # build feature list automatically
    if isinstance(nb_features, int):
        if nb_levels is None:
            raise ValueError('must provide unet nb_levels if nb_features is an integer')
        feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
        nb_features = [
            np.repeat(feats[:-1], nb_conv_per_level),
            np.repeat(np.flip(feats), nb_conv_per_level)
        ]
    elif nb_levels is not None:
        raise ValueError('cannot use nb_levels if nb_features is not an integer')

    ndims = len(input_tensor.get_shape()) - 2
    assert ndims in (1, 2, 3), 'ndims should be one of 1, 2, or 3. found: %d' % ndims
    MaxPooling = getattr(KL, 'MaxPooling%dD' % ndims)

    # extract any surplus (full resolution) decoder convolutions
    enc_nf, dec_nf = nb_features
    nb_dec_convs = len(enc_nf)
    final_convs = dec_nf[nb_dec_convs:]
    dec_nf = dec_nf[:nb_dec_convs]
    nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

    if isinstance(max_pool, int):
        max_pool = [max_pool] * nb_levels

    # configure encoder (down-sampling path)
    enc_layers = []
    last = input_tensor
    for level in range(nb_levels - 1):
        for conv in range(nb_conv_per_level):
            nf = enc_nf[level * nb_conv_per_level + conv]
            layer_name = '%s_enc_conv_%d_%d' % (name, level, conv)
            last = build_conv_block(last, nf, hyp_tensor, name=layer_name)
        enc_layers.append(last)
        
        # temporarily use maxpool since downsampling doesn't exist in keras
        last = MaxPooling(max_pool[level], name='%s_enc_pooling_%d' % (name, level))(last)

    # configure decoder (up-sampling path)
    for level in range(nb_levels - 1):
        real_level = nb_levels - level - 2
        for conv in range(nb_conv_per_level):
            nf = dec_nf[level * nb_conv_per_level + conv]
            layer_name = '%s_dec_conv_%d_%d' % (name, real_level, conv)
            last = build_conv_block(last, nf, hyp_tensor, name=layer_name)
        if not half_res or level < (nb_levels - 2):
            layer_name = '%s_dec_upsample_%d' % (name, real_level)
            last = vxm.tf.networks._upsample_block(last, enc_layers.pop(), factor=max_pool[real_level], name=layer_name)

    # now we take care of any remaining convolutions
    for num, nf in enumerate(final_convs):
        layer_name = '%s_dec_final_conv_%d' % (name, num)
        last = build_conv_block(last, nf, hyp_tensor, name=layer_name)

    return last


class HyperMorphModel(LoadableModel):
    """
    HyperMorph model that extends the standard VxmDense model from the VoxelMorph library.
    """

    @store_config_args
    def __init__(self,
        inshape,
        nb_unet_features=None,
        nb_unet_levels=None,
        unet_feat_mult=1,
        nb_unet_conv_per_level=1,
        int_steps=7,
        int_downsize=2,
        src_feats=1,
        trg_feats=1,
        unet_half_res=True,
        hyp_model=True,
        nb_hyp=1,
        hyp_layers=2,
        hyp_units=32,
        hyp_activation='relu',
        name='vxm_dense',
        set_hyp_as_weight=False):

        # make inputs
        ndims = len(inshape)
        source = tf.keras.Input(shape=(*inshape, src_feats), name='%s_source_input' % name)
        target = tf.keras.Input(shape=(*inshape, trg_feats), name='%s_target_input' % name)
        unet_input = KL.concatenate([source, target], name='%s_input_concat' % name)

        # make hypernetwork
        hyp_input = tf.keras.Input(shape=[nb_hyp], name='%s_hyp_input' % name)
        hyp_last_dense = hyp_input
        for n in range(hyp_layers):
            hyp_last_dense = KL.Dense(hyp_units, activation=hyp_activation, name='%s_hyp_dense_%d' % (name, n + 1))(hyp_last_dense)

        # build core unet model and grab inputs
        unet_output = build_unet(
            input_tensor=unet_input,
            hyp_tensor=hyp_last_dense,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            nb_conv_per_level=nb_unet_conv_per_level,
            half_res=unet_half_res,
            name='%s_unet' % name
        )

        # transform unet output into a flow field
        flow = build_conv_block(
            unet_output,
            ndims,
            hyp_last_dense,
            activate=False,
            kernel_initializer=KI.RandomNormal(mean=0.0, stddev=1e-5),
            name='%s_flow' % name
        )

        if not unet_half_res:
            # optionally resize for integration
            if int_steps > 0 and int_downsize > 1:
                flow = vxm.layers.RescaleTransform(1 / int_downsize, name='%s_flow_resize' % name)(flow)

        preint_flow = flow
        pos_flow = flow

        # integrate to produce diffeomorphic warp (i.e. treat flow as a stationary velocity field)
        if int_steps > 0:
            pos_flow = vxm.layers.VecInt(method='ss', name='%s_flow_int' % name, int_steps=int_steps)(pos_flow)

            # resize to final resolution
            if int_downsize > 1:
                pos_flow = vxm.layers.RescaleTransform(int_downsize, name='%s_diffflow' % name)(pos_flow)

        # warp image with flow field
        y_source = vxm.layers.SpatialTransformer(name='%s_transformer' % name)([source, pos_flow])

        # make model
        inputs = [source, target, hyp_input]
        super().__init__(name=name, inputs=inputs, outputs=[y_source, preint_flow])

        # cache pointers to layers and tensors for future reference
        self.references = LoadableModel.ReferenceContainer()
        self.references.y_source = y_source
        self.references.pos_flow = pos_flow
        self.references.hyp_input = hyp_input

    def get_registration_model(self):
        """
        Returns a reconfigured model to predict only the warped image and final transform.
        """
        return tf.keras.Model(self.inputs, [self.references.y_source, self.references.pos_flow])

