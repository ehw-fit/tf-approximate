##========== Copyright (c) 2020, Filip Vaverka, All rights reserved. =========##
##
## Purpose:     Approximate Conv2D Keras layer incorporating batch level
##              quantization.
##              Gradients are computed with accurate Conv2D implementation.
##
## $NoKeywords: $ApproxTF $fake_approx_convolutional.py
## $Date:       $2020-02-25
##============================================================================##

import os
from abc import abstractmethod
import tensorflow as tf

from tensorflow.python.keras import activations
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import backend

from tensorflow.python.framework import ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_nn_ops

from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.layers.convolutional import Conv
from tensorflow.python.util.tf_export import keras_export
from tensorflow.python.keras.engine.input_spec import InputSpec

approx_op_module = tf.load_op_library('libApproxGPUOpsTF.so')


class _BaseNonAtrousApproxConvolution(object):
    def __init__(self,
                 input_shape,
                 filter_shape,  # pylint: disable=redefined-builtin
                 padding,
                 data_format=None,
                 strides=None,
                 name=None):
        filter_shape = filter_shape.with_rank(input_shape.ndims)
        self.padding = padding
        self.name = name
        input_shape = input_shape.with_rank(filter_shape.ndims)
        if input_shape.ndims is None:
            raise ValueError("Rank of convolution must be known")
        if input_shape.ndims < 3 or input_shape.ndims > 5:
            raise ValueError(
                "`input` and `filter` must have rank at least 3 and at most 5")
        conv_dims = input_shape.ndims - 2
        if strides is None:
            strides = [1] * conv_dims
        elif len(strides) != conv_dims:
            raise ValueError("len(strides)=%d, but should be %d" % (len(strides),
                                                                    conv_dims))

        if conv_dims == 2:
            if data_format is None or data_format == "NHWC":
                data_format = "NHWC"
                strides = [1] + list(strides) + [1]
            elif data_format == "NCHW":
                strides = [1, 1] + list(strides)
            else:
                raise ValueError("data_format must be \"NHWC\" or \"NCHW\".")
            self.strides = strides
            self.data_format = data_format

    @abstractmethod
    def __call__(self, input, filter):  # pylint: disable=redefined-builtin
        pass


class _NonAtrousApproxConvolutionWithMinMaxVars(_BaseNonAtrousApproxConvolution):
    def __init__(self,
                 input_shape,
                 filter_shape,  # pylint: disable=redefined-builtin
                 padding,
                 data_format=None,
                 strides=None,
                 name=None,
                 num_bits=8,
                 mul_map_file=''):
        super(_NonAtrousApproxConvolutionWithMinMaxVars, self).__init__(
            input_shape=input_shape,
            filter_shape=filter_shape,
            padding=padding,
            data_format=data_format,
            strides=strides,
            name=name)

        self.num_bits = num_bits
        self.mul_map_file = mul_map_file
        self.conv_op = approx_op_module.approx_conv2d_with_min_max_vars
        self.input_min = None
        self.input_max = None
        self.filter_min = None
        self.filter_max = None
        self.quantized_filter = None

    def set_min_max_vars(self, input_min, input_max, filter_min, filter_max):
        self.input_min = input_min
        self.input_max = input_max
        self.filter_min = filter_min
        self.filter_max = filter_max

    def set_quantized_filter(self, quantized_filter):
        self.quantized_filter = quantized_filter

    def __call__(self, input, filter):
        # NOTE: "filter" is ignored because OP requires kernel passed through Fake Quantization, which is specified in
        #       self.quantized_filter (before this method is called).
        return self.conv_op(
            input=input,
            filter=self.quantized_filter,
            input_min=self.input_min, input_max=self.input_max,
            filter_min=self.filter_min, filter_max=self.filter_max,
            strides=self.strides,
            num_bits=self.num_bits,
            mul_map_file=self.mul_map_file,
            padding=self.padding,
            data_format=self.data_format,
            name=self.name)


class FakeApproxConvolution2D(nn_ops.Convolution):
    def __init__(self,
                 input_shape,
                 filter_shape,
                 padding,
                 strides=None,
                 dilation_rate=None,
                 name=None,
                 data_format=None,
                 num_bits=8,
                 mul_map_file=''):
        self.num_bits = num_bits
        self.mul_map_file = mul_map_file

        self.input_min = None
        self.input_max = None
        self.filter_min = None
        self.filter_max = None
        self.quantized_filter = None

        super(FakeApproxConvolution2D, self).__init__(
            input_shape=input_shape,
            filter_shape=filter_shape,
            padding=padding,
            strides=strides,
            dilation_rate=dilation_rate,
            name=name,
            data_format=data_format)

    def set_min_max_vars(self, input_min, input_max, filter_min, filter_max):
        self.conv_op.call.set_min_max_vars(input_min, input_max, filter_min, filter_max)

    def set_quantized_filter(self, quantized_filter):
        self.conv_op.call.set_quantized_filter(quantized_filter)

    def _build_op(self, _, padding):
        conv_op = _NonAtrousApproxConvolutionWithMinMaxVars(
            self.input_shape,
            filter_shape=self.filter_shape,
            padding=padding,
            data_format=self.data_format,
            strides=self.strides,
            name=self.name,
            num_bits=self.num_bits,
            mul_map_file=self.mul_map_file)

        return conv_op


@keras_export('keras.layers.FakeApproxConv2D', 'keras.layers.FakeApproxConvolution2D')
class FakeApproxConv2D(Conv):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 num_bits=8,
                 mul_map_file='',
                 **kwargs):
        self.num_bits = num_bits
        self.mul_map_file = mul_map_file

        super(FakeApproxConv2D, self).__init__(
            rank=2,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activations.get(activation),
            use_bias=use_bias,
            kernel_initializer=initializers.get(kernel_initializer),
            bias_initializer=initializers.get(bias_initializer),
            kernel_regularizer=regularizers.get(kernel_regularizer),
            bias_regularizer=regularizers.get(bias_regularizer),
            activity_regularizer=regularizers.get(activity_regularizer),
            kernel_constraint=constraints.get(kernel_constraint),
            bias_constraint=constraints.get(bias_constraint),
            **kwargs)

    def build(self, input_shape):
        super().build(input_shape)
        self._convolution_op = None  # Native convolution was assigned as this point and will be replaced with our own.

    def call(self, inputs):
        if self.padding == 'causal':
            op_padding = 'valid'
        else:
            op_padding = self.padding

        if not isinstance(op_padding, (list, tuple)):
            op_padding = op_padding.upper()

        # Use Fake Quantization to quantize/dequantize input of the convolution
        input_min = tf.math.reduce_min(inputs)
        input_max = tf.math.reduce_max(inputs)
        quantized_input = tf.quantization.fake_quant_with_min_max_vars(inputs, input_min, input_max,
                                                                       num_bits=self.num_bits)

        # Use Fake Quantization to quantize/dequantize kernel of the convolution
        kernel_min = tf.math.reduce_min(self.kernel)
        kernel_max = tf.math.reduce_max(self.kernel)
        quantized_kernel = tf.quantization.fake_quant_with_min_max_vars(self.kernel, kernel_min, kernel_max,
                                                                        num_bits=self.num_bits)

        if self._convolution_op is None:
            self._convolution_op = FakeApproxConvolution2D(
                inputs.shape,
                filter_shape=self.kernel.shape,
                dilation_rate=self.dilation_rate,
                strides=self.strides,
                padding=op_padding,
                data_format=conv_utils.convert_data_format(self.data_format,
                                                           self.rank + 2),
                num_bits=self.num_bits,
                mul_map_file=self.mul_map_file)

        # Set additional inputs used by inner convolution operation
        self._convolution_op.set_min_max_vars(input_min, input_max, kernel_min, kernel_max)
        self._convolution_op.set_quantized_filter(quantized_kernel)

        # Reuse standard call for convolution (with quantized inputs)
        return super().call(quantized_input)


@ops.RegisterGradient("FakeApproxConv2D")
def _FakeApproxConv2DGrad(op, grad):
    """Gradient function for FakeApproxConv2D."""
    dilations = op.get_attr("dilations")
    strides = op.get_attr("strides")
    padding = op.get_attr("padding")
    explicit_paddings = op.get_attr("explicit_paddings")
    use_cudnn_on_gpu = True  # op.get_attr("use_cudnn_on_gpu")
    data_format = op.get_attr("data_format")
    shape_0, shape_1 = array_ops.shape_n([op.inputs[0], op.inputs[1]])

    # We call the gen_nn_ops backprop functions instead of nn_ops backprop
    # functions for performance reasons in Eager mode. gen_nn_ops functions take a
    # `explicit_paddings` parameter, but nn_ops functions do not. So if were were
    # to use the nn_ops functions, we would have to convert `padding` and
    # `explicit_paddings` into a single `padding` parameter, increasing overhead
    # in Eager mode.
    return [
        gen_nn_ops.conv2d_backprop_input(
            shape_0,
            op.inputs[1],
            grad,
            dilations=dilations,
            strides=strides,
            padding=padding,
            explicit_paddings=explicit_paddings,
            use_cudnn_on_gpu=use_cudnn_on_gpu,
            data_format=data_format),
        gen_nn_ops.conv2d_backprop_filter(
            op.inputs[0],
            shape_1,
            grad,
            dilations=dilations,
            strides=strides,
            padding=padding,
            explicit_paddings=explicit_paddings,
            use_cudnn_on_gpu=use_cudnn_on_gpu,
            data_format=data_format)
    ]
