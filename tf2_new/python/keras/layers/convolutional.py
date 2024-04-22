##========== Copyright (c) 2020, Filip Vaverka, All rights reserved. =========##
##
## Purpose:     Approximate Conv2D Keras layers exposing min/max inputs.
##              These inputs are connected to quantization range variables
##              used by FakeQuantWithMinMaxVars.
##
## $NoKeywords: $ApproxTF $convolutional.py
## $Date:       $2020-09-13
##============================================================================##

import os
import six
from abc import abstractmethod

from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import backend
from tensorflow.python.framework import (tensor_shape, ops)
from tensorflow.python.ops import (array_ops, gen_nn_ops, nn_ops)
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.util.tf_export import keras_export
from tensorflow.python.keras.engine.input_spec import InputSpec
from python.ops import nn_ops as approx_nn_ops
from python.keras import backend as approx_backend

##============================================================================##
## Approximate Conv2D layer with min/max variables
##============================================================================##
@keras_export('keras.layers.ApproxConv2DWithMinMaxVars')
class ApproxConv2DWithMinMaxVars(Conv2D):
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
            approx_num_bits=8,
            approx_mul_table_file='',
            **kwargs):
        super(ApproxConv2DWithMinMaxVars, self).__init__(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)
        if approx_num_bits > 8 or approx_num_bits <= 0:
            raise ValueError('Maximum supported bit-width of `ApproxConv2DWithMinMaxVars` is 8. '
                             'Received bit-width:', str(approx_num_bits))

        self.approx_num_bits = approx_num_bits
        self.approx_mul_table_file = approx_mul_table_file

    def build(self, input_shape):
        # if not isinstance(input_shape, (list, tuple)) or len(input_shape) != 5:
        #     raise ValueError('`ApproxConv2DWithMinMaxVars` requires 5 inputs: '
        #                      'input, input_min, input_max, filter_min, filter_max.')

        # super(ApproxConv2DWithMinMaxVars, self).build(input_shape[0])
        super(ApproxConv2DWithMinMaxVars, self).build(input_shape)

        self._convolution_op = None
        self.built = False
    
    def call(self, inputs):
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 5:
            raise ValueError('`ApproxConv2DWithMinMaxVars` requires 5 inputs: '
                             'input, input_min, input_max, filter_min, filter_max.')

        call_input_shape = inputs[0].get_shape()

        # Convert Keras formats to TF native formats.
        if self.padding == 'causal':
            tf_padding = 'VALID'  # Causal padding handled in `call`.
        elif isinstance(self.padding, six.string_types):
            tf_padding = self.padding.upper()
        else:
            tf_padding = self.padding
        
        tf_dilations = list(self.dilation_rate)
        tf_strides = list(self.strides)

        self._convolution_op = approx_nn_ops.ApproxConvolutionWithMinMaxVars(
            call_input_shape,
            filter_shape=self.kernel.shape,
            input_min=inputs[1], input_max=inputs[2],
            filter_min=inputs[3], filter_max=inputs[4],
            dilation_rate=tf_dilations,
            strides=tf_strides,
            padding=tf_padding,
            data_format=self._tf_data_format,
            approx_num_bits=self.approx_num_bits,
            approx_mul_table_file=self.approx_mul_table_file)

        return super(ApproxConv2DWithMinMaxVars, self).call(inputs[0])


##============================================================================##
## Approximate DepthwiseConv2D layer with min/max variables
##============================================================================##
@keras_export('keras.layers.ApproxDepthwiseConv2DWithMinMaxVars')
class ApproxDepthwiseConv2DWithMinMaxVars(Conv2D):
    def __init__(self,
            kernel_size,
            strides=(1, 1),
            padding='valid',
            depth_multiplier=1,
            data_format=None,
            activation=None,
            use_bias=True,
            depthwise_initializer='glorot_uniform',
            bias_initializer='zeros',
            depthwise_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            depthwise_constraint=None,
            bias_constraint=None,
            approx_num_bits=8,
            approx_mul_table_file='',
            **kwargs):
        super(ApproxDepthwiseConv2DWithMinMaxVars, self).__init__(
            filters=None,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            activation=activation,
            use_bias=use_bias,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            bias_constraint=bias_constraint,
            **kwargs)
        self.depth_multiplier = depth_multiplier
        self.depthwise_initializer = initializers.get(depthwise_initializer)
        self.depthwise_regularizer = regularizers.get(depthwise_regularizer)
        self.depthwise_constraint  = constraints.get(depthwise_constraint)
        self.bias_initializer = initializers.get(bias_initializer)

        if approx_num_bits > 8 or approx_num_bits <= 0:
            raise ValueError('Maximum supported bit-width of `FakeApproxDepthwiseConv2D` is 8. '
                             'Received bit-width:', str(approx_num_bits))

        self.approx_num_bits = approx_num_bits
        self.approx_mul_table_file = approx_mul_table_file
    
    def build(self, inputs_shape):
        # if not isinstance(inputs_shape, (list, tuple)) or len(inputs_shape) != 5:
        #     raise ValueError('`ApproxDepthwiseConv2DWithMinMaxVars` requires 5 inputs: '
        #                      'input, input_min, input_max, filter_min, filter_max.')
        
        # input_shape = inputs_shape[0]
        input_shape = inputs_shape

        if len(input_shape) < 4:
            raise ValueError('Inputs to `FakeApproxDepthwiseConv2D` should have rank 4. '
                             'Received input shape:', str(input_shape))
        
        input_shape = tensor_shape.TensorShape(input_shape)
        channel_axis = self._get_channel_axis()

        if input_shape.dims[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs to '
                             '`FakeApproxDepthwiseConv2D` '
                             'should be defined. Found `None`.')
        
        input_dim = int(input_shape[channel_axis])
        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
        
        self.depthwise_kernel = self.add_weight(
            shape=depthwise_kernel_shape,
            initializer=self.depthwise_initializer,
            name='depthwise_kernel',
            regularizer=self.depthwise_regularizer,
            constraint=self.depthwise_constraint)
        
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(input_dim * self.depth_multiplier,),
                initializer=self.bias_initializer,
                name='bias',
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint)
        else:
            self.bias = None
        
        self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
        self.built = True
    
    def call(self, inputs):
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 5:
            raise ValueError('`ApproxDepthwiseConv2DWithMinMaxVars` requires 5 inputs: '
                             'input, input_min, input_max, filter_min, filter_max.')

        outputs = approx_backend.approx_depthwise_conv2d_with_min_max_vars(
            inputs[0],
            self.depthwise_kernel,
            input_min=inputs[1], input_max=inputs[2],
            kernel_min=inputs[3], kernel_max=inputs[4],
            strides=self.strides,
            padding=self.padding,
            dilation_rate=self.dilation_rate,
            data_format=self.data_format,
            approx_num_bits=self.approx_num_bits,
            approx_mul_table_file=self.approx_mul_table_file)
        
        if self.use_bias:
            outputs = backend.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)
        
        if self.activation is not None:
            return self.activation(outputs)
        
        return outputs
    
    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            rows = input_shape[2]
            cols = input_shape[3]
            out_filters = input_shape[1] * self.depth_multiplier
        elif self.data_format == 'channels_last':
            rows = input_shape[1]
            cols = input_shape[2]
            out_filters = input_shape[3] * self.depth_multiplier

            rows = conv_utils.conv_output_length(rows, self.kernel_size[0],
                                                self.padding,
                                                self.strides[0])
            cols = conv_utils.conv_output_length(cols, self.kernel_size[1],
                                                self.padding,
                                                self.strides[1])
        if self.data_format == 'channels_first':
            return (input_shape[0], out_filters, rows, cols)
        elif self.data_format == 'channels_last':
            return (input_shape[0], rows, cols, out_filters)

    def get_config(self):
        config = super(ApproxDepthwiseConv2DWithMinMaxVars, self).get_config()
        config.pop('filters')
        config.pop('kernel_initializer')
        config.pop('kernel_regularizer')
        config.pop('kernel_constraint')
        config['depth_multiplier'] = self.depth_multiplier
        config['depthwise_initializer'] = initializers.serialize(
            self.depthwise_initializer)
        config['depthwise_regularizer'] = regularizers.serialize(
            self.depthwise_regularizer)
        config['depthwise_constraint'] = constraints.serialize(
            self.depthwise_constraint)
        return config
