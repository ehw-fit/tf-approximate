##========== Copyright (c) 2020, Filip Vaverka, All rights reserved. =========##
##
## Purpose:     Tensorflow backend for approximated operations.
##
## $NoKeywords: $ApproxTF $backend.py
## $Date:       $2020-09-10
##============================================================================##

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

from tensorflow.python.keras import backend_config
from tensorflow.python.framework import (tensor_shape, ops)
from tensorflow.python.ops import (array_ops, gen_nn_ops, nn_ops)
from tensorflow.python.util.tf_export import keras_export

from tensorflow.python.keras.backend import (
    _preprocess_conv2d_input,
    _preprocess_padding, 
    image_data_format)

dir_path = os.path.dirname(os.path.realpath(__file__))
approx_op_module = tf.load_op_library(os.path.join(dir_path, '..', '..', 'build', 'libApproxGPUOpsTF.so'))


@keras_export('keras.backend.approx_conv2d_with_min_max_vars')
def approx_conv2d_with_min_max_vars(x,
        kernel,
        input_min, input_max,
        kernel_min, kernel_max,
        strides=(1, 1),
        padding='valid',
        data_format=None,
        dilation_rate=(1, 1),
        approx_num_bits=8,
        approx_mul_table_file=''):
    # if data_format is None:
    #     data_format = image_data_format()
    
    # if data_format not in {'channels_first', 'channels_last'}:
    #     raise ValueError('Unknown data_format: ' + str(data_format))

    # x, tf_data_format = _preprocess_conv2d_input(x, data_format)
    # padding = _preprocess_padding(padding)
    x, tf_data_format = _preprocess_conv2d_input(x, data_format)
    padding = _preprocess_padding(padding.lower())
    if len(dilation_rate) < 4:
        if tf_data_format == 'NHWC':
            dilation_rate = (1,) + tuple(dilation_rate) + (1,)
        else:
            dilation_rate = (1, 1) + tuple(dilation_rate)
        
    x = approx_op_module.approx_conv2d_with_min_max_vars(
        input=x,
        filter=kernel,
        input_min=input_min, input_max=input_max,
        filter_min=kernel_min, filter_max=kernel_max,
        dilations=dilation_rate,
        strides=strides,
        padding=padding,
        data_format=data_format,
        num_bits=approx_num_bits,
        mul_map_file=approx_mul_table_file)
    
    # if data_format == 'channels_first' and tf_data_format == 'NHWC':
    #     x = array_ops.transpose(x, (0, 3, 1, 2))
    
    return x

        
@keras_export('keras.backend.fake_approx_conv2d')
def fake_approx_conv2d(x,
        kernel,
        strides=(1, 1),
        padding='valid',
        data_format=None,
        dilation_rate=(1, 1),
        approx_num_bits=8,
        approx_mul_table_file=''):
    input_min = tf.math.reduce_min(x)
    input_max = tf.math.reduce_max(x)
    quantized_input = tf.quantization.fake_quant_with_min_max_vars(x, input_min, input_max,
                                                                    num_bits=approx_num_bits)
    
    kernel_min = tf.math.reduce_min(kernel)
    kernel_max = tf.math.reduce_max(kernel)
    quantized_kernel = tf.quantization.fake_quant_with_min_max_vars(kernel, kernel_min, kernel_max,
                                                                    num_bits=approx_num_bits)
        
    if data_format == "channels_last":
        data_format = "NHWC"
    if data_format == "channels_first":
        data_format = "NCHW"
        
    # if data_format == 'NHWC':
    #     strides = (1,) + tuple(strides) + (1,)
    #     dilation_rate = (1,) + tuple(dilation_rate) + (1,)
    # else:
    #     strides = (1, 1) + tuple(strides)
    #     dilation_rate = (1, 1) + tuple(dilation_rate)    
    
    x = approx_conv2d_with_min_max_vars(quantized_input, quantized_kernel,
        input_min, input_max,
        kernel_min, kernel_max,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        approx_num_bits=approx_num_bits,
        approx_mul_table_file=approx_mul_table_file)
    return x

@keras_export('keras.backend.fake_approx_conv_2d')
def fake_approx_conv_2d(x,
        kernel,
        strides=(1, 1),
        padding='valid',
        data_format=None,
        dilation_rate=(1, 1),
        approx_num_bits=8,
        approx_mul_table_file=''):
        
        strides = (1,) + tuple(strides) + (1,)
        
        return fake_approx_conv2d(
            x,
            kernel,
            strides,
            padding,
            data_format,
            dilation_rate,
            approx_num_bits,
            approx_mul_table_file
        )

@keras_export('keras.backend.fake_approx_per_channel_conv2d')
def fake_approx_per_channel_conv2d(x,
        kernel,
        strides=(1, 1),
        padding='valid',
        data_format=None,
        dilation_rate=(1, 1),
        approx_num_bits=8,
        approx_mul_table_file=''):
    input_min = tf.math.reduce_min(x)
    input_max = tf.math.reduce_max(x)
    quantized_input = tf.quantization.fake_quant_with_min_max_vars(x, input_min, input_max,
                                                                    num_bits=approx_num_bits)
    kernel_min = tf.math.reduce_min(kernel, axis=[0, 1, 2])
    kernel_max = tf.math.reduce_max(kernel, axis=[0, 1, 2])
    quantized_kernel = tf.quantization.fake_quant_with_min_max_vars_per_channel(kernel, kernel_min, kernel_max,
                                                                                num_bits=approx_num_bits)
    
    x = approx_conv2d_with_min_max_vars(quantized_input, quantized_kernel,
        input_min, input_max,
        kernel_min, kernel_max,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        approx_num_bits=approx_num_bits,
        approx_mul_table_file=approx_mul_table_file)
    return x


@keras_export('keras.backend.approx_depthwise_conv2d_with_min_max_vars')
def approx_depthwise_conv2d_with_min_max_vars(x,
        depthwise_kernel,
        input_min, input_max,
        kernel_min, kernel_max,
        strides=(1, 1),
        padding='valid',
        data_format=None,
        dilation_rate=(1, 1),
        approx_num_bits=8,
        approx_mul_table_file=''):
    if data_format is None:
        data_format = image_data_format()

    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format: ' + str(data_format))

    x, tf_data_format = _preprocess_conv2d_input(x, data_format)
    padding = _preprocess_padding(padding)

    if tf_data_format == 'NHWC':
        strides = (1,) + tuple(strides) + (1,)
        dilations = (1,) + tuple(dilation_rate) + (1,)
    else:
        strides = (1, 1) + tuple(strides)
        dilations = (1, 1) + tuple(dilation_rate)
    
    x = approx_op_module.approx_depthwise_conv2d_with_min_max_vars(
        x,
        depthwise_kernel,
        input_min, input_max,
        kernel_min, kernel_max,
        strides=strides,
        padding=padding,
        dilations=dilations,
        data_format=tf_data_format,
        num_bits=approx_num_bits,
        mul_map_file=approx_mul_table_file)
    
    if data_format == 'channels_first' and tf_data_format == 'NHWC':
        x = array_ops.transpose(x, (0, 3, 1, 2))
    
    return x


@keras_export('keras.backend.fake_approx_depthwise_conv2d')
def fake_approx_depthwise_conv2d(x,
        depthwise_kernel,
        strides=(1, 1),
        padding='valid',
        data_format=None,
        dilation_rate=(1, 1),
        approx_num_bits=8,
        approx_mul_table_file=''):
    input_min = tf.math.reduce_min(x)
    input_max = tf.math.reduce_max(x)
    quantized_input = tf.quantization.fake_quant_with_min_max_vars(x, input_min, input_max,
                                                                    num_bits=approx_num_bits)
    
    kernel_min = tf.math.reduce_min(depthwise_kernel)
    kernel_max = tf.math.reduce_max(depthwise_kernel)
    quantized_kernel = tf.quantization.fake_quant_with_min_max_vars(depthwise_kernel, 
                                                                    kernel_min, kernel_max,
                                                                    num_bits=approx_num_bits)
    
    x = approx_depthwise_conv2d_with_min_max_vars(quantized_input, quantized_kernel,
        input_min, input_max,
        kernel_min, kernel_max,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        approx_num_bits=approx_num_bits,
        approx_mul_table_file=approx_mul_table_file)
    return x

@keras_export('keras.backend.fake_approx_separable_conv2d')
def fake_approx_separable_conv2d(x,
                                 depthwise_filter,
                                 pointwise_filter,
                                 strides,
                                 padding,
                                 rate=None,
                                 name=None,
                                 data_format=None,
                                 dilations=(1,1),
                                 approx_num_bits=8,
                                 approx_mul_table_file=''):
    
    if rate is None:
        rate = [1, 1]
    
    # def op(input_converted, _, padding):
    #     return fake_approx_depthwise_conv2d(
    #         x=input_converted,
    #         depthwise_kernel=depthwise_filter,
    #         strides=strides,
    #         padding=padding,
    #         dilation_rate=dilations,
    #         data_format=data_format,
    #         approx_num_bits=approx_num_bits,
    #         approx_mul_table_file=approx_mul_table_file
    #     )

    # with space to batch carries out the problems with dilations, if dilation == 1, it does nothing
    # depthwise = nn_ops.with_space_to_batch(
    #     input=x,
    #     filter_shape=array_ops.shape(depthwise_filter),
    #     dilation_rate=rate,
    #     padding=padding,
    #     data_format=data_format,
    #     op=op
    # )
    
    depthwise = fake_approx_depthwise_conv2d(
        x=x,
        depthwise_kernel=depthwise_filter,
        strides=strides,
        padding=padding,
        dilation_rate=dilations,
        data_format=data_format,
        approx_num_bits=approx_num_bits,
        approx_mul_table_file=approx_mul_table_file
        )
   
    # return fake_approx_conv2d(
    #     x=depthwise,
    #     kernel=pointwise_filter,
    #     strides=[1,1,1,1],
    #     padding="valid",
    #     data_format=data_format,
    #     approx_num_bits=approx_num_bits,
    #     approx_mul_table_file=approx_mul_table_file
    # )
    
    return nn_ops.conv2d(
        input=depthwise,
        filter=pointwise_filter,
        strides=[1,1,1,1],
        padding="VALID",
        data_format=data_format,
    )




@ops.RegisterGradient("ApproxConv2DWithMinMaxVars")
def _ApproxConv2DWithMinMaxVarsGrad(op, grad):
    """Gradient function for ApproxConv2DWithMinMaxVars."""
    dilations         = op.get_attr("dilations")
    strides           = op.get_attr("strides")
    padding           = op.get_attr("padding")
    explicit_paddings = op.get_attr("explicit_paddings")
    use_cudnn_on_gpu  = True
    data_format       = op.get_attr("data_format")
    shape_0, shape_1  = array_ops.shape_n([op.inputs[0], op.inputs[1]])

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
            data_format=data_format
        )] + [tf.zeros_like(op.inputs[i]) for i in range(2,6)]


@ops.RegisterGradient("ApproxDepthwiseConv2DWithMinMaxVars")
def _ApproxDepthwiseConv2DWithMinMaxVarsGrad(op, grad):
    """Gradient function for ApproxDepthwiseConv2DWithMinMaxVars."""
    dilations         = op.get_attr("dilations")
    strides           = op.get_attr("strides")
    padding           = op.get_attr("padding")
    data_format       = op.get_attr("data_format")
    shape_0, shape_1  = array_ops.shape_n([op.inputs[0], op.inputs[1]])

    return [
        nn_ops.depthwise_conv2d_native_backprop_input(
            shape_0,
            op.inputs[1],
            grad,
            dilations=dilations,
            strides=strides,
            padding=padding,
            data_format=data_format),
        nn_ops.depthwise_conv2d_native_backprop_filter(
            op.inputs[0],
            shape_1,
            grad,
            dilations=dilations,
            strides=strides,
            padding=padding,
            data_format=data_format)
        ] + [tf.zeros_like(op.inputs[i]) for i in range(2,6)]
