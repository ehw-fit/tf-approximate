##========== Copyright (c) 2020, Filip Vaverka, All rights reserved. =========##
##
## Purpose:     Approximate extensions to nn_ops.py from Tensorflow
##
## $NoKeywords: $ApproxTF $nn_ops.py
## $Date:       $2020-09-11
##============================================================================##

from numpy import percentile
from tensorflow.python.ops.nn_ops import (Convolution, _NonAtrousConvolution, 
    convert_padding, _get_sequence)
from tensorflow.python.ops.gen_nn_ops import conv2d
from tensorflow.python.framework import smart_cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.util import deprecation
from python.keras import backend as approx_backend
from tensorflow.python.util.tf_export import keras_export


##============================================================================##
## Fake Approx Convolution Operation
##============================================================================##
class _NonAtrousFakeApproxConvolution(_NonAtrousConvolution):
    def __init__(self,
            input_shape,
            filter_shape,
            padding,
            data_format=None,
            strides=None,
            name=None,
            approx_num_bits=8,
            approx_mul_table_file='',
            per_channel=False):
        super(_NonAtrousFakeApproxConvolution, self).__init__(
            input_shape,
            filter_shape,
            padding,
            data_format,
            strides,
            name)
        self.approx_num_bits = approx_num_bits
        self.approx_mul_table_file = approx_mul_table_file
        self.per_channel = per_channel

        input_shape = input_shape.with_rank(filter_shape.ndims)
        conv_dims = input_shape.ndims - 2

        if conv_dims != 2:
            raise ValueError("Only 2D convolutions are supported (%d requested)." % conv_dims)

        if per_channel:
            self.conv_op = fake_approx_perchannel_conv_2d
        else:
            self.conv_op = fake_approx_perbatch_conv_2d

    # Note that we need this adapter since argument names for conv1d don't match
    # those for gen_nn_ops.conv2d and gen_nn_ops.conv3d.
    # pylint: disable=redefined-builtin
    def _conv1d(self, input, filter, strides, padding, data_format, name):
        raise NotImplementedError("This operation is not supported")
    # pylint: enable=redefined-builtin

    def __call__(self, inp, filter):  # pylint: disable=redefined-builtin
        return self.conv_op(
            input=inp,
            filter=filter,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            name=self.name,
            approx_num_bits=self.approx_num_bits,
            approx_mul_table_file=self.approx_mul_table_file)



class FakeApproxConvolution(Convolution):
    def __init__(self,
            input_shape,
            filter_shape,
            padding,
            strides=None,
            dilation_rate=None,
            name=None,
            data_format=None,
            approx_num_bits=8,
            approx_mul_table_file='',
            per_channel=False):
        self.approx_num_bits = approx_num_bits
        self.approx_mul_table_file = approx_mul_table_file
        self.per_channel = per_channel

        super(FakeApproxConvolution, self).__init__(
            input_shape=input_shape,
            filter_shape=filter_shape,
            padding=padding,
            strides=strides,
            dilation_rate=dilation_rate,
            name=name,
            data_format=data_format)
    
    def _build_op(self, _, padding):

        return _NonAtrousFakeApproxConvolution(
            self.input_shape,
            filter_shape=self.filter_shape,
            padding=padding,
            data_format=self.data_format,
            strides=self.strides,
            name=self.name,
            approx_num_bits=self.approx_num_bits,
            approx_mul_table_file=self.approx_mul_table_file,
            per_channel=self.per_channel)


def fake_approx_perbatch_conv_2d(
        input, 
        filter,
        strides,
        padding,
        data_format,
        dilations=[1, 1],
        name=None,
        filters=None,
        approx_num_bits=8,
        approx_mul_table_file=''
        ):
    filter = deprecation.deprecated_argument_lookup(
        "filters", filters, "filter", filter)
    padding, _ = convert_padding(padding)
    
    if data_format is None:
        data_format = "NHWC"
    
    channel_index = 1 if data_format.startswith("NC") else 3

    strides   = _get_sequence(strides, 2, channel_index, "strides")
    dilations = _get_sequence(dilations, 2, channel_index, "dilations")

    conv_op = approx_backend.fake_approx_conv2d
    #print("### approx per batch")

    assert(approx_mul_table_file != "")
    
    return conv_op(
        input,
        filter,
        strides,
        padding,
        data_format=data_format,
        dilation_rate=dilations,
        approx_num_bits=approx_num_bits,
        approx_mul_table_file=approx_mul_table_file)

def fake_approx_perchannel_conv_2d(
        input, 
        filter,
        strides,
        padding,
        data_format,
        dilations=[1, 1],
        name=None,
        filters=None,
        approx_num_bits=8,
        approx_mul_table_file=''):
    filter = deprecation.deprecated_argument_lookup(
        "filters", filters, "filter", filter)
    padding, _ = convert_padding(padding)
    
    if data_format == "channels_last":
        data_format = "NHWC"
    if data_format == "channels_first":
        data_format = "NCHW"
        
    if data_format is None:
        data_format = "NHWC"
    
    channel_index = 1 if data_format.startswith("NC") else 3

    strides   = _get_sequence(strides, 2, channel_index, "strides")
    dilations = _get_sequence(dilations, 2, channel_index, "dilations")

    conv_op = approx_backend.fake_approx_per_channel_conv2d
    #print("### approx per channel")

    return conv_op(
        input,
        filter,
        strides,
        padding,
        data_format=data_format,
        dilation_rate=dilations,
        approx_num_bits=approx_num_bits,
        approx_mul_table_file=approx_mul_table_file)


##============================================================================##
## Approx Convolution Operation with Min/Max Variables exposed
##============================================================================##
class _NonAtrousApproxConvolutionWithMinMaxVars(_NonAtrousConvolution):
    def __init__(self,
            input_shape,
            filter_shape,
            input_min, input_max,
            filter_min, filter_max,
            padding,
            data_format=None,
            strides=None,
            name=None,
            approx_num_bits=8,
            approx_mul_table_file=''):
        super(_NonAtrousApproxConvolutionWithMinMaxVars, self).__init__(
            input_shape,
            filter_shape,
            padding,
            data_format,
            strides,
            name)
        self.input_min = input_min
        self.input_max = input_max
        self.filter_min = filter_min
        self.filter_max = filter_max
        self.approx_num_bits = approx_num_bits
        self.approx_mul_table_file = approx_mul_table_file

        input_shape = input_shape.with_rank(filter_shape.ndims)
        conv_dims = input_shape.ndims - 2

        if conv_dims != 2:
            raise ValueError("Only 2D convolutions are supported (%d requested)." % conv_dims)

        self.conv_op = approx_conv_2d_with_min_max_vars
    
    def __call__(self, input, filter):
        return self.conv_op(
            input=input,
            filter=filter,
            input_min=self.input_min,   input_max=self.input_max,
            filter_min=self.filter_min, filter_max=self.filter_max,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            name=self.name,
            approx_num_bits=self.approx_num_bits,
            approx_mul_table_file=self.approx_mul_table_file)


class ApproxConvolutionWithMinMaxVars(Convolution):
    def __init__(self,
            input_shape,
            filter_shape,
            padding,
            input_min, input_max,
            filter_min, filter_max,
            strides=None,
            dilation_rate=None,
            name=None,
            data_format=None,
            approx_num_bits=8,
            approx_mul_table_file=''):
        self.input_min = input_min
        self.input_max = input_max
        self.filter_min = filter_min
        self.filter_max = filter_max
        self.approx_num_bits = approx_num_bits
        self.approx_mul_table_file = approx_mul_table_file

        super(ApproxConvolutionWithMinMaxVars, self).__init__(
            input_shape=input_shape,
            filter_shape=filter_shape,
            padding=padding,
            strides=strides,
            dilation_rate=dilation_rate,
            name=name,
            data_format=data_format)
    
    def _build_op(self, _, padding):
        return _NonAtrousApproxConvolutionWithMinMaxVars(
            self.input_shape,
            filter_shape=self.filter_shape,
            input_min=self.input_min, input_max=self.input_max,
            filter_min=self.filter_min, filter_max=self.filter_max,
            padding=padding,
            data_format=self.data_format,
            strides=self.strides,
            name=self.name,
            approx_num_bits=self.approx_num_bits,
            approx_mul_table_file=self.approx_mul_table_file)


def approx_conv_2d_with_min_max_vars(
        input,
        filter,
        input_min, input_max,
        filter_min, filter_max,
        strides,
        padding,
        data_format,
        dilations=[1, 1],
        name=None,
        filters=None,
        approx_num_bits=8,
        approx_mul_table_file=''):
    filter = deprecation.deprecated_argument_lookup(
        "filters", filters, "filter", filter)
    padding, _ = convert_padding(padding)

    if data_format is None:
        data_format = "NHWC"
    
    channel_index = 1 if data_format.startswith("NC") else 3

    strides   = _get_sequence(strides, 2, channel_index, "strides")
    dilations = _get_sequence(dilations, 2, channel_index, "dilations")

    return approx_backend.approx_conv2d_with_min_max_vars(
        input,
        filter,
        input_min, input_max,
        filter_min, filter_max,
        strides,
        padding,
        data_format=data_format,
        dilation_rate=dilations,
        approx_num_bits=approx_num_bits,
        approx_mul_table_file=approx_mul_table_file)


##============================================================================##
## Tunable Approx Convolution Operation with Min/Max Variables exposed
##============================================================================##
class _NonAtrousTunableApproxConvolutionWithMinMaxVars(_NonAtrousConvolution):
    def __init__(self,
            input_shape,
            filter_shape,
            input_min, input_max,
            filter_min, filter_max,
            padding,
            tuning_phase,
            data_format=None,
            strides=None,
            name=None,
            approx_num_bits=8,
            approx_mul_table_file=''):
        super(_NonAtrousTunableApproxConvolutionWithMinMaxVars, self).__init__(
            input_shape,
            filter_shape,
            padding,
            data_format,
            strides,
            name)
        self.input_min = input_min
        self.input_max = input_max
        self.filter_min = filter_min
        self.filter_max = filter_max
        self.approx_num_bits = approx_num_bits
        self.approx_mul_table_file = approx_mul_table_file
        self.tuning_phase = tuning_phase

        input_shape = input_shape.with_rank(filter_shape.ndims)
        conv_dims = input_shape.ndims - 2

        if conv_dims != 2:
            raise ValueError("Only 2D convolutions are supported (%d requested)." % conv_dims)

        self.conv_op = conv2d
        self.approx_conv_op = approx_conv_2d_with_min_max_vars
    
    def __call__(self, input, filter):
        return control_flow_ops.cond(self.tuning_phase, 
            lambda: self.approx_conv_op(
                        input=input,
                        filter=filter,
                        input_min=self.input_min, input_max=self.input_max,
                        filter_min=self.filter_min, filter_max=self.filter_max,
                        strides=self.strides,
                        padding=self.padding,
                        data_format=self.data_format,
                        name=self.name,
                        approx_num_bits=self.approx_num_bits,
                        approx_mul_table_file=self.approx_mul_table_file),
            lambda: self.conv_op(
                        input=input, 
                        filter=filter,
                        strides=self.strides,
                        padding=self.padding,
                        data_format=self.data_format,
                        name=self.name))


class TunableApproxConvolutionWithMinMaxVars(Convolution):
    def __init__(self,
            input_shape,
            filter_shape,
            padding,
            input_min, input_max,
            filter_min, filter_max,
            tuning_phase,
            strides=None,
            dilation_rate=None,
            name=None,
            data_format=None,
            approx_num_bits=8,
            approx_mul_table_file=''):
        self.input_min = input_min
        self.input_max = input_max
        self.filter_min = filter_min
        self.filter_max = filter_max
        self.approx_num_bits = approx_num_bits
        self.approx_mul_table_file = approx_mul_table_file
        self.tuning_phase = tuning_phase

        super(TunableApproxConvolutionWithMinMaxVars, self).__init__(
            input_shape=input_shape,
            filter_shape=filter_shape,
            padding=padding,
            strides=strides,
            dilation_rate=dilation_rate,
            name=name,
            data_format=data_format)
    
    def _build_op(self, _, padding):
        return _NonAtrousTunableApproxConvolutionWithMinMaxVars(
            self.input_shape,
            filter_shape=self.filter_shape,
            input_min=self.input_min, input_max=self.input_max,
            filter_min=self.filter_min, filter_max=self.filter_max,
            padding=padding,
            tuning_phase=self.tuning_phase,
            data_format=self.data_format,
            strides=self.strides,
            name=self.name,
            approx_num_bits=self.approx_num_bits,
            approx_mul_table_file=self.approx_mul_table_file)
