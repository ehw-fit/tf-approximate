from test_script import describe_layer
import tensorflow as tf
import numpy as np
import argparse
import os
from python.keras.layers.fake_convolutional import FakeApproxConv2D, FakeApproxDepthwiseConv2D
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.5)


dir_path = os.path.dirname(os.path.realpath(__file__))
approx_mul_file = os.path.join(dir_path, 'test_mul_table.bin')
approx_op_lib_file = os.path.join(
    dir_path, '..', 'build', 'libApproxGPUOpsTF.so')


class TestApproxConv2D(object):
    def __init__(self, test_op_module, input_shape, filter_shape, stride):
        self.test_op_module = test_op_module

        self.input_data = np.random.rand(*input_shape).astype('float32')
        self.input_tensor = None
        self.input_bound_tensors = (None, None)

        self.filter_data = np.random.rand(*filter_shape).astype('float32')
        self.filter_tensor = None
        self.filter_bound_tensors = (None, None)

        self.stride = stride

        self.output_ref_tensor = None
        self.output_tensor = None

    def __build(self):
        self.input_tensor = tf.placeholder(
            'float32', self.input_data.shape)  # tf.constant(self.input_data)
        self.input_bound_tensors = (tf.constant(
            0.0, dtype='float32'), tf.constant(1.0, dtype='float32'))

        self.filter_tensor = tf.placeholder(
            'float32', self.filter_data.shape)  # tf.constant(self.filter_data)
        self.filter_bound_tensors = (tf.constant(
            0.0, dtype='float32'), tf.constant(1.0, dtype='float32'))

        input_tensor_quant = tf.quantization.fake_quant_with_min_max_vars(self.input_tensor,
                                                                          self.input_bound_tensors[0],
                                                                          self.input_bound_tensors[1], num_bits=8)
        filter_tensor_quant = tf.quantization.fake_quant_with_min_max_vars(self.filter_tensor,
                                                                           self.filter_bound_tensors[0],
                                                                           self.filter_bound_tensors[1], num_bits=8)

        self.output_ref_tensor = tf.nn.conv2d(
            input_tensor_quant, filter_tensor_quant, self.stride, 'VALID')
        self.output_tensor = self.test_op_module.approx_conv2d_with_min_max_vars(input_tensor_quant, filter_tensor_quant,
                                                                                 *self.input_bound_tensors, *self.filter_bound_tensors,
                                                                                 self.stride, 8, approx_mul_file, 'VALID')

    def run(self, device):
        with tf.device('/{}'.format(device)):
            self.__build()
            with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
                inputs_dict = {
                    self.input_tensor: self.input_data,
                    self.filter_tensor: self.filter_data
                }

                sess.run(tf.compat.v1.global_variables_initializer())

                result = sess.run(tf.abs(self.output_tensor - self.output_ref_tensor) /
                                  tf.reduce_max(tf.abs(self.output_ref_tensor)), feed_dict=inputs_dict)
                print('{}: Linf Error: {}'.format(device, np.max(result)))


class TestApproxConv2DPerAxis(object):
    def __init__(self, test_op_module, input_data, filter_data, stride):
        self.test_op_module = test_op_module

        self.input_data = input_data
        self.input_tensor = None
        self.input_bound_tensors = (None, None)

        self.filter_data = filter_data
        self.filter_tensor = None
        self.filter_bound_tensors = (None, None)

        self.stride = stride

        self.output_ref_tensor = None
        self.outpt_tensor = None

    def __build(self):
        self.input_tensor = tf.placeholder('float32', self.input_data.shape)
        self.input_bound_tensors = (
            tf.math.reduce_min(self.input_tensor),
            tf.math.reduce_max(self.input_tensor)
        )

        self.filter_tensor = tf.placeholder('float32', self.filter_data.shape)
        self.filter_bound_tensors = (
            tf.math.reduce_min(self.filter_tensor, axis=[0, 1, 2]),
            tf.math.reduce_max(self.filter_tensor, axis=[0, 1, 2])
        )

        input_tensor_quant = tf.quantization.fake_quant_with_min_max_vars(self.input_tensor,
                                                                          self.input_bound_tensors[0],
                                                                          self.input_bound_tensors[1], num_bits=8)
        filter_tensor_quant = tf.quantization.fake_quant_with_min_max_vars_per_channel(self.filter_tensor,
                                                                                       self.filter_bound_tensors[0],
                                                                                       self.filter_bound_tensors[1], num_bits=8)

        self.output_ref_tensor = tf.nn.conv2d(
            input_tensor_quant, filter_tensor_quant, self.stride, 'VALID')
        self.outpt_tensor = self.test_op_module.approx_conv2d_with_min_max_vars(input_tensor_quant, filter_tensor_quant,
                                                                                *self.input_bound_tensors, *self.filter_bound_tensors,
                                                                                self.stride, 8, approx_mul_file, 'VALID')

    def run(self, device):
        with tf.device('/{}'.format(device)):
            self.__build()
            with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
                inputs_dict = {
                    self.input_tensor:  self.input_data,
                    self.filter_tensor: self.filter_data
                }

                sess.run(tf.compat.v1.global_variables_initializer())

                result = sess.run(tf.abs(self.outpt_tensor - self.output_ref_tensor) /
                                  tf.reduce_max(tf.abs(self.output_ref_tensor)), feed_dict=inputs_dict)
                print('{}: Linf Error: {}'.format(device, np.max(result)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'file', type=str, help='File (NPZ) to open')

    args = parser.parse_args()

    npz = np.load(args.file, allow_pickle=True)

    test_input = npz["input"].copy()
    #test_input = test_input - 1
    bias = "weights_1" in npz
    strides = 2

    print("shape: ", test_input.shape)
    print("input range: ", test_input.min(), test_input.max())
    print("shape: ", npz["weights_0"].shape)
    print("strides: ", strides)
    print("bias: ", bias)


    test_op_module = tf.load_op_library(approx_op_lib_file)

    #test = TestApproxConv2DPerAxis(test_op_module, test_input, npz["weights_0"], [1, 1, 1, 1])

    inputs = tf.keras.Input(shape=test_input.shape[1:])
    layer1 = tf.keras.layers.Conv2D(filters=npz["filters"],
                                    kernel_size=npz["kernel_size"],
                                    strides=strides,
                                    use_bias=bias)

    y1 = layer1(inputs)

    # print("layer 1")
    # print(describe_layer(layer1))
    # print("=" * 80)

    per_channel = True
    layer2 = FakeApproxConv2D(filters=npz["filters"],
                              kernel_size=npz["kernel_size"],
                              approx_mul_table_file=approx_mul_file,
                              per_channel=per_channel,
                              strides=strides,
                              use_bias=bias)

    y2 = layer2(inputs)

    # print("layer 2")
    # print(describe_layer(layer2))
    # print("=" * 80)

    model = tf.keras.Model(inputs=inputs, outputs=[y1, y2])
    if bias:
        layer1.set_weights([npz["weights_0"], npz["weights_1"]])
        layer2.set_weights([npz["weights_0"], npz["weights_1"]])
    else:
        layer1.set_weights([npz["weights_0"]])
        layer2.set_weights([npz["weights_0"]])

    # model.summary()
    r1, r2 = model(test_input)

    # print(r1.shape)
    # print(r2.shape)
    print("difference max", np.max(np.abs(r1 - r2) / np.max(r1)))
    print("difference median", np.median(np.abs(r1 - r2) / np.max(r1)))
    # test = TestApproxConv2D(test_op_module, [1, 256, 256, 3], [3, 3, 3, 2], [1, 1, 1, 1])
    # test.run(args.device)
