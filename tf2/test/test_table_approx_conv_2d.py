import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import argparse
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


dir_path = os.path.dirname(os.path.realpath(__file__))
approx_mul_file = os.path.join(dir_path, 'test_mul_table.bin')


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
        self.input_tensor = tf.placeholder('float32', self.input_data.shape)  # tf.constant(self.input_data)
        self.input_bound_tensors = (tf.constant(0.0, dtype='float32'), tf.constant(1.0, dtype='float32'))

        self.filter_tensor = tf.placeholder('float32', self.filter_data.shape)  # tf.constant(self.filter_data)
        self.filter_bound_tensors = (tf.constant(0.0, dtype='float32'), tf.constant(1.0, dtype='float32'))

        input_tensor_quant = tf.quantization.fake_quant_with_min_max_vars(self.input_tensor,
                                                                          self.input_bound_tensors[0],
                                                                          self.input_bound_tensors[1], num_bits=8)
        filter_tensor_quant = tf.quantization.fake_quant_with_min_max_vars(self.filter_tensor,
                                                                           self.filter_bound_tensors[0],
                                                                           self.filter_bound_tensors[1], num_bits=8)

        self.output_ref_tensor = tf.nn.conv2d(input_tensor_quant, filter_tensor_quant, self.stride, 'SAME')
        self.output_tensor = self.test_op_module.approx_conv2d_with_min_max_vars(input_tensor_quant, filter_tensor_quant,
                                                                                 *self.input_bound_tensors, *self.filter_bound_tensors,
                                                                                 self.stride, 8, approx_mul_file, 'SAME')

    def run(self, device):
        with tf.device('/{}'.format(device)):
            self.__build()
            with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True)) as sess:
                inputs_dict = {
                    self.input_tensor: self.input_data,
                    self.filter_tensor: self.filter_data
                }

                sess.run(tf.compat.v1.global_variables_initializer())

                result = sess.run(tf.abs(self.output_tensor - self.output_ref_tensor) / tf.reduce_max(tf.abs(self.output_ref_tensor)), feed_dict=inputs_dict)
                print('{}: Linf Error: {}'.format(device, np.max(result)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, help='Tensorflow device to be used (ie. \'cpu:0\', \'gpu:0\', ...).', default='cpu:0')

    args = parser.parse_args()

    test_op_module = tf.load_op_library('libApproxGPUOpsTF.so')

    test = TestApproxConv2D(test_op_module, [1, 256, 256, 1], [8, 8, 1, 1], [1, 1, 1, 1])
    test.run(args.device)
