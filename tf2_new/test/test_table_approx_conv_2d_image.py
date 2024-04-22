import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, help='Tensorflow device to be used (ie. \'cpu:0\',\'gpu:0\',...).', default='cpu:0')

    args = parser.parse_args()

    #tf.logging.set_verbosity(tf.logging.FATAL)
    tf.compat.v1.disable_v2_behavior()
    tf.debugging.set_log_device_placement(True)
    test_op_module = tf.load_op_library('libApproxGPUOpsTF.so')

    # kernelData = np.zeros((8, 8, 3, 3), dtype='float32')
    # kernelData[:, :, :] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    # kernelData = kernelData / (8*8)

    kernelData = np.zeros((8, 8, 3, 3), dtype='float32')
    kernelData = np.random.rand(8, 8, 3, 3) - 0.5
    print(np.sum(kernelData, (0, 1, 2)))

    img = Image.open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'lena_std.png'))
    img.load()
    imageData = np.asarray(img, dtype='float32')
    if imageData.shape[2] > 3:
        imageData = imageData[:, :, 0:3]



    batchSize = 2

    data = np.zeros((batchSize,) + imageData.shape, dtype='float32')
    for i in range(0, batchSize):
        data[i, :] = imageData[:]

    data[1, :] = data[1, :] * 0.5

    data[:] = data[:] * (1.0 / 255.0) - 1.5

    print('DATA: {0}'.format(data[0]))

    dataInTensor     = tf.constant(data)
    dataKernelTensor = tf.constant(kernelData, dtype='float32')

    # input_min = tf.Variable(0, dtype='float32')
    # input_max = tf.Variable(1, dtype='float32')
    # filter_min = tf.Variable(0, dtype='float32')
    # filter_max = tf.Variable(1, dtype='float32')

    input_min = tf.reduce_min(dataInTensor)
    input_max = tf.reduce_max(dataInTensor)
    filter_min = tf.reduce_min(dataKernelTensor)
    filter_max = tf.reduce_max(dataKernelTensor)

    dataInTensorQ = tf.quantization.fake_quant_with_min_max_vars(dataInTensor, input_min, input_max, num_bits=8)
    dataKernelTensorQ = tf.quantization.fake_quant_with_min_max_vars(dataKernelTensor, filter_min, filter_max, num_bits=8)

    dataOutTensor = test_op_module.approx_conv2d_with_min_max_vars(dataInTensorQ, dataKernelTensorQ,
                                                                   input_min, input_max, filter_min, filter_max,
                                                                   [1, 1, 1, 1], 8, '../test/test_mul_table.bin', 'SAME')
    # dataOutTensor    = test_op_module.approx_conv2d(dataInTensorQ, dataKernelTensorQ, [1, 1, 1, 1], 'SAME')
    dataOutRefTensor = tf.nn.conv2d(dataInTensorQ, dataKernelTensorQ, [1, 1, 1, 1], 'SAME')

    with tf.device('/{}'.format(args.device)):
        with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True)) as sess:
            sess.run(tf.compat.v1.global_variables_initializer())

            dataOut = sess.run(tf.abs(dataOutTensor - dataOutRefTensor) / tf.reduce_max(tf.abs(dataOutRefTensor)))
            print(sess.run(tf.reduce_sum(dataKernelTensorQ, (0, 1, 2))))
            print(dataOut.shape)
            np.savetxt('tmp.txt', dataOut.flatten())
            print(np.max(dataOut))
            # dataOut = sess.run(dataOutTensor)

            for i in range(0, batchSize):
                out = Image.fromarray(np.asarray(np.squeeze(dataOut[i, :] * 255.0), dtype='uint8'), 'RGB')
                out.save('lena_out_{0}.png'.format(i))

            file_writer = tf.summary.FileWriter('tflogs', sess.graph)
            file_writer.flush()
