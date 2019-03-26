from __future__ import division
from __future__ import print_function

import argparse
import functools
import itertools
import os
import time

import cifar10
#import cifar10_model
#import cifar10_utils
import numpy as np
import six
from six.moves import xrange, reduce  # pylint: disable=redefined-builtin
import tensorflow as tf

import json, gzip

tf.logging.set_verbosity(tf.logging.INFO)

tf.load_op_library('../axqconv/axqconv.so')


def main(data_dir, graph, mult, batch_size=1000, iterations=10, tune=True, write_log = False, **unused):
    tf.reset_default_graph()
        
    logdir = "log"
    subset = "validation"
    subset = "eval"
    input_layer = "input"
    output_layer = "resnet/tower_0/fully_connected/dense/BiasAdd:0"
    use_distortion = False

    num_intra_threads = 2

    sess_config = tf.ConfigProto(
      allow_soft_placement=True,
      intra_op_parallelism_threads=num_intra_threads)

    with tf.Session() as sess:
        dataset = cifar10.Cifar10DataSet(data_dir, subset, use_distortion)
        image_batch, label_batch = dataset.make_batch(batch_size)
        
        # assignment of attributes at protobuf level
        idop = 0
        with tf.gfile.GFile(graph, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            for node in graph_def.node:
                if node.op != "AxConv2D": continue
                tf.logging.info("%s = %s (tune = %s)" % (node.name, mult, tune))
                val = node.attr.get_or_create("AxMult")
                val.s = str.encode(mult)
                
                val = node.attr.get_or_create("AxTune")
                val.b = tune
                #node.attr["AxMult"] = val
                idop += 1 
                #node.attr["AxMult"] = None

        # load graph
        predictions = tf.import_graph_def(graph_def, name="TestedNet", input_map = {input_layer: image_batch},  return_elements = [output_layer])

        graph = sess.graph

        labels = tf.reshape(label_batch, shape=[batch_size])
        probs = tf.reshape(predictions, shape=[batch_size, 11])

        mval = tf.argmax(probs, 1, output_type=tf.int32)
        equality = tf.equal(mval, labels)
        accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))

        if write_log:
            writer = tf.summary.FileWriter(logdir)
            writer.add_graph(sess.graph)
            merged = tf.summary.merge_all()

        acc, cnt = 0, 0
        bacc = 0
        for it in range(iterations):
            start = time.time()
            amean = sess.run(accuracy)

            tf.logging.info("Mean accuracy (run %d): %.5f in %f sec", it, amean, time.time() - start)
            bacc += amean

        tf.logging.info("results;mult=%s;tune=%s;accuracy=%.3f" % (str(mult), str(tune), float(bacc) / float(it + 1)))
        return float(bacc) / float(it + 1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()  
    parser.add_argument(
        'graph',
        type=str,
        help='Frozen graph.')

    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='The directory where the CIFAR-10 input data is stored.')

    parser.add_argument(
        '--batch-size',
        type=int,
        default=1000,
        help='Number of images in the batch.')

        
    parser.add_argument(
        '--iterations',
        type=int,
        default=10,
        help='Number of iterations of batches, batch_size * iterations <= data_size; batch_size >> iterations.')

    parser.add_argument(  
        '--mult',
        type=str,
        required=True,
        help='Name of multiplier; (eg mul8u_1JFF) - for more see ../axqconv/axmult/*.c.')

    parser.add_argument(  
        '--tune',
        type=bool,
        help='Use weight adaptation algoritm.')
    

    args = parser.parse_args()
    main(**vars(args))

    


