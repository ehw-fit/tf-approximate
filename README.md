# TensorFlow Approximate Layers

## Overview
This library extends TensorFlow library by **AxConv2D** layer that implements **QuantizedConv2D** layer with approximate multiplier. The approximate multipliers implemeted in C/C++ are stored in [axqcovn/axmult](axqconv/axmult) folder. The layers optionaly allow to use *weight tuning algorithm* that tries to modify weights of the layer to minimize mean arithmetic error of the multipliers.

For more details see paper: [ArXiv ...](). If you use this library in your work, please use a following reference

```bibtex
@article {
    tbd,
}
```

In contrast with standard implementation, the proposed layer introduces additional two parameters: AxMult(str) and AxTune(bool).

Note that the library was used for inference path only.

## Usage
The library is implmeneted symbolic library that is dynamically included. Firstly it must be build

```bash
cd axqconv
make
```

Then it can be included to the run.

```python
import tensorflow as tf
tf.load_op_library('../axqconv/axqconv.so')
```

## Example
An example is given in [example](example) folder. It approximates ResNet neural network trained for CIFAR-10 dataset.

Firstly, the dataset must be downloaded and preprocessed.

    python generate_cifar10_tfrecords.py --data-dir=${PWD}/cifar-10-data

Then, the frozen network must be quantized and the layer replaced by approximate implementation. Note that this step can be skipped because the quantized ResNet-8  (resnet_8_quant_ax.pb) is included in cifar-10-graphs folder.

```bash
tensorflow/bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
            --in_graph=${fn}.pb \
            --out_graph=${fn}_quant_ax.pb \
            --outputs='resnet/tower_0/fully_connected/dense/BiasAdd:0' \
            --inputs='input' \
            --transforms='
            add_default_attributes
            strip_unused_nodes(type=float, shape="1,299,299,3")
            remove_nodes(op=Identity, op=CheckNumerics)
            fold_constants(ignore_errors=true)
            fold_batch_norms
            quantize_weights
            quantize_nodes
            strip_unused_nodes
            sort_by_execution_order
            rename_op(old_op_name=QuantizedConv2D,new_op_name=AxConv2D)'
```

The example script cifar10_ax_inference.py approximates input NN with the same multiplier (uniform structure). 

```bash
# usage: cifar10_ax_inference.py [-h] --data-dir DATA_DIR
#                                [--batch-size BATCH_SIZE]
#                                [--iterations ITERATIONS] --mult MULT
#                                [--tune TUNE]
#                                graph
# 
# positional arguments:
#   graph                 Frozen graph.
# 
# optional arguments:
#   -h, --help            show this help message and exit
#   --data-dir DATA_DIR   The directory where the CIFAR-10 input data is stored.
#   --batch-size BATCH_SIZE
#                         Number of images in the batch.
#   --iterations ITERATIONS
#                         Number of iterations of batches, batch_size *
#                         iterations <= data_size; batch_size >> iterations.
#   --mult MULT           Name of multiplier; (eg mul8u_1JFF) - for more see
#                         ../axqconv/axmult/*.c.
#   --tune TUNE           Use weight adaptation algoritm.

python cifar10_ax_inference.py cifar-10-graphs/resnet_8_quant_ax.pb --data-dir cifar-10-data/ --tune true --mult mul8u_1JFF
# Expected output:
# INFO:tensorflow:resnet/tower_0/conv2d/Conv2D/eightbit = mul8u_1JFF (tune = True)
# INFO:tensorflow:resnet/tower_0/stage/residual_v1/conv2d/Conv2D/eightbit = mul8u_1JFF (tune = True)
# INFO:tensorflow:resnet/tower_0/stage/residual_v1/conv2d_1/Conv2D/eightbit =  mul8u_1JFF (tune = True)
# INFO:tensorflow:resnet/tower_0/stage_1/residual_v1/conv2d/Conv2D/eightbit =  mul8u_1JFF (tune = True)
# INFO:tensorflow:resnet/tower_0/stage_1/residual_v1/conv2d_1/Conv2D/eightbit = mul8u_1JFF (tune = True)
# INFO:tensorflow:resnet/tower_0/stage_2/residual_v1/conv2d/Conv2D/eightbit =  mul8u_1JFF (tune = True)
# INFO:tensorflow:resnet/tower_0/stage_2/residual_v1/conv2d_1/Conv2D/eightbit = mul8u_1JFF (tune = True)
# INFO:tensorflow:Mean accuracy (run 0): 0.83400 in 83.854471 sec
# INFO:tensorflow:Mean accuracy (run 1): 0.83400 in 82.986569 sec
# INFO:tensorflow:Mean accuracy (run 2): 0.82500 in 82.558742 sec
# INFO:tensorflow:Mean accuracy (run 3): 0.81700 in 82.657452 sec
# INFO:tensorflow:Mean accuracy (run 4): 0.84300 in 82.476791 sec
# INFO:tensorflow:Mean accuracy (run 5): 0.83500 in 83.072191 sec
# INFO:tensorflow:Mean accuracy (run 6): 0.83800 in 83.239446 sec
# INFO:tensorflow:Mean accuracy (run 7): 0.82900 in 82.900470 sec
# INFO:tensorflow:Mean accuracy (run 8): 0.83800 in 83.148296 sec
# INFO:tensorflow:Mean accuracy (run 9): 0.83300 in 83.009785 sec
# INFO:tensorflow:results;mult=mul8u_1JFF;tune=True;accuracy=0.833
```

The input multiplier can be one from following list (see [EvoApproxLib](https://ehw.fit.vutbr.cz/evoapproxlib) for more details)
  *  mul8u_1JFF - *accurate*
  *  mul8u_DM1
  *  mul8u_GS2
  *  mul8u_YX7
  *  mul8u_QKX
  *  mul8u_2HH
  *  mul8u_17C8
  *  mul8u_L40
  *  mul8u_150Q
  *  mul8u_13QR
  *  mul8u_7C1
  *  mul8u_2AC
  *  mul8u_199Z
  *  mul8u_12N4
  *  mul8u_1AGV
  *  mul8u_PKY
  *  mul8u_19DB
  *  mul8u_CK5
  *  mul8u_17QU
  *  mul8u_FTA
  *  mul8u_KEM
  *  mul8u_125K
  *  mul8u_18DU
  *  mul8u_JQQ
  *  mul8u_JV3
  *  mul8u_QJD
  *  mul8u_2P7
  *  mul8u_185Q
  *  mul8u_14VP
  *  mul8u_96D
  *  mul8u_NGR
  *  mul8u_ZFB
  *  mul8u_17KS
  *  mul8u_EXZ
  *  mul8u_Y48
  *  mul8u_1446
