# TensorFlow Approximate Layers for GPUs

## Overview
This library extends TensorFlow library by **ApproxConv2DWithMinMaxVars** layer that implements **Conv2D** with approximate multiplier. The layer is intended to be used together with **FakeQuantWithMinMaxVars** layers (on inputs) to experiment with approximate/quantized convolutions in FP32 CNNs.

Although the TF layer **ApproxConv2DWithMinMaxVars** can be used directly we also provide Keras wrapper of the layer. The Keras wrapper **FakeApproxConv2D** ([python/keras/layers/fake_approx_convolutional.py](python/keras/layers/fake_approx_convolutional.py)) includes **FakeQuantWithMinMaxVars** to quantize the inputs. Both input and filter ranges (used for quantization) are computed for each batch independently.

## Usage
The library is implemented as dynamically loaded plugin for TensorFlow. The library can be either downloaded as CUDA enabled Singularity container [tf-approximate-gpu.sif](https://ehw.fit.vutbr.cz/tf-approximate/tf-approximate-gpu.sif) based on "tensorflow/tensorflow:latest-gpu-py3" image or built from sources

```bash
mkdir build
cd build
cmake ..
make
```

The prerequisites to build the library is working installation of CUDA SDK (10.0+) and TensorFlow (2.1.0+) with GPU (CUDA) support enabled. The build system provides switch to disable CUDA support, which causes the layer to fallback to **SLOW** CPU implementation: cmake -DTFAPPROX_ALLOW_GPU_CONV=OFF ..

Finally built library can be used as
```python
import tensorflow as tf
approx_op_module = tf.load_op_library('libApproxGPUOpsTF.so')
```

## Example
An example provided in [examples](examples) shows usage of Keras layer **FakeApproxConv2D**, which allows to simulate convolution using approximate 8x8 bit multiplier defined as arbitrary binary table.

The first step is to train the network using FP32 as usual and store resulting weights:
```bash
python ../examples/fake_approx_train.py
```

The approximation then can be experimented with by replacing **Conv2D** with **FakeApproxConv2D** layers in the Keras model and loading trained weights back:
```bash
python ../examples/fake_approx_eval.py --mtab_file ../examples/axmul_8x8/mul8u_L40.bin
```
Leaving the argument "mtab_file" (multiplication table file) out will cause **FakeApproxConv2D** layers to use accurate 8x8 bit multiplication.

The prebuilt Singularity container can be used by adding following prefix to previous commands:
```bash
singularity run tf-approximate-gpu.sif
```
