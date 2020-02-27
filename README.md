# TensorFlow Approximate Layers

## Overview
This library extends TensorFlow library providing Approximate Convolutional (ApproxConv) layers, i.e. layers with reduced precision (typically 8 bits) implemented using approximate circuits (multipliers). The proposed layer enables to specify via parameter which approximate multiplier should be used (e.g. a multiplier from the [EvoApproxLib](//github.com/ehw-fit/evoapproxlib)). To maximize the throughput, the layer expects availability of a model of approximate multiplier in form of a Truth Table.



## Accelerated version (TensorFlow 2, GPU, optionally CPU)
This library extends TensorFlow library by ApproxConv2DWithMinMaxVars layer that implements Conv2D with approximate multiplier. The layer is intended to be used together with FakeQuantWithMinMaxVars layers (on inputs) to experiment with approximate/quantized convolutions in FP32 CNNs. The code can be executed on GPU or CPU but it is recommended to use a GPU to maximize the throughput. 


![Application overview](overview.png)

This is the most recent version of the approximate layers for TensorFlow. This implementation provides ~ 200x speedup with respect to the previous CPU-based version. We published the source codes as well as *docker or singularity container* with pre-build TensorFlow and the libraries for NVIDIA GPUs. The source codes and application examples are given in the [tf2](tf2) folder. For more details please see the paper [arXiv:2002.09481](https://arxiv.org/abs/2002.09481)

### Performance of the accelerated version
![Speed comparison](gpu_speedup.png)
Note that the evaluation was performed on Intel Xeon E5-2620 CPU equipped with NVIDIA GTX 1080 GPU, TensorFlow 1.X, and NVIDIA CUDA Toolkit 10.1.

*F. Vaverka, V. Mrazek, Z. Vasicek and L. Sekanina. __"TFApprox: Towards a Fast Emulation of DNN Approximate Hardware Accelerators on GPU"__. 2020 Design, Automation and Test in Europe Conference (DATE), Grenoble, FR, 2020.*


```bibtex
@INPROCEEDINGS{8942068,
    author={F. {Vaverka} and V. {Mrazek} and Z. {Vasicek} and L. {Sekanina} and M. A. {Hanif} and M. {Shafique}},
    booktitle={2020 Design, Automation and Test in Europe Conference (DATE)},
    title={TFApprox: Towards a Fast Emulation of DNN Approximate Hardware Accelerators on GPU},
    year={2020},
    volume={},
    number={},
    pages={4},
}
```


## Basic implementation (TensorFlow 1.14, CPU only)
This repository provides two versions of the approximate layers. The first is based on a simple CPU implementation from the TensorFlow library and is located in [tf1](tf1) folder. In this version, a **AxConv2D** layer is implemented, that extends **QuantizedConv2D** layer with approximate multiplier. The basic usage is shown in the [README](tf1/README.md) file.

For more details see paper: [10.1109/ICCAD45719.2019.8942068](https://dx.doi.org/10.1109/ICCAD45719.2019.8942068) or [arXiv:1907.07229](https://arxiv.org/abs/1907.07229) . If you use this library in your work, please use a following reference

*V. Mrazek, Z. Vasicek, L. Sekanina, M. A. Hanif and M. Shafique, __"ALWANN: Automatic Layer-Wise Approximation of Deep Neural Network Accelerators without Retraining,"__ 2019 IEEE/ACM International Conference on Computer-Aided Design (ICCAD), Westminster, CO, USA, 2019, pp. 1-8.*

```bibtex
@INPROCEEDINGS{8942068,
    author={V. {Mrazek} and Z. {Vasicek} and L. {Sekanina} and M. A. {Hanif} and M. {Shafique}},
    booktitle={2019 IEEE/ACM International Conference on Computer-Aided Design (ICCAD)},
    title={ALWANN: Automatic Layer-Wise Approximation of Deep Neural Network Accelerators without Retraining},
    year={2019},
    volume={},
    number={},
    pages={1-8},
    keywords={approximate computing;deep neural networks;computational path;ResNet;CIFAR-10},
    doi={10.1109/ICCAD45719.2019.8942068},
    ISSN={1933-7760},
    month={Nov},
}
```

