# TensorFlow Approximate Layers

## Overview
This library extends TensorFlow library by appoximate convolutional layers. The approximation is introduced by employing arbitrary approximate multuiplier (e.g. from the EvoApproxLib library of approximate components).

## CPU version
This repository provides two versions of the approximate layers. The first is based on a simple CPU implementation from the TensorFlow library and is located in [cpu](CPU) folder. In this version, a **AxConv2D** layer is implemented, that extends **QuantizedConv2D** layer with approximate multiplier. The application examples are given in the [cpu/README.md](README) file.

For more details see paper: [arXiv:1907.07229](https://arxiv.org/abs/1907.07229). If you use this library in your work, please use a following reference

    MRAZEK Vojtěch, VASICEK Zdenek, SEKANINA Lukas, HANIF Muhammad Abdullah and SHAFIQUE Muhammad. ALWANN: Automatic Layer-Wise Approximation of Deep Neural Network Accelerators without Retraining. To appear in ICCAD'19 2019.

```bibtex
@article {
    author = {Vojtech Mrazek and Zdenek Vasicek and Lukas Sekanina and Muhammad Abdullah Hanif and Muhammad Shafique,
   title = {ALWANN: Automatic Layer-Wise Approximation of Deep Neural
	Network Accelerators without Retraining},
   pages = {8},
   year = {2019},,
   booktitle = {ICCAD'19 (to appear)}
}
```

## GPU version
Since the CPU version uses a basic implementation of the convolution, it is not very effective. However, this version was reimplemented in order to achieve a performance of the inference more comparable to the accurate implementation (200x speedup w.r.t. CPU version).


## Speed comparison
| Tables        | Are           | Cool  |
| ------------- |:-------------:| -----:|
| col 3 is      | right-aligned | $1600 |
| col 2 is      | centered      |   $12 |
| zebra stripes | are neat      |    $1 |



| 2x DNN parameters |  | 2x Accurate Conv2D | | 2x Approximate AxConv2D ||   2x Approx. overhead | |  2x Speedup GPU vs CPU |
| DNN | L | # MACs | CPU | GPU | CPU |  GPU |  CPU |  GPU |  Accurate |  Approximate |
| --------- | --------- | --------- | --------- | --------- | CPU |  GPU |  CPU |  GPU |  Accurate |  Approximate |


% generovano skriptem na gitu: scripts/tab_all.py
ResNet-8 |   7 | $ 21\times10^6$ | \tti{0.2}{4.4}|  \tti{1.8}{0.2} | \tti{0.2}{341} | \tti{1.7}{1.5} | 337 s  | 1.2 s | 2.3 $\times$  | \mainResult{106.8 $\times$}  \\
ResNet-14 |  13 | $ 35\times10^6$ | \tti{0.2}{7.4}|  \tti{1.9}{0.3} | \tti{0.2}{724} | \tti{1.8}{3.1} | 718 s  | 2.7 s | 3.5 $\times$  | \mainResult{148.8 $\times$}  \\
ResNet-20 |  19 | $ 49\times10^6$ | \tti{0.2}{10.4}|  \tti{1.8}{0.5} | \tti{0.2}{1105} | \tti{1.8}{4.7} | 1096 s  | 4.3 s | 4.7 $\times$  | \mainResult{170.2 $\times$}  \\
ResNet-26 |  25 | $ 63\times10^6$ | \tti{0.2}{13.4}|  \tti{1.9}{0.6} | \tti{0.2}{1489} | \tti{1.8}{6.2} | 1477 s  | 5.6 s | 5.5 $\times$  | \mainResult{185.0 $\times$}  \\
ResNet-32 |  31 | $ 77\times10^6$ | \tti{0.3}{16.3}|  \tti{1.9}{0.7} | \tti{0.3}{1876} | \tti{1.9}{7.9} | 1861 s  | 7.3 s | 6.5 $\times$  | \mainResult{191.0 $\times$}  \\
ResNet-38 |  37 | $ 91\times10^6$ | \tti{0.3}{19.3}|  \tti{1.9}{0.8} | \tti{0.3}{2259} | \tti{1.9}{9.4} | 2241 s  | 8.6 s | 7.3 $\times$  | \mainResult{200.1 $\times$}  \\
ResNet-44 |  43 | $106\times10^6$ | \tti{0.3}{22.3}|  \tti{1.9}{0.9} | \tti{0.3}{2640} | \tti{2.0}{10.9} | 2620 s  | 10.0 s | 8.0 $\times$  | \mainResult{205.6 $\times$}  \\
ResNet-50 |  49 | $120\times10^6$ | \tti{0.3}{25.2}|  \tti{1.9}{1.1} | \tti{0.3}{3025} | \tti{2.0}{12.6} | 3003 s  | 11.7 s | 8.6 $\times$  | \mainResult{207.2 $\times$}  \\
ResNet-56 |  55 | $134\times10^6$ | \tti{0.3}{28.1}|  \tti{1.9}{1.2} | \tti{0.3}{3409} | \tti{2.0}{13.9} | 3384 s  | 12.8 s | 9.2 $\times$  | \mainResult{214.4 $\times$}  \\
ResNet-62 |  61 | $148\times10^6$ | \tti{0.3}{31.1}|  \tti{1.9}{1.3} | \tti{0.3}{3796} | \tti{2.3}{15.5} | 3767 s  | 14.7 s | 10.0 $\times$  | \mainResult{213.2 $\times$}  \\
\bottomrule