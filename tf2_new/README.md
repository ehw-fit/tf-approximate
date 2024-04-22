# TensorFlow Approximate Layers for GPUs
## Modules
```bash
ml CUDA/10.2.89-GCC-8.3.0-2.32 CMake/3.15.3-GCCcore-8.3.0 Anaconda3
```
## Build
```bash
mkdir build
cd build
cmake .. -DTFAPPROX_CUDA_ARCHS="75"
make
```
Note: Table with CUDA GPUs [on Wikipedia](https://en.wikipedia.org/wiki/CUDA#GPUs_supported).

## Test numpy
Code for dump of data of layer "l" in file test\_numpy\_keras.py

```python
fn = K.function([model.input], [model.layers[l].input, model.layers[l].output])
for b in [0, 1]:
    res = {}
    res["input"], res["output"] = fn([x_test[(b) * 128: (b+1) * 128]])
    for i, j in enumerate(model.layers[l].weights):
        res[f"weights_{i}"] = j
    res[f"bias"] = model.layers[l].bias

    res[f"filters"] = model.layers[l].filters
    res[f"kernel_size"] = model.layers[l].kernel_size
    
    np.savez_compressed(f"tmp/layer_{l}_batch_{b}.npz", 
        **res)
```
