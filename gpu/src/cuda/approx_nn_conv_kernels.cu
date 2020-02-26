//========== Copyright (c) 2019, Filip Vaverka, All rights reserved. =========//
//
// Purpose:     Implementation of (non-)approximated GEMM and Im2Col kernels
//              running on GPU/CUDA devices.
//
// $NoKeywords: $ApproxGPUOpsTF $approx_nn_conv_kernels.cu
// $Date:       $2019-12-19
//============================================================================//

#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "gpu_kernel_helper.h"
#include "approx_nn_conv_kernels.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

#define TILE_DIM 8

//----------------------------------------------------------------------------//
// Non-approximated kernels
//----------------------------------------------------------------------------//
template<typename T1, typename T2, typename T3>
__global__ void ApproxGemmCudaKernelCombined(size_t m, size_t n, size_t k,
                                             const T1 *a, size_t lda, const T2 *b, size_t ldb,
                                             T3 *c, size_t ldc);

template<typename T, typename AT>
struct ApproxConvGEMMKernelCombined<GPUDevice, T, AT, NullApproxOpType_t> {
    void operator()(const GPUDevice &d, const NullApproxOpType_t<GPUDevice, T, AT> &approxOp,
                    int m, int n, int k, T alpha,
                    const T *a, int lda,
                    const T *b, int ldb,
                    T beta, T *c, int ldc);
};

template<typename T, typename AT>
void ApproxConvGEMMKernelCombined<GPUDevice, T, AT, NullApproxOpType_t>::operator()(
    const GPUDevice &d, const NullApproxOpType_t<GPUDevice, T, AT> &,
    int m, int n, int k, T alpha,
    const T *a, int lda,
    const T *b, int ldb,
    T beta, T *c, int ldc)
{
    dim3 blockSize(8, 8, 1);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y, 1);
    
    ApproxGemmCudaKernelCombined<T, T, T>
        <<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, 
                                                 a, lda, 
                                                 b, ldb, 
                                                 c, ldc);
}

template struct ApproxConvGEMMKernelCombined<GPUDevice, Eigen::half, Eigen::half, NullApproxOpType_t>;
template struct ApproxConvGEMMKernelCombined<GPUDevice, float,       float,       NullApproxOpType_t>;
template struct ApproxConvGEMMKernelCombined<GPUDevice, double,      double,      NullApproxOpType_t>;

template<typename T1, typename T2, typename T3>
__global__ void ApproxGemmCudaKernelCombined(size_t m, size_t n, size_t k,
                                             const T1 *a, size_t lda, const T2 *b, size_t ldb,
                                             T3 *c, size_t ldc)
{
    T3 value(0);

    int Row = blockIdx.y*TILE_DIM + threadIdx.y;
    int Col = blockIdx.x*TILE_DIM + threadIdx.x;

    __shared__ T1 As[TILE_DIM][TILE_DIM];
    __shared__ T2 Bs[TILE_DIM][TILE_DIM];

    for (int i = 0; i < (TILE_DIM + k - 1)/TILE_DIM; ++i) {

         if (i*TILE_DIM + threadIdx.x < k && Row < m)
             As[threadIdx.y][threadIdx.x] = a[Row*lda + i*TILE_DIM + threadIdx.x];
         else
             As[threadIdx.y][threadIdx.x] = T1(0);

         if (i*TILE_DIM + threadIdx.y < k && Col < n)
             Bs[threadIdx.y][threadIdx.x] = b[(i*TILE_DIM + threadIdx.y)*ldb + Col];
         else
             Bs[threadIdx.y][threadIdx.x] = T2(0);

         __syncthreads();

         for (int n = 0; n < TILE_DIM; ++n)
             value += As[threadIdx.y][n] * Bs[n][threadIdx.x];

         __syncthreads();
    }

    if (Row < m && Col < n)
        c[((blockIdx.y * blockDim.y  + threadIdx.y)*ldc) +
           (blockIdx.x * blockDim.x) + threadIdx.x] = value;
}

template<typename T1, typename T2, typename T3>
__global__ void ApproxGemmCudaKernel(size_t m, size_t n, size_t k,
                                     const T1 *a, size_t lda, const T2 *b, size_t ldb,
                                     T3 *c, size_t ldc);

template<typename T>
struct ApproxConvGEMMKernel<GPUDevice, T, T, NullApproxOpType_t> {
    void operator()(const GPUDevice &d, const NullApproxOpType_t<GPUDevice, T, T> &,
                    int m, int n, int k, T alpha,
                    const T *a, int lda,
                    const T *b, int ldb,
                    const T *aCoeff, const T *bCoeffs,
                    T beta, T *c, int ldc);
};

template<typename T>
void ApproxConvGEMMKernel<GPUDevice, T, T, NullApproxOpType_t>::operator()(
    const GPUDevice &d, const NullApproxOpType_t<GPUDevice, T, T> &,
    int m, int n, int k, T alpha,
    const T *a, int lda,
    const T *b, int ldb,
    const T *aCoeff, const T *bCoeffs,
    T beta, T *c, int ldc)
{
    dim3 blockSize(8, 8, 1);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y, 1);
    
    ApproxGemmCudaKernel<T, T, T>
        <<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, 
                                                 a, lda, 
                                                 b, ldb, 
                                                 c, ldc);
}

template struct ApproxConvGEMMKernel<GPUDevice, Eigen::half, Eigen::half, NullApproxOpType_t>;
template struct ApproxConvGEMMKernel<GPUDevice, float,       float,       NullApproxOpType_t>;
template struct ApproxConvGEMMKernel<GPUDevice, double,      double,      NullApproxOpType_t>;

template<typename T1, typename T2, typename T3>
__global__ void ApproxGemmCudaKernel(size_t m, size_t n, size_t k,
                                     const T1 *a, size_t lda, const T2 *b, size_t ldb,
                                     T3 *c, size_t ldc)
{
    T3 value(0);

    int Row = blockIdx.y*TILE_DIM + threadIdx.y;
    int Col = blockIdx.x*TILE_DIM + threadIdx.x;

    __shared__ T1 As[TILE_DIM][TILE_DIM];
    __shared__ T2 Bs[TILE_DIM][TILE_DIM];

    for (int i = 0; i < (TILE_DIM + k - 1)/TILE_DIM; ++i) {

         if (i*TILE_DIM + threadIdx.x < k && Row < m)
             As[threadIdx.y][threadIdx.x] = a[Row*lda + i*TILE_DIM + threadIdx.x];
         else
             As[threadIdx.y][threadIdx.x] = T1(0);

         if (i*TILE_DIM + threadIdx.y < k && Col < n)
             Bs[threadIdx.y][threadIdx.x] = b[(i*TILE_DIM + threadIdx.y)*ldb + Col];
         else
             Bs[threadIdx.y][threadIdx.x] = T2(0);

         __syncthreads();

         for (int n = 0; n < TILE_DIM; ++n)
             value += As[threadIdx.y][n] * Bs[n][threadIdx.x];

         __syncthreads();
    }

    if (Row < m && Col < n)
        c[((blockIdx.y * blockDim.y  + threadIdx.y)*ldc) +
           (blockIdx.x * blockDim.x) + threadIdx.x] = value;
}


// Non-Approximated Image-to-Columns kernel
template<typename T, typename AT>
__global__ void ApproxConvIm2ColCudaKernel(const T *in,
                    int c, int w, int h, int ow, int oh,
                    int kw, int kh, int pw, int ph, int sw, int sh,
                    int dw, int dh, int po, int pc, AT *out);

template<typename T, typename AT>
struct ApproxConvIm2ColKernel<GPUDevice, T, AT, NullApproxOpType_t> {
    void operator()(const GPUDevice &d, const NullApproxOpType_t<GPUDevice, T, AT> &,
                    const T *in,
                    int c, int w, int h, int ow, int oh,
                    int kw, int kh, int pw, int ph, int sw, int sh,
                    int dw, int dh, int po, int pc, AT *out, T *);
};

template<typename T, typename AT>
void ApproxConvIm2ColKernel<GPUDevice, T, AT, NullApproxOpType_t>::operator()(
    const GPUDevice &d, const NullApproxOpType_t<GPUDevice, T, AT> &,
    const T *in,
    int c, int w, int h, int ow, int oh,
    int kw, int kh, int pw, int ph, int sw, int sh,
    int dw, int dh, int po, int pc, AT *out, T *)
{
    //unsigned pc = ow * oh;
    unsigned pl = kw * kh * c;
    
    unsigned blockSize = 256;
    unsigned gridSize  = (pc * pl + blockSize - 1) / blockSize;
    
    ApproxConvIm2ColCudaKernel<T>
        <<<gridSize, blockSize, 0, d.stream()>>>(in, c, w, h,
                                                 ow, oh,
                                                 kw, kh,
                                                 pw, ph,
                                                 sw, sh,
                                                 dw, dh,
                                                 po, pc, out);
}

template struct ApproxConvIm2ColKernel<GPUDevice, Eigen::half, Eigen::half, NullApproxOpType_t>;
template struct ApproxConvIm2ColKernel<GPUDevice, float,       float,       NullApproxOpType_t>;
template struct ApproxConvIm2ColKernel<GPUDevice, double,      double,      NullApproxOpType_t>;

template<typename T, typename AT>
__global__ void ApproxConvIm2ColCudaKernel(const T *in,
                    int c, int w, int h, int ow, int oh,
                    int kw, int kh, int pw, int ph, int sw, int sh,
                    int dw, int dh, int po, int pc, AT *out)
{
    //unsigned pc = ow * oh;
    unsigned pl = kw * kh * c;
    
    for(unsigned tId = blockIdx.x * blockDim.x + threadIdx.x; tId < pc*pl; tId += blockDim.x * gridDim.x)
    {
        unsigned patchId = (tId + po*pl) / pl;
        unsigned outB    = (patchId / ow) / oh;
        unsigned outH    = (patchId / ow) % oh;
        unsigned outW    = patchId % ow;
        
        unsigned valueId = (tId + po*pl) % pl;
        unsigned offsetH = valueId / (kw * c);
        unsigned offsetW = (valueId / c) % kw;
        unsigned offsetC = valueId % c;
        
        unsigned inH = outH * sh - ph + offsetH * dh;
        unsigned inW = outW * sw - pw + offsetW * dw;
        
        if(inH >= 0 && inW >= 0 && inH < h && inW < w)
            out[tId] = in[((outB * h + inH) * w + inW) * c + offsetC];
        else
            out[tId] = T(0);
    }
}

// Non-Approximated Filter correction coefficients kernels
template<typename T, typename AT>
struct ApproxFilterCorrCoeff<Eigen::GpuDevice, T, AT, NullApproxOpType_t> {
    void operator()(const Eigen::GpuDevice &d, const NullApproxOpType_t<Eigen::GpuDevice, T, AT> &approxOp,
                    ConstTensor4<T> filterCoeffs, Flat<T> outCorrCoeffs);
};

template<typename T, typename AT>
void ApproxFilterCorrCoeff<Eigen::GpuDevice, T, AT, NullApproxOpType_t>::operator()(
    const Eigen::GpuDevice &d, const NullApproxOpType_t<Eigen::GpuDevice, T, AT> &, 
    ConstTensor4<T> filterCoeffs, Flat<T> outCorrCoeffs)
{
    // Result should not be used in "NullApproxOpType_t" case
    // Eigen::array<int, 3> dims = {0, 1, 2};
    // outCorrCoeffs.device(d) = filterCoeffs.sum(dims);
}

template struct ApproxFilterCorrCoeff<GPUDevice, Eigen::half, Eigen::half, NullApproxOpType_t>;
template struct ApproxFilterCorrCoeff<GPUDevice, float,       float,       NullApproxOpType_t>;
template struct ApproxFilterCorrCoeff<GPUDevice, double,      double,      NullApproxOpType_t>;

//----------------------------------------------------------------------------//
// Lookup table approximate kernels (8-bit inputs)
//----------------------------------------------------------------------------//

template<typename T, typename AT>
using GpuOpQuantProps_t = typename TableApproxOpType_t<GPUDevice, T, AT>::OpQuantProps_t;

// Approximated Filter correction coefficients kernels
template<typename T, typename AT>
struct ApproxFilterCorrCoeff<Eigen::GpuDevice, T, AT, TableApproxOpType_t> {
    void operator()(const Eigen::GpuDevice &d, const TableApproxOpType_t<Eigen::GpuDevice, T, AT> &approxOp,
                    ConstTensor4<T> filterCoeffs, Flat<T> outCorrCoeffs);
};

template<typename T, typename AT>
void ApproxFilterCorrCoeff<Eigen::GpuDevice, T, AT, TableApproxOpType_t>::operator()(
    const Eigen::GpuDevice &d, const TableApproxOpType_t<Eigen::GpuDevice, T, AT> &approxOp, 
    ConstTensor4<T> filterCoeffs, Flat<T> outCorrCoeffs)
{
    Eigen::array<int, 3> dims = {0, 1, 2};
    outCorrCoeffs.device(d) = filterCoeffs.sum(dims);
}

template struct ApproxFilterCorrCoeff<GPUDevice, float, uint8, TableApproxOpType_t>;

// Approximated GEMM Combined
template<typename T, typename AT>
__global__ void ApproxGemmCudaKernelCombined(size_t m, size_t n, size_t k,
                                             const T *a, size_t lda, const T *b, size_t ldb,
                                             cudaTextureObject_t lookupTable,
                                             T *c, size_t ldc, const GpuOpQuantProps_t<T, AT> quantProps);

template<typename T, typename AT>
struct ApproxConvGEMMKernelCombined<GPUDevice, T, AT, TableApproxOpType_t> {
    void operator()(const GPUDevice &d, const TableApproxOpType_t<GPUDevice, T, AT> &approxOp,
                    int m, int n, int k, T alpha,
                    const T *a, int lda,
                    const T *b, int ldb,
                    T beta, T *c, int ldc);
};

template<typename T, typename AT>
void ApproxConvGEMMKernelCombined<GPUDevice, T, AT, TableApproxOpType_t>::operator()(
    const GPUDevice &d, const TableApproxOpType_t<GPUDevice, T, AT> &approxOp,
    int m, int n, int k, T alpha,
    const T *a, int lda,
    const T *b, int ldb,
    T beta, T *c, int ldc)
{
    dim3 blockSize(8, 8, 1);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y, 1);
    
    cudaTextureObject_t lookupTableTexObj = 0;
    
    ApproxGemmCudaKernelCombined<T, AT>
        <<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, 
                                                 a, lda, 
                                                 b, ldb, 
                                                 lookupTableTexObj,
                                                 c, ldc, approxOp.quantProps);
}

template struct ApproxConvGEMMKernelCombined<GPUDevice, float, uint8, TableApproxOpType_t>;

template<typename T, typename AT>
__global__ void ApproxGemmCudaKernelCombined(size_t m, size_t n, size_t k,
                                             const T *a, size_t lda, const T *b, size_t ldb,
                                             cudaTextureObject_t lookupTable,
                                             T *c, size_t ldc, const GpuOpQuantProps_t<T, AT> quantProps)
{
    T value(0);

    int Row = blockIdx.y*TILE_DIM + threadIdx.y;
    int Col = blockIdx.x*TILE_DIM + threadIdx.x;

    __shared__ uint As[TILE_DIM][TILE_DIM];
    __shared__ uint Bs[TILE_DIM][TILE_DIM];

    for (int i = 0; i < (TILE_DIM + k - 1)/TILE_DIM; ++i) {

         if (i*TILE_DIM + threadIdx.x < k && Row < m)
             As[threadIdx.y][threadIdx.x] = AT((a[Row*lda + i*TILE_DIM + threadIdx.x] - quantProps.input.offset) * quantProps.input.invScale + T(0.5));
         else
             As[threadIdx.y][threadIdx.x] = 0;

         if (i*TILE_DIM + threadIdx.y < k && Col < n)
             Bs[threadIdx.y][threadIdx.x] = AT((b[(i*TILE_DIM + threadIdx.y)*ldb + Col] - quantProps.filter.offset) * quantProps.filter.invScale + T(0.5));
         else
             Bs[threadIdx.y][threadIdx.x] = 0;

         __syncthreads();

         for (int n = 0; n < TILE_DIM; ++n)
         {
             uint tableFetchIdx = (As[threadIdx.y][n] << 8) | Bs[n][threadIdx.x];
             value += float(tex1Dfetch<ushort>(lookupTable, tableFetchIdx));
         }

         __syncthreads();
    }

    if (Row < m && Col < n)
    {
        float patchCorrection  = 1.0f;
        float filterCorrection = 1.0f;
        
        value = value * quantProps.s1xS2 + quantProps.m2 * patchCorrection +
            quantProps.m1 * filterCorrection - T(k) * quantProps.m1xM2;
        
        c[((blockIdx.y * blockDim.y  + threadIdx.y)*ldc) +
           (blockIdx.x * blockDim.x) + threadIdx.x] = value;
    }
}

// Approximated GEMM
template<typename T, typename AT>
__global__ void ApproxGemmCudaKernel(int m, int n, int k,
    const AT *a, int lda,
    const T *b, int ldb,
    cudaTextureObject_t lookupTable,
    const T *patchSums, const T *filterSums,
    T *c, int ldc, const GpuOpQuantProps_t<T, AT> quantProps);

template<typename T, typename AT>
struct ApproxConvGEMMKernel<Eigen::GpuDevice, T, AT, TableApproxOpType_t> {
    void operator()(const Eigen::GpuDevice &d, const TableApproxOpType_t<Eigen::GpuDevice, T, AT> &approxOp,
                    int m, int n, int k, T alpha,
                    const AT *a, int lda,
                    const T *b, int ldb,
                    const T *aCoeffs, const T *bCoeffs,
                    T beta, T *c, int ldc);
};

template<typename T, typename AT>
void ApproxConvGEMMKernel<Eigen::GpuDevice, T, AT, TableApproxOpType_t>::operator()(
    const Eigen::GpuDevice &d, const TableApproxOpType_t<Eigen::GpuDevice, T, AT> &approxOp,
    int m, int n, int k, T alpha,
    const AT *a, int lda,
    const T *b, int ldb,
    const T *aCoeffs, const T *bCoeffs,
    T beta, T *c, int ldc)
{
    dim3 blockSize(8, 8, 1);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y, 1);
    
    //std::cout << "TEX: " << approxOp.GetLookupData() << std::endl;
    
    ApproxGemmCudaKernel<T, AT>
        <<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, 
                                                 a, lda, 
                                                 b, ldb, 
                                                 approxOp.GetLookupData(),
                                                 aCoeffs, bCoeffs,
                                                 c, ldc, approxOp.quantProps);
}

template<typename T, typename AT>
__global__ void ApproxGemmCudaKernel(int m, int n, int k,
    const AT *a, int lda,
    const T *b, int ldb,
    cudaTextureObject_t lookupTable,
    const T *patchSums, const T *filterSums,
    T *c, int ldc, const GpuOpQuantProps_t<T, AT> quantProps)
{
    T value = T(0);

    int Row = blockIdx.y*TILE_DIM + threadIdx.y;
    int Col = blockIdx.x*TILE_DIM + threadIdx.x;

    __shared__ uint As[TILE_DIM][TILE_DIM];
    __shared__ uint Bs[TILE_DIM][TILE_DIM];

    for (int i = 0; i < (TILE_DIM + k - 1)/TILE_DIM; ++i) {

         if (i*TILE_DIM + threadIdx.x < k && Row < m)
             As[threadIdx.y][threadIdx.x] = a[Row*lda + i*TILE_DIM + threadIdx.x];
         else
             As[threadIdx.y][threadIdx.x] = 0;

         if (i*TILE_DIM + threadIdx.y < k && Col < n)
             Bs[threadIdx.y][threadIdx.x] = AT(((b[(i*TILE_DIM + threadIdx.y)*ldb + Col] - quantProps.filter.offset) * quantProps.filter.invScale) + T(0.5));
         else
             Bs[threadIdx.y][threadIdx.x] = 0;

         __syncthreads();

         for (int n = 0; n < TILE_DIM; ++n)
         {
             //value += T(As[threadIdx.y][n] * Bs[n][threadIdx.x]);
             uint tableFetchIdx = (As[threadIdx.y][n] << 8) | Bs[n][threadIdx.x];
             value += float(tex1Dfetch<ushort>(lookupTable, tableFetchIdx));
         }

         __syncthreads();
    }

    if (Row < m && Col < n)
    {
        float patchCorrection  = patchSums[blockIdx.y * blockDim.y  + threadIdx.y];
        float filterCorrection = filterSums[blockIdx.x * blockDim.x + threadIdx.x];
        
        value = value * quantProps.s1xS2 + quantProps.m2 * patchCorrection +
            quantProps.m1 * filterCorrection - T(k) * quantProps.m1xM2;
        
        c[((blockIdx.y * blockDim.y  + threadIdx.y)*ldc) +
           (blockIdx.x * blockDim.x) + threadIdx.x] = value;
    }
}

template struct ApproxConvGEMMKernel<GPUDevice, float, uint8, TableApproxOpType_t>;

// Approximated Im-2-Col
template<typename T>
__device__ void PreScan(T value, volatile T *sdata, int tid, int n);

template<typename T, typename AT>
__global__ void ApproxConvIm2ColCudaKernel(const T *in, 
                                           int c, int w, int h, int ow, int oh,
                                           int kw, int kh, int pw, int ph, int sw, int sh,
                                           int dw, int dh, int po, int pc, AT *out, T *outCoeffs, 
                                           const GpuOpQuantProps_t<T, AT> quantProps);

template<typename T, typename AT>
struct ApproxConvIm2ColKernel<Eigen::GpuDevice, T, AT, TableApproxOpType_t> {
    void operator()(const Eigen::GpuDevice &d, const TableApproxOpType_t<Eigen::GpuDevice, T, AT> &approxOp,
                    const T *in,
                    int c, int w, int h, int ow, int oh,
                    int kw, int kh, int pw, int ph, int sw, int sh,
                    int dw, int dh, int po, int pc, AT *out, T *outCoeffs);
};

template<typename T, typename AT>
void ApproxConvIm2ColKernel<Eigen::GpuDevice, T, AT, TableApproxOpType_t>::operator()(
    const Eigen::GpuDevice &d, const TableApproxOpType_t<Eigen::GpuDevice, T, AT> &approxOp, 
    const T *in,
    int c, int w, int h, int ow, int oh,
    int kw, int kh, int pw, int ph, int sw, int sh,
    int dw, int dh, int po, int pc, AT *out, T *outCoeffs)
{
    unsigned pl = kw * kh * c;
    
    unsigned blockSize = 256;
    unsigned gridSize  = (pc * pl + blockSize - 1) / blockSize;
    
    cudaMemset(outCoeffs, 0, pc * sizeof(T));
    
    ApproxConvIm2ColCudaKernel<T, AT>
        <<<gridSize, blockSize, blockSize * sizeof(T), d.stream()>>>(in, c, w, h,
                                                                     ow, oh,
                                                                     kw, kh,
                                                                     pw, ph,
                                                                     sw, sh,
                                                                     dw, dh,
                                                                     po, pc, out, outCoeffs, approxOp.quantProps);
}

template<typename T>
__device__ void PreScan(T value, volatile T *sdata, int tid, int n)
{
    int offset = 1;
    sdata[tid] = value;
    
    for(int d = n >> 1; d > 0; d >>= 1)
    {
        __syncthreads();
        
        if(tid < d)
        {
            int ai = offset*(2*tid+1)-1;
            int bi = offset*(2*tid+2)-1;
            
            sdata[bi] += sdata[ai];
        }
        
        offset *= 2;
    }
    
    if(tid == 0)
        sdata[n - 1] = T(0);
    
    for(int d = 1; d < n; d *= 2)
    {
        offset >>= 1;
        __syncthreads();
        if(tid < d)
        {
            int ai = offset*(2*tid+1)-1;
            int bi = offset*(2*tid+2)-1;
            
            float t = sdata[ai];
            sdata[ai] = sdata[bi];
            sdata[bi] += t;
        }
    }
    
    __syncthreads();
}

template<typename T, typename AT>
__global__ void ApproxConvIm2ColCudaKernel(const T *in, 
                                           int c, int w, int h, int ow, int oh,
                                           int kw, int kh, int pw, int ph, int sw, int sh,
                                           int dw, int dh, int po, int pc, AT *out, T *outCoeffs, 
                                           const GpuOpQuantProps_t<T, AT> quantProps)
{
    extern __shared__ float sdata[];
    unsigned pl = kw * kh * c;
    
    for(unsigned tId = blockIdx.x * blockDim.x + threadIdx.x; tId < pc*pl; tId += blockDim.x * gridDim.x)
    {
        unsigned patchId = (tId + po*pl) / pl;
        unsigned outB    = (patchId / ow) / oh;
        unsigned outH    = (patchId / ow) % oh;
        unsigned outW    = patchId % ow;
        
        unsigned valueId = (tId + po*pl) % pl;
        unsigned offsetH = valueId / (kw * c);
        unsigned offsetW = (valueId / c) % kw;
        unsigned offsetC = valueId % c;
        
        unsigned inH = outH * sh - ph + offsetH * dh;
        unsigned inW = outW * sw - pw + offsetW * dw;
        
        
        T value = T(0);
        
        if(inH >= 0 && inW >= 0 && inH < h && inW < w)
            value = in[((outB * h + inH) * w + inW) * c + offsetC];
        out[tId] = AT(((value - quantProps.input.offset) * quantProps.input.invScale) + T(0.5));
        
        PreScan<T>(value, sdata, threadIdx.x, blockDim.x);
        
        if((valueId == pl - 1) || (threadIdx.x == blockDim.x - 1))
        {
            T sumValue = sdata[threadIdx.x] + value;
            /*if(threadIdx.x > pl)
                sumValue -= sdata[threadIdx.x - valueId];*/
            if(threadIdx.x > valueId)
                sumValue -= sdata[threadIdx.x - valueId];
            
            atomicAdd(&outCoeffs[patchId - po], sumValue);
        }
    }
}

template struct ApproxConvIm2ColKernel<GPUDevice, float, uint8, TableApproxOpType_t>;

#endif // GOOGLE_CUDA
