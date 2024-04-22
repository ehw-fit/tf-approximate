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
#include "approx_ops_quant_data.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#define GEMM_TILE_DIM 8

using namespace tensorflow;
using GPUDevice = Eigen::GpuDevice;

//----------------------------------------------------------------------------//
// Non-approximated kernels
//----------------------------------------------------------------------------//
template<typename T1, typename T2, typename T3>
__global__ void ApproxGemmCudaKernelCombined(size_t m, size_t n, size_t k,
                                             const T1 *a, size_t lda, const T2 *b, size_t ldb,
                                             T3 *c, size_t ldc, int totalBlocksCount);

template<typename T, typename AT>
struct ApproxConvGEMMKernelCombined<NullApproxConvOpQuantData<GPUDevice, T, AT> > {
    void operator()(const GPUDevice &d, const NullApproxConvOpQuantData<GPUDevice, T, AT> &approxOpData,
                    int m, int n, int k, T alpha,
                    const T *a, int lda,
                    const T *b, int ldb,
                    T beta, T *c, int ldc);
};

template<typename T, typename AT>
void ApproxConvGEMMKernelCombined<NullApproxConvOpQuantData<GPUDevice, T, AT> >::operator()(
    const GPUDevice &d, const NullApproxConvOpQuantData<GPUDevice, T, AT> &,
    int m, int n, int k, T alpha,
    const T *a, int lda,
    const T *b, int ldb,
    T beta, T *c, int ldc)
{
    const dim3 blockSize(GEMM_TILE_DIM, GEMM_TILE_DIM, 1);
    const int numBlocksX = ((n + blockSize.x - 1) / blockSize.x);
    const int numBlocksY = ((m + blockSize.y - 1) / blockSize.y);
    int numBlocks  = numBlocksX * numBlocksY;

    const int maxBlocks = d.getNumGpuMultiProcessors() * 
        d.maxGpuThreadsPerMultiProcessor() / (blockSize.x*blockSize.y);
    
    if(numBlocks > maxBlocks)
        numBlocks = maxBlocks;
    
    dim3 gridSize(numBlocks, 1, 1);
    
    ApproxGemmCudaKernelCombined<T, T, T>
        <<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, 
                                                 a, lda, 
                                                 b, ldb, 
                                                 c, ldc, numBlocksX*numBlocksY);
}

template struct ApproxConvGEMMKernelCombined<NullApproxConvOpQuantData<GPUDevice, Eigen::half, Eigen::half> >;
template struct ApproxConvGEMMKernelCombined<NullApproxConvOpQuantData<GPUDevice,       float,       float> >;
template struct ApproxConvGEMMKernelCombined<NullApproxConvOpQuantData<GPUDevice,      double,      double> >;

template<typename T1, typename T2, typename T3>
__global__ void ApproxGemmCudaKernelCombined(size_t m, size_t n, size_t k,
                                             const T1 *a, size_t lda, const T2 *b, size_t ldb,
                                             T3 *c, size_t ldc, int totalBlocksCount)
{
    __shared__ T1 As[GEMM_TILE_DIM][GEMM_TILE_DIM];
    __shared__ T2 Bs[GEMM_TILE_DIM][GEMM_TILE_DIM];

    const int numTilesX = (n + blockDim.x - 1) / blockDim.x;
    for(int tileIdx = blockIdx.x; tileIdx < totalBlocksCount; tileIdx += gridDim.x)
    {
        const int tileY = tileIdx / numTilesX;
        const int tileX = tileIdx % numTilesX;

        T3 value(0);

        int Row = tileY * GEMM_TILE_DIM + threadIdx.y;
        int Col = tileX * GEMM_TILE_DIM + threadIdx.x;

        for (int i = 0; i < (GEMM_TILE_DIM + k - 1) / GEMM_TILE_DIM; ++i)
        {
            if (i*GEMM_TILE_DIM + threadIdx.x < k && Row < m)
                As[threadIdx.y][threadIdx.x] = a[Row*lda + i*GEMM_TILE_DIM + threadIdx.x];
            else
                As[threadIdx.y][threadIdx.x] = T1(0);

            if (i*GEMM_TILE_DIM + threadIdx.y < k && Col < n)
                Bs[threadIdx.y][threadIdx.x] = b[(i*GEMM_TILE_DIM + threadIdx.y)*ldb + Col];
            else
                Bs[threadIdx.y][threadIdx.x] = T2(0);

            __syncthreads();

            for (int n = 0; n < GEMM_TILE_DIM; ++n)
                value += As[threadIdx.y][n] * Bs[n][threadIdx.x];

            __syncthreads();
        }

        if (Row < m && Col < n)
            c[((tileY * blockDim.y  + threadIdx.y)*ldc) +
               (tileX * blockDim.x) + threadIdx.x] = value;
    }
}

template<typename T1, typename T2, typename T3>
__global__ void ApproxGemmCudaKernel(size_t m, size_t n, size_t k,
                                     const T1 *a, size_t lda, const T2 *b, size_t ldb,
                                     T3 *c, size_t ldc, int totalBlocksCount);

template<typename T>
struct ApproxConvGEMMKernel<NullApproxConvOpQuantData<GPUDevice, T, T> > {
    void operator()(const GPUDevice &d, const NullApproxConvOpQuantData<GPUDevice, T, T> &,
                    int m, int n, int k, T alpha,
                    const T *a, int lda,
                    const T *b, int ldb,
                    T beta, T *c, int ldc);
};

template<typename T>
void ApproxConvGEMMKernel<NullApproxConvOpQuantData<GPUDevice, T, T> >::operator()(
    const GPUDevice &d, const NullApproxConvOpQuantData<GPUDevice, T, T> &,
    int m, int n, int k, T alpha,
    const T *a, int lda,
    const T *b, int ldb,
    T beta, T *c, int ldc)
{
    const dim3 blockSize(GEMM_TILE_DIM, GEMM_TILE_DIM, 1);
    const int numBlocksX = ((n + blockSize.x - 1) / blockSize.x);
    const int numBlocksY = ((m + blockSize.y - 1) / blockSize.y);
    int numBlocks  = numBlocksX * numBlocksY;

    const int maxBlocks = d.getNumGpuMultiProcessors() * 
        d.maxGpuThreadsPerMultiProcessor() / (blockSize.x*blockSize.y);
    
    if(numBlocks > maxBlocks)
        numBlocks = maxBlocks;
    
    dim3 gridSize(numBlocks, 1, 1);
    
    ApproxGemmCudaKernel<T, T, T>
        <<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, 
                                                 a, lda, 
                                                 b, ldb, 
                                                 c, ldc, numBlocksX * numBlocksY);
}

template struct ApproxConvGEMMKernel<NullApproxConvOpQuantData<GPUDevice, Eigen::half, Eigen::half> >;
template struct ApproxConvGEMMKernel<NullApproxConvOpQuantData<GPUDevice,       float,       float> >;
template struct ApproxConvGEMMKernel<NullApproxConvOpQuantData<GPUDevice,      double,      double> >;

template<typename T1, typename T2, typename T3>
__global__ void ApproxGemmCudaKernel(size_t m, size_t n, size_t k,
                                     const T1 *a, size_t lda, const T2 *b, size_t ldb,
                                     T3 *c, size_t ldc, int totalBlocksCount)
{
    const int numTilesX = (n + blockDim.x - 1) / blockDim.x;
    for(int tileIdx = blockIdx.x; tileIdx < totalBlocksCount; tileIdx += gridDim.x)
    {
        const int tileY = tileIdx / numTilesX;
        const int tileX = tileIdx % numTilesX;

        T3 value(0);

        int Row = tileY * GEMM_TILE_DIM + threadIdx.y;
        int Col = tileX * GEMM_TILE_DIM + threadIdx.x;

        __shared__ T1 As[GEMM_TILE_DIM][GEMM_TILE_DIM];
        __shared__ T2 Bs[GEMM_TILE_DIM][GEMM_TILE_DIM];

        for (int i = 0; i < (GEMM_TILE_DIM + k - 1)/GEMM_TILE_DIM; ++i) {

            if (i*GEMM_TILE_DIM + threadIdx.x < k && Row < m)
                As[threadIdx.y][threadIdx.x] = a[Row*lda + i*GEMM_TILE_DIM + threadIdx.x];
            else
                As[threadIdx.y][threadIdx.x] = T1(0);

            if (i*GEMM_TILE_DIM + threadIdx.y < k && Col < n)
                Bs[threadIdx.y][threadIdx.x] = b[(i*GEMM_TILE_DIM + threadIdx.y)*ldb + Col];
            else
                Bs[threadIdx.y][threadIdx.x] = T2(0);

            __syncthreads();

            for (int n = 0; n < GEMM_TILE_DIM; ++n)
                value += As[threadIdx.y][n] * Bs[n][threadIdx.x];

            __syncthreads();
        }

        if (Row < m && Col < n)
            c[((tileY * blockDim.y  + threadIdx.y)*ldc) +
               (tileX * blockDim.x) + threadIdx.x] = value;
    }
}


// Non-Approximated Image-to-Columns kernel
template<typename T, typename AT>
__global__ void ApproxConvIm2ColCudaKernel(const T *in,
                    int c, int w, int h, int ow, int oh,
                    int kw, int kh, int pw, int ph, int sw, int sh,
                    int dw, int dh, int po, int pc, AT *out);

template<typename T, typename AT>
struct ApproxConvIm2ColKernel<NullApproxConvOpQuantData<GPUDevice, T, AT> > {
    void operator()(const GPUDevice &d, const NullApproxConvOpQuantData<GPUDevice, T, AT>  &,
                    const T *in,
                    int c, int w, int h, int ow, int oh,
                    int kw, int kh, int pw, int ph, int sw, int sh,
                    int dw, int dh, int po, int pc, AT *out);
};

template<typename T, typename AT>
void ApproxConvIm2ColKernel<NullApproxConvOpQuantData<GPUDevice, T, AT> >::operator()(
    const GPUDevice &d, const NullApproxConvOpQuantData<GPUDevice, T, AT>  &,
    const T *in,
    int c, int w, int h, int ow, int oh,
    int kw, int kh, int pw, int ph, int sw, int sh,
    int dw, int dh, int po, int pc, AT *out)
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

template struct ApproxConvIm2ColKernel<NullApproxConvOpQuantData<GPUDevice, Eigen::half, Eigen::half> >;
template struct ApproxConvIm2ColKernel<NullApproxConvOpQuantData<GPUDevice,       float,       float> >;
template struct ApproxConvIm2ColKernel<NullApproxConvOpQuantData<GPUDevice,      double,      double> >;

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

//----------------------------------------------------------------------------//
// Lookup table approximate kernels (8-bit inputs)
//----------------------------------------------------------------------------//

template<typename T, typename AT>
using GpuOpQuantProps_t = typename TableApproxOpType_t<GPUDevice, T, AT>::OpQuantProps_t;

template<typename T, typename AT>
using GpuOpQuantPropsData_t = typename TableApproxConvOpQuantData<GPUDevice, T, AT>::OpQuantProps_t;

// Approximated GEMM Combined
template<typename T, typename AT>
__global__ void ApproxGemmCudaKernelCombined(size_t m, size_t n, size_t k,
                                             const T *a, size_t lda, const T *b, size_t ldb,
                                             cudaTextureObject_t lookupTable,
                                             T *c, size_t ldc, const GpuOpQuantPropsData_t<T, AT> quantProps, 
                                             int totalBlocksCount);

template<typename T, typename AT>
struct ApproxConvGEMMKernelCombined<TableApproxConvOpQuantData<GPUDevice, T, AT> > {
    void operator()(const GPUDevice &d, const TableApproxConvOpQuantData<GPUDevice, T, AT> &approxOpData,
                    int m, int n, int k, T alpha,
                    const T *a, int lda,
                    const T *b, int ldb,
                    T beta, T *c, int ldc);
};

template<typename T, typename AT>
void ApproxConvGEMMKernelCombined<TableApproxConvOpQuantData<GPUDevice, T, AT> >::operator()(
    const GPUDevice &d, const TableApproxConvOpQuantData<GPUDevice, T, AT> &approxOpData,
    int m, int n, int k, T alpha,
    const T *a, int lda,
    const T *b, int ldb,
    T beta, T *c, int ldc)
{
    const typename TableApproxConvOpQuantData<GPUDevice, T, AT>::OpQuantProps_t &quantProps = approxOpData.GetQuantProps();
    const typename TableApproxConvOpQuantData<GPUDevice, T, AT>::ApproxOpType_t &approxOp   = approxOpData.GetApproxOp();

    const dim3 blockSize(8, 8, 1);
    const int numBlocksX = ((n + blockSize.x - 1) / blockSize.x);
    const int numBlocksY = ((m + blockSize.y - 1) / blockSize.y);
    int numBlocks  = numBlocksX * numBlocksY;

    const int maxBlocks = d.getNumGpuMultiProcessors() * 
        d.maxGpuThreadsPerMultiProcessor() / (blockSize.x*blockSize.y);
    
    if(numBlocks > maxBlocks)
        numBlocks = maxBlocks;
    
    dim3 gridSize(numBlocks, 1, 1);
    
    ApproxGemmCudaKernelCombined<T, AT>
        <<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, 
                                                 a, lda, 
                                                 b, ldb, 
                                                 approxOp.GetLookupData(),
                                                 c, ldc, quantProps, numBlocksX*numBlocksY);
}

template struct ApproxConvGEMMKernelCombined<TableApproxConvOpQuantData<GPUDevice, float, uint8> >;

/*template<typename T, typename AT>
__global__ void ApproxGemmCudaKernelCombined(size_t m, size_t n, size_t k,
                                             const T *a, size_t lda, const T *b, size_t ldb,
                                             cudaTextureObject_t lookupTable,
                                             T *c, size_t ldc, const GpuOpQuantProps_t<T, AT> quantProps)
{
    T value(0);
    T corrSum(0);

    int Row = blockIdx.y*GEMM_TILE_DIM + threadIdx.y;
    int Col = blockIdx.x*GEMM_TILE_DIM + threadIdx.x;

    __shared__ T    As[GEMM_TILE_DIM][GEMM_TILE_DIM];
    __shared__ uint Bs[GEMM_TILE_DIM][GEMM_TILE_DIM];

    for (int i = 0; i < (GEMM_TILE_DIM + k - 1)/GEMM_TILE_DIM; ++i) {

         if (i*GEMM_TILE_DIM + threadIdx.x < k && Row < m)
             As[threadIdx.y][threadIdx.x] = a[Row*lda + i*GEMM_TILE_DIM + threadIdx.x];
         else
             As[threadIdx.y][threadIdx.x] = 0;

         if (i*GEMM_TILE_DIM + threadIdx.y < k && Col < n)
             Bs[threadIdx.y][threadIdx.x] = ClampBitWidth<AT, 8>((b[(i*GEMM_TILE_DIM + threadIdx.y)*ldb + Col] - quantProps.filter.offset) * quantProps.filter.invScale + T(0.5));
         else
             Bs[threadIdx.y][threadIdx.x] = 0;

         __syncthreads();

         for (int n = 0; n < GEMM_TILE_DIM; ++n)
         {
             uint patchValue = ClampBitWidth<AT, 8>((As[threadIdx.y][n] - quantProps.input.offset) * quantProps.input.invScale + T(0.5));
             uint tableFetchIdx = (patchValue << 8) | Bs[n][threadIdx.x];
             value   += float(tex1Dfetch<ushort>(lookupTable, tableFetchIdx));
             corrSum += As[threadIdx.y][n];
         }

         __syncthreads();
    }

    if (Row < m && Col < n)
    {
        float patchCorrection = corrSum;
        float filterCorrection = quantProps.pFilterCorr[blockIdx.x * blockDim.x + threadIdx.x];
        
        value = value * quantProps.s1xS2 + quantProps.m2 * patchCorrection +
            quantProps.m1 * filterCorrection - T(k) * quantProps.m1xM2;
        
        c[((blockIdx.y * blockDim.y  + threadIdx.y)*ldc) +
           (blockIdx.x * blockDim.x) + threadIdx.x] = value;
    }
}*/

template<typename T, typename AT>
__global__ void ApproxGemmCudaKernelCombined(size_t m, size_t n, size_t k,
                                             const T *a, size_t lda, const T *b, size_t ldb,
                                             cudaTextureObject_t lookupTable,
                                             T *c, size_t ldc, const GpuOpQuantPropsData_t<T, AT> quantProps,
                                             int totalBlocksCount)
{
    const int numTilesX = (n + blockDim.x - 1) / blockDim.x;
    for(int tileIdx = blockIdx.x; tileIdx < totalBlocksCount; tileIdx += gridDim.x)
    {
        const int tileY = tileIdx / numTilesX;
        const int tileX = tileIdx % numTilesX;

        T value(0);
        T inputCorrSum(0);
        T filterCorrSum(0);

        int Row = tileY * GEMM_TILE_DIM + threadIdx.y;
        int Col = tileX * GEMM_TILE_DIM + threadIdx.x;

        const T filterOffset   = (quantProps.filterMode == 1 && Col < n) ? quantProps.pFilter[Col*3 + 2] : quantProps.pFilter[2];
        const T filterInvScale = (quantProps.filterMode == 1 && Col < n) ? quantProps.pFilter[Col*3 + 1] : quantProps.pFilter[1];
        const T s1xS2          = (quantProps.filterMode == 1 && Col < n) ? quantProps.pS1xS2[Col] : quantProps.pS1xS2[0];
        const T m1xM2          = (quantProps.filterMode == 1 && Col < n) ? quantProps.pM1xM2[Col] : quantProps.pM1xM2[0];

        __shared__ T As[GEMM_TILE_DIM][GEMM_TILE_DIM];
        __shared__ T Bs[GEMM_TILE_DIM][GEMM_TILE_DIM];

        for(int i = 0; i < (GEMM_TILE_DIM + k - 1) / GEMM_TILE_DIM; ++i)
        {
            if (i*GEMM_TILE_DIM + threadIdx.x < k && Row < m)
                As[threadIdx.y][threadIdx.x] = a[Row*lda + i*GEMM_TILE_DIM + threadIdx.x];
            else
                As[threadIdx.y][threadIdx.x] = 0;

            if (i*GEMM_TILE_DIM + threadIdx.y < k && Col < n)
                Bs[threadIdx.y][threadIdx.x] = b[(i*GEMM_TILE_DIM + threadIdx.y)*ldb + Col];
            else
                Bs[threadIdx.y][threadIdx.x] = 0;

            __syncthreads();

            for (int n = 0; n < GEMM_TILE_DIM; ++n)
            {
                uint patchValue  = ClampBitWidth<AT, 8>((As[threadIdx.y][n] - quantProps.pInput[2]) * quantProps.pInput[1] + T(0.5));
                uint filterValue = ClampBitWidth<AT, 8>((Bs[n][threadIdx.x] - filterOffset) * filterInvScale + T(0.5));
                uint tableFetchIdx = (patchValue << 8) | filterValue;
                value   += float(tex1Dfetch<ushort>(lookupTable, tableFetchIdx));

                inputCorrSum  += As[threadIdx.y][n];
                filterCorrSum += Bs[n][threadIdx.x];
            }

            __syncthreads();
        }

        if (Row < m && Col < n)
        {        
            value = value * s1xS2 + filterOffset * inputCorrSum +
                quantProps.pInput[2] * filterCorrSum - T(k) * m1xM2;
            
            c[((tileY * blockDim.y  + threadIdx.y)*ldc) +
               (tileX * blockDim.x) + threadIdx.x] = value;
        }
    }
}

// Approximated GEMM
template<typename T, typename AT>
__global__ void ApproxGemmCudaKernel(int m, int n, int k,
    const AT *a, int lda,
    const T *b, int ldb,
    cudaTextureObject_t lookupTable,
    const T *patchSums, const T *filterSums,
    T *c, int ldc, const GpuOpQuantPropsData_t<T, AT> quantProps,
    int totalBlocksCount);

template<typename T, typename AT>
struct ApproxConvGEMMKernel<TableApproxConvOpQuantData<GPUDevice, T, AT> > {
    void operator()(const Eigen::GpuDevice &d, const TableApproxConvOpQuantData<GPUDevice, T, AT> &approxOp,
                    int m, int n, int k, T alpha,
                    const AT *a, int lda,
                    const T *b, int ldb,
                    T beta, T *c, int ldc);
};

template<typename T, typename AT>
void ApproxConvGEMMKernel<TableApproxConvOpQuantData<GPUDevice, T, AT> >::operator()(
    const Eigen::GpuDevice &d, const TableApproxConvOpQuantData<GPUDevice, T, AT> &approxOpData,
    int m, int n, int k, T alpha,
    const AT *a, int lda,
    const T *b, int ldb,
    T beta, T *c, int ldc)
{
    const typename TableApproxConvOpQuantData<GPUDevice, T, AT>::OpQuantProps_t &quantProps = approxOpData.GetQuantProps();
    const typename TableApproxConvOpQuantData<GPUDevice, T, AT>::ApproxOpType_t &approxOp   = approxOpData.GetApproxOp();

    const dim3 blockSize(GEMM_TILE_DIM, GEMM_TILE_DIM, 1);
    const int numBlocksX = ((n + blockSize.x - 1) / blockSize.x);
    const int numBlocksY = ((m + blockSize.y - 1) / blockSize.y);
    int numBlocks  = numBlocksX * numBlocksY;

    const int maxBlocks = d.getNumGpuMultiProcessors() * 
        d.maxGpuThreadsPerMultiProcessor() / (blockSize.x*blockSize.y);
    
    if(numBlocks > maxBlocks)
        numBlocks = maxBlocks;
    
    dim3 gridSize(numBlocks, 1, 1);
    
    ApproxGemmCudaKernel<T, AT>
        <<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, 
                                                 a, lda, 
                                                 b, ldb, 
                                                 approxOp.GetLookupData(),
                                                 c, ldc, quantProps, numBlocksX * numBlocksY);
}

template<typename T, typename AT>
__global__ void ApproxGemmCudaKernel(int m, int n, int k,
    const AT *a, int lda,
    const T *b, int ldb,
    cudaTextureObject_t lookupTable,
    T *c, int ldc, const GpuOpQuantPropsData_t<T, AT> quantProps,
    int totalBlocksCount)
{
    const int numTilesX = (n + blockDim.x - 1) / blockDim.x;
    for(int tileIdx = blockIdx.x; tileIdx < totalBlocksCount; tileIdx += gridDim.x)
    {
        const int tileY = tileIdx / numTilesX;
        const int tileX = tileIdx % numTilesX;

        T value = T(0);

        int Row = tileY * GEMM_TILE_DIM + threadIdx.y;
        int Col = tileX * GEMM_TILE_DIM + threadIdx.x;

        const T filterOffset   = (quantProps.filterMode == 1 && Col < n) ? quantProps.pFilter[Col*3 + 2] : quantProps.pFilter[2];
        const T filterInvScale = (quantProps.filterMode == 1 && Col < n) ? quantProps.pFilter[Col*3 + 1] : quantProps.pFilter[1];
        const T s1xS2          = (quantProps.filterMode == 1 && Col < n) ? quantProps.pS1xS2[Col] : quantProps.pS1xS2[0];
        const T m1xM2          = (quantProps.filterMode == 1 && Col < n) ? quantProps.pM1xM2[Col] : quantProps.pM1xM2[0];

        __shared__ uint As[GEMM_TILE_DIM][GEMM_TILE_DIM];
        __shared__ uint Bs[GEMM_TILE_DIM][GEMM_TILE_DIM];

        for(int i = 0; i < (GEMM_TILE_DIM + k - 1)/GEMM_TILE_DIM; ++i)
        {
            if (i*GEMM_TILE_DIM + threadIdx.x < k && Row < m)
                As[threadIdx.y][threadIdx.x] = a[Row*lda + i*GEMM_TILE_DIM + threadIdx.x];
            else
                As[threadIdx.y][threadIdx.x] = 0;

            if (i*GEMM_TILE_DIM + threadIdx.y < k && Col < n)
                Bs[threadIdx.y][threadIdx.x] = ClampBitWidth<AT, 8>(((b[(i*GEMM_TILE_DIM + threadIdx.y)*ldb + Col] - filterOffset) * filterInvScale) + T(0.5));
            else
                Bs[threadIdx.y][threadIdx.x] = 0;

            __syncthreads();

            for (int n = 0; n < GEMM_TILE_DIM; ++n)
            {
                //value += T(As[threadIdx.y][n] * Bs[n][threadIdx.x]);
                uint tableFetchIdx = (As[threadIdx.y][n] << 8) | Bs[n][threadIdx.x];
                value += float(tex1Dfetch<ushort>(lookupTable, tableFetchIdx));
            }

            __syncthreads();
        }

        if (Row < m && Col < n)
        {
            float patchCorrection  = quantProps.pInputCorr[tileY * blockDim.y  + threadIdx.y];
            float filterCorrection = quantProps.pFilterCorr[tileX * blockDim.x + threadIdx.x];
            
            value = value * s1xS2 + filterOffset * patchCorrection +
                quantProps.pInput[2] * filterCorrection - T(k) * m1xM2;
            
            c[((tileY * blockDim.y  + threadIdx.y)*ldc) +
               (tileX * blockDim.x) + threadIdx.x] = value;
        }
    }
}

template struct ApproxConvGEMMKernel<TableApproxConvOpQuantData<GPUDevice, float, uint8> >;

// Approximated Im-2-Col
template<typename T>
__device__ void PreScan(T value, volatile T *sdata, int tid, int n);

template<typename T, typename AT>
__global__ void ApproxConvIm2ColCudaKernel(const T *in, 
                                           int c, int w, int h, int ow, int oh,
                                           int kw, int kh, int pw, int ph, int sw, int sh,
                                           int dw, int dh, int po, int pc, AT *out, T *outCoeffs, 
                                           const GpuOpQuantPropsData_t<T, AT> quantProps);

template<typename T, typename AT>
struct ApproxConvIm2ColKernel<TableApproxConvOpQuantData<GPUDevice, T, AT> > {
    void operator()(const Eigen::GpuDevice &d, const TableApproxConvOpQuantData<GPUDevice, T, AT> &approxOpData,
                    const T *in,
                    int c, int w, int h, int ow, int oh,
                    int kw, int kh, int pw, int ph, int sw, int sh,
                    int dw, int dh, int po, int pc, AT *out);
};

template<typename T, typename AT>
void ApproxConvIm2ColKernel<TableApproxConvOpQuantData<GPUDevice, T, AT> >::operator()(
    const Eigen::GpuDevice &d, const TableApproxConvOpQuantData<GPUDevice, T, AT> &approxOpData, 
    const T *in,
    int c, int w, int h, int ow, int oh,
    int kw, int kh, int pw, int ph, int sw, int sh,
    int dw, int dh, int po, int pc, AT *out)
{
    const typename TableApproxConvOpQuantData<GPUDevice, T, AT>::OpQuantProps_t &quantProps = approxOpData.GetQuantProps();
    const typename TableApproxConvOpQuantData<GPUDevice, T, AT>::ApproxOpType_t &approxOp   = approxOpData.GetApproxOp();
    T *outCoeffs = quantProps.pInputCorr;

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
                                                                     po, pc, out, outCoeffs, quantProps);
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
                                           const GpuOpQuantPropsData_t<T, AT> quantProps)
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
        //out[tId] = AT(((value - quantProps.input.offset) * quantProps.input.invScale) + T(0.5));
        out[tId] = ClampBitWidth<AT, 8>(((value - quantProps.pInput[2]) * quantProps.pInput[1]) + T(0.5));
        
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

template struct ApproxConvIm2ColKernel<TableApproxConvOpQuantData<GPUDevice, float, uint8> >;

#endif // GOOGLE_CUDA
