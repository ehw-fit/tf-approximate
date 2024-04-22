//========== Copyright (c) 2019, Filip Vaverka, All rights reserved. =========//
//
// Purpose:     Implementation of (non-)approximated GEMM and Im2Col kernels
//              running on CPU devices.
//
// $NoKeywords: $ApproxGPUOpsTF $approx_nn_conv_kernels.cpp
// $Date:       $2019-12-19
//============================================================================//

#define EIGEN_STACK_ALLOCATION_LIMIT 0
#define EIGEN_USE_THREADS
//#undef ABSL_HAVE_STD_STRING_VIEW

#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/numeric_op.h>
#include <tensorflow/core/framework/tensor.h>
#include <third_party/eigen3/unsupported/Eigen/CXX11/Tensor>
#include <third_party/eigen3/Eigen/Core>
#include <tensorflow/core/framework/tensor_types.h>
#include <tensorflow/core/platform/types.h>

#include "approx_nn_conv_kernels.h"

using namespace tensorflow;
using CPUDevice = Eigen::ThreadPoolDevice;

//----------------------------------------------------------------------------//
// Non-approximated kernels
//----------------------------------------------------------------------------//
template<typename T, typename AT>
void ApproxConvGEMMKernelCombined<NullApproxConvOpQuantData<CPUDevice, T, AT> >::operator()(
    const CPUDevice &d, const NullApproxConvOpQuantData<CPUDevice, T, AT> &, 
    int m, int n, int k, T alpha, 
    const T *a, int lda, 
    const T *b, int ldb, 
    T beta, T *c, int ldc) 
{
    const size_t aIStride = size_t(lda);
    const size_t aLStride = 1;
    const size_t bJStride = 1;
    const size_t bLStride = size_t(ldb);
    const size_t cIStride = size_t(ldc);
    const size_t cJStride = 1;

    for(size_t j = 0; j < size_t(n); ++j)
    {
        for(size_t i = 0; i < size_t(m); ++i)
        {
            T total(0);

            for(size_t l = 0; l < size_t(k); ++l)
            {
                const size_t aIndex = ((i * aIStride) + (l * aLStride));
                const T aValue = a[aIndex];

                const size_t bIndex = ((j * bJStride) + (l * bLStride));
                const T bValue = b[bIndex];

                total += (aValue * bValue);
            }

            const size_t cIndex = ((i * cIStride) + (j * cJStride));
            c[cIndex] = total;
        }
    }
}

template struct ApproxConvGEMMKernelCombined<NullApproxConvOpQuantData<CPUDevice, Eigen::half, Eigen::half> >;
template struct ApproxConvGEMMKernelCombined<NullApproxConvOpQuantData<CPUDevice,       float,       float> >;
template struct ApproxConvGEMMKernelCombined<NullApproxConvOpQuantData<CPUDevice,      double,      double> >;

template<typename T, typename AT>
void ApproxConvGEMMKernel<NullApproxConvOpQuantData<CPUDevice, T, AT> >::operator()(
        const CPUDevice &d, const NullApproxConvOpQuantData<CPUDevice, T, AT> &,
        int m, int n, int k, T alpha,
        const AT *a, int lda,
        const T *b, int ldb,
        T beta, T *c, int ldc)
{
    const size_t aIStride = size_t(lda);
    const size_t aLStride = 1;
    const size_t bJStride = 1;
    const size_t bLStride = size_t(ldb);
    const size_t cIStride = size_t(ldc);
    const size_t cJStride = 1;

    for(size_t j = 0; j < size_t(n); ++j)
    {
        for(size_t i = 0; i < size_t(m); ++i)
        {
            T total(0);

            for(size_t l = 0; l < size_t(k); ++l)
            {
                const size_t aIndex = ((i * aIStride) + (l * aLStride));
                const T aValue = a[aIndex];

                const size_t bIndex = ((j * bJStride) + (l * bLStride));
                const T bValue = b[bIndex];

                total += ((aValue) * (bValue));
            }

            const size_t cIndex = ((i * cIStride) + (j * cJStride));
            c[cIndex] = T(total);
        }
    }
}

template struct ApproxConvGEMMKernel<NullApproxConvOpQuantData<CPUDevice, Eigen::half, Eigen::half> >;
template struct ApproxConvGEMMKernel<NullApproxConvOpQuantData<CPUDevice,       float,       float> >;
template struct ApproxConvGEMMKernel<NullApproxConvOpQuantData<CPUDevice,      double,      double> >;

template<typename T, typename AT>
void ApproxConvIm2ColKernel<NullApproxConvOpQuantData<CPUDevice, T, AT> >::operator()(
        const CPUDevice &d, const NullApproxConvOpQuantData<CPUDevice, T, AT> &,
        const T *in,
        int c, int w, int h, int ow, int oh,
        int kw, int kh, int pw, int ph, int sw, int sh,
        int dw, int dh, int po, int pc, AT *out)
{
    const int pl = c * kw * kh;

    for(int tId = 0; tId < pc*pl; ++tId)
    {
        int patchId = (tId + po*pl) / pl;
        int outB    = (patchId / ow) / oh;
        int outH    = (patchId / ow) % oh;
        int outW    = patchId % ow;

        int valueId = (tId + po*pl) % pl;
        int offsetH = valueId / (kw * c);
        int offsetW = (valueId / c) % kw;
        int offsetC = valueId % c;

        int inH = outH * sh - ph + offsetH * dh;
        int inW = outW * sw - pw + offsetW * dw;

        if(inH >= 0 && inW >= 0 && inH < h && inW < w)
            out[tId] = in[((outB * h + inH) * w + inW) * c + offsetC];
        else
            out[tId] = T(0);
    }
}

template struct ApproxConvIm2ColKernel<NullApproxConvOpQuantData<CPUDevice, Eigen::half, Eigen::half> >;
template struct ApproxConvIm2ColKernel<NullApproxConvOpQuantData<CPUDevice,       float,       float> >;
template struct ApproxConvIm2ColKernel<NullApproxConvOpQuantData<CPUDevice,      double,      double> >;

//----------------------------------------------------------------------------//
// Lookup table approximate kernels (8-bit inputs)
//----------------------------------------------------------------------------//
template<typename T, typename AT>
void ApproxConvGEMMKernelCombined<TableApproxConvOpQuantData<CPUDevice, T, AT> >::operator()(
        const CPUDevice &d, const TableApproxConvOpQuantData<CPUDevice, T, AT> &approxOpData,
        int m, int n, int k, T alpha,
        const T *a, int lda,
        const T *b, int ldb,
        T beta, T *c, int ldc)
{
    const typename TableApproxConvOpQuantData<CPUDevice, T, AT>::OpQuantProps_t &quantProps = approxOpData.GetQuantProps();
    const typename TableApproxConvOpQuantData<CPUDevice, T, AT>::ApproxOpType_t &lookup     = approxOpData.GetApproxOp();

    const size_t aIStride = size_t(lda);
    const size_t aLStride = 1;
    const size_t bJStride = 1;
    const size_t bLStride = size_t(ldb);
    const size_t cIStride = size_t(ldc);
    const size_t cJStride = 1;

    for(size_t j = 0; j < size_t(n); ++j)
    {
        const T filterOffset   = (quantProps.filterMode == 1) ? quantProps.pFilter[j*3 + 2] : quantProps.pFilter[2];
        const T filterInvScale = (quantProps.filterMode == 1) ? quantProps.pFilter[j*3 + 1] : quantProps.pFilter[1];
        const T s1xS2          = (quantProps.filterMode == 1) ? quantProps.pS1xS2[j] : quantProps.pS1xS2[0];
        const T m1xM2          = (quantProps.filterMode == 1) ? quantProps.pM1xM2[j] : quantProps.pM1xM2[0];

        for(size_t i = 0; i < size_t(m); ++i)
        {
            T total(0);
            T aSum(0);
            T bSum(0);

            for(size_t l = 0; l < size_t(k); ++l)
            {
                const size_t aIndex = ((i * aIStride) + (l * aLStride));
                const uint32 aValue = AT(((a[aIndex] - quantProps.pInput[2]) * quantProps.pInput[1]) + T(0.5));
                aSum += a[aIndex];

                const size_t bIndex = ((j * bJStride) + (l * bLStride));
                const uint32 bValue = AT(((b[bIndex] - filterOffset) * filterInvScale) + T(0.5));
                bSum += b[bIndex];

                const uint32 cValue = lookup.lookupTable[(aValue << lookup.bitWidth) | bValue];
                total += T(cValue);
            }

            const size_t cIndex = ((i * cIStride) + (j * cJStride));
            c[cIndex] = total * s1xS2 +
                    filterOffset * aSum +
                    quantProps.pInput[2] * bSum -
                    T(k) * m1xM2;
        }
    }
}

template struct ApproxConvGEMMKernelCombined<TableApproxConvOpQuantData<CPUDevice, float, uint8> >;

template<typename T, typename AT>
void ApproxConvGEMMKernel<TableApproxConvOpQuantData<CPUDevice, T, AT> >::operator()(
        const CPUDevice &d, const TableApproxConvOpQuantData<CPUDevice, T, AT> &approxOpData,
        int m, int n, int k, T alpha,
        const AT *a, int lda,
        const T *b, int ldb,
        T beta, T *c, int ldc)
{
    const typename TableApproxConvOpQuantData<CPUDevice, T, AT>::OpQuantProps_t &quantProps = approxOpData.GetQuantProps();
    const typename TableApproxConvOpQuantData<CPUDevice, T, AT>::ApproxOpType_t &lookup     = approxOpData.GetApproxOp();

    const size_t aIStride = size_t(lda);
    const size_t aLStride = 1;
    const size_t bJStride = 1;
    const size_t bLStride = size_t(ldb);
    const size_t cIStride = size_t(ldc);
    const size_t cJStride = 1;

    for(size_t j = 0; j < size_t(n); ++j)
    {
        const T filterOffset   = (quantProps.filterMode == 1) ? quantProps.pFilter[j*3 + 2] : quantProps.pFilter[2];
        const T filterInvScale = (quantProps.filterMode == 1) ? quantProps.pFilter[j*3 + 1] : quantProps.pFilter[1];
        const T s1xS2          = (quantProps.filterMode == 1) ? quantProps.pS1xS2[j] : quantProps.pS1xS2[0];
        const T m1xM2          = (quantProps.filterMode == 1) ? quantProps.pM1xM2[j] : quantProps.pM1xM2[0];

        for(size_t i = 0; i < size_t(m); ++i)
        {
            T total(0);

            for(size_t l = 0; l < size_t(k); ++l)
            {
                const size_t aIndex = ((i * aIStride) + (l * aLStride));
                const uint32 aValue = a[aIndex];

                const size_t bIndex = ((j * bJStride) + (l * bLStride));
                const uint32 bValue = AT((b[bIndex] - filterOffset) * filterInvScale + T(0.5));

                const uint32 cValue = lookup.lookupTable[(aValue << lookup.bitWidth) | bValue];
                total += T(cValue);
            }

            const size_t cIndex = ((i * cIStride) + (j * cJStride));
            c[cIndex] = total * s1xS2 +
                    filterOffset * quantProps.pInputCorr[i] +
                    quantProps.pInput[2] * quantProps.pFilterCorr[j] -
                    T(k) * m1xM2;
        }
    }
}

template struct ApproxConvGEMMKernel<TableApproxConvOpQuantData<CPUDevice, float, uint8> >;

template<typename T, typename AT>
void ApproxConvIm2ColKernel<TableApproxConvOpQuantData<Eigen::ThreadPoolDevice, T, AT> >::operator()(
        const CPUDevice &d, const TableApproxConvOpQuantData<Eigen::ThreadPoolDevice, T, AT> &approxOpData,
        const T *in,
        int c, int w, int h, int ow, int oh,
        int kw, int kh, int pw, int ph, int sw, int sh,
        int dw, int dh, int po, int pc, AT *out)
{
    const typename TableApproxConvOpQuantData<CPUDevice, T, AT>::OpQuantProps_t &quantProps = approxOpData.GetQuantProps();
    const typename TableApproxConvOpQuantData<CPUDevice, T, AT>::ApproxOpType_t &lookup     = approxOpData.GetApproxOp();

    T *outCoeffs = quantProps.pInputCorr;

    const int pl = c * kw * kh;

    for(int i = 0; i < pc; ++i)
        outCoeffs[i] = T(0);

    for(int tId = 0; tId < pc*pl; ++tId)
    {
        int patchId = (tId + po*pl) / pl;
        int outB    = (patchId / ow) / oh;
        int outH    = (patchId / ow) % oh;
        int outW    = patchId % ow;

        int valueId = (tId + po*pl) % pl;
        int offsetH = valueId / (kw * c);
        int offsetW = (valueId / c) % kw;
        int offsetC = valueId % c;

        int inH = outH * sh - ph + offsetH * dh;
        int inW = outW * sw - pw + offsetW * dw;

        T value = T(0);

        if(inH >= 0 && inW >= 0 && inH < h && inW < w)
            value = in[((outB * h + inH) * w + inW) * c + offsetC];

        out[tId] = AT((value - quantProps.pInput[2]) * quantProps.pInput[1] + T(0.5));
        outCoeffs[patchId - po] += value;
    }
}

template struct ApproxConvIm2ColKernel<TableApproxConvOpQuantData<CPUDevice, float, uint8> >;
