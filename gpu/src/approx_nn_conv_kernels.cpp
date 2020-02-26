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
void ApproxConvGEMMKernelCombined<CPUDevice, T, AT, NullApproxOpType_t>::operator()(
        const CPUDevice &d, const NullApproxOpType_t<CPUDevice, T, AT> &,
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

template struct ApproxConvGEMMKernelCombined<CPUDevice, Eigen::half, Eigen::half, NullApproxOpType_t>;
template struct ApproxConvGEMMKernelCombined<CPUDevice, float,       float,       NullApproxOpType_t>;
template struct ApproxConvGEMMKernelCombined<CPUDevice, double,      double,      NullApproxOpType_t>;

template<typename T, typename AT>
void ApproxConvGEMMKernel<CPUDevice, T, AT, NullApproxOpType_t>::operator()(
        const CPUDevice &d, const NullApproxOpType_t<CPUDevice, T, AT> &,
        int m, int n, int k, T alpha,
        const AT *a, int lda,
        const T *b, int ldb,
        const T *, const T *,
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

template struct ApproxConvGEMMKernel<CPUDevice, Eigen::half, Eigen::half, NullApproxOpType_t>;
template struct ApproxConvGEMMKernel<CPUDevice, float,       float,       NullApproxOpType_t>;
template struct ApproxConvGEMMKernel<CPUDevice, double,      double,      NullApproxOpType_t>;

template<typename T, typename AT>
void ApproxConvIm2ColKernel<CPUDevice, T, AT, NullApproxOpType_t>::operator()(
        const CPUDevice &d, const NullApproxOpType_t<CPUDevice, T, AT> &,
        const T *in,
        int c, int w, int h, int ow, int oh,
        int kw, int kh, int pw, int ph, int sw, int sh,
        int dw, int dh, int po, int pc, AT *out, T *)
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

template struct ApproxConvIm2ColKernel<CPUDevice, Eigen::half, Eigen::half, NullApproxOpType_t>;
template struct ApproxConvIm2ColKernel<CPUDevice, float,       float,       NullApproxOpType_t>;
template struct ApproxConvIm2ColKernel<CPUDevice, double,      double,      NullApproxOpType_t>;

template <typename T>
using ConstScalar = typename tensorflow::TTypes<T>::ConstScalar;

template<typename T, typename AT>
void ApproxFilterCorrCoeff<Eigen::ThreadPoolDevice, T, AT, NullApproxOpType_t>::operator()(
        const Eigen::ThreadPoolDevice &d, const NullApproxOpType_t<Eigen::ThreadPoolDevice, T, AT> &,
        ConstTensor4<T>, Flat<T> outCorrCoeffs)
{
    outCorrCoeffs.setZero();
}

template struct ApproxFilterCorrCoeff<CPUDevice, Eigen::half, Eigen::half, NullApproxOpType_t>;
template struct ApproxFilterCorrCoeff<CPUDevice, float,       float,       NullApproxOpType_t>;
template struct ApproxFilterCorrCoeff<CPUDevice, double,      double,      NullApproxOpType_t>;

//----------------------------------------------------------------------------//
// Lookup table approximate kernels (8-bit inputs)
//----------------------------------------------------------------------------//
template<typename T, typename AT>
void ApproxConvGEMMKernelCombined<CPUDevice, T, AT, TableApproxOpType_t>::operator()(
        const CPUDevice &d, const TableApproxOpType_t<CPUDevice, T, AT> &lookup,
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
            T aSum(0);
            T bSum(0);

            for(size_t l = 0; l < size_t(k); ++l)
            {
                const size_t aIndex = ((i * aIStride) + (l * aLStride));
                const uint32 aValue = AT(((a[aIndex] - lookup.quantProps.input.offset) * lookup.quantProps.input.invScale) + T(0.5));
                aSum += a[aIndex];

                const size_t bIndex = ((j * bJStride) + (l * bLStride));
                const uint32 bValue = AT(((b[bIndex] - lookup.quantProps.filter.offset) * lookup.quantProps.filter.invScale) + T(0.5));
                bSum += b[bIndex];

                const uint32 cValue = lookup.lookupTable[(aValue << lookup.bitWidth) | bValue];
                total += T(cValue);
            }

            const size_t cIndex = ((i * cIStride) + (j * cJStride));
            c[cIndex] = total * lookup.quantProps.s1xS2 +
                    lookup.quantProps.m2 * aSum +
                    lookup.quantProps.m1 * bSum -
                    T(k) * lookup.quantProps.m1xM2;
        }
    }
}

template struct ApproxConvGEMMKernelCombined<CPUDevice, float, uint8, TableApproxOpType_t>;

template<typename T, typename AT>
void ApproxConvGEMMKernel<CPUDevice, T, AT, TableApproxOpType_t>::operator()(
        const CPUDevice &d, const TableApproxOpType_t<CPUDevice, T, AT> &lookup,
        int m, int n, int k, T alpha,
        const AT *a, int lda,
        const T *b, int ldb,
        const T *aCoeffs, const T *bCoeffs,
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
                const uint32 aValue = a[aIndex];

                const size_t bIndex = ((j * bJStride) + (l * bLStride));
                const uint32 bValue = AT((b[bIndex] - lookup.quantProps.filter.offset) * lookup.quantProps.filter.invScale + T(0.5));

                const uint32 cValue = lookup.lookupTable[(aValue << lookup.bitWidth) | bValue];
                total += T(cValue);
            }

            const size_t cIndex = ((i * cIStride) + (j * cJStride));
            c[cIndex] = total * lookup.quantProps.s1xS2 +
                    lookup.quantProps.m2 * aCoeffs[i] +
                    lookup.quantProps.m1 * bCoeffs[j] -
                    T(k) * lookup.quantProps.m1xM2;
        }
    }
}

template struct ApproxConvGEMMKernel<CPUDevice, float, uint8, TableApproxOpType_t>;

template<typename T, typename AT>
void ApproxConvIm2ColKernel<CPUDevice, T, AT, TableApproxOpType_t>::operator()(
        const CPUDevice &d, const TableApproxOpType_t<CPUDevice, T, AT> &lookup,
        const T *in,
        int c, int w, int h, int ow, int oh,
        int kw, int kh, int pw, int ph, int sw, int sh,
        int dw, int dh, int po, int pc, AT *out, T *outCoeffs)
{
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

        out[tId] = AT((value - lookup.quantProps.input.offset) * lookup.quantProps.input.invScale + T(0.5));
        outCoeffs[patchId - po] += value;
    }
}

template struct ApproxConvIm2ColKernel<CPUDevice, float, uint8, TableApproxOpType_t>;

template<typename T, typename AT>
void ApproxFilterCorrCoeff<CPUDevice, T, AT, TableApproxOpType_t>::operator()(
        const Eigen::ThreadPoolDevice &d, const TableApproxOpType_t<Eigen::ThreadPoolDevice, T, AT> &approxOp,
        ConstTensor4<T> filterCoeffs, Flat<T> outCorrCoeffs) {
    Eigen::array<int, 3> dims = {0, 1, 2};
    outCorrCoeffs.device(d) = filterCoeffs.sum(dims);
}

template struct ApproxFilterCorrCoeff<CPUDevice, float, uint8, TableApproxOpType_t>;
