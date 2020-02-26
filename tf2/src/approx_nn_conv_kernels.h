//========== Copyright (c) 2019, Filip Vaverka, All rights reserved. =========//
//
// Purpose:     Implementation of (non-)approximated GEMM and Im2Col kernels
//              running on CPU or GPU/CUDA devices.
//
// $NoKeywords: $ApproxGPUOpsTF $approx_nn_conv_kernels.h
// $Date:       $2019-12-19
//============================================================================//

#pragma once

#ifndef APPROX_NN_CONV_KERNELS_H
#define APPROX_NN_CONV_KERNELS_H

#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/platform/types.h>

#include "approx_ops_types.h"

template<typename T>
using ConstFlat = typename tensorflow::TTypes<T>::ConstFlat;

template<typename T>
using Flat = typename tensorflow::TTypes<T>::Flat;

template<typename T>
using ConstTensor4 = typename tensorflow::TTypes<T, 4>::ConstTensor;

template<typename Device, typename T, typename AT, template<typename, typename, typename> class ApproxOpType>
struct ApproxConvGEMMKernelCombined {
    void operator()(const Device &d, const ApproxOpType<Device, T, AT> &approxOp,
                    int m, int n, int k, T alpha,
                    const T *a, int lda,
                    const T *b, int ldb,
                    T beta, T *c, int ldc);
};

template<typename Device, typename T, typename AT, template<typename, typename, typename> class ApproxOpType>
struct ApproxConvGEMMKernel {
    void operator()(const Device &d, const ApproxOpType<Device, T, AT> &approxOp,
                    int m, int n, int k, T alpha,
                    const AT *a, int lda,
                    const T *b, int ldb,
                    const T *aCoeffs, const T *bCoeffs,
                    T beta, T *c, int ldc);
};

template<typename Device, typename T, typename AT, template<typename, typename, typename> class ApproxOpType>
struct ApproxConvIm2ColKernel {
    void operator()(const Device &d, const ApproxOpType<Device, T, AT> &approxOp,
                    const T *in,
                    int c, int w, int h, int ow, int oh,
                    int kw, int kh, int pw, int ph, int sw, int sh,
                    int dw, int dh, int po, int pc, AT *out, T *outCoeffs);
};

template<typename Device, typename T, typename AT, template<typename, typename, typename> class ApproxOpType>
struct ApproxFilterCorrCoeff {
    void operator()(const Device &d, const ApproxOpType<Device, T, AT> &approxOp,
                    ConstTensor4<T> filterCoeffs, Flat<T> outCorrCoeffs);
};

// Declarations of Host (CPU) kernel specializations with no approximation
template<typename T, typename AT>
struct ApproxConvGEMMKernelCombined<Eigen::ThreadPoolDevice, T, AT, NullApproxOpType_t> {
    void operator()(const Eigen::ThreadPoolDevice &d, const NullApproxOpType_t<Eigen::ThreadPoolDevice, T, AT> &approxOp,
                    int m, int n, int k, T alpha,
                    const T *a, int lda,
                    const T *b, int ldb,
                    T beta, T *c, int ldc);
};

template<typename T, typename AT>
struct ApproxConvGEMMKernel<Eigen::ThreadPoolDevice, T, AT, NullApproxOpType_t> {
    void operator()(const Eigen::ThreadPoolDevice &d, const NullApproxOpType_t<Eigen::ThreadPoolDevice, T, AT> &approxOp,
                    int m, int n, int k, T alpha,
                    const AT *a, int lda,
                    const T *b, int ldb,
                    const T *aCoeffs, const T *bCoeffs,
                    T beta, T *c, int ldc);
};

template<typename T, typename AT>
struct ApproxConvIm2ColKernel<Eigen::ThreadPoolDevice, T, AT, NullApproxOpType_t> {
    void operator()(const Eigen::ThreadPoolDevice &d, const NullApproxOpType_t<Eigen::ThreadPoolDevice, T, AT> &approxOp,
                    const T *in,
                    int c, int w, int h, int ow, int oh,
                    int kw, int kh, int pw, int ph, int sw, int sh,
                    int dw, int dh, int po, int pc, AT *out, T *outCoeffs);
};

template<typename T, typename AT>
struct ApproxFilterCorrCoeff<Eigen::ThreadPoolDevice, T, AT, NullApproxOpType_t> {
    void operator()(const Eigen::ThreadPoolDevice &d, const NullApproxOpType_t<Eigen::ThreadPoolDevice, T, AT> &approxOp,
                    ConstTensor4<T> filterCoeffs, Flat<T> outCorrCoeffs);
};

// Declarations of Host (CPU) kernel specializations with Lookup table approximation
template<typename T, typename AT>
struct ApproxConvGEMMKernelCombined<Eigen::ThreadPoolDevice, T, AT, TableApproxOpType_t> {
    void operator()(const Eigen::ThreadPoolDevice &d, const TableApproxOpType_t<Eigen::ThreadPoolDevice, T, AT> &approxOp,
                    int m, int n, int k, T alpha,
                    const T *a, int lda,
                    const T *b, int ldb,
                    T beta, T *c, int ldc);
};

template<typename T, typename AT>
struct ApproxConvGEMMKernel<Eigen::ThreadPoolDevice, T, AT, TableApproxOpType_t> {
    void operator()(const Eigen::ThreadPoolDevice &d, const TableApproxOpType_t<Eigen::ThreadPoolDevice, T, AT> &approxOp,
                    int m, int n, int k, T alpha,
                    const AT *a, int lda,
                    const T *b, int ldb,
                    const T *aCoeffs, const T *bCoeffs,
                    T beta, T *c, int ldc);
};

template<typename T, typename AT>
struct ApproxConvIm2ColKernel<Eigen::ThreadPoolDevice, T, AT, TableApproxOpType_t> {
    void operator()(const Eigen::ThreadPoolDevice &d, const TableApproxOpType_t<Eigen::ThreadPoolDevice, T, AT> &approxOp,
                    const T *in,
                    int c, int w, int h, int ow, int oh,
                    int kw, int kh, int pw, int ph, int sw, int sh,
                    int dw, int dh, int po, int pc, AT *out, T *outCoeffs);
};

template<typename T, typename AT>
struct ApproxFilterCorrCoeff<Eigen::ThreadPoolDevice, T, AT, TableApproxOpType_t> {
    void operator()(const Eigen::ThreadPoolDevice &d, const TableApproxOpType_t<Eigen::ThreadPoolDevice, T, AT> &approxOp,
                    ConstTensor4<T> filterCoeffs, Flat<T> outCorrCoeffs);
};

// Declarations of Device (GPU) kernel specializations
#ifdef GOOGLE_CUDA

template<typename T, typename AT, template<typename, typename, typename> class ApproxOpType>
struct ApproxConvGEMMKernelCombined<Eigen::GpuDevice, T, AT, ApproxOpType> {
    void operator()(const Eigen::GpuDevice &d, const ApproxOpType<Eigen::GpuDevice, T, AT> &approxOp,
                    int m, int n, int k, T alpha,
                    const T *a, int lda,
                    const T *b, int ldb,
                    T beta, T *c, int ldc);
};

template<typename T, typename AT, template<typename, typename, typename> class ApproxOpType>
struct ApproxConvGEMMKernel<Eigen::GpuDevice, T, AT, ApproxOpType> {
    void operator()(const Eigen::GpuDevice &d, const ApproxOpType<Eigen::GpuDevice, T, AT> &approxOp,
                    int m, int n, int k, T alpha,
                    const AT *a, int lda,
                    const T *b, int ldb,
                    const T *aCoeffs, const T *bCoeffs,
                    T beta, T *c, int ldc);
};

/*extern template struct ApproxConvGEMMKernel<Eigen::GpuDevice, Eigen::half, Eigen::half, NullApproxOpType_t>;
extern template struct ApproxConvGEMMKernel<Eigen::GpuDevice, float,       float,       NullApproxOpType_t>;
extern template struct ApproxConvGEMMKernel<Eigen::GpuDevice, double,      double,      NullApproxOpType_t>;*/

template<typename T, typename AT, template<typename, typename, typename> class ApproxOpType>
struct ApproxConvIm2ColKernel<Eigen::GpuDevice, T, AT, ApproxOpType> {
    void operator()(const Eigen::GpuDevice &d, const ApproxOpType<Eigen::GpuDevice, T, AT> &approxOp,
                    const T *in,
                    int c, int w, int h, int ow, int oh,
                    int kw, int kh, int pw, int ph, int sw, int sh,
                    int dw, int dh, int po, int pc, AT *out, T *outCoeffs);
};

/*extern template struct ApproxConvIm2ColKernel<Eigen::GpuDevice, Eigen::half, Eigen::half, NullApproxOpType_t>;
extern template struct ApproxConvIm2ColKernel<Eigen::GpuDevice, float,       float,       NullApproxOpType_t>;
extern template struct ApproxConvIm2ColKernel<Eigen::GpuDevice, double,      double,      NullApproxOpType_t>;*/

template<typename T, typename AT, template<typename, typename, typename> class ApproxOpType>
struct ApproxFilterCorrCoeff<Eigen::GpuDevice, T, AT, ApproxOpType> {
    void operator()(const Eigen::GpuDevice &d, const ApproxOpType<Eigen::GpuDevice, T, AT> &approxOp,
                    ConstTensor4<T> filterCoeffs, Flat<T> outCorrCoeffs);
};

#endif // GOOGLE_CUDA

#endif // APPROX_NN_CONV_KERNELS_H
