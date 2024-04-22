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
//#undef ABSL_HAVE_STD_STRING_VIEW

#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/platform/types.h>

#include "approx_ops_quant_data.h"

template<typename T>
using ConstFlat = typename tensorflow::TTypes<T>::ConstFlat;

template<typename T>
using Flat = typename tensorflow::TTypes<T>::Flat;

template<typename T>
using ConstTensor4 = typename tensorflow::TTypes<T, 4>::ConstTensor;

template<typename Device, typename T, typename AT>
using NullApproxConvOpQuantData = ApproxConvOpQuantData<NullApproxOpType_t<Device, T, AT> >;

template<typename Device, typename T, typename AT>
using TableApproxConvOpQuantData = ApproxConvOpQuantData<TableApproxOpType_t<Device, T, AT> >;


template<typename ApproxOpData>
struct ApproxConvGEMMKernelCombined {
    typedef typename ApproxOpData::DeviceType_t Device;
    typedef typename ApproxOpData::DataType_t T;

    void operator()(const typename ApproxOpData::DeviceType_t &d, const ApproxOpData &approxOpData,
                    int m, int n, int k, T alpha,
                    const T *a, int lda,
                    const T *b, int ldb,
                    T beta, T *c, int ldc);
};

template<typename ApproxOpData>
struct ApproxConvGEMMKernel {
    typedef typename ApproxOpData::DeviceType_t Device;
    typedef typename ApproxOpData::DataType_t T;
    typedef typename ApproxOpData::QuantDataType_t AT;

    void operator()(const Device &d, const ApproxOpData &approxOpData,
                    int m, int n, int k, T alpha,
                    const AT *a, int lda,
                    const T *b, int ldb,
                    T beta, T *c, int ldc);
};

template<typename ApproxOpData>
struct ApproxConvIm2ColKernel {
    typedef typename ApproxOpData::DeviceType_t Device;
    typedef typename ApproxOpData::DataType_t T;
    typedef typename ApproxOpData::QuantDataType_t AT;

    void operator()(const Device &d, const ApproxOpData &approxOp,
                    const T *in,
                    int c, int w, int h, int ow, int oh,
                    int kw, int kh, int pw, int ph, int sw, int sh,
                    int dw, int dh, int po, int pc, AT *out);
};

// Declarations of Host (CPU) kernel specializations with no approximation
template<typename T, typename AT>
struct ApproxConvGEMMKernelCombined<NullApproxConvOpQuantData<Eigen::ThreadPoolDevice, T, AT> > {
    void operator()(const Eigen::ThreadPoolDevice &d, const NullApproxConvOpQuantData<Eigen::ThreadPoolDevice, T, AT> &approxOpData,
                    int m, int n, int k, T alpha,
                    const T *a, int lda,
                    const T *b, int ldb,
                    T beta, T *c, int ldc);
};

template<typename T, typename AT>
struct ApproxConvGEMMKernel<NullApproxConvOpQuantData<Eigen::ThreadPoolDevice, T, AT> > {
    void operator()(const Eigen::ThreadPoolDevice &d, const NullApproxConvOpQuantData<Eigen::ThreadPoolDevice, T, AT> &approxOpData,
                    int m, int n, int k, T alpha,
                    const AT *a, int lda,
                    const T *b, int ldb,
                    T beta, T *c, int ldc);
};

template<typename T, typename AT>
struct ApproxConvIm2ColKernel<NullApproxConvOpQuantData<Eigen::ThreadPoolDevice, T, AT> > {
    void operator()(const Eigen::ThreadPoolDevice &d, const NullApproxConvOpQuantData<Eigen::ThreadPoolDevice, T, AT> &approxOpData,
                    const T *in,
                    int c, int w, int h, int ow, int oh,
                    int kw, int kh, int pw, int ph, int sw, int sh,
                    int dw, int dh, int po, int pc, AT *out);
};

// Declarations of Host (CPU) kernel specializations with Lookup table approximation
template<typename T, typename AT>
struct ApproxConvGEMMKernelCombined<TableApproxConvOpQuantData<Eigen::ThreadPoolDevice, T, AT> > {
    void operator()(const Eigen::ThreadPoolDevice &d, const TableApproxConvOpQuantData<Eigen::ThreadPoolDevice, T, AT> &approxOpData,
                    int m, int n, int k, T alpha,
                    const T *a, int lda,
                    const T *b, int ldb,
                    T beta, T *c, int ldc);
};

template<typename T, typename AT>
struct ApproxConvGEMMKernel<TableApproxConvOpQuantData<Eigen::ThreadPoolDevice, T, AT> > {
    void operator()(const Eigen::ThreadPoolDevice &d, const TableApproxConvOpQuantData<Eigen::ThreadPoolDevice, T, AT> &approxOpData,
                    int m, int n, int k, T alpha,
                    const AT *a, int lda,
                    const T *b, int ldb,
                    T beta, T *c, int ldc);
};

template<typename T, typename AT>
struct ApproxConvIm2ColKernel<TableApproxConvOpQuantData<Eigen::ThreadPoolDevice, T, AT> > {
    void operator()(const Eigen::ThreadPoolDevice &d, const TableApproxConvOpQuantData<Eigen::ThreadPoolDevice, T, AT> &approxOpData,
                    const T *in,
                    int c, int w, int h, int ow, int oh,
                    int kw, int kh, int pw, int ph, int sw, int sh,
                    int dw, int dh, int po, int pc, AT *out);
};

// Declarations of Device (GPU) kernel specializations
#ifdef GOOGLE_CUDA

template<template<typename> class ApproxOpData, template<typename, typename, typename> class ApproxOpType, typename T, typename AT>
struct ApproxConvGEMMKernelCombined<ApproxOpData<ApproxOpType<Eigen::GpuDevice, T, AT> > > {
    void operator()(const Eigen::GpuDevice &d, const ApproxOpData<ApproxOpType<Eigen::GpuDevice, T, AT> > &approxOpData,
                    int m, int n, int k, T alpha,
                    const T *a, int lda,
                    const T *b, int ldb,
                    T beta, T *c, int ldc);
};


template<template<typename> class ApproxOpData, template<typename, typename, typename> class ApproxOpType, typename T, typename AT>
struct ApproxConvGEMMKernel<ApproxOpData<ApproxOpType<Eigen::GpuDevice, T, AT> > > {
    void operator()(const Eigen::GpuDevice &d, const ApproxOpData<ApproxOpType<Eigen::GpuDevice, T, AT> > &approxOpData,
                    int m, int n, int k, T alpha,
                    const AT *a, int lda,
                    const T *b, int ldb,
                    T beta, T *c, int ldc);
};

template<template<typename> class ApproxOpData, template<typename, typename, typename> class ApproxOpType, typename T, typename AT>
struct ApproxConvIm2ColKernel<ApproxOpData<ApproxOpType<Eigen::GpuDevice, T, AT> > > {
    void operator()(const Eigen::GpuDevice &d, const ApproxOpData<ApproxOpType<Eigen::GpuDevice, T, AT> > &approxOpData,
                    const T *in,
                    int c, int w, int h, int ow, int oh,
                    int kw, int kh, int pw, int ph, int sw, int sh,
                    int dw, int dh, int po, int pc, AT *out);
};

#endif // GOOGLE_CUDA

#endif // APPROX_NN_CONV_KERNELS_H
