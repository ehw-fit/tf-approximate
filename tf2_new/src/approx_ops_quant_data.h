//========== Copyright (c) 2021, Filip Vaverka, All rights reserved. =========//
//
// Purpose:     Quantization data for approximate operation.
//
// $NoKeywords: $ApproxGPUOpsTF $approx_ops_quant_data.h
// $Date:       $2021-05-30
//============================================================================//

#pragma once

#ifndef __APPROX_OPS_QUANT_DATA_H__
#define __APPROX_OPS_QUANT_DATA_H__
//#undef ABSL_HAVE_STD_STRING_VIEW

#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/tensor_types.h>
#include <tensorflow/core/kernels/fake_quant_ops_functor.h>

#include "approx_ops_types.h"

template<typename ApproxOpType>
class ApproxConvOpQuantData;

template<template<typename, typename, typename> class ApproxOpType, typename Device, typename T, typename QT>
class ApproxConvOpQuantData<ApproxOpType<Device, T, QT> >
{
public:
    typedef Device DeviceType_t;
    typedef T DataType_t;
    typedef QT QuantDataType_t;
    typedef ApproxOpType<Device, T, QT> ApproxOpType_t;

    struct OpQuantProps_t {
        T *pInput;  ///< [[scale, 1/scale, offset], ...]
        T *pFilter; ///< [[scale, 1/scale, offset], ...]
        int filterMode; ///< Nonzero for per-axis filter quantization

        T *pS1xS2;
        T *pM1xM2;

        T *pInputCorr;
        T *pFilterCorr;
    };

    ApproxConvOpQuantData(tensorflow::OpKernelContext *ctx,
        const ApproxOpType_t &approxOpType);

    ~ApproxConvOpQuantData() {}

    void ComputeQuantProps(const int inputMinIdx, const int inputMaxIdx,
        const int filterMinIdx, const int filterMaxIdx);
    void ComputeFilterCorrCoefficients(const int filterIdx);
    void AllocateInputCorrCoefficients(const tensorflow::int64 size);
    
    const OpQuantProps_t &GetQuantProps() const { return m_opQuantProps; }
    const ApproxOpType_t &GetApproxOp() const { return m_approxOpType; }

protected:
    tensorflow::OpKernelContext *m_pCtx;
    const ApproxOpType_t &m_approxOpType;

    tensorflow::Tensor m_quantPropsInput;
    tensorflow::Tensor m_quantPropsFilter;
    tensorflow::Tensor m_quantPropsS1xS2;
    tensorflow::Tensor m_quantPropsM1xM2;

    tensorflow::Tensor m_corrCoeffInput;
    tensorflow::Tensor m_corrCoeffFiler;

    OpQuantProps_t m_opQuantProps;
};

template<typename Device, typename T, typename QT>
class ApproxConvOpQuantData<NullApproxOpType_t<Device, T, QT> >
{
public:
    typedef Device DeviceType_t;
    typedef T DataType_t;
    typedef QT QuantDataType_t;
    typedef NullApproxOpType_t<Device, T, QT> ApproxOpType_t;

    struct OpQuantProps_t {};

    ApproxConvOpQuantData(tensorflow::OpKernelContext *ctx,
        const ApproxOpType_t &approxOpType)
        : m_approxOpType(approxOpType) 
    {}

    ~ApproxConvOpQuantData() {}

    void ComputeQuantProps(const int inputMinIdx, const int inputMaxIdx,
        const int filterMinIdx, const int filterMaxIdx) {}
    void ComputeFilterCorrCoefficients(const int filterIdx) {}
    void AllocateInputCorrCoefficients(const tensorflow::int64 size) {}

    const OpQuantProps_t &GetQuantProps() const { return m_opQuantProps; }
    const ApproxOpType_t &GetApproxOp() const { return m_approxOpType; }

protected:
    OpQuantProps_t m_opQuantProps;
    const ApproxOpType_t &m_approxOpType;
};

#if GOOGLE_CUDA

extern template class ApproxConvOpQuantData<TableApproxOpType_t<Eigen::GpuDevice, float, uint8_t> >;
extern template class ApproxConvOpQuantData<NullApproxOpType_t<Eigen::GpuDevice, float, uint8_t> >;

#endif // GOOGLE_CUDA

template<template<typename, typename, typename> class ApproxOpType, typename Device, typename T, typename QT>
ApproxConvOpQuantData<ApproxOpType<Device, T, QT> >::ApproxConvOpQuantData(
        tensorflow::OpKernelContext *ctx, const ApproxOpType<Device, T, QT> &approxOpType)
    : m_pCtx(ctx)
    , m_approxOpType(approxOpType)
{
    m_opQuantProps.pInputCorr = nullptr;
    m_opQuantProps.pFilterCorr = nullptr;
}

template<typename T, unsigned N>
void AllocateTensorTemp(tensorflow::OpKernelContext *ctx, const std::array<tensorflow::int64, N> &shape, tensorflow::Tensor &buffer) {
    using namespace tensorflow;

    TensorShape tensorShape;
    TensorShapeUtils::MakeShape(shape.data(), shape.size(), &tensorShape);
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(), tensorShape, &buffer));
}

template<typename Device>
void CopyToDeviceSync(const Device &d, void *pDst, const void *pSrc, size_t size) {
    memcpy(pDst, pSrc, size);
}

template<>
void CopyToDeviceSync(const Eigen::GpuDevice &d, void *pDst, const void *pSrc, size_t size);

template<typename Device, typename T>
struct ComputeQuantPropsFunctor {
    void operator()(const Device &d, tensorflow::ConstVec<T> min, tensorflow::ConstVec<T> max, 
        const int quantMin, const int quantMax, typename tensorflow::TTypes<T>::Matrix outputs) {
            using namespace tensorflow;

            typename Eigen::Tensor<T, 2, Eigen::RowMajor> outputHost(outputs.dimensions());

            for(Index i = 0; i < min.size(); ++i)
            {
                const T minVal = min(i);
                const T maxVal = max(i);

                if(minVal == 0.0f && maxVal == 0.0f) {
                    auto chip = outputHost.template chip<0>(i);
                    chip.device(d) = chip.constant(T(0));
                    continue;
                }

                float nudgedMin, nudgedMax, nudgedScale, invNudgedScale;
                Nudge(minVal, maxVal, quantMin, quantMax, &nudgedMin, &nudgedMax, &nudgedScale, &invNudgedScale);

                typename Eigen::Tensor<T, 1, Eigen::RowMajor> nudgedValues(3);
                nudgedValues.setValues({ nudgedScale, T(1) / nudgedScale, nudgedMin });

                auto chip = outputHost.template chip<0>(i);
                chip = nudgedValues;
            }

            CopyToDeviceSync(d, outputs.data(), outputHost.data(), outputs.size() * sizeof(T));            
    }

    void operator()(const Device &d, tensorflow::ConstScalar<T> min, tensorflow::ConstScalar<T> max,
        const int quantMin, const int quantMax, typename tensorflow::TTypes<T>::Matrix outputs) {
            using namespace tensorflow;

            const T minVal = min();
            const T maxVal = max();

            if(minVal == 0.0f && maxVal == 0.0f) {
                auto chip = outputs.template chip<0>(0);
                chip.device(d) = chip.constant(T(0));
                return;
            }

            typename Eigen::Tensor<T, 2, Eigen::RowMajor> outputHost(outputs.dimensions());

            float nudgedMin, nudgedMax, nudgedScale, invNudgedScale;
            Nudge(minVal, maxVal, quantMin, quantMax, &nudgedMin, &nudgedMax, &nudgedScale, &invNudgedScale);

            typename Eigen::Tensor<T, 1, Eigen::RowMajor> nudgedValues(3);
            nudgedValues.setValues({ nudgedScale, T(1) / nudgedScale, nudgedMin });

            auto chip = outputHost.template chip<0>(0);
            chip = nudgedValues;

            CopyToDeviceSync(d, outputs.data(), outputHost.data(), outputs.size() * sizeof(T));
    }
};

template<template<typename, typename, typename> class ApproxOpType, typename Device, typename T, typename QT>
void ApproxConvOpQuantData<ApproxOpType<Device, T, QT> >::ComputeQuantProps(
    const int inputMinIdx, const int inputMaxIdx,
    const int filterMinIdx, const int filterMaxIdx)
{
    using namespace tensorflow;

    const Tensor &inputMin = m_pCtx->input(inputMinIdx);
    const Tensor &inputMax = m_pCtx->input(inputMaxIdx);

    const Tensor &filterMin = m_pCtx->input(filterMinIdx);
    const Tensor &filterMax = m_pCtx->input(filterMaxIdx);

    CHECK(inputMin.IsSameSize(inputMax))   << "Inconsistent input range (min, max) shape.";
    CHECK(filterMin.IsSameSize(filterMax)) << "Inconsistent filter range (min, max) shape.";

    const int64 inputQuantParamsCount  = (inputMin.dims() == 0)  ? 1 : inputMin.dim_size(0);
    const int64 filterQuantParamsCount = (filterMin.dims() == 0) ? 1 : filterMin.dim_size(0);

    CHECK(inputQuantParamsCount == 1) << "Per-axis quantization is supported only for filters.";

    AllocateTensorTemp<T, 2>(m_pCtx, {inputQuantParamsCount,  3}, m_quantPropsInput);
    AllocateTensorTemp<T, 2>(m_pCtx, {filterQuantParamsCount, 3}, m_quantPropsFilter);

    if(inputQuantParamsCount == 1)
        ComputeQuantPropsFunctor<Device, T>()(m_pCtx->eigen_device<Device>(), inputMin.scalar<T>(), inputMax.scalar<T>(),
            m_approxOpType.quantMin, m_approxOpType.quantMax, m_quantPropsInput.matrix<T>());
    else
        ComputeQuantPropsFunctor<Device, T>()(m_pCtx->eigen_device<Device>(), inputMin.vec<T>(), inputMax.vec<T>(), 
            m_approxOpType.quantMin, m_approxOpType.quantMax, m_quantPropsInput.matrix<T>());
    
    if(filterQuantParamsCount == 1)
        ComputeQuantPropsFunctor<Device, T>()(m_pCtx->eigen_device<Device>(), filterMin.scalar<T>(), filterMax.scalar<T>(), 
            m_approxOpType.quantMin, m_approxOpType.quantMax, m_quantPropsFilter.matrix<T>());
    else
        ComputeQuantPropsFunctor<Device, T>()(m_pCtx->eigen_device<Device>(), filterMin.vec<T>(), filterMax.vec<T>(), 
            m_approxOpType.quantMin, m_approxOpType.quantMax, m_quantPropsFilter.matrix<T>());
    
    AllocateTensorTemp<T, 2>(m_pCtx, { m_quantPropsInput.dim_size(0), m_quantPropsFilter.dim_size(0) }, m_quantPropsS1xS2);
    AllocateTensorTemp<T, 2>(m_pCtx, { m_quantPropsInput.dim_size(0), m_quantPropsFilter.dim_size(0) }, m_quantPropsM1xM2);
    
    {
        const Device &d = m_pCtx->eigen_device<Device>();
        Eigen::array<Eigen::IndexPair<long>, 0> emptyIndexList = {};

        typename TTypes<T>::Matrix quantPropsInput  = m_quantPropsInput.matrix<T>();
        typename TTypes<T>::Matrix quantPropsFilter = m_quantPropsFilter.matrix<T>();
        
        typename TTypes<T>::Matrix quantPropsS1xS2  = m_quantPropsS1xS2.matrix<T>();
        quantPropsS1xS2.device(d) = quantPropsInput.template chip<1>(0).contract(quantPropsFilter.template chip<1>(0), emptyIndexList);

        typename TTypes<T>::Matrix quantPropsM1xM2  = m_quantPropsM1xM2.matrix<T>();
        quantPropsM1xM2.device(d) = quantPropsInput.template chip<1>(2).contract(quantPropsFilter.template chip<1>(2), emptyIndexList);
    }

    m_opQuantProps.pInput  = m_quantPropsInput.flat<T>().data();
    m_opQuantProps.pFilter = m_quantPropsFilter.flat<T>().data();
    m_opQuantProps.pS1xS2  = m_quantPropsS1xS2.flat<T>().data();
    m_opQuantProps.pM1xM2  = m_quantPropsM1xM2.flat<T>().data();
    m_opQuantProps.filterMode = (filterQuantParamsCount == 1) ? 0 : 1;
}

template<template<typename, typename, typename> class ApproxOpType, typename Device, typename T, typename QT>
void ApproxConvOpQuantData<ApproxOpType<Device, T, QT> >::ComputeFilterCorrCoefficients(const int filterIdx)
{
    using namespace tensorflow;

    const Tensor &filter = m_pCtx->input(filterIdx);
    const int64 filtersCount = m_pCtx->input(filterIdx).dim_size(3);

    AllocateTensorTemp<T, 1>(m_pCtx, {filtersCount}, m_corrCoeffFiler);

    ApproxFilterCorrCoeff<Device, T, QT, ApproxOpType>()(m_pCtx->eigen_device<Device>(),
        m_approxOpType, filter.tensor<T, 4>(), m_corrCoeffFiler.flat<T>());
    
    m_opQuantProps.pFilterCorr = m_corrCoeffFiler.flat<T>().data();
}

template<template<typename, typename, typename> class ApproxOpType, typename Device, typename T, typename QT>
void ApproxConvOpQuantData<ApproxOpType<Device, T, QT> >::AllocateInputCorrCoefficients(const tensorflow::int64 size)
{
    using namespace tensorflow;

    AllocateTensorTemp<T, 1>(m_pCtx, {size}, m_corrCoeffInput);

    m_opQuantProps.pInputCorr = m_corrCoeffInput.flat<T>().data();
}

#endif // __APPROX_OPS_QUANT_DATA_H__
