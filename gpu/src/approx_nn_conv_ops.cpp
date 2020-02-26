/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

//========== Copyright (c) 2019, Filip Vaverka, All rights reserved. =========//
//
// Purpose:     Modified implementation of standard TF's 2D Convolution ops
//              allowing simulation of approximated convolutions.
//
// $NoKeywords: $ApproxGPUOpsTF $approx_nn_conv_ops.cpp
// $Date:       $2019-12-19
//============================================================================//

#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/ops_util.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/framework/numeric_op.h>
#include <tensorflow/core/framework/bounds_check.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/register_types.h>

#include <tensorflow/core/util/padding.h>
#include <tensorflow/core/util/tensor_format.h>
#include <tensorflow/core/lib/strings/numbers.h>
#include <tensorflow/core/lib/strings/str_util.h>

#include "approx_nn_conv_kernels.h"

#include "approx_nn_conv_ops.h"

#if GOOGLE_CUDA
#include "gpu_utils.h"
#endif // GOOGLE_CUDA

using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

/**
 * Helper to map C/C++ types to TF buffer types.
 */
template<typename T>
struct TempBufferType_t;

template<>
struct TempBufferType_t<Eigen::half> { static DataType Get() { return DataType::DT_HALF; } };

template<>
struct TempBufferType_t<float> { static DataType Get() { return DataType::DT_FLOAT; } };

template<>
struct TempBufferType_t<double> { static DataType Get() { return DataType::DT_DOUBLE; } };

template<>
struct TempBufferType_t<uint8> { static DataType Get() { return DataType::DT_UINT8; } };

/**
 * Generic convolution OP launcher.
 */
template<typename Device, typename T, typename AT, template<typename, typename, typename> class ApproxOpType>
struct LaunchGeneric {
    void operator()(OpKernelContext *ctx, const Tensor &input, const Tensor &filter,
                    int rowPadding, int colPadding, int rowStride, int colStride,
                    int rowDilation, int colDilation, const Padding &padding,
                    const std::vector<int64> &explicitPaddings, Tensor *output,
                    TensorFormat dataFormat, const ApproxOpType<Device, T, AT> &approxOp)
    {
        CHECK(dataFormat == FORMAT_NHWC) << "Generic conv implementation only "
                                            "supports NHWC tensor format for now.";

        if(filter.dim_size(0) == 1 && filter.dim_size(1) == 1 && rowStride == 1 &&
                colStride == 1 && (padding == SAME || padding == VALID))
        {
            const int m = int(input.dim_size(0) * input.dim_size(1) * input.dim_size(2));
            const int n = int(filter.dim_size(3));
            const int k = int(input.dim_size(3));
            const int lda = k;
            const int ldb = int(filter.dim_size(3));
            const int ldc = int(filter.dim_size(3));

            ApproxConvGEMMKernelCombined<Device, T, AT, ApproxOpType>()(
                        ctx->eigen_device<Device>(), approxOp,
                        m, n, k, T(0),
                        input.flat<T>().data(), lda,
                        filter.flat<T>().data(), ldb,
                        T(0), output->flat<T>().data(), ldc);
        }
        else if(filter.dim_size(0) == input.dim_size(1) &&
                filter.dim_size(1) == input.dim_size(2) && rowDilation == 1 &&
                colDilation == 1 && padding == VALID)
        {
            const int m = int(input.dim_size(0));
            const int n = int(filter.dim_size(3));
            const int k = int(input.dim_size(1) * input.dim_size(2) * input.dim_size(3));
            const int lda = k;
            const int ldb = int(filter.dim_size(3));
            const int ldc = int(filter.dim_size(3));

            ApproxConvGEMMKernelCombined<Device, T, AT, ApproxOpType>()(
                        ctx->eigen_device<Device>(), approxOp,
                        m, n, k, T(0),
                        input.flat<T>().data(), lda,
                        filter.flat<T>().data(), ldb,
                        T(0), output->flat<T>().data(), ldc);
        }
        else
        {
            const int64 kMaxChunkSize = (16 * 1024 * 1024) / sizeof(T);
            int64 patchLength  = filter.dim_size(0) * filter.dim_size(1) * filter.dim_size(2);
            OP_REQUIRES(ctx, (patchLength) <= kMaxChunkSize,
                        errors::InvalidArgument("ApproxConv2DOp: Patch too large for buffer"));

            int64 totalPatchesCount = output->dim_size(0) * output->dim_size(1) * output->dim_size(2);
            const int64 patchesPerChunk = kMaxChunkSize / patchLength;

            Tensor im2ColTmpBuffer;

            {
                TensorShape im2ColTmpBufferShape;
                TensorShapeUtils::MakeShape(&kMaxChunkSize, 1, &im2ColTmpBufferShape);
                OP_REQUIRES_OK(ctx, ctx->allocate_temp(TempBufferType_t<AT>::Get(), im2ColTmpBufferShape, &im2ColTmpBuffer));
            }

            Tensor patchCoeffTmpBuffer;

            {
                TensorShape shape;
                TensorShapeUtils::MakeShape(&patchesPerChunk, 1, &shape);
                OP_REQUIRES_OK(ctx, ctx->allocate_temp(TempBufferType_t<T>::Get(), shape, &patchCoeffTmpBuffer));
            }

            Tensor filterCoeffTmpBuffer;

            {
                TensorShape shape;
                int numFilters = int(filter.dim_size(3));
                TensorShapeUtils::MakeShape(&numFilters, 1, &shape);
                OP_REQUIRES_OK(ctx, ctx->allocate_temp(TempBufferType_t<T>::Get(), shape, &filterCoeffTmpBuffer));
            }

            AT *pIm2ColBufferData = im2ColTmpBuffer.flat<AT>().data();
            T *pPatchCoeffData    = patchCoeffTmpBuffer.flat<T>().data();
            T *pFilterCoeffData   = filterCoeffTmpBuffer.flat<T>().data();

            ApproxFilterCorrCoeff<Device, T, AT, ApproxOpType>()(
                        ctx->eigen_device<Device>(), approxOp, filter.tensor<T, 4>(), filterCoeffTmpBuffer.flat<T>());
            /*for(unsigned i = 0; i < filterCoeffTmpBuffer.shape().num_elements(); ++i)
                std::cout << pFilterCoeffData[i] << std::endl;*/

            if(padding == EXPLICIT)
            {
                ctx->SetStatus(errors::Unimplemented("Explicit padding is not implemented."));
            }
            else
            {
                T *pOutputData = output->flat<T>().data();

                for(int64 i = 0; i < totalPatchesCount; i += patchesPerChunk)
                {
                    int64 batch = i / (output->dim_size(1) * output->dim_size(2));
                    int patchOffset = int(i % (output->dim_size(1) * output->dim_size(2)));
                    int patchesCount = int(std::min(patchesPerChunk, totalPatchesCount - i));

                    const T *pInputData = input.flat<T>().data() + batch * input.dim_size(1) * input.dim_size(2) * input.dim_size(3);

                    ApproxConvIm2ColKernel<Device, T, AT, ApproxOpType>()(
                                ctx->eigen_device<Device>(), approxOp,
                                pInputData,
                                input.dim_size(3), input.dim_size(2), input.dim_size(1),
                                output->dim_size(2), output->dim_size(1),
                                filter.dim_size(1), filter.dim_size(0), colPadding, rowPadding, colStride, rowStride,
                                colDilation, rowDilation, patchOffset, patchesCount, pIm2ColBufferData, pPatchCoeffData);
                    /*for(unsigned i = 0; i < patchesCount; ++i)
                        std::cout << "PCC: " << pPatchCoeffData[i] << std::endl;*/

                    const int m = patchesCount;
                    const int n = int(filter.dim_size(3));
                    const int k = int(patchLength);
                    const int lda = int(patchLength);
                    const int ldb = int(filter.dim_size(3));
                    const int ldc = int(filter.dim_size(3));

                    ApproxConvGEMMKernel<Device, T, AT, ApproxOpType>()(
                                ctx->eigen_device<Device>(), approxOp,
                                m, n, k, T(0),
                                pIm2ColBufferData, lda,
                                filter.flat<T>().data(), ldb,
                                pPatchCoeffData, pFilterCoeffData,
                                T(0), pOutputData, ldc);

                    pOutputData += patchesPerChunk * filter.dim_size(3);
                }
            }
        }
    }
};

/**
 * Launch approximate 2D convolution on CPU device.
 */
template<typename T, typename AT, template<typename, typename, typename> class ApproxOpType>
struct LaunchApproxConv2DOp<CPUDevice, T, AT, ApproxOpType> {
    void operator()(OpKernelContext *ctx,
                    const Tensor &input, const Tensor &filter,
                    int rowPadding, int colPadding,
                    int rowDilation, int colDilation, int rowStride, int colStride,
                    const Padding &padding, const std::vector<int64> &explicitPaddings, Tensor *output,
                    TensorFormat dataFormat, const ApproxOpType<CPUDevice, T, AT> &approxOp)
    {
        //std::cout << "ApproxConv2DOp on CPU" << std::endl;

        if(dataFormat != FORMAT_NHWC)
        {
            ctx->SetStatus(errors::Unimplemented("Generic conv implementation only supports "
                                                 "NHWC tensor format for now."));
            return;
        }

        const int64 inDepth = GetTensorDim(input, dataFormat, 'C');
        OP_REQUIRES(ctx, inDepth == filter.dim_size(2),
                    errors::Unimplemented("Generic conv implementation does not "
                                          "support grouped convolutions for now."));

        for(int64 explicitPadding: explicitPaddings)
        {
            if(!FastBoundsCheck(explicitPadding, std::numeric_limits<int>::max())) {
                ctx->SetStatus(errors::InvalidArgument("filter too large"));
                return;
            }
        }

        LaunchGeneric<CPUDevice, T, AT, ApproxOpType>()(ctx, input, filter,
                                                        rowPadding, colPadding, rowStride, colStride,
                                                        rowDilation, colDilation, padding,
                                                        explicitPaddings, output, dataFormat, approxOp);
    }
};

#define TF_REQUIRES(EXP, STATUS)                \
  do {                                          \
    if (!TF_PREDICT_TRUE(EXP)) return (STATUS); \
} while (false)

Status tensorflow::InitApproxConv2DParameters(const OpKernelConstruction *ctx,
                                              ApproxConv2DParameters *params)
{
    TF_RETURN_IF_ERROR(ctx->GetAttr("dilations", &params->dilations));
    TF_RETURN_IF_ERROR(ctx->GetAttr("strides", &params->strides));
    TF_RETURN_IF_ERROR(ctx->GetAttr("padding", &params->padding));
    if(ctx->HasAttr("explicit_paddings"))
    {
        TF_RETURN_IF_ERROR(ctx->GetAttr("explicit_paddings", &params->explicitPaddings));
    }

    string dataFormatString;
    TF_RETURN_IF_ERROR(ctx->GetAttr("data_format", &dataFormatString));
    TF_REQUIRES(FormatFromString(dataFormatString, &params->dataFormat),
                errors::InvalidArgument("Invalid data format"));

    const auto &strides = params->strides;
    const auto &dilations = params->dilations;
    const auto &dataFormat = params->dataFormat;

    TF_REQUIRES(dilations.size() == 4,
                errors::InvalidArgument("Sliding window dilations field must "
                                        "specify 4 dimensions"));

    TF_REQUIRES(strides.size() == 4,
                errors::InvalidArgument("Sliding window strides must "
                                        "specify 4 dimensions"));

    const int64 strideN = GetTensorDim(strides, dataFormat, 'N');
    const int64 strideC = GetTensorDim(strides, dataFormat, 'C');
    const int64 strideH = GetTensorDim(strides, dataFormat, 'H');
    const int64 strideW = GetTensorDim(strides, dataFormat, 'W');

    TF_REQUIRES(strideN == 1 && strideC == 1,
                errors::InvalidArgument("Current implementation does not yet support "
                                        "strides in the batch and depth dimensions."));
    TF_REQUIRES(strideH > 0 && strideW > 0,
                errors::InvalidArgument("Row and column strides should be larger than 0."));

    const int64 dilationN = GetTensorDim(dilations, dataFormat, 'N');
    const int64 dilationC = GetTensorDim(dilations, dataFormat, 'C');
    const int64 dilationH = GetTensorDim(dilations, dataFormat, 'H');
    const int64 dilationW = GetTensorDim(dilations, dataFormat, 'W');

    TF_REQUIRES(dilationN == 1 && dilationC == 1,
                errors::InvalidArgument("Current implementation does not yet support "
                                        "dilations in the batch and depth dimensions."));

    TF_REQUIRES(dilationH > 0 && dilationW > 0,
                errors::InvalidArgument("Dilated rates should be larger than 0."));

    TF_RETURN_IF_ERROR(CheckValidPadding(params->padding, params->explicitPaddings,
                                         /*num_dims=*/4, dataFormat));

    return Status::OK();
}

Status tensorflow::ComputeApproxConv2DDimensions(const ApproxConv2DParameters &params,
                                                 const Tensor &input, const Tensor &filter,
                                                 ApproxConv2DDimensions *dimensions)
{
    TF_REQUIRES(input.dims() == 4,
                errors::InvalidArgument("input must be 4-dimensional",
                                        input.shape().DebugString()));
    TF_REQUIRES(filter.dims() == 4,
                errors::InvalidArgument("filter must be 4-dimensional",
                                        filter.shape().DebugString()));

    for(int i = 0; i < 3; ++i)
    {
        TF_REQUIRES(FastBoundsCheck(filter.dim_size(i), std::numeric_limits<int>::max()),
                    errors::InvalidArgument("filter is too large"));
    }

    const int64 inDepthRaw = GetTensorDim(input, params.dataFormat, 'C');
    const int64 patchDepthRaw = filter.dim_size(2);

    TF_REQUIRES(FastBoundsCheck(inDepthRaw, std::numeric_limits<int>::max()),
                errors::InvalidArgument("Input depth is too large"));
    TF_REQUIRES(FastBoundsCheck(patchDepthRaw, std::numeric_limits<int>::max()),
                errors::InvalidArgument("Patch depth too large"));

    const int inDepth = static_cast<int>(inDepthRaw);
    const int patchDepth = static_cast<int>(patchDepthRaw);
    TF_REQUIRES(inDepth % patchDepth == 0,
                errors::InvalidArgument(
                    "input depth must be evenly divisible by filter depth: ",
                    inDepth, " vs ", patchDepth));

    const int outDepth = static_cast<int>(filter.dim_size(3));

    const int64 inputRowsRaw = GetTensorDim(input, params.dataFormat, 'H');
    TF_REQUIRES(FastBoundsCheck(inputRowsRaw, std::numeric_limits<int>::max()),
                errors::InvalidArgument("Input rows too large"));
    const int inputRows = static_cast<int>(inputRowsRaw);
    const int filterRows = static_cast<int>(filter.dim_size(0));

    const int64 inputColsRaw = GetTensorDim(input, params.dataFormat, 'W');
    TF_REQUIRES(FastBoundsCheck(inputColsRaw, std::numeric_limits<int>::max()),
                errors::InvalidArgument("Input cols too large"));
    const int inputCols = static_cast<int>(inputColsRaw);
    const int filterCols = static_cast<int>(filter.dim_size(1));

    const int64 batchRaw = GetTensorDim(input, params.dataFormat, 'N');
    TF_REQUIRES(FastBoundsCheck(batchRaw, std::numeric_limits<int>::max()),
                errors::InvalidArgument("batch is too large"));
    const int batch = static_cast<int>(batchRaw);

    const int strideRows = GetTensorDim(params.strides, params.dataFormat, 'H');
    const int strideCols = GetTensorDim(params.strides, params.dataFormat, 'W');
    const int dilationRows = GetTensorDim(params.dilations, params.dataFormat, 'H');
    const int dilationCols = GetTensorDim(params.dilations, params.dataFormat, 'W');

    int64 padRowsBefore, padRowsAfter, padColsBefore, padColsAfter;
    if(params.padding == Padding::EXPLICIT)
    {
        GetExplicitPaddingForDim(params.explicitPaddings, params.dataFormat, 'H',
                                 &padRowsBefore, &padRowsAfter);
        GetExplicitPaddingForDim(params.explicitPaddings, params.dataFormat, 'W',
                                 &padColsBefore, &padColsAfter);
    }

    int64 outRows = 0, outCols = 0;
    TF_RETURN_IF_ERROR(GetWindowedOutputSizeVerboseV2(
                           inputRows, filterRows, dilationRows, strideRows, params.padding,
                           &outRows, &padRowsBefore, &padRowsAfter));
    TF_RETURN_IF_ERROR(GetWindowedOutputSizeVerboseV2(
                           inputCols, filterCols, dilationCols, strideCols, params.padding,
                           &outCols, &padColsBefore, &padColsAfter));

    dimensions->batch = batch;
    dimensions->inputRows = inputRows;
    dimensions->inputCols = inputCols;
    dimensions->inDepth = inDepth;
    dimensions->filterRows = filterRows;
    dimensions->filterCols = filterCols;
    dimensions->patchDepth = patchDepth;
    dimensions->outDepth = outDepth;
    dimensions->strideRows = strideRows;
    dimensions->strideCols = strideCols;
    dimensions->dilationRows = dilationRows;
    dimensions->dilationCols = dilationCols;
    dimensions->outRows = outRows;
    dimensions->outCols = outCols;
    dimensions->padRowsBefore = padRowsBefore;
    dimensions->padRowsAfter = padRowsAfter;
    dimensions->padColsBefore = padColsBefore;
    dimensions->padColsAfter = padColsAfter;

    return Status::OK();
}

#undef TF_REQUIRES

/**
 * Approximate 2D convolution for generic device.
 */
template<typename Device, typename T, typename AT, template<typename, typename, typename> class ApproxOpType>
class ApproxConv2DOp : public OpKernel {
public:
    explicit ApproxConv2DOp(OpKernelConstruction *ctx) : OpKernel(ctx)
        , m_approxOp(ctx)
    {
        OP_REQUIRES_OK(ctx, InitApproxConv2DParameters(ctx, &m_params));
    }

    void Compute(OpKernelContext *ctx) override {
        const Tensor &input = ctx->input(0);
        const Tensor &filter = ctx->input(1);

        ApproxConv2DDimensions dimensions;
        OP_REQUIRES_OK(ctx, ComputeApproxConv2DDimensions(m_params, input, filter,
                                                          &dimensions));

        TensorShape outShape = ShapeFromFormat(
                    m_params.dataFormat, dimensions.batch, dimensions.outRows,
                    dimensions.outCols, dimensions.outDepth);

        Tensor *output = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, outShape, &output));

        VLOG(2) << "ApproxConv2D: inDepth = " << dimensions.inDepth
                << ", parchDepth = " << dimensions.patchDepth
                << ", inputCols = " << dimensions.inputCols
                << ", filterCols = " << dimensions.filterCols
                << ", inputRows = " << dimensions.inputRows
                << ", filterRows = " << dimensions.filterRows
                << ", strideRows = " << dimensions.strideRows
                << ", strideCols = " << dimensions.strideCols
                << ", dilationRows = " << dimensions.dilationRows
                << ", dilationCols = " << dimensions.dilationCols
                << ", outDepth = " << dimensions.outDepth;

        //std::cout << "PADDING: " << dimensions.padRowsBefore << " " << dimensions.padColsBefore << std::endl;

        if(outShape.num_elements() == 0)
            return;

        m_approxOp.Update(ctx);

        // TODO: Handle "explicit" padding
        m_launcher(ctx, input, filter,
                   dimensions.padRowsBefore, dimensions.padColsBefore,
                   dimensions.dilationRows, dimensions.dilationCols,
                   dimensions.strideRows, dimensions.strideCols, m_params.padding,
                   m_params.explicitPaddings, output, m_params.dataFormat, m_approxOp);
    }

private:
    ApproxConv2DParameters m_params;
    LaunchApproxConv2DOp<Device, T, AT, ApproxOpType> m_launcher;
    ApproxOpType<Device, T, AT> m_approxOp;

    TF_DISALLOW_COPY_AND_ASSIGN(ApproxConv2DOp);
};

#define REGISTER_CPU(T) \
    REGISTER_KERNEL_BUILDER( \
        Name("ApproxConv2D").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
        ApproxConv2DOp<CPUDevice, T, T, NullApproxOpType_t>);

#ifdef ALLOW_CPU_FOR_APPROX_CONV
TF_CALL_half(REGISTER_CPU);
TF_CALL_float(REGISTER_CPU);
TF_CALL_double(REGISTER_CPU);
#endif // ALLOW_CPU_FOR_APPROX_CONV
#undef REGISTER_CPU

namespace tensorflow {
template struct LaunchApproxConv2DOp<CPUDevice, Eigen::half, Eigen::half, NullApproxOpType_t>;
template struct LaunchApproxConv2DOp<CPUDevice, float,       float,       NullApproxOpType_t>;
template struct LaunchApproxConv2DOp<CPUDevice, double,      double,      NullApproxOpType_t>;
}

// Register approximated 2D Convolution with min/max variable inputs on GPU
#define REGISTER_CPU(T) \
    REGISTER_KERNEL_BUILDER( \
        Name("ApproxConv2DWithMinMaxVars") \
            .Device(DEVICE_CPU)      \
            .TypeConstraint<T>("T"), \
        ApproxConv2DOp<CPUDevice, T, uint8, TableApproxOpType_t>);

#ifdef ALLOW_CPU_FOR_APPROX_CONV
TF_CALL_float(REGISTER_CPU);
#endif // ALLOW_CPU_FOR_APPROX_CONV
#undef REGISTER_CPU

namespace tensorflow {
template struct LaunchApproxConv2DOp<CPUDevice, float, uint8, TableApproxOpType_t>;
}

#if GOOGLE_CUDA

/**
 * Specialization of Approximate 2D convolution for generic device to CUDA GPUs.
 */
template<typename T, typename AT, template<typename, typename, typename> class ApproxOpType>
void LaunchApproxConv2DOp<GPUDevice, T, AT, ApproxOpType>::operator()(OpKernelContext *ctx,
                            const Tensor &inputParam, const Tensor &filter,
                            int rowPadding, int colPadding,
                            int rowDilation, int colDilation, int rowStride, int colStride,
                            const Padding &padding, const std::vector<int64> &explicitPaddings, Tensor *output,
                            TensorFormat dataFormat, const ApproxOpType<GPUDevice, T, AT> &approxOp)
{
    //std::cout << "ApproxConv2DOp on GPU" << std::endl;

    auto *stream = ctx->op_device_context()->stream();
    OP_REQUIRES(ctx, stream, errors::Internal("No GPU stream available."));

    Tensor input = inputParam;
    const int64 inBatch  = GetTensorDim(input, dataFormat, 'N');
    int64 inRows         = GetTensorDim(input, dataFormat, 'H');
    int64 inCols         = GetTensorDim(input, dataFormat, 'W');
    const int64 inDepths = GetTensorDim(input, dataFormat, 'C');

    const int64 patchRows   = filter.dim_size(0);
    const int64 patchCols   = filter.dim_size(1);
    const int64 patchDepths = filter.dim_size(2);

    bool isGroupedConvolution = (patchDepths != inDepths);

    // Upload lookup table to the GPU and create 1D texture for lookups


    if(patchRows == 1 && patchCols == 1 && !isGroupedConvolution &&
            rowDilation == 1 && colDilation == 1 && rowStride == 1 &&
            colStride == 1 && dataFormat == FORMAT_NHWC &&
            (padding == VALID || padding == SAME))
    {
        const int m = int(inBatch * inRows * inCols);
        const int k = int(patchDepths);
        const int n = int(filter.dim_size(3));

        /*auto pA = AsDeviceMemory(input.template flat<T>().data(), input.template flat<T>().size());
        auto pB = AsDeviceMemory(filter.template flat<T>().data(), filter.template flat<T>().size());
        auto pC = AsDeviceMemory(output->template flat<T>().data(), output->template flat<T>().size());*/

        ApproxConvGEMMKernelCombined<GPUDevice, T, AT, ApproxOpType>()(ctx->eigen_device<GPUDevice>(), approxOp,
                                                                       m, n, k, T(0),
                                                                       input.flat<T>().data(), k,
                                                                       filter.flat<T>().data(), n,
                                                                       T(0), output->flat<T>().data(), n);

        return;
    }
    else if(patchRows == inRows && patchCols == inCols &&
            !isGroupedConvolution && rowDilation == 1 &&
            colDilation == 1 && padding == VALID &&
            dataFormat == FORMAT_NHWC)
    {
        const int m = int(inBatch);
        const int k = int(patchRows * patchCols * patchDepths);
        const int n = int(filter.dim_size(3));

        /*auto pA = AsDeviceMemory(input.template flat<T>().data(), input.template flat<T>().size());
        auto pB = AsDeviceMemory(filter.template flat<T>().data(), filter.template flat<T>().size());
        auto pC = AsDeviceMemory(output->template flat<T>().data(), output->template flat<T>().size());*/

        ApproxConvGEMMKernelCombined<GPUDevice, T, AT, ApproxOpType>()(ctx->eigen_device<GPUDevice>(), approxOp,
                                                                       m, n, k, T(0),
                                                                       input.flat<T>().data(), k,
                                                                       filter.flat<T>().data(), n,
                                                                       T(0), output->flat<T>().data(), n);

        return;
    }
    else
    {
        const int64 kMaxChunkSize = (16 * 1024 * 1024) / sizeof(T);
        int64 patchLength  = filter.dim_size(0) * filter.dim_size(1) * filter.dim_size(2);
        OP_REQUIRES(ctx, (patchLength) <= kMaxChunkSize,
                    errors::InvalidArgument("ApproxConv2DOp: Patch too large for buffer"));

        int64 totalPatchesCount = output->dim_size(0) * output->dim_size(1) * output->dim_size(2);
        const int64 patchesPerChunk = kMaxChunkSize / patchLength;

        Tensor im2ColTmpBuffer;

        {
            TensorShape shape;
            TensorShapeUtils::MakeShape(&kMaxChunkSize, 1, &shape);
            OP_REQUIRES_OK(ctx, ctx->allocate_temp(TempBufferType_t<AT>::Get(), shape, &im2ColTmpBuffer));
        }

        Tensor patchCoeffTmpBuffer;

        {
            TensorShape shape;
            TensorShapeUtils::MakeShape(&patchesPerChunk, 1, &shape);
            OP_REQUIRES_OK(ctx, ctx->allocate_temp(TempBufferType_t<T>::Get(), shape, &patchCoeffTmpBuffer));
        }

        Tensor filterCoeffTmpBuffer;

        {
            TensorShape shape;
            int numFilters = int(filter.dim_size(3));
            TensorShapeUtils::MakeShape(&numFilters, 1, &shape);
            OP_REQUIRES_OK(ctx, ctx->allocate_temp(TempBufferType_t<T>::Get(), shape, &filterCoeffTmpBuffer));
        }

        AT *pIm2ColBufferData = im2ColTmpBuffer.flat<AT>().data();
        T *pPatchCoeffData    = patchCoeffTmpBuffer.flat<T>().data();
        T *pFilterCoeffData   = filterCoeffTmpBuffer.flat<T>().data();

        ApproxFilterCorrCoeff<GPUDevice, T, AT, ApproxOpType>()(
                    ctx->eigen_device<GPUDevice>(), approxOp, filter.tensor<T, 4>(), filterCoeffTmpBuffer.flat<T>());

        if(padding == EXPLICIT)
        {
            ctx->SetStatus(errors::Unimplemented("Explicit padding is not implemented."));
        }
        else
        {
            T *pOutputData = output->flat<T>().data();

            for(int64 i = 0; i < totalPatchesCount; i += patchesPerChunk)
            {
                int64 batch = i / (output->dim_size(1) * output->dim_size(2));
                int patchOffset = int(i % (output->dim_size(1) * output->dim_size(2)));
                int patchesCount = int(std::min(patchesPerChunk, totalPatchesCount - i));

                const T *pInputData = input.flat<T>().data() + batch * input.dim_size(1) * input.dim_size(2) * input.dim_size(3);

                ApproxConvIm2ColKernel<GPUDevice, T, AT, ApproxOpType>()(
                            ctx->eigen_device<GPUDevice>(), approxOp,
                            pInputData, input.dim_size(3), input.dim_size(2), input.dim_size(1),
                            output->dim_size(2), output->dim_size(1),
                            filter.dim_size(1), filter.dim_size(0), colPadding, rowPadding, colStride, rowStride,
                            colDilation, rowDilation, patchOffset, patchesCount, pIm2ColBufferData, pPatchCoeffData);

                const int m = patchesCount;
                const int n = int(filter.dim_size(3));
                const int k = int(patchLength);
                const int lda = int(patchLength);
                const int ldb = int(filter.dim_size(3));
                const int ldc = int(filter.dim_size(3));

                ApproxConvGEMMKernel<GPUDevice, T, AT, ApproxOpType>()(
                            ctx->eigen_device<GPUDevice>(), approxOp,
                            m, n, k, T(0),
                            pIm2ColBufferData, lda,
                            filter.flat<T>().data(), ldb,
                            pPatchCoeffData, pFilterCoeffData,
                            T(0), pOutputData, ldc);

                pOutputData += patchesPerChunk * filter.dim_size(3);
            }
        }
    }
}

// Register normal approximated 2D Convolution
#define REGISTER_GPU(T) \
    REGISTER_KERNEL_BUILDER( \
        Name("ApproxConv2D").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
        ApproxConv2DOp<GPUDevice, T, T, NullApproxOpType_t>);

#ifdef ALLOW_GPU_FOR_APPROX_CONV
TF_CALL_half(REGISTER_GPU);
TF_CALL_float(REGISTER_GPU);
TF_CALL_double(REGISTER_GPU);
#endif // ALLOW_GPU_FOR_APPROX_CONV
#undef REGISTER_GPU

namespace tensorflow {
template struct LaunchApproxConv2DOp<GPUDevice, Eigen::half, Eigen::half, NullApproxOpType_t>;
template struct LaunchApproxConv2DOp<GPUDevice, float,       float,       NullApproxOpType_t>;
template struct LaunchApproxConv2DOp<GPUDevice, double,      double,      NullApproxOpType_t>;
}

// Register approximated 2D Convolution with min/max variable inputs on GPU
#define REGISTER_GPU(T) \
    REGISTER_KERNEL_BUILDER( \
        Name("ApproxConv2DWithMinMaxVars") \
            .Device(DEVICE_GPU)        \
            .TypeConstraint<T>("T")    \
            .HostMemory("input_min")   \
            .HostMemory("input_max")   \
            .HostMemory("filter_min")  \
            .HostMemory("filter_max"), \
        ApproxConv2DOp<GPUDevice, T, uint8, TableApproxOpType_t>);

#ifdef ALLOW_GPU_FOR_APPROX_CONV
TF_CALL_float(REGISTER_GPU);
#endif // ALLOW_GPU_FOR_APPROX_CONV

namespace tensorflow {
template struct LaunchApproxConv2DOp<GPUDevice, float, uint8, TableApproxOpType_t>;
}

#endif // GOOGLE_CUDA
