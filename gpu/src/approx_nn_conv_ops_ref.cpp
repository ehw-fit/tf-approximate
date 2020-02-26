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

#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/ops_util.h>
#include <tensorflow/core/framework/numeric_op.h>
#include <tensorflow/core/framework/bounds_check.h>
#include <tensorflow/core/framework/register_types.h>

#include <tensorflow/core/util/padding.h>

#include "approx_ops_types.h"

using namespace tensorflow;
using CPUDevice = Eigen::ThreadPoolDevice;

template<class T, class AT, template<typename, typename, typename> class ApproxOpType, class TConvFunctor>
class ApproxConv2DRefOp : public OpKernel {
public:
    explicit ApproxConv2DRefOp(OpKernelConstruction *ctx) : OpKernel(ctx)
      , m_approxOp(ctx)
    {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("strides", &m_strides));
        string dataFormat;
        OP_REQUIRES_OK(ctx, ctx->GetAttr("data_format", &dataFormat));
        OP_REQUIRES(ctx, FormatFromString(dataFormat, &m_dataFormat),
                    errors::InvalidArgument("Invalid data format"));
        OP_REQUIRES(ctx, m_dataFormat == FORMAT_NHWC,
                    errors::InvalidArgument("Data format not supported by this kernel", dataFormat));
        OP_REQUIRES(ctx, m_strides.size() == 4,
                    errors::InvalidArgument("Sliding window strides field must "
                                            "specify 4 dimensions"));
        const int64 strideN = GetTensorDim(m_strides, m_dataFormat, 'N');
        const int64 strideC = GetTensorDim(m_strides, m_dataFormat, 'C');

        OP_REQUIRES(ctx, strideN == 1 && strideC == 1,
                    errors::InvalidArgument("Current implementation does not yet support "
                                            "strides in the batch and depth dimensions."));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("padding", &m_padding));
    }

    void Compute(OpKernelContext *ctx) override
    {
        const Tensor &input = ctx->input(0);
        const Tensor &filter = ctx->input(1);

        OP_REQUIRES(ctx, input.dims() == 4,
                    errors::InvalidArgument("input must be 4-dimensional",
                                            input.shape().DebugString()));
        OP_REQUIRES(ctx, filter.dims() == 4,
                    errors::InvalidArgument("filter must be 4-dimensional: ",
                                            filter.shape().DebugString()));

        for(int i = 0; i < 3; ++i)
        {
            OP_REQUIRES(ctx, FastBoundsCheck(filter.dim_size(i), std::numeric_limits<int>::max()),
                        errors::InvalidArgument("filter too large"));
        }

        const int64 inDepth = GetTensorDim(input, m_dataFormat, 'C');
        OP_REQUIRES(ctx, inDepth == filter.dim_size(2),
                    errors::InvalidArgument("input and filter must have same depth: ",
                                            inDepth, " vs ", filter.dim_size(2)));

        const int outDepth = static_cast<int>(filter.dim_size(3));

        const int64 inputRowsRaw = GetTensorDim(input, m_dataFormat, 'H');
        OP_REQUIRES(ctx, FastBoundsCheck(inputRowsRaw, std::numeric_limits<int>::max()),
                    errors::InvalidArgument("Input rows too large"));
        const int inputRows  = static_cast<int>(inputRowsRaw);
        const int filterRows = static_cast<int>(filter.dim_size(0));

        const int64 inputColsRaw = GetTensorDim(input, m_dataFormat, 'W');
        OP_REQUIRES(ctx, FastBoundsCheck(inputColsRaw, std::numeric_limits<int>::max()),
                    errors::InvalidArgument("Input cols too large"));
        const int inputCols  = static_cast<int>(inputColsRaw);
        const int filterCols = static_cast<int>(filter.dim_size(1));

        const int64 batchRaw = GetTensorDim(input, m_dataFormat, 'N');
        OP_REQUIRES(ctx, FastBoundsCheck(batchRaw, std::numeric_limits<int>::max()),
                    errors::InvalidArgument("batch is too large"));
        const int batch = static_cast<int>(batchRaw);

        const int strideRows = GetTensorDim(m_strides, m_dataFormat, 'H');
        const int strideCols = GetTensorDim(m_strides, m_dataFormat, 'W');

        int64 outRows = 0, outCols = 0, padRows = 0, padCols = 0;
        OP_REQUIRES_OK(ctx, GetWindowedOutputSize(inputRows, filterRows, strideRows,
                                                  m_padding, &outRows, &padRows));
        OP_REQUIRES_OK(ctx, GetWindowedOutputSize(inputCols, filterCols, strideCols,
                                                  m_padding, &outCols, &padCols));

        TensorShape outShape = ShapeFromFormat(m_dataFormat, batch,
                                               outRows, outCols, outDepth);

        Tensor *output = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, outShape, &output));

        VLOG(2) << "ApproxConv2D: inDepth = " << inDepth
                << ", inputCols = " << inputCols
                << ", filterCols = " << filterCols
                << ", inputRows = " << inputRows
                << ", filterRows = " << filterRows
                << ", strideRows = " << strideRows
                << ", strideCols = " << strideCols
                << ", outDepth = " << outDepth;

        if(outShape.num_elements() == 0)
            return;

        m_approxOp.Update(ctx);

        TConvFunctor convFunctor;
        convFunctor(ctx, m_approxOp,
                    input.flat<T>().data(), batch, inputRows, inputCols,
                    inDepth, filter.flat<T>().data(), filterRows, filterCols,
                    outDepth, strideRows, strideCols, m_padding,
                    output->flat<T>().data(), outRows, outCols);
    }

private:
    ApproxOpType<CPUDevice, T, AT> m_approxOp;
    std::vector<int32> m_strides;
    Padding m_padding;
    TensorFormat m_dataFormat;

    TF_DISALLOW_COPY_AND_ASSIGN(ApproxConv2DRefOp);
};

// Data layout:
// inputData  = [inputBatches, inputHeight,  inputWidth,  inputDepth]
// filterData = [filterHeight, filterWidth,  inputDepth,  filterCount]
// outputData = [inputBatches, outputHeight, outputWidth, filterCount]
template<class T, class AT, template<typename, typename, typename> class ApproxOpType>
class ReferenceApproxConvFunctor {
public:
    void operator()(OpKernelContext *ctx, const ApproxOpType<CPUDevice, T, AT> &approxOp,
                    const T *inputData, int inputBatches, int inputHeight, int inputWidth, int inputDepth,
                    const T *filterData, int filterHeight, int filterWidth, int filterCount,
                    int strideRows, int strideCols, Padding padding,
                    T *outputData, int outputHeight, int outputWidth);
};

// Approximation-less reference convolution functor
template<class T, class AT>
class ReferenceApproxConvFunctor<T, AT, NullApproxOpType_t> {
public:
    void operator()(OpKernelContext *ctx, const NullApproxOpType_t<CPUDevice, T, AT> &approxOp,
                    const T *inputData, int inputBatches, int inputHeight, int inputWidth, int inputDepth,
                    const T *filterData, int filterHeight, int filterWidth, int filterCount,
                    int strideRows, int strideCols, Padding padding,
                    T *outputData, int outputHeight, int outputWidth)
    {
        int filterLeftOffset, filterTopOffset;

        if(padding == VALID)
        {
            filterLeftOffset = ((outputWidth - 1)  * strideCols + filterWidth  - inputWidth  + 1) / 2;
            filterTopOffset  = ((outputHeight - 1) * strideRows + filterHeight - inputHeight + 1) / 2;
        }
        else
        {
            filterLeftOffset = ((outputWidth - 1)  * strideCols + filterWidth  - inputWidth)  / 2;
            filterTopOffset  = ((outputHeight - 1) * strideRows + filterHeight - inputHeight) / 2;
        }

        // Walk through all 3D images in the batch.
        for(int batch = 0; batch < inputBatches; ++batch)
        {
            // Walk through all pixels and channels in the output 3D image.
            for(int outY = 0; outY < outputHeight; ++outY)
            {
                for(int outX = 0; outX < outputWidth; ++outX)
                {
                    // Number of output channels is given by number of specified filters.
                    for(int outChannel = 0; outChannel < filterCount; ++outChannel)
                    {
                        const int inXOrigin = (outX * strideCols) - filterLeftOffset;
                        const int inYOrigin = (outY * strideRows) - filterTopOffset;

                        T total(0);

                        // Multiply filter kernel with patch of input image around its current position,
                        // summing the results into current channel of the output pixel.
                        for(int filterY = 0; filterY < filterHeight; ++filterY)
                        {
                            for(int filterX = 0; filterX < filterWidth; ++filterX)
                            {
                                // Filter has same number of channels as an input image.
                                for(int inChannel = 0; inChannel < inputDepth; ++inChannel)
                                {
                                    const int inX = inXOrigin + filterX;
                                    const int inY = inYOrigin + filterY;

                                    T inputValue;

                                    if((inX >= 0) && (inX < inputWidth) && (inY >= 0) && (inY < inputHeight))
                                    {
                                        inputValue = inputData[(batch * inputHeight * inputWidth * inputDepth) + (inY * inputWidth * inputDepth) + (inX * inputDepth) + inChannel];
                                    }
                                    else
                                    {
                                        inputValue = T(0);
                                    }

                                    const T filterValue = filterData[(filterY * filterWidth * inputDepth * filterCount) + (filterX * inputDepth * filterCount) + (inChannel * filterCount) + outChannel];
                                    total += (inputValue * filterValue);
                                }
                            }
                        }

                        outputData[(batch * outputHeight * outputWidth * filterCount) + (outY * outputWidth * filterCount) + (outX * filterCount) + outChannel] = total;
                    }
                }
            }
        }
    }
};

// Lookup table approximated reference convolution functor
template<class T, class AT>
class ReferenceApproxConvFunctor<T, AT, TableApproxOpType_t> {
public:
    void operator()(OpKernelContext *ctx, const TableApproxOpType_t<CPUDevice, T, AT> &approxOp,
                    const T *inputData, int inputBatches, int inputHeight, int inputWidth, int inputDepth,
                    const T *filterData, int filterHeight, int filterWidth, int filterCount,
                    int strideRows, int strideCols, Padding padding,
                    T *outputData, int outputHeight, int outputWidth)
    {
        int filterLeftOffset, filterTopOffset;

        if(padding == VALID)
        {
            filterLeftOffset = ((outputWidth - 1)  * strideCols + filterWidth  - inputWidth  + 1) / 2;
            filterTopOffset  = ((outputHeight - 1) * strideRows + filterHeight - inputHeight + 1) / 2;
        }
        else
        {
            filterLeftOffset = ((outputWidth - 1)  * strideCols + filterWidth  - inputWidth)  / 2;
            filterTopOffset  = ((outputHeight - 1) * strideRows + filterHeight - inputHeight) / 2;
        }

        // Compute filter coefficient sums

        // Walk through all 3D images in the batch.
        for(int batch = 0; batch < inputBatches; ++batch)
        {
            // Walk through all pixels and channels in the output 3D image.
            for(int outY = 0; outY < outputHeight; ++outY)
            {
                for(int outX = 0; outX < outputWidth; ++outX)
                {
                    // Number of output channels is given by number of specified filters.
                    for(int outChannel = 0; outChannel < filterCount; ++outChannel)
                    {
                        const int inXOrigin = (outX * strideCols) - filterLeftOffset;
                        const int inYOrigin = (outY * strideRows) - filterTopOffset;

                        T total(0);
                        T inputPatchTotal(0);
                        T filterPatchTotal(0);

                        // Multiply filter kernel with patch of input image around its current position,
                        // summing the results into current channel of the output pixel.
                        for(int filterY = 0; filterY < filterHeight; ++filterY)
                        {
                            for(int filterX = 0; filterX < filterWidth; ++filterX)
                            {
                                // Filter has same number of channels as an input image.
                                for(int inChannel = 0; inChannel < inputDepth; ++inChannel)
                                {
                                    const int inX = inXOrigin + filterX;
                                    const int inY = inYOrigin + filterY;

                                    T inputValue;

                                    if((inX >= 0) && (inX < inputWidth) && (inY >= 0) && (inY < inputHeight))
                                        inputValue = inputData[(batch * inputHeight * inputWidth * inputDepth) + (inY * inputWidth * inputDepth) + (inX * inputDepth) + inChannel];
                                    else
                                        inputValue = T(0);

                                    const T filterValue = filterData[(filterY * filterWidth * inputDepth * filterCount) + (filterX * inputDepth * filterCount) + (inChannel * filterCount) + outChannel];

                                    inputPatchTotal  += inputValue;
                                    filterPatchTotal += filterValue; // TODO: This can be factored out as it should be constant accross all patches

                                    // Quantize values and use lookup table to compute the product
                                    const uint32 inputValueQ  = AT(((inputValue - approxOp.quantProps.input.offset) * approxOp.quantProps.input.invScale) + T(0.5));
                                    const uint32 filterValueQ = AT(((filterValue - approxOp.quantProps.filter.offset) * approxOp.quantProps.filter.invScale) + T(0.5));
                                    const uint32 prodValueQ   = approxOp.lookupTable[(inputValueQ << approxOp.bitWidth) | filterValueQ];

                                    total += T(prodValueQ);
                                }
                            }
                        }

                        total = total * approxOp.quantProps.s1xS2 +
                                approxOp.quantProps.m2 * inputPatchTotal +
                                approxOp.quantProps.m1 * filterPatchTotal -
                                T(filterHeight * filterWidth * inputDepth) * approxOp.quantProps.m1xM2;
                        outputData[(batch * outputHeight * outputWidth * filterCount) + (outY * outputWidth * filterCount) + (outX * filterCount) + outChannel] = total;
                    }
                }
            }
        }
    }
};

// Register approximation-less reference convolution functor
#define REGISTER_CPU(T)                                                     \
    REGISTER_KERNEL_BUILDER(                                                \
        Name("ApproxConv2D").Device(DEVICE_CPU).TypeConstraint<T>("T"),     \
        ApproxConv2DRefOp<T, uint8, NullApproxOpType_t, ReferenceApproxConvFunctor<T, uint8, NullApproxOpType_t> >);

#ifdef FORCE_REF_CPU_APPROX_CONV
TF_CALL_half(REGISTER_CPU);
TF_CALL_float(REGISTER_CPU);
TF_CALL_double(REGISTER_CPU);
#endif // USE_GEMM_FOR_CONV
#undef REGISTER_CPU

// Register lookup table approximated reference convolution functor
#define REGISTER_CPU(T)                                                     \
    REGISTER_KERNEL_BUILDER(                                                \
        Name("ApproxConv2DWithMinMaxVars")                                  \
            .Device(DEVICE_CPU)                                             \
            .TypeConstraint<T>("T")                                         \
            .HostMemory("input_min")                                        \
            .HostMemory("input_max")                                        \
            .HostMemory("filter_min")                                       \
            .HostMemory("filter_max"),                                      \
        ApproxConv2DRefOp<T, uint8, TableApproxOpType_t, ReferenceApproxConvFunctor<T, uint8, TableApproxOpType_t> >);

#ifdef FORCE_REF_CPU_APPROX_CONV
TF_CALL_float(REGISTER_CPU);
#endif // USE_GEMM_FOR_CONV
#undef REGISTER_CPU
