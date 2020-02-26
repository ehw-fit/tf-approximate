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

#include "approx_gemm_functors.h"

using namespace tensorflow;

template<class T1, class T2, class T3, class TGemmFunctor>
class Im2ColApproxConvFunctor {
public:
    void operator()(OpKernelContext *ctx, const T1 *inputData,
                    int inputBatches, int inputHeight, int inputWidth, int inputDepth,
                    const T2 *filterData, int filterHeight, int filterWidth, int filterCount,
                    int strideRows, int strideCols, Padding padding,
                    T3 *outputData, int outputHeight, int outputWidth) {
        if((inputBatches <= 0) || (inputWidth <= 0) || (inputHeight <= 0) ||
                (inputDepth <= 0)) {
            LOG(WARNING) << "ApproxConv2D was called with bad input dimensions: "
                         << inputBatches << ", " << inputHeight << ", "
                         << inputWidth << ", " << inputDepth;
            return;
        }

        if((filterWidth <= 0) || (filterHeight <= 0) || (filterCount <= 0)) {
            LOG(WARNING) << "ApproxConv2D was called with bad filter dimensions: "
                         << filterWidth << ", " << filterHeight << ", "
                         << filterCount;
            return;
        }

        if((outputWidth <= 0) || (outputHeight <= 0)) {
            LOG(WARNING) << "ApproxConv2D was called with bad output width or height: "
                         << outputWidth << ", " << outputHeight;
            return;
        }

        if(filterHeight == 1 && filterWidth == 1 && strideRows == 1 &&
                strideCols == 1) {
            const int m = inputBatches * inputHeight * inputWidth;
            const int n = filterCount;
            const int k = inputDepth;
            const int lda = k;
            const int ldb = filterCount;
            const int ldc = filterCount;

            TGemmFunctor gemmFunctor;
            gemmFunctor(ctx, m, n, k, inputData, lda, filterData, ldb,
                        outputData, ldc);
            return;
        }
        else if(filterHeight == inputHeight && filterWidth == inputWidth &&
                padding == VALID) {
            const int m = inputBatches;
            const int n = filterCount;
            const int k = inputHeight * inputWidth * inputDepth;
            const int lda = k;
            const int ldb = filterCount;
            const int ldc = filterCount;

            TGemmFunctor gemmFunctor;
            gemmFunctor(ctx, m, n, k, inputData, lda, filterData, ldb,
                        outputData, ldc);
            return;
        }

        int filterLeftOffset, filterTopOffset;
        if(padding == VALID) {
            filterLeftOffset = ((outputWidth  - 1) * strideCols + filterWidth  - inputWidth  + 1) / 2;
            filterTopOffset  = ((outputHeight - 1) * strideRows + filterHeight - inputHeight + 1) / 2;
        }
        else {
            filterLeftOffset = ((outputWidth  - 1) * strideCols + filterWidth  - inputWidth)  / 2;
            filterTopOffset  = ((outputHeight - 1) * strideRows + filterHeight - inputHeight) / 2;
        }

        const int64 kMaxChunkSize = 16 * 1024 * 1024;
        const int filterValueCount = filterWidth * filterHeight * inputDepth;
        OP_REQUIRES(ctx, (filterValueCount * sizeof(T1)) <= kMaxChunkSize,
                    errors::InvalidArgument("Im2ColApprox patch too large for buffer"));
        const int64 patchesPerChunk = kMaxChunkSize / (filterValueCount * sizeof(T1));
        const int64 chunkValueCount = (kMaxChunkSize + (sizeof(T1) - 1)) / sizeof(T1);

        Tensor im2ColTmpBuffer;
        TensorShape im2ColTmpBufferShape;
        TensorShapeUtils::MakeShape(&chunkValueCount, 1, &im2ColTmpBufferShape);
        OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataType::DT_FLOAT, im2ColTmpBufferShape, &im2ColTmpBuffer));

        T1 *pIm2ColBufferData = im2ColTmpBuffer.flat<T1>().data();

        const int64 patchCount = (inputBatches * outputHeight * outputWidth);
        const int64 chunkCount = (patchCount + (patchesPerChunk - 1)) / patchesPerChunk;

        for(int64 chunkIndex = 0; chunkIndex < chunkCount; ++chunkIndex)
        {
            const int64 patchIndexStart = chunkIndex * patchesPerChunk;
            const int64 patchIndexEnd   = std::min(patchIndexStart + patchesPerChunk, patchCount);

            for(int64 patchIndex = patchIndexStart; patchIndex < patchIndexEnd; ++patchIndex)
            {
                const int64 batch = patchIndex / (outputHeight * outputWidth);
                const int64 outY  = (patchIndex / outputWidth) % outputHeight;
                const int64 outX  = patchIndex % outputWidth;

                const T1 *inputBatchStart = inputData + (batch * inputHeight * inputWidth * inputDepth);

                const int inYOrigin = (outY * strideRows) - filterTopOffset;
                const int inXOrigin = (outX * strideCols) - filterLeftOffset;
                const int patchIndexWithinChunk = patchIndex % patchesPerChunk;
                T1 *im2colPatchStart = pIm2ColBufferData + (patchIndexWithinChunk * filterValueCount);

                for(int filterY = 0; filterY < filterHeight; ++filterY)
                {
                    const int inY = inYOrigin + filterY;
                    T1 *im2ColRowStart = im2colPatchStart + (filterY * filterWidth * inputDepth);

                    if((inY < 0) || (inY >= inputHeight))
                    {
                        T1 *im2ColRowEnd = im2ColRowStart + (filterWidth * inputDepth);
                        std::fill(im2ColRowStart, im2ColRowEnd, T1(0));
                    }
                    else
                    {
                        const int inXEnd = inXOrigin + filterWidth;
                        const int leftZeroCount   = std::max(0, 0 - inXOrigin);
                        const int rightZeroCount  = std::max(0, inXEnd - inputWidth);
                        const int centerCopyCount = filterWidth - (leftZeroCount + rightZeroCount);

                        if(leftZeroCount > 0)
                        {
                            T1 *im2ColLeftStart = im2ColRowStart;
                            T1 *im2ColLeftEnd   = im2ColLeftStart + (leftZeroCount * inputDepth);

                            std::fill(im2ColLeftStart, im2ColLeftEnd, T1(0));
                        }

                        if(centerCopyCount > 0)
                        {
                            const T1 *inputRowStart = inputBatchStart + (inY * inputWidth * inputDepth) +
                                    (std::max(0, inXOrigin) * inputDepth);
                            const T1 *inputRowEnd   = inputRowStart + (centerCopyCount * inputDepth);
                            T1 *im2ColCenterStart = im2ColRowStart + (leftZeroCount * inputDepth);

                            std::copy(inputRowStart, inputRowEnd, im2ColCenterStart);
                        }

                        if(rightZeroCount > 0)
                        {
                            T1 *im2ColRightStart = im2ColRowStart   + ((leftZeroCount + centerCopyCount) * inputDepth);
                            T1 *im2ColRightEnd   = im2ColRightStart + (rightZeroCount * inputDepth);

                            std::fill(im2ColRightStart, im2ColRightEnd, T1(0));
                        }
                    }
                }
            }

            const int howManyPatches = patchIndexEnd - patchIndexStart;
            const int m = howManyPatches;
            const int n = filterCount;
            const int k = filterValueCount;
            const int lda = filterValueCount;
            const int ldb = filterCount;
            const int ldc = filterCount;

            T3 *chunkOutputData = outputData + (patchIndexStart * filterCount);
            TGemmFunctor gemmFunctor;
            gemmFunctor(ctx, m, n, k, pIm2ColBufferData, lda, filterData, ldb,
                        chunkOutputData, ldc);
        }
    }
};

template<class T, class TConvFunctor>
class ApproxConv2DUsingGemmOp : public BinaryOp<T> {
public:
    explicit ApproxConv2DUsingGemmOp(OpKernelConstruction *ctx) : BinaryOp<T>(ctx)
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

        TConvFunctor convFunctor;
        convFunctor(ctx, input.flat<T>().data(), batch, inputRows, inputCols,
                    inDepth, filter.flat<T>().data(), filterRows, filterCols,
                    outDepth, strideRows, strideCols, m_padding,
                    output->flat<T>().data(), outRows, outCols);
    }

private:
    std::vector<int32> m_strides;
    Padding m_padding;
    TensorFormat m_dataFormat;

    TF_DISALLOW_COPY_AND_ASSIGN(ApproxConv2DUsingGemmOp);
};

#define REGISTER_CPU(T)                                                     \
    REGISTER_KERNEL_BUILDER(                                                \
        Name("ApproxConv2D").Device(DEVICE_CPU).TypeConstraint<T>("T"),     \
        ApproxConv2DUsingGemmOp<T, Im2ColApproxConvFunctor<T, T, T, ReferenceGemmFunctor<T, T, T> > >);

#if 0
TF_CALL_half(REGISTER_CPU);
TF_CALL_float(REGISTER_CPU);
#endif // USE_GEMM_FOR_CONV
#undef REGISTER_CPU
