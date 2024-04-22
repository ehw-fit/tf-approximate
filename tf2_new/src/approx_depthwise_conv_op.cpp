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

#define EIGEN_USE_THREADS

#include <algorithm>
#include <cmath>
#include <type_traits>
//#undef ABSL_HAVE_STD_STRING_VIEW

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
//#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/util/use_cudnn.h"
#include "tensorflow/core/util/work_sharder.h"
#include "tensorflow/core/framework/kernel_shape_util.h"

#if GOOGLE_CUDA
//#include "third_party/gpus/cudnn/cudnn.h"
#include <tensorflow/core/platform/stream_executor.h>
#endif // GOOGLE_CUDA

#include "approx_nn_conv_ops.h"
#include "approx_depthwise_conv_op.h"
#include "approx_nn_conv_kernels.h"
#include "approx_ops_types.h"
#include "approx_ops_quant_data.h"

template<typename T, typename AT>
using CpuConvOpQuantData = ApproxConvOpQuantData<TableApproxOpType_t<Eigen::ThreadPoolDevice, T, AT> >;

template<typename Device, typename T, typename AT>
using TableApproxConvOpQuantData = ApproxConvOpQuantData<TableApproxOpType_t<Device, T, AT> >;

template<typename Packet>
inline EIGEN_DEVICE_FUNC Packet PClamp(const Packet &value, const Packet &minVal, const Packet &maxVal) {
    using namespace Eigen::internal;
    return pmin(pmax(value, minVal), maxVal);
}

template<typename OutPacket, unsigned width, typename InPacket>
inline EIGEN_DEVICE_FUNC OutPacket PClampBitWidth(const InPacket &value) {
    using namespace Eigen::internal;
    typedef typename unpacket_traits<InPacket>::type Type;
    return pcast<InPacket, OutPacket>(PClamp(value, pset1<InPacket>(Type(0)), pset1<InPacket>(Type((1 << width) - 1))));
}

namespace tensorflow {

// In depthwise convolution, one input is convolved into depth_multipler
// outputs and the outputs don't need to be reduced again like what regular
// convolution does.
//  However, the way to apply filters to inputs is exactly the same as the
// regular convolution. Please refer to the regular convolution kernels for
// more details.

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

// Computes the vectorized product of 'input_buffer' and 'filter' and stores
// result in 'output' at location specified by 'out_r' and 'out_c'.
//
// EX:
//   in_depth = 3, depth_multiplier = 2, filter [2, 2], register_width = 4
//   Both 'input_buffer' and 'filter' are padded to register-width boundaries.
//
//   input_buffer [rows, cols, in_depth, depth_multiplier]
//     [a0, a0, a1, a1] [a2, a2, 0, 0] [b0, b0, b1, b1] [b2, b2, 0, 0]
//     [e0, e0, e1, e1] [e2, e2, 0, 0] [f0, f0, f1, f1] [f2, f2, 0, 0]
//
//   filter [rows, cols, in_depth, depth_multiplier]
//     [u0, v0, w0, x0] [y0, z0, 0, 0] [u1, v1, w1, x1] [y1, z1, 0, 0]
//     [u2, v2, w2, x2] [y2, z2, 0, 0] [u3, v3, w3, x3] [y3, z3, 0, 0]
//
//   First output register [in_depth, depth_multiplier]
//     [q0, q1, q2, q3] = ([a0, a0, a1, a1] x [u0, v0, w0, x0]) +
//                        ([b0, b0, b1, b1] x [u1, v1, w1, x1]) +
//                        ([e0, e0, e1, e1] x [u2, v2, w2, x2]) +
//                        ([f0, f0, f1, f1] x [u3, v3, w3, x3])
//
// TODO(andydavis) Experiment with processing multiple inputs per input buffer.
template <typename T>
struct ApproxDepthwiseConv2DKernel {
    static void Run(const ApproxDepthwiseArgs& args,
                    const int64 padded_filter_inner_dim_size, const int64 out_r,
                    const int64 out_c, const T* filter, const T* input_buffer,
                    T* output, TensorFormat data_format, const CpuConvOpQuantData<T, uint8> &approxOpData) {

        typedef typename Eigen::internal::packet_traits<T>::type Packet;
        typedef typename Eigen::internal::packet_traits<int>::type QuantPacket;
        static const int64 kPacketSize = (sizeof(Packet) / sizeof(T));

        const typename CpuConvOpQuantData<T, uint8>::OpQuantProps_t &quantProps = approxOpData.GetQuantProps();
        const typename CpuConvOpQuantData<T, uint8>::ApproxOpType_t &lookup     = approxOpData.GetApproxOp();

        const int64 out_depth = args.out_depth;
        const int64 filter_spatial_size = args.filter_rows * args.filter_cols;
        const int64 output_scalar_size = out_depth % kPacketSize;
        const int64 output_vectorized_size =
                (out_depth / kPacketSize) * kPacketSize;
        const int64 base_output_index = (out_r * args.out_cols + out_c) * out_depth;

        for (int i = 0; i < output_vectorized_size; i += kPacketSize) {
            // Reset accumulator.
            auto vaccum  = Eigen::internal::pset1<Packet>(static_cast<T>(0));
            auto corrSum = Eigen::internal::pset1<Packet>(static_cast<T>(0));

            auto filterOffset   = Eigen::internal::pset1<Packet>(static_cast<T>(0));
            auto filterInvScale = Eigen::internal::pset1<Packet>(static_cast<T>(0));
            auto s1xS2          = Eigen::internal::pset1<Packet>(static_cast<T>(0));
            auto m1xM2          = Eigen::internal::pset1<Packet>(static_cast<T>(0));

            if(quantProps.filterMode == 1)
            {
                T filterOffsets[kPacketSize];
                T filterInvScales[kPacketSize];
                T s1xS2s[kPacketSize];
                T m1xM2s[kPacketSize];

                for(int j = 0; j < kPacketSize; ++j)
                {
                    const int multiplier = (i + j) % args.depth_multiplier;
                    filterOffsets[j]   = quantProps.pFilter[multiplier*3 + 2];
                    filterInvScales[j] = quantProps.pFilter[multiplier*3 + 1];
                    s1xS2s[j]          = quantProps.pS1xS2[multiplier];
                    m1xM2s[j]          = quantProps.pM1xM2[multiplier];
                }

                filterOffset   = Eigen::internal::ploadu<Packet>(filterOffsets);
                filterInvScale = Eigen::internal::ploadu<Packet>(filterInvScales);
                s1xS2          = Eigen::internal::ploadu<Packet>(s1xS2s);
                m1xM2          = Eigen::internal::ploadu<Packet>(m1xM2s);
            }
            else
            {
                filterOffset   = Eigen::internal::pset1<Packet>(quantProps.pFilter[2]);
                filterInvScale = Eigen::internal::pset1<Packet>(quantProps.pFilter[1]);
                s1xS2          = Eigen::internal::pset1<Packet>(quantProps.pS1xS2[0]);
                m1xM2          = Eigen::internal::pset1<Packet>(quantProps.pM1xM2[0]);
            }

            T filterCorrs[kPacketSize];

            for(int j = 0; j < kPacketSize; ++j)
            {
                const int multiplier = (i + j) % args.depth_multiplier;
                filterCorrs[j] = quantProps.pFilterCorr[multiplier];
            }

            auto filterCorr = Eigen::internal::ploadu<Packet>(filterCorrs);

            for (int j = 0; j < filter_spatial_size; ++j) {
                // Calculate index.
                const int64 index = i + j * padded_filter_inner_dim_size;
                // Load filter.
                // TODO(andydavis) Unroll 'out_c' loop in caller so we can load
                // multiple inputs here to amortize the cost of each filter block load.
                const auto filter_block =
                        Eigen::internal::ploadu<Packet>(filter + index);
                // Load input.
                const auto data_block =
                        Eigen::internal::ploadu<Packet>(input_buffer + index);
                
                // Vector precise multiply-add.
                /*vaccum = Eigen::internal::pmadd<Packet>(filter_block, data_block, vaccum);*/

                // Update input correction sum
                corrSum = Eigen::internal::padd(data_block, corrSum);

                // Quantize filter and data blocks
                const auto filter_block_quant = PClampBitWidth<QuantPacket, 8>(Eigen::internal::padd(
                    Eigen::internal::pmul(Eigen::internal::psub(filter_block, filterOffset), 
                                          filterInvScale), Eigen::internal::pset1<Packet>(T(0.5))));
                const auto data_block_quant   = PClampBitWidth<QuantPacket, 8>(Eigen::internal::padd(
                    Eigen::internal::pmul(Eigen::internal::psub(data_block, Eigen::internal::pset1<Packet>(quantProps.pInput[2])), 
                                          Eigen::internal::pset1<Packet>(quantProps.pInput[1])), Eigen::internal::pset1<Packet>(T(0.5))));
                const auto lookup_idx = Eigen::internal::por(Eigen::internal::plogical_shift_left<8>(filter_block_quant), data_block_quant);

                // Sequentially evaluate multiplications using lookup table
                T approx_mul_u[kPacketSize];
                int lookup_idx_u[kPacketSize];

                Eigen::internal::pstoreu<int>(lookup_idx_u, lookup_idx);

                for(int k = 0; k < kPacketSize; ++k)
                    approx_mul_u[k] = lookup.lookupTable[lookup_idx_u[k]];
                
                vaccum = Eigen::internal::padd<Packet>(Eigen::internal::ploadu<Packet>(approx_mul_u), vaccum);
            }

            // Add correction coefficients
            vaccum = Eigen::internal::pmul(vaccum, s1xS2);
            vaccum = Eigen::internal::padd(vaccum, Eigen::internal::pmul(filterOffset, corrSum));
            vaccum = Eigen::internal::padd(vaccum, Eigen::internal::pmul(Eigen::internal::pset1<Packet>(quantProps.pInput[2]), filterCorr));
            vaccum = Eigen::internal::padd(vaccum, Eigen::internal::pmul(Eigen::internal::pset1<Packet>(T(filter_spatial_size)), m1xM2));

            // Store vector accumulator to output.
            Eigen::internal::pstoreu<T>(output + base_output_index + i, vaccum);
        }

        if (output_scalar_size > 0) {
            auto vaccum = Eigen::internal::pset1<Packet>(static_cast<T>(0));
            auto corrSum = Eigen::internal::pset1<Packet>(static_cast<T>(0));
            const int multiplier = output_vectorized_size % args.depth_multiplier;

            const T filterOffset   = (quantProps.filterMode == 1) ? quantProps.pFilter[multiplier*3 + 2] : quantProps.pFilter[2];
            const T filterInvScale = (quantProps.filterMode == 1) ? quantProps.pFilter[multiplier*3 + 1] : quantProps.pFilter[1];
            const T s1xS2          = (quantProps.filterMode == 1) ? quantProps.pS1xS2[multiplier] : quantProps.pS1xS2[0];
            const T m1xM2          = (quantProps.filterMode == 1) ? quantProps.pM1xM2[multiplier] : quantProps.pM1xM2[0];

            for (int j = 0; j < filter_spatial_size; ++j) {
                const int64 index =
                        output_vectorized_size + j * padded_filter_inner_dim_size;
                const auto filter_block =
                        Eigen::internal::ploadu<Packet>(filter + index);
                const auto data_block =
                        Eigen::internal::ploadu<Packet>(input_buffer + index);
                //vaccum = Eigen::internal::pmadd<Packet>(filter_block, data_block, vaccum);
                
                // Update input correction sum
                corrSum = Eigen::internal::padd(data_block, corrSum);

                // Quantize filter and data blocks
                const auto filter_block_quant = PClampBitWidth<QuantPacket, 8>(Eigen::internal::padd(
                    Eigen::internal::pmul(Eigen::internal::psub(filter_block, Eigen::internal::pset1<Packet>(filterOffset)), 
                                          Eigen::internal::pset1<Packet>(filterInvScale)), Eigen::internal::pset1<Packet>(T(0.5))));
                const auto data_block_quant   = PClampBitWidth<QuantPacket, 8>(Eigen::internal::padd(
                    Eigen::internal::pmul(Eigen::internal::psub(data_block, Eigen::internal::pset1<Packet>(quantProps.pInput[2])), 
                                          Eigen::internal::pset1<Packet>(quantProps.pInput[1])), Eigen::internal::pset1<Packet>(T(0.5))));
                const auto lookup_idx = Eigen::internal::por(Eigen::internal::plogical_shift_left<8>(filter_block_quant), data_block_quant);

                // Sequentially evaluate multiplications using lookup table
                T approx_mul_u[kPacketSize];
                int lookup_idx_u[kPacketSize];

                Eigen::internal::pstoreu<int>(lookup_idx_u, lookup_idx);

                for(int k = 0; k < output_scalar_size; ++k)
                    approx_mul_u[k] = lookup.lookupTable[lookup_idx_u[k]];
                
                vaccum = Eigen::internal::padd<Packet>(Eigen::internal::ploadu<Packet>(approx_mul_u), vaccum);
            }

            // Add correction coefficients
            vaccum = Eigen::internal::pmul(vaccum, Eigen::internal::pset1<Packet>(s1xS2));
            vaccum = Eigen::internal::padd(vaccum, Eigen::internal::pmul(Eigen::internal::pset1<Packet>(filterOffset), corrSum));
            vaccum = Eigen::internal::padd(vaccum, Eigen::internal::pmul(Eigen::internal::pset1<Packet>(quantProps.pInput[2]), Eigen::internal::ploadu<Packet>(quantProps.pFilterCorr + multiplier)));
            vaccum = Eigen::internal::padd(vaccum, Eigen::internal::pset1<Packet>(T(filter_spatial_size) * m1xM2));

            // Load accumulator into an array and loop through output.
            T out_buf[kPacketSize];
            Eigen::internal::pstoreu<T>(out_buf, vaccum);
            const int64 last_output_index =
                    base_output_index + output_vectorized_size;

            for (int j = 0; j < output_scalar_size; ++j) {
                output[last_output_index + j] = out_buf[j];
            }
        }
    }
};

// Computes the depthwise conv2d of 'input' by 'depthwise_filter' and stores
// the result in 'output'. This implementation trades off copying small patches
// of the input to achieve better data alignment, which enables vectorized
// load/store and multiply-add operations (see comments at InputBufferCopyOp and
// DepthwiseConv2DKernel for details).
//
// TODO(andydavis) Evaluate the performance of processing multiple input
// patches in the inner loop.
// TODO(andydavis) Consider a zero-copy implementation for the case when
// 'in_depth' is a multiple of register width, and 'depth_multipler' is one.
// TODO(andydavis) Evaluate the performance of alternative implementations.
template<template<typename> class ApproxOpData, template<typename, typename, typename> class ApproxOpType, typename T>
struct LaunchApproxDepthwiseConvOp<ApproxOpData<ApproxOpType<CPUDevice, T, uint8> > > {
    typedef typename Eigen::internal::packet_traits<T>::type Packet;

    void operator()(OpKernelContext* ctx, const ApproxDepthwiseArgs& args,
                    const T* input, const T* depthwise_filter, T* output,
                    TensorFormat data_format, const ApproxOpData<ApproxOpType<CPUDevice, T, uint8> > &approxOpData) {
        OP_REQUIRES(ctx, data_format == FORMAT_NHWC,
                    errors::Unimplemented(
                        "Depthwise convolution on CPU is only supported for NHWC format"));

        static const int64 kPacketSize = (sizeof(Packet) / sizeof(T));

        // Pad 'depthwise_filter' to vector register width (if needed).
        const bool pad_filter = (args.out_depth % kPacketSize) == 0 ? false : true;
        Tensor padded_filter;

        if (pad_filter) {
            // Allocate space for padded filter.
            const int64 filter_spatial_size = args.filter_rows * args.filter_cols;
            const int64 padded_filter_inner_dim_size =
                    ((args.out_depth + kPacketSize - 1) / kPacketSize) * kPacketSize;
            OP_REQUIRES_OK(
                        ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                                TensorShape({filter_spatial_size,
                                                             padded_filter_inner_dim_size}),
                                                &padded_filter));

            // Write out padded filter.
            functor::ApproxDepthwiseFilterPadOp<T>()(
                        args, depthwise_filter, padded_filter.template flat<T>().data());
        }

        const T* filter_data =
                pad_filter ? padded_filter.template flat<T>().data() : depthwise_filter;

        // Computes one shard of depthwise conv2d output.
        auto shard = [&ctx, &args, &input, &filter_data, &output, data_format, &approxOpData](
                int64 start, int64 limit) {

            static const int64 kPacketSize = (sizeof(Packet) / sizeof(T));
            const int64 input_image_size =
                    args.in_rows * args.in_cols * args.in_depth;
            const int64 output_image_size =
                    args.out_rows * args.out_cols * args.out_depth;
            const int64 filter_spatial_size = args.filter_rows * args.filter_cols;
            const int64 padded_filter_inner_dim_size =
                    ((args.out_depth + kPacketSize - 1) / kPacketSize) * kPacketSize;

            // Allocate buffer for local input regions.
            Tensor input_buffer;
            OP_REQUIRES_OK(
                        ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
            TensorShape({filter_spatial_size,
                         padded_filter_inner_dim_size}), &input_buffer));
            T* input_buffer_data = input_buffer.template flat<T>().data();

            for (int64 i = start; i < limit; ++i) {
                const int64 b = i / args.out_rows;
                const int64 in_base = b * input_image_size;
                const int64 out_base = b * output_image_size;

                const int64 out_r = i % args.out_rows;

                for (int64 out_c = 0; out_c < args.out_cols; ++out_c) {
                    // Populate 'input_buffer_data' with data from local input region.
                    functor::ApproxDepthwiseInputCopyOp<T>()(args, padded_filter_inner_dim_size,
                                                             out_r, out_c, input + in_base,
                                                             input_buffer_data);

                    // Process buffered input across all filters and store to output.
                    ApproxDepthwiseConv2DKernel<T>::Run(
                        args, padded_filter_inner_dim_size, out_r, out_c, filter_data,
                        input_buffer_data, output + out_base, data_format, approxOpData);
                }
            }
        };

        const int64 total_shards = args.batch * args.out_rows;

        // Empirically tested to give reasonable performance boosts at batch size 1
        // without reducing throughput at batch size 32.
        const float kCostMultiplier = 2.5f;

        // TODO(andydavis): Estimate shard cost (in cycles) based on the number of
        // flops/loads/stores required to compute one shard.
        const int64 shard_cost = kCostMultiplier * args.out_cols * args.out_depth;

        auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
        Shard(worker_threads.num_threads, worker_threads.workers, total_shards,
              shard_cost, shard);
    }
};

// Extern template instantiated in approx_nn_conv_ops.cpp.
/*extern template struct LaunchApproxConv2DOp<CPUDevice, Eigen::half, Eigen::half, TableApproxOpType_t>;
extern template struct LaunchApproxConv2DOp<CPUDevice, float, float, TableApproxOpType_t>;
extern template struct LaunchApproxConv2DOp<CPUDevice, double, double, TableApproxOpType_t>;*/

#if GOOGLE_CUDA

// Extern template instantiated in approx_nn_conv_ops.cpp.
/*extern template struct LaunchApproxConv2DOp<GPUDevice, Eigen::half, Eigen::half, TableApproxOpType_t>;
extern template struct LaunchApproxConv2DOp<GPUDevice, float, float, TableApproxOpType_t>;
extern template struct LaunchApproxConv2DOp<GPUDevice, double, double, TableApproxOpType_t>;*/

// Extern template instantiated in approx_depthwise_conv_op_gpu.cu.
//extern template struct LaunchApproxDepthwiseConvOp<GPUDevice, Eigen::half, TableApproxOpType_t>;
extern template struct LaunchApproxDepthwiseConvOp<TableApproxConvOpQuantData<GPUDevice, float, uint8> >;
extern template struct LaunchApproxDepthwiseConvOp<TableApproxConvOpQuantData<GPUDevice, double, uint8> >;

#endif // GOOGLE_CUDA

template <typename Device, typename T, template<typename, typename, typename> class ApproxOpType>
class ApproxDepthwiseConv2DNativeOp : public OpKernel {
public:
    explicit ApproxDepthwiseConv2DNativeOp(OpKernelConstruction* context)
        : OpKernel(context)
        , m_approxOp(context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
        string data_format;
        OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
        OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
        errors::InvalidArgument("Invalid data format"));

        OP_REQUIRES(context, strides_.size() == 4,
                    errors::InvalidArgument("Sliding window strides field must "
                                            "specify 4 dimensions"));
        stride_ = GetTensorDim(strides_, data_format_, 'H');
        const int64 stride_w = GetTensorDim(strides_, data_format_, 'W');
        const int64 stride_n = GetTensorDim(strides_, data_format_, 'N');
        const int64 stride_c = GetTensorDim(strides_, data_format_, 'C');

        OP_REQUIRES(context, stride_ == stride_w,
        errors::InvalidArgument(
                        "Current implementation only supports equal length "
                        "strides in the row and column dimensions."));
        OP_REQUIRES(context, (stride_n == 1 && stride_c == 1),
                    errors::InvalidArgument("Current implementation does not yet support "
                                            "strides in the batch and depth dimensions."));
        OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));

        dtype_ = DataTypeToEnum<T>::value;
    }

    void Compute(OpKernelContext* context) override {
        typedef ApproxConvOpQuantData<ApproxOpType<Device, T, uint8> > ApproxConvOpData_t;

        // Input tensor is of the following dimensions:
        // [ batch, in_rows, in_cols, in_depth ]
        const Tensor& input = context->input(0);

        // Input filter is of the following dimensions:
        // [ filter_rows, filter_cols, in_depth, depth_multiplier]
        const Tensor& filter = context->input(1);

        // For 2D convolution, there should be 4 dimensions.
        OP_REQUIRES(context, input.dims() == 4,
                    errors::InvalidArgument("input must be 4-dimensional",
                                            input.shape().DebugString()));
        OP_REQUIRES(context, filter.dims() == 4,
                    errors::InvalidArgument("filter must be 4-dimensional: ",
                                            filter.shape().DebugString()));

        // in_depth for input and filter must match.
        const int64 in_depth = GetTensorDim(input, data_format_, 'C');
        OP_REQUIRES(context, in_depth == filter.dim_size(2),
                    errors::InvalidArgument(
                        "input and filter must have the same depth: ", in_depth,
                        " vs ", filter.dim_size(2)));

        // The last dimension for filter is depth multiplier.
        const int32 depth_multiplier = filter.dim_size(3);

        // The output depth is input depth x depth multipler
        const int32 out_depth = in_depth * depth_multiplier;

        const int64 input_rows_raw = GetTensorDim(input, data_format_, 'H');
        OP_REQUIRES(context,
                    FastBoundsCheck(input_rows_raw, std::numeric_limits<int32>::max()),
                    errors::InvalidArgument("Input rows too large"));
        const int32 input_rows = static_cast<int32>(input_rows_raw);
        const int32 filter_rows = filter.dim_size(0);

        const int64 input_cols_raw = GetTensorDim(input, data_format_, 'W');
        OP_REQUIRES(context,
                    FastBoundsCheck(input_cols_raw, std::numeric_limits<int32>::max()),
                    errors::InvalidArgument("Input cols too large"));
        const int32 input_cols = static_cast<int32>(input_cols_raw);
        const int32 filter_cols = filter.dim_size(1);

        // The first dimension for input is batch.
        const int32 batch = input.dim_size(0);

        int64 out_rows = 0, out_cols = 0, pad_rows = 0, pad_cols = 0;
        OP_REQUIRES_OK(context,
                       GetWindowedOutputSize(input_rows, filter_rows, stride_,
                                             padding_, &out_rows, &pad_rows));
        OP_REQUIRES_OK(context,
                       GetWindowedOutputSize(input_cols, filter_cols, stride_,
                                             padding_, &out_cols, &pad_cols));
        TensorShape out_shape =
                ShapeFromFormat(data_format_, batch, out_rows, out_cols, out_depth);
        OP_REQUIRES(context,
                    (!std::is_same<Device, GPUDevice>::value ||
                     FastBoundsCheck(out_shape.num_elements(),
                                     std::numeric_limits<int32>::max())),
                    errors::InvalidArgument("Output elements too large for GPU kernel"));

        Tensor* output = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));

        /*Tensor filterCoeffTmpBuffer;

        {
            typedef typename Eigen::internal::packet_traits<T>::type Packet;
            static const int64 kPacketSize = (sizeof(Packet) / sizeof(T));

            TensorShape shape;
            int64 numFiltersPadded = ((filter.dim_size(3) + kPacketSize - 1) / kPacketSize) * kPacketSize;
            TensorShapeUtils::MakeShape(&numFiltersPadded, 1, &shape);
            OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::v(), shape, &filterCoeffTmpBuffer));
        }*/

        // If there is nothing to compute, return.
        if (out_shape.num_elements() == 0) {
            return;
        }

        VLOG(2) << "DepthwiseConv2dNative: "
                << " Input: [" << batch << ", " << input_rows << ", " << input_cols
                << ", " << in_depth << "]; Filter: [" << filter_rows << ", "
                << filter_cols << ", " << in_depth << ", " << depth_multiplier
                << "]; Output: [" << batch << ", " << out_rows << ", " << out_cols
                << ", " << out_depth << "], stride = " << stride_
                << ", pad_rows = " << pad_rows << ", pad_cols = " << pad_cols;
        
        ApproxConvOpData_t quantData(context, m_approxOp);
        quantData.ComputeQuantProps(2, 3, 4, 5);
        quantData.ComputeFilterCorrCoefficients(1);

        /*m_approxOp.Update(context, nullptr, filterCoeffTmpBuffer.template flat<T>().data());

        ApproxFilterCorrCoeff<Device, T, uint8, ApproxOpType>()(context->eigen_device<Device>(), m_approxOp, 
            filter.tensor<T, 4>(), filterCoeffTmpBuffer.Slice(0, filter.dim_size(3)).flat<T>());*/

        ApproxDepthwiseArgs args;
        args.batch = batch;
        args.in_rows = input_rows;
        args.in_cols = input_cols;
        args.in_depth = in_depth;
        args.filter_rows = filter_rows;
        args.filter_cols = filter_cols;
        args.depth_multiplier = depth_multiplier;
        args.stride = stride_;
        args.pad_rows = pad_rows;
        args.pad_cols = pad_cols;
        args.out_rows = out_rows;
        args.out_cols = out_cols;
        args.out_depth = out_depth;

        auto input_ptr = input.template flat<T>().data();
        auto filter_ptr = filter.template flat<T>().data();
        auto output_ptr = output->template flat<T>().data();
        LaunchApproxDepthwiseConvOp<ApproxConvOpData_t>()(context, args, input_ptr, filter_ptr,
                                                 output_ptr, data_format_, quantData);
    }

protected:
    bool use_cudnn_grouped_conv_;

private:
    std::vector<int32> strides_;
    Padding padding_;
    TensorFormat data_format_;

    int64 stride_;  // in height/width dimension.

    DataType dtype_;

    ApproxOpType<Device, T, uint8> m_approxOp;

    TF_DISALLOW_COPY_AND_ASSIGN(ApproxDepthwiseConv2DNativeOp);
};

#define REGISTER_CPU_KERNEL(T)                                                       \
  REGISTER_KERNEL_BUILDER(                                                           \
      Name("ApproxDepthwiseConv2DWithMinMaxVars").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      ApproxDepthwiseConv2DNativeOp<CPUDevice, T, TableApproxOpType_t>)

//TF_CALL_half(REGISTER_CPU_KERNEL);
TF_CALL_float(REGISTER_CPU_KERNEL);
/*#if !defined(PLATFORM_WINDOWS) || !defined(_DEBUG)
TF_CALL_double(REGISTER_CPU_KERNEL);
#endif*/

#if GOOGLE_CUDA

#define REGISTER_GPU_KERNEL(T)                                                       \
  REGISTER_KERNEL_BUILDER(                                                           \
      Name("ApproxDepthwiseConv2DWithMinMaxVars") \
          .Device(DEVICE_GPU)        \
          .TypeConstraint<T>("T")    \
          .HostMemory("input_min")   \
          .HostMemory("input_max")   \
          .HostMemory("filter_min")  \
          .HostMemory("filter_max"), \
      ApproxDepthwiseConv2DNativeOp<GPUDevice, T, TableApproxOpType_t>)

//TF_CALL_half(REGISTER_GPU_KERNEL);
TF_CALL_float(REGISTER_GPU_KERNEL);
//TF_CALL_double(REGISTER_GPU_KERNEL);

#endif // GOOGLE_CUDA

}
