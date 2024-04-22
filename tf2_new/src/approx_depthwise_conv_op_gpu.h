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

#pragma once

#ifndef APPROX_DEPTHWISE_CONV_OP_GPU_H
#define APPROX_DEPTHWISE_CONV_OP_GPU_H

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
//#undef ABSL_HAVE_STD_STRING_VIEW

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
//#include "third_party/cub/util_ptx.cuh"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/types.h"
#include "gpu_kernel_helper.h"
#include "tensorflow/core/util/tensor_format.h"

#include "approx_depthwise_conv_op.h"

#include "approx_ops_types.h"
#include "approx_ops_quant_data.h"

#if defined(_MSC_VER) && !defined(__clang__)
#define UNROLL
#define NOUNROLL
#else
#define UNROLL _Pragma("unroll")
#define NOUNROLL _Pragma("nounroll")
#endif

namespace tensorflow {

namespace detail {
template <typename T>
struct PseudoHalfType {
  using Type = T;
};
template <>
struct PseudoHalfType<Eigen::half> {
  using Type = float;
};
} // namespace detail

using Eigen::GpuDevice;

template<typename T, typename AT>
using GpuConvOpQuantData = ApproxConvOpQuantData<TableApproxOpType_t<GpuDevice, T, AT> >;

template<typename Device, typename T, typename AT>
using TableApproxConvOpQuantData = ApproxConvOpQuantData<TableApproxOpType_t<Device, T, AT> >;

template<typename T, typename AT>
using GpuOpQuantProps_t = typename TableApproxOpType_t<GpuDevice, T, AT>::OpQuantProps_t;

template<typename T, typename AT>
using GpuOpQuantPropsData_t = typename TableApproxConvOpQuantData<GpuDevice, T, AT>::OpQuantProps_t;

// Returns whether depthwise convolution forward or backward input pass can be
// performed using the faster ('Small') variant of the kernel.
inline EIGEN_DEVICE_FUNC bool CanLaunchApproxDepthwiseConv2DGPUSmall(
        const ApproxDepthwiseArgs& args) {
    return  args.depth_multiplier == 1 && args.stride == 1 && args.in_rows <= 32 &&
            args.in_cols <= 32 && args.in_rows == args.out_rows &&
            args.in_cols == args.out_cols && args.pad_rows >= 0 &&
            args.pad_rows < args.filter_rows && args.pad_cols >= 0 &&
            args.pad_cols < args.filter_cols &&
            args.filter_rows * args.filter_cols <=
                (args.in_rows + 1) / 2 * args.in_cols;
}

// Returns whether depthwise convolution backward filter pass can be performed
// using the faster ('Small') variant of the kernel.
inline EIGEN_DEVICE_FUNC bool CanLaunchApproxDepthwiseConv2DBackpropFilterGPUSmall(
        const ApproxDepthwiseArgs& args, const int block_height) {
    return  args.depth_multiplier == 1 && args.stride == 1 && args.in_rows <= 32 &&
            args.in_cols <= 32 && args.in_rows == args.out_rows &&
            args.in_cols == args.out_cols && args.pad_rows >= 0 &&
            args.pad_rows < args.filter_rows && args.pad_cols >= 0 &&
            args.pad_cols < args.filter_cols && block_height <= args.in_rows &&
            args.filter_rows * args.filter_cols <= args.in_cols * block_height;
}

// The DepthwiseConv2dGPUKernels perform either forward or backprop input
// convolution depending on a template argument of this enum.
enum ApproxDepthwiseConv2DDirection { DIRECTION_FORWARD, DIRECTION_BACKWARD };

// A Cuda kernel to compute the depthwise convolution forward pass
// in NHWC format.
template <typename T, int kKnownFilterWidth, int kKnownFilterHeight, int kKnownDepthMultiplier>
__global__ void __launch_bounds__(1024, 2)
    ApproxDepthwiseConv2DGPUKernelNHWC(const ApproxDepthwiseArgs args, const T* input,
                                       const T* filter, T* output, int num_outputs, 
                                       cudaTextureObject_t lookupTable, const GpuOpQuantPropsData_t<T, uint8> quantProps) {
    typedef typename detail::PseudoHalfType<T>::Type S;
    const int in_height = args.in_rows;
    const int in_width = args.in_cols;
    const int in_depth = args.in_depth;
    const int filter_height =
            kKnownFilterHeight < 0 ? args.filter_rows : kKnownFilterHeight;
    const int filter_width =
            kKnownFilterWidth < 0 ? args.filter_cols : kKnownFilterWidth;
    const int depth_multiplier =
            kKnownDepthMultiplier < 0 ? args.depth_multiplier : kKnownDepthMultiplier;
    const int stride = args.stride;
    const int pad_height = args.pad_rows;
    const int pad_width = args.pad_cols;
    const int out_height = args.out_rows;
    const int out_width = args.out_cols;
    const int out_depth = args.out_depth;

    CUDA_1D_KERNEL_LOOP(thread_id, num_outputs) {
        // Compute the indexes of this thread in the output.
        const int out_channel = thread_id % out_depth;
        const int out_col = (thread_id / out_depth) % out_width;
        const int out_row = (thread_id / out_depth / out_width) % out_height;
        const int batch = thread_id / out_depth / out_width / out_height;

        // Compute the input depth and the index of depth multiplier.
        const int in_channel = out_channel / depth_multiplier;
        const int multiplier = out_channel % depth_multiplier;

        // Decide if all input is valid, if yes, we can skip the boundary checks
        // for each input.
        const int input_row_start = out_row * stride - pad_height;
        const int input_col_start = out_col * stride - pad_width;
        const int input_row_end = input_row_start + filter_height;
        const int input_col_end = input_col_start + filter_width;

        const T filterOffset   = (quantProps.filterMode == 1) ? quantProps.pFilter[multiplier*3 + 2] : quantProps.pFilter[2];
        const T filterInvScale = (quantProps.filterMode == 1) ? quantProps.pFilter[multiplier*3 + 1] : quantProps.pFilter[1];
        const T s1xS2          = (quantProps.filterMode == 1) ? quantProps.pS1xS2[multiplier] : quantProps.pS1xS2[0];
        const T m1xM2          = (quantProps.filterMode == 1) ? quantProps.pM1xM2[multiplier] : quantProps.pM1xM2[0];

        S sum     = static_cast<S>(0);
        S corrSum = static_cast<S>(0);

        const int input_offset_temp = in_height * batch;
        if (input_row_start >= 0 && input_col_start >= 0 &&
                input_row_end < in_height && input_col_end < in_width) {
            UNROLL for (int filter_row = 0; filter_row < filter_height; ++filter_row) {
                const int in_row = input_row_start + filter_row;
                const int filter_offset_temp = filter_width * filter_row;

                UNROLL for (int filter_col = 0; filter_col < filter_width; ++filter_col) {
                    const int in_col = input_col_start + filter_col;

                    const int input_offset =
                            in_channel +
                            in_depth * (in_col + in_width * (in_row + input_offset_temp));
                    const int filter_offset =
                            multiplier +
                            depth_multiplier *
                            (in_channel + in_depth * (filter_col + filter_offset_temp));

                    /*sum += static_cast<S>(ldg(input + input_offset)) *
                            static_cast<S>(ldg(filter + filter_offset));*/
                    
                    uint valueA = ClampBitWidth<uint8, 8>((ldg(input + input_offset) - quantProps.pInput[2]) * quantProps.pInput[1] + T(0.5));
                    uint valueB = ClampBitWidth<uint8, 8>((ldg(filter + filter_offset) - filterOffset) * filterInvScale + T(0.5));
                    uint tableFetchIdx = (valueA << 8) | valueB;
                    sum     += S(tex1Dfetch<ushort>(lookupTable, tableFetchIdx));
                    corrSum += S(ldg(input + input_offset));
                }
            }
        }
        else {
            UNROLL for (int filter_row = 0; filter_row < filter_height; ++filter_row) {
                const int in_row = input_row_start + filter_row;
                const int filter_offset_temp = filter_width * filter_row;

                UNROLL for (int filter_col = 0; filter_col < filter_width; ++filter_col) {
                    const int in_col = input_col_start + filter_col;
                    if (in_row >= 0 && in_row < in_height && in_col >= 0 &&
                            in_col < in_width) {
                        const int in_col = input_col_start + filter_col;

                        const int input_offset =
                                in_channel +
                                in_depth * (in_col + in_width * (in_row + input_offset_temp));
                        const int filter_offset =
                                multiplier +
                                depth_multiplier *
                                (in_channel + in_depth * (filter_col + filter_offset_temp));
                    
                        /*sum += static_cast<S>(ldg(input + input_offset)) *
                                static_cast<S>(ldg(filter + filter_offset));*/
                        
                        uint valueA = ClampBitWidth<uint8, 8>((ldg(input + input_offset) - quantProps.pInput[2]) * quantProps.pInput[1] + T(0.5));
                        uint valueB = ClampBitWidth<uint8, 8>((ldg(filter + filter_offset) - filterOffset) * filterInvScale + T(0.5));
                        uint tableFetchIdx = (valueA << 8) | valueB;
                        sum     += S(tex1Dfetch<ushort>(lookupTable, tableFetchIdx));
                        corrSum += S(ldg(input + input_offset));
                    }
                }
            }
        }

        output[thread_id] = static_cast<T>(sum) * s1xS2 + filterOffset * T(corrSum) +
            quantProps.pInput[2] * quantProps.pFilterCorr[multiplier] - T(filter_width * filter_height) * m1xM2;
    }
}

// CUDA kernel to compute the depthwise convolution forward pass in NHWC format,
// tailored for small images up to 32x32. Stride and depth multiplier must be 1.
// Padding must be 'SAME', which allows to reuse the index computation. Only
// use this kernel if CanLaunchDepthwiseConv2dGPUSmall(args) returns true.
// Tiles of the input and filter tensors are loaded into shared memory before
// performing the convolution. Each thread handles two elements per iteration,
// one each in the lower and upper half of a tile.
// Backprop input direction is the same as forward direction with the filter
// rotated by 180°.
// T is the tensors' data type. S is the math type the kernel uses. This is the
// same as T for all cases but pseudo half (which has T=Eigen::half, S=float).
template <typename T, ApproxDepthwiseConv2DDirection kDirection,
          int kKnownFilterWidth, int kKnownFilterHeight, int kBlockDepth,
          bool kKnownEvenHeight>
__global__ __launch_bounds__(1024, 2) void ApproxDepthwiseConv2DGPUKernelNHWCSmall(
        const ApproxDepthwiseArgs args, const T* input, const T* filter, T* output,
        cudaTextureObject_t lookupTable, const GpuOpQuantPropsData_t<T, uint8> quantProps) {
    typedef typename detail::PseudoHalfType<T>::Type S;
    assert(CanLaunchApproxDepthwiseConv2DGPUSmall(args));
    // Holds block plus halo and filter data for blockDim.x depths.
    GPU_DYNAMIC_SHARED_MEM_DECL(8, unsigned char, shared_memory);
    static_assert(sizeof(S) <= 8, "Insufficient alignment detected");
    S* const shared_data = reinterpret_cast<S*>(shared_memory);

    const int num_batches = args.batch;
    const int in_height = args.in_rows;
    const int in_width = args.in_cols;
    const int in_depth = args.in_depth;
    const int filter_height =
            kKnownFilterHeight < 0 ? args.filter_rows : kKnownFilterHeight;
    const int filter_width =
            kKnownFilterWidth < 0 ? args.filter_cols : kKnownFilterWidth;
    const int pad_height = args.pad_rows;
    const int pad_width = args.pad_cols;

    assert(blockDim.x == kBlockDepth);
    assert(blockDim.y == args.in_cols);
    const int block_height = blockDim.z;

    // These values are the same for all threads and could
    // be precomputed on the CPU.
    const int block_size = block_height * in_width * kBlockDepth;
    const int in_row_size = in_width * in_depth;
    const int in_size = in_height * in_row_size;
    const int in_increment = (in_width - 1) * kBlockDepth;
    const int filter_pixels = filter_height * filter_width;
    const int tile_width = in_width + filter_width - 1;
    const int even_height = kKnownEvenHeight || (1 & ~in_height);
    const int tile_height = in_height + filter_height - even_height;
    const int tile_row_size = tile_width * kBlockDepth;
    const int tile_size = tile_height * tile_row_size;
    const int tile_offset = block_height * tile_row_size;
    const int pad_offset = pad_height * tile_width + pad_width;
    const int batch_blocks = (in_depth + kBlockDepth - 1) / kBlockDepth;
    const int in_blocks = batch_blocks * num_batches;
    const int tensor_offset =
            kKnownEvenHeight ? in_size / 2 : block_height * in_row_size;

    const int thread_depth = threadIdx.x;
    const int thread_col = threadIdx.y;
    const int thread_row = threadIdx.z;

    // Position in block.
    const int thread_pix = thread_row * in_width + thread_col;
    const int thread_idx = thread_pix * kBlockDepth + thread_depth;

    // Initialize tile, in particular the padding.
    for (int i = thread_idx; i < tile_size; i += block_size) {
        shared_data[i] = S();
    }
    __syncthreads();

    // Position in tensors.
    const int tensor_idx = thread_pix * in_depth + thread_depth;

    // Position in (padded) shared memory.
    const int data_pix = thread_row * tile_width + thread_col;
    const int data_idx = data_pix * kBlockDepth + thread_depth;

    // Position in shared memory, offset by pad_height / pad_width.
    const int tile_pix = data_pix + pad_offset;
    const int tile_idx = tile_pix * kBlockDepth + thread_depth;

    const int max_channel = in_depth - thread_depth;
    const int filter_write_offset =
            thread_pix < filter_pixels ? tile_size + thread_idx : 0;
    const int filter_read_offset =
            tile_size + thread_depth +
            (kDirection == DIRECTION_FORWARD ? 0 : filter_pixels * kBlockDepth);
    const bool skip_second =
            !kKnownEvenHeight && thread_row + (in_height & 1) == block_height;

    for (int b = blockIdx.x; b < in_blocks; b += gridDim.x) {
        const int batch = b / batch_blocks;
        const int block = b - batch * batch_blocks;

        const int start_channel = block * kBlockDepth;
        const int filter_offset = tensor_idx + start_channel;
        const int inout_offset = batch * in_size + filter_offset;
        const bool channel_in_range = start_channel < max_channel;

        if (channel_in_range) {
            const T* const in_ptr = inout_offset + input;
            S* const tile_ptr = tile_idx + shared_data;
            tile_ptr[0] = static_cast<S>(ldg(in_ptr));
            if (!skip_second) {
                tile_ptr[tile_offset] = static_cast<S>(ldg(tensor_offset + in_ptr));
            }

            if (filter_write_offset != 0) {
                shared_data[filter_write_offset] =
                        static_cast<S>(ldg(filter_offset + filter));
            }
        }

        // Note: the condition to reach this is uniform across the entire block.
        __syncthreads();

        if (channel_in_range) {
            S sum1     = S();
            S corrSum1 = S();
            S sum2     = S();
            S corrSum2 = S();
            int shared_offset = data_idx;
            const S* filter_ptr = filter_read_offset + shared_data;

            UNROLL for (int r = 0; r < filter_height; ++r) {
                UNROLL for (int c = 0; c < filter_width; ++c) {
                    if (kDirection == DIRECTION_BACKWARD) {
                        filter_ptr -= kBlockDepth;
                    }

                    const S filter_value = *filter_ptr;
                    const S* const tile_ptr = shared_offset + shared_data;
                    /*sum1 += filter_value * tile_ptr[0];
                    sum2 += filter_value * tile_ptr[tile_offset];*/

                    uint filterValue = ClampBitWidth<uint8, 8>((filter_value          - quantProps.pFilter[2]) * quantProps.pFilter[1] + T(0.5));
                    uint value1      = ClampBitWidth<uint8, 8>((tile_ptr[0]           - quantProps.pInput[2])  * quantProps.pInput[1]  + T(0.5));
                    uint value2      = ClampBitWidth<uint8, 8>((tile_ptr[tile_offset] - quantProps.pInput[2])  * quantProps.pInput[1]  + T(0.5));
                    sum1     += S(tex1Dfetch<ushort>(lookupTable, (filterValue << 8) | value1));
                    corrSum1 += S(tile_ptr[0]);
                    sum2     += S(tex1Dfetch<ushort>(lookupTable, (filterValue << 8) | value2));
                    corrSum2 += S(tile_ptr[tile_offset]);

                    shared_offset += kBlockDepth;

                    if (kDirection == DIRECTION_FORWARD) {
                        filter_ptr += kBlockDepth;
                    }
                }
                shared_offset += in_increment;
            }

            T* const out_ptr = inout_offset + output;
            out_ptr[0] = static_cast<T>(sum1) * quantProps.pS1xS2[0] + quantProps.pFilter[2] * T(corrSum1) +
                quantProps.pInput[2] * quantProps.pFilterCorr[0] - T(filter_width * filter_height) * quantProps.pM1xM2[0];

            if (!skip_second) {
                out_ptr[tensor_offset] = static_cast<T>(sum2) * quantProps.pS1xS2[0] + quantProps.pFilter[2] * T(corrSum2) +
                    quantProps.pInput[2] * quantProps.pFilterCorr[0] - T(filter_width * filter_height) * quantProps.pM1xM2[0];
            }
        }

        // Note: the condition to reach this is uniform across the entire block.
        __syncthreads();
    }
}

// A Cuda kernel to compute the depthwise convolution forward pass
// in NCHW format.
template <typename T, int kKnownFilterWidth, int kKnownFilterHeight,
          int kKnownDepthMultiplier>
__global__ void __launch_bounds__(1024, 2)
    ApproxDepthwiseConv2DGPUKernelNCHW(const ApproxDepthwiseArgs args, const T* input,
                                       const T* filter, T* output, int num_outputs, 
                                       cudaTextureObject_t lookupTable, const GpuOpQuantPropsData_t<T, uint8> quantProps) {

    typedef typename detail::PseudoHalfType<T>::Type S;
    const int in_height = args.in_rows;
    const int in_width = args.in_cols;
    const int in_depth = args.in_depth;
    const int filter_height =
            kKnownFilterHeight < 0 ? args.filter_rows : kKnownFilterHeight;
    const int filter_width =
            kKnownFilterWidth < 0 ? args.filter_cols : kKnownFilterWidth;
    const int depth_multiplier =
            kKnownDepthMultiplier < 0 ? args.depth_multiplier : kKnownDepthMultiplier;
    const int stride = args.stride;
    const int pad_height = args.pad_rows;
    const int pad_width = args.pad_cols;
    const int out_height = args.out_rows;
    const int out_width = args.out_cols;
    const int out_depth = args.out_depth;

    CUDA_1D_KERNEL_LOOP(thread_id, num_outputs) {
        // Compute the indexes of this thread in the output.
        //
        // We want coalesced reads so we make sure that each warp reads
        // a contiguous chunk of memory.
        //
        // THIS IS PROBABLY WRONG, we are not doing coalesced reads
        // into the input, because of the depth multiplier division...
        const int out_col = thread_id % out_width;
        const int out_row = (thread_id / out_width) % out_height;
        const int out_channel = (thread_id / out_width / out_height) % out_depth;
        const int batch = thread_id / out_width / out_height / out_depth;

        // Compute the input depth and the index of depth multiplier
        // based off the output depth index that this thread is
        // computing n.
        const int in_channel = out_channel / depth_multiplier;
        const int multiplier = out_channel % depth_multiplier;

        // Data is stored in the following format (let's assume we
        // flatten the height and width into one contiguous dimension
        // called "P".
        //
        // B1C1P1 B1C1P2 ..... B1C2P1 B1C2P2 ....
        // B2C1P1 B2C1P2 ..... B2C2P1 B2C2P2 ....
        //
        // Each row contains in_depth * in_height * in_width values
        // for each sample in the batch.
        //
        // We can further flatten it into:
        //
        // B1C1P1 B1C1P2 .....
        // B1C2P1 B1C2P2 ....
        // B2C1P1 B2C1P2 .....
        // B2C2P1 B2C2P2 ....
        //
        // where each row is a contiguous array of all of the spatial
        // pixels for a given batch and input depth.  The following
        // loop unrolls across the filter dimensions for a given thread,
        // indexing into the filter value and the corresponding input
        // patch.
        //
        // We can compute the index into the patch once right here.
        const int input_offset_temp =
                (batch * in_depth + in_channel) * (in_height * in_width);

        // Finally, we can iterate over the spatial dimensions and perform the
        // convolution, writing into the output at the end.
        //
        // We perform an additional optimization, where we can determine
        // whether the patch fits within the image indices statically, and
        // avoid boundary checking within the loop.
        const int input_row_start = out_row * stride - pad_height;
        const int input_col_start = out_col * stride - pad_width;
        const int input_row_end = input_row_start + filter_height;
        const int input_col_end = input_col_start + filter_width;

        const T filterOffset   = (quantProps.filterMode == 1) ? quantProps.pFilter[multiplier*3 + 2] : quantProps.pFilter[2];
        const T filterInvScale = (quantProps.filterMode == 1) ? quantProps.pFilter[multiplier*3 + 1] : quantProps.pFilter[1];
        const T s1xS2          = (quantProps.filterMode == 1) ? quantProps.pS1xS2[multiplier] : quantProps.pS1xS2[0];
        const T m1xM2          = (quantProps.filterMode == 1) ? quantProps.pM1xM2[multiplier] : quantProps.pM1xM2[0];

        S sum     = static_cast<S>(0);
        S corrSum = static_cast<S>(0);
        if (input_row_start >= 0 && input_col_start >= 0 &&
                input_row_end < in_height && input_col_end < in_width) {
            // Loop that doesn't need to check for boundary conditions.
            UNROLL for (int filter_row = 0; filter_row < filter_height; ++filter_row) {
                const int in_row = input_row_start + filter_row;
                const int filter_offset_temp = filter_width * filter_row;
                UNROLL for (int filter_col = 0; filter_col < filter_width; ++filter_col) {
                    const int in_col = input_col_start + filter_col;

                    const int input_offset =
                            (input_offset_temp) + (in_row * in_width) + in_col;
                    const int filter_offset =
                            multiplier +
                            depth_multiplier *
                            (in_channel + in_depth * (filter_col + filter_offset_temp));

                    /*sum += static_cast<S>(ldg(input + input_offset)) *
                            static_cast<S>(ldg(filter + filter_offset));*/

                    uint valueA = ClampBitWidth<uint8, 8>((ldg(input + input_offset)   - quantProps.pInput[2])  * quantProps.pInput[1]  + T(0.5));
                    uint valueB = ClampBitWidth<uint8, 8>((ldg(filter + filter_offset) - filterOffset) * filterInvScale + T(0.5));
                    uint tableFetchIdx = (valueA << 8) | valueB;
                    sum     += S(tex1Dfetch<ushort>(lookupTable, tableFetchIdx));
                    corrSum += S(ldg(input + input_offset));
                }
            }
        }
        else {
            // Loop that needs to check for boundary conditions.
            UNROLL for (int filter_row = 0; filter_row < filter_height; ++filter_row) {
                const int in_row = input_row_start + filter_row;
                const int filter_offset_temp = filter_width * filter_row;
                UNROLL for (int filter_col = 0; filter_col < filter_width; ++filter_col) {
                    const int in_col = input_col_start + filter_col;
                    // TODO(vrv): the in_row check can be done outside of this loop;
                    // benchmark both methods to determine the better decision.
                    if (in_row >= 0 && in_row < in_height && in_col >= 0 &&
                            in_col < in_width) {
                        const int in_col = input_col_start + filter_col;

                        // input_offset_temp indexes into the start of memory
                        // where the spatial data starts.
                        const int input_offset =
                                (input_offset_temp) + (in_row * in_width) + in_col;

                        const int filter_offset =
                                multiplier +
                                depth_multiplier *
                                (in_channel + in_depth * (filter_col + filter_offset_temp));

                        /*sum += static_cast<S>(ldg(input + input_offset)) *
                                static_cast<S>(ldg(filter + filter_offset));*/
                        
                        uint valueA = ClampBitWidth<uint8, 8>((ldg(input + input_offset)   - quantProps.pInput[2])  * quantProps.pInput[1]  + T(0.5));
                        uint valueB = ClampBitWidth<uint8, 8>((ldg(filter + filter_offset) - filterOffset) * filterInvScale+ T(0.5));
                        uint tableFetchIdx = (valueA << 8) | valueB;
                        sum     += S(tex1Dfetch<ushort>(lookupTable, tableFetchIdx));
                        corrSum += S(ldg(input + input_offset));
                    }
                }
            }
        }

        output[thread_id] = static_cast<T>(sum) * s1xS2 + filterOffset * T(corrSum) +
            quantProps.pInput[2] * quantProps.pFilterCorr[multiplier] - T(filter_width * filter_height) * m1xM2;
    }
}

// CUDA kernel to compute the depthwise convolution forward pass in NCHW format,
// tailored for small images up to 32x32. Stride and depth multiplier must be 1.
// Padding must be 'SAME', which allows to reuse the index computation. Only
// use this kernel if CanLaunchDepthwiseConv2dGPUSmall(args) returns true.
// Tiles of the input and filter tensors are loaded into shared memory before
// performing the convolution. Each thread handles two elements per iteration,
// one each in the lower and upper half of a tile.
// Backprop input direction is the same as forward direction with the filter
// rotated by 180°.
// T is the tensors' data type. S is the math type the kernel uses. This is the
// same as T for all cases but pseudo half (which has T=Eigen::half, S=float).
template <typename T, ApproxDepthwiseConv2DDirection kDirection,
          int kKnownFilterWidth, int kKnownFilterHeight, int kBlockDepth,
          bool kKnownEvenHeight>
__global__ __launch_bounds__(1024, 2) void ApproxDepthwiseConv2DGPUKernelNCHWSmall(
        const ApproxDepthwiseArgs args, const T* input, const T* filter, T* output,
        cudaTextureObject_t lookupTable, const GpuOpQuantPropsData_t<T, uint8> quantProps) {
    typedef typename detail::PseudoHalfType<T>::Type S;
    assert(CanLaunchDepthwiseConv2dGPUSmall(args));
    // Holds block plus halo and filter data for blockDim.z depths.
    GPU_DYNAMIC_SHARED_MEM_DECL(8, unsigned char, shared_memory);
    static_assert(sizeof(S) <= 8, "Insufficient alignment detected");
    S* const shared_data = reinterpret_cast<S*>(shared_memory);

    const int num_batches = args.batch;
    const int in_height = args.in_rows;
    const int in_width = args.in_cols;
    const int in_depth = args.in_depth;
    const int filter_height =
            kKnownFilterHeight < 0 ? args.filter_rows : kKnownFilterHeight;
    const int filter_width =
            kKnownFilterWidth < 0 ? args.filter_cols : kKnownFilterWidth;
    const int pad_height = args.pad_rows;
    const int pad_width = args.pad_cols;

    // Fixed blockDim.z, tailored for maximum grid size for images of size 16x16.
    assert(blockDim.x == args.in_cols);
    assert(blockDim.z == kBlockDepth);
    const int block_height = blockDim.y;

    // These values are the same for all threads and could
    // be precomputed on the CPU.
    const int block_pixels = in_width * block_height;
    const int block_size = block_pixels * kBlockDepth;
    const int in_pixels = in_width * in_height;
    const int in_increment = in_width - 1;
    const int filter_pixels = filter_height * filter_width;
    const int tile_width = in_width + filter_width - 1;
    const int even_height = kKnownEvenHeight || (1 & ~in_height);
    const int tile_height = in_height + filter_height - even_height;
    const int tile_pixels = tile_width * tile_height;
    const int tile_size = tile_pixels * kBlockDepth;
    const int tile_offset = block_height * tile_width;
    const int pad_offset = pad_height * tile_width + pad_width;
    const int in_total_depth = in_depth * num_batches;
    const int in_blocks = (in_total_depth + kBlockDepth - 1) / kBlockDepth;

    const int thread_col = threadIdx.x;
    const int thread_row = threadIdx.y;
    const int thread_depth = threadIdx.z;

    // Position in block.
    const int thread_pix = thread_row * in_width + thread_col;
    const int thread_idx = thread_depth * block_pixels + thread_pix;

    // Initialize tile, in particular the padding.
    for (int i = thread_idx; i < tile_size; i += block_size) {
        shared_data[i] = S();
    }
    __syncthreads();

    // Position in tensors.
    const int tensor_idx = thread_depth * in_pixels + thread_pix;

    // Position in (padded) shared memory.
    const int data_pix = thread_row * tile_width + thread_col;
    const int data_idx = thread_depth * tile_pixels + data_pix;

    // Position in shared memory, offset by pad_height / pad_width.
    const int tile_idx = data_idx + pad_offset;

    // Filter is always in HWCK format, irrespective of the input/output format.
    const int filter_pix = thread_idx / kBlockDepth;
    const int filter_channel = thread_idx % kBlockDepth;
    const int filter_idx = filter_pix * in_depth;

    const int max_channel = in_total_depth - thread_depth;
    const int filter_write_offset =
            filter_pix < filter_pixels ? tile_size + thread_idx : 0;
    const int filter_read_offset =
            tile_size + thread_depth +
            (kDirection == DIRECTION_FORWARD ? 0 : filter_pixels * kBlockDepth);
    const bool skip_second =
            !kKnownEvenHeight && thread_row + (in_height & 1) == block_height;

    for (int b = blockIdx.x; b < in_blocks; b += gridDim.x) {
        const int channel = b * kBlockDepth;

        const int inout_offset = channel * in_pixels + tensor_idx;
        const bool channel_in_range = channel < max_channel;

        if (channel_in_range) {
            const T* const in_ptr = inout_offset + input;
            S* const tile_ptr = tile_idx + shared_data;
            tile_ptr[0] = static_cast<S>(ldg(in_ptr));
            if (!skip_second) {
                tile_ptr[tile_offset] = static_cast<S>(ldg(block_pixels + in_ptr));
            }
        }

        if (filter_write_offset != 0) {
            const int filter_offset =
                    filter_idx + (channel + filter_channel) % in_depth;
            shared_data[filter_write_offset] =
                    static_cast<S>(ldg(filter_offset + filter));
        }

        // Note: the condition to reach this is uniform across the entire block.
        __syncthreads();

        if (channel_in_range) {
            S sum1     = S();
            S corrSum1 = S();
            S sum2     = S();
            S corrSum2 = S();
            int shared_offset = data_idx;
            const S* filter_ptr = filter_read_offset + shared_data;
            UNROLL for (int r = 0; r < filter_height; ++r) {
                UNROLL for (int c = 0; c < filter_width; ++c) {
                    if (kDirection == DIRECTION_BACKWARD) {
                        filter_ptr -= kBlockDepth;
                    }

                    const S filter_value = *filter_ptr;
                    const S* const tile_ptr = shared_offset + shared_data;
                    /*sum1 += filter_value * tile_ptr[0];
                    sum2 += filter_value * tile_ptr[tile_offset];*/

                    uint filterValue = ClampBitWidth<uint8, 8>((filter_value          - quantProps.pFilter[2]) * quantProps.pFilter[1] + T(0.5));
                    uint value1      = ClampBitWidth<uint8, 8>((tile_ptr[0]           - quantProps.pInput[2])  * quantProps.pInput[1]  + T(0.5));
                    uint value2      = ClampBitWidth<uint8, 8>((tile_ptr[tile_offset] - quantProps.pInput[2])  * quantProps.pInput[1]  + T(0.5));
                    sum1     += S(tex1Dfetch<ushort>(lookupTable, (filterValue << 8) | value1));
                    corrSum1 += S(tile_ptr[0]);
                    sum2     += S(tex1Dfetch<ushort>(lookupTable, (filterValue << 8) | value2));
                    corrSum2 += S(tile_ptr[tile_offset]);

                    ++shared_offset;

                    if (kDirection == DIRECTION_FORWARD) {
                        filter_ptr += kBlockDepth;
                    }
                }

                shared_offset += in_increment;
            }

            T* const out_ptr = inout_offset + output;
            out_ptr[0] = static_cast<T>(sum1) * quantProps.pS1xS2[0] + quantProps.pFilter[2] * T(corrSum1) +
                quantProps.pInput[2] * quantProps.pFilterCorr[0] - T(filter_width * filter_height) * quantProps.pM1xM2[0];

            if (!skip_second) {
                out_ptr[block_pixels] = static_cast<T>(sum2) * quantProps.pS1xS2[0] + quantProps.pFilter[2] * T(corrSum2) +
                quantProps.pInput[2] * quantProps.pFilterCorr[0] - T(filter_width * filter_height) * quantProps.pM1xM2[0];
            }
        }

        // Note: the condition to reach this is uniform across the entire block.
        __syncthreads();
    }
}

template <typename T, ApproxDepthwiseConv2DDirection kDirection,
          int kKnownFilterWidth, int kKnownFilterHeight, int kBlockDepth,
          bool kKnownEvenHeight>
Status LaunchApproxDepthwiseConv2DGPUSmall(OpKernelContext* ctx,
                                           const ApproxDepthwiseArgs& args, const T* input,
                                           const T* filter, T* output,
                                           TensorFormat data_format, 
                                           const TableApproxConvOpQuantData<GpuDevice, T, uint8> &approxOpData) {
    typedef typename detail::PseudoHalfType<T>::Type S;
    const int block_height = (args.in_rows + 1) / 2;
    dim3 block_dim;
    int block_count;
    void (*kernel)(const ApproxDepthwiseArgs, const T*, const T*, T*, cudaTextureObject_t, const GpuOpQuantPropsData_t<T, uint8>);

    const typename TableApproxConvOpQuantData<GpuDevice, T, uint8>::OpQuantProps_t &quantProps = approxOpData.GetQuantProps();
    const typename TableApproxConvOpQuantData<GpuDevice, T, uint8>::ApproxOpType_t &approxOp   = approxOpData.GetApproxOp();

    switch (data_format) {
    case FORMAT_NHWC:
        block_dim = dim3(kBlockDepth, args.in_cols, block_height);
        block_count =
                args.batch * DivUp(args.out_depth, kBlockDepth) * kBlockDepth;
        kernel =
                ApproxDepthwiseConv2DGPUKernelNHWCSmall<T, kDirection, kKnownFilterWidth,
                    kKnownFilterHeight, kBlockDepth,
                    kKnownEvenHeight>;
    break;
    case FORMAT_NCHW:
        block_dim = dim3(args.in_cols, block_height, kBlockDepth);
        block_count =
                DivUp(args.batch * args.out_depth, kBlockDepth) * kBlockDepth;
        kernel =
                ApproxDepthwiseConv2DGPUKernelNCHWSmall<T, kDirection, kKnownFilterWidth,
                    kKnownFilterHeight, kBlockDepth,
                    kKnownEvenHeight>;
    break;
    default:
        return errors::InvalidArgument("FORMAT_", ToString(data_format),
                                       " is not supported");
    }

    const int tile_width = args.in_cols + args.filter_cols - 1;
    const int tile_height = block_height * 2 + args.filter_rows - 1;
    const int tile_pixels = tile_height * tile_width;
    const int filter_pixels = args.filter_rows * args.filter_cols;
    const int shared_memory_size =
            kBlockDepth * (tile_pixels + filter_pixels) * sizeof(S);
    const int num_outputs = args.out_rows * args.out_cols * block_count;
    auto device = ctx->eigen_gpu_device();
    GpuLaunchConfig config = GetGpuLaunchConfigFixedBlockSize(
                num_outputs, device, kernel, shared_memory_size,
                block_dim.x * block_dim.y * block_dim.z);
    TF_CHECK_OK(GpuLaunchKernel(kernel, config.block_count, block_dim,
                                shared_memory_size, device.stream(), args, input,
                                filter, output, approxOp.GetLookupData(), quantProps));

    return Status::OK();
}

// Returns whether the context's GPU supports efficient fp16 math.
/*
inline bool HasFastHalfMath(OpKernelContext* ctx) {
    int major, minor;

    ctx->op_device_context()
            ->stream()
            ->parent()
            ->GetDeviceDescription()
            .cuda_compute_capability(&major, &minor);

    auto cuda_arch = major * 100 + minor * 10;

    // GPUs before sm_53 don't support fp16 math, and sm_61's fp16 math is slow.
    return cuda_arch >= 530 && cuda_arch != 610;
}
*/
inline bool HasFastHalfMath(OpKernelContext* ctx) {
  se::CudaComputeCapability compute_capability =
      ctx->op_device_context()->stream()->GetCudaComputeCapability();
  // GPUs before sm_53 don't support fp16 math, and sm_61's fp16 math is slow.
  return compute_capability.IsAtLeast(5, 3) &&
         compute_capability != se::CudaComputeCapability{6, 1};
}

template <typename T, ApproxDepthwiseConv2DDirection kDirection,
          int kKnownFilterWidth, int kKnownFilterHeight, int kBlockDepth>
Status LaunchApproxDepthwiseConv2DGPUSmall(OpKernelContext* ctx,
                                           const ApproxDepthwiseArgs& args, const T* input,
                                           const T* filter, T* output,
                                           TensorFormat data_format, const TableApproxConvOpQuantData<GpuDevice, T, uint8> &approxOpData) {
    if (args.in_rows & 1) {
        return LaunchApproxDepthwiseConv2DGPUSmall<T, kDirection, kKnownFilterWidth,
                kKnownFilterHeight, kBlockDepth, false>(ctx, args, input, filter,
                                                        output, data_format, approxOpData);
    }
    else {
        return LaunchApproxDepthwiseConv2DGPUSmall<T, kDirection, kKnownFilterWidth,
                kKnownFilterHeight, kBlockDepth, true>(ctx, args, input, filter,
                                                       output, data_format, approxOpData);
    }
}

template <typename T, ApproxDepthwiseConv2DDirection kDirection,
          int kKnownFilterWidth, int kKnownFilterHeight>
Status LaunchApproxDepthwiseConv2DGPUSmall(OpKernelContext* ctx,
                                           const ApproxDepthwiseArgs& args, const T* input,
                                           const T* filter, T* output,
                                           TensorFormat data_format, const TableApproxConvOpQuantData<GpuDevice, T, uint8> &approxOpData) {
    // Maximize (power of two) kBlockDepth while keeping a block within 1024
    // threads (2 pixels per thread).
    const int block_pixels = (args.in_rows + 1) / 2 * args.in_cols;
    if (block_pixels > 256) {
        return LaunchApproxDepthwiseConv2DGPUSmall<T, kDirection, kKnownFilterWidth,
                kKnownFilterHeight, 2>(ctx, args, input, filter, output, data_format, approxOpData);
    }
    else if (block_pixels > 128) {
        return LaunchApproxDepthwiseConv2DGPUSmall<T, kDirection, kKnownFilterWidth,
                kKnownFilterHeight, 4>(ctx, args, input, filter, output, data_format, approxOpData);
    }
    else {
        return LaunchApproxDepthwiseConv2DGPUSmall<T, kDirection, kKnownFilterWidth,
                kKnownFilterHeight, 8>(ctx, args, input, filter, output, data_format, approxOpData);
    }
}

template <typename T, int kKnownFilterWidth, int kKnownFilterHeight,
          int kKnownDepthMultiplier>
Status LaunchApproxDepthwiseConv2DGPU(OpKernelContext* ctx, const ApproxDepthwiseArgs& args,
                                      const T* input, const T* filter, T* output,
                                      TensorFormat data_format, const TableApproxConvOpQuantData<GpuDevice, T, uint8> &approxOpData) {
    void (*kernel)(const ApproxDepthwiseArgs, const T*, const T*, T*, int, cudaTextureObject_t, const GpuOpQuantPropsData_t<T, uint8>);
    switch (data_format) {
    case FORMAT_NHWC:
        kernel =
                ApproxDepthwiseConv2DGPUKernelNHWC<T, kKnownFilterWidth, kKnownFilterHeight,
                    kKnownDepthMultiplier>;
    break;
    case FORMAT_NCHW:
        kernel =
                ApproxDepthwiseConv2DGPUKernelNCHW<T, kKnownFilterWidth, kKnownFilterHeight,
                    kKnownDepthMultiplier>;
    break;
    default:
        return errors::InvalidArgument("FORMAT_", ToString(data_format),
                                       " is not supported");
    }

    const typename TableApproxConvOpQuantData<GpuDevice, T, uint8>::OpQuantProps_t &quantProps = approxOpData.GetQuantProps();
    const typename TableApproxConvOpQuantData<GpuDevice, T, uint8>::ApproxOpType_t &approxOp   = approxOpData.GetApproxOp();

    const int num_outputs =
            args.batch * args.out_rows * args.out_cols * args.out_depth;
    auto device = ctx->eigen_gpu_device();
    GpuLaunchConfig config =
            GetGpuLaunchConfig(num_outputs, device, kernel, 0, 0);
    // The compile-time constant version runs faster with a single block.
    const int max_block_count = kKnownFilterWidth < 0 || kKnownFilterHeight < 0 ||
            kKnownDepthMultiplier < 0 ? std::numeric_limits<int>::max()
                                      : device.getNumGpuMultiProcessors();

    TF_CHECK_OK(GpuLaunchKernel(kernel,
                                std::min(max_block_count, config.block_count),
                                config.thread_per_block, 0, device.stream(), args,
                                input, filter, output, num_outputs, approxOp.GetLookupData(), quantProps));

    return Status::OK();
}

template <typename T, int kKnownFilterWidth, int kKnownFilterHeight>
Status LaunchApproxDepthwiseConv2DGPU(OpKernelContext* ctx, const ApproxDepthwiseArgs& args,
                                      const T* input, const T* filter, T* output,
                                      TensorFormat data_format, const TableApproxConvOpQuantData<GpuDevice, T, uint8> &approxOpData) {
    if (args.depth_multiplier == 1) {
        if (CanLaunchApproxDepthwiseConv2DGPUSmall(args)) {
            return LaunchApproxDepthwiseConv2DGPUSmall<T, DIRECTION_FORWARD,
                    kKnownFilterWidth, kKnownFilterHeight>(ctx, args, input,
                                                           filter, output, data_format, approxOpData);
        }

        return LaunchApproxDepthwiseConv2DGPU<T, kKnownFilterWidth, kKnownFilterHeight, 1>(
                    ctx, args, input, filter, output, data_format, approxOpData);
    } else {
        return LaunchApproxDepthwiseConv2DGPU<T, kKnownFilterWidth, kKnownFilterHeight, -1>(
                    ctx, args, input, filter, output, data_format, approxOpData);
    }
}

// A simple launch pad to launch the Cuda kernel for depthwise convolution.
template<template<typename> class ApproxOpData, template<typename, typename, typename> class ApproxOpType, typename T>
void LaunchApproxDepthwiseConvOp<ApproxOpData<ApproxOpType<Eigen::GpuDevice, T, uint8> > >::operator()(OpKernelContext* ctx,
                                                           const ApproxDepthwiseArgs& args,
                                                           const T* input,
                                                           const T* filter, T* output,
                                                           TensorFormat data_format, const ApproxOpData<ApproxOpType<Eigen::GpuDevice, T, uint8> > &approxOpData) {
    if (args.filter_rows == 3 && args.filter_cols == 3) {
        OP_REQUIRES_OK(ctx, LaunchApproxDepthwiseConv2DGPU<T, 3, 3>(
                           ctx, args, input, filter, output, data_format, approxOpData));
    } else {
        OP_REQUIRES_OK(ctx, LaunchApproxDepthwiseConv2DGPU<T, -1, -1>(
                           ctx, args, input, filter, output, data_format, approxOpData));
    }
}
}

#endif // GOOGLE_CUDA

#endif // APPROX_DEPTHWISE_CONV_OP_GPU_H
