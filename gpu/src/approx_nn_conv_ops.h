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
// $NoKeywords: $ApproxGPUOpsTF $approx_nn_conv_ops.h
// $Date:       $2019-12-19
//============================================================================//

#pragma once

#ifndef APPROX_NN_CONV_OPS_H
#define APPROX_NN_CONV_OPS_H

#include <vector>

#include <tensorflow/core/framework/resource_mgr.h>
#include <tensorflow/core/util/tensor_format.h>

namespace tensorflow {

class OpKernelContext;

template<typename Device, typename T, typename AT, template<typename, typename, typename> class ApproxOpType>
struct LaunchApproxConv2DOp {
    void operator()(OpKernelContext *ctx,
                    const Tensor &input, const Tensor &filter,
                    int rowPadding, int colPadding,
                    int rowDilation, int colDilation, int rowStride, int colStride,
                    const Padding &padding, const std::vector<int64> &explicitPaddings, Tensor *output,
                    TensorFormat dataFormat, const ApproxOpType<Device, T, AT> &approxOp);
};

#ifdef GOOGLE_CUDA

template<typename T, typename AT, template<typename, typename, typename> class ApproxOpType>
struct LaunchApproxConv2DOp<Eigen::GpuDevice, T, AT, ApproxOpType> {
    void operator()(OpKernelContext *ctx,
                    const Tensor &input, const Tensor &filter,
                    int rowPadding, int colPadding,
                    int rowDilation, int colDilation, int rowStride, int colStride,
                    const Padding &padding, const std::vector<int64> &explicitPaddings, Tensor *output,
                    TensorFormat dataFormat, const ApproxOpType<Eigen::GpuDevice, T, AT> &approxOp);
};

#endif // GOOGLE_CUDA

struct ApproxConv2DParameters {
    std::vector<int32> dilations;
    std::vector<int32> strides;
    Padding padding;
    TensorFormat dataFormat;
    std::vector<int64> explicitPaddings;
};

struct ApproxConv2DDimensions {
    int batch;
    int inputRows;
    int inputCols;
    int inDepth;

    int filterRows;
    int filterCols;
    int patchDepth;
    int outDepth;

    int strideRows;
    int strideCols;

    int dilationRows;
    int dilationCols;

    int64 outRows;
    int64 outCols;
    int64 padRowsBefore;
    int64 padRowsAfter;
    int64 padColsBefore;
    int64 padColsAfter;
};

Status InitApproxConv2DParameters(const OpKernelConstruction *ctx,
                                  ApproxConv2DParameters *params);

Status ComputeApproxConv2DDimensions(const ApproxConv2DParameters &params,
                                     const Tensor &input, const Tensor &filter,
                                     ApproxConv2DDimensions *dimensions);

}

#endif // APPROX_NN_CONV_OPS_H
