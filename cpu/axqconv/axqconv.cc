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

// Implements quantized eight-bit versions of the convolution operations.

#include <algorithm>
#include <vector>

#define EIGEN_USE_THREADS

#define GEMMLOWP_ALLOW_SLOW_SCALAR_FALLBACK
//#include "public/gemmlowp.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
//#include "tensorflow/core/kernels/conv_ops.h"
//#include "tensorflow/core/kernels/meta_support.h"
//#include "tensorflow/core/kernels/ops_util.h"
//#include "tensorflow/core/kernels/quantization_utils.h"
#include "quantization_utils.h"
//#include "tensorflow/core/kernels/reference_gemm.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/padding.h"


#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/framework/bounds_check.h"
//#include "conv_ops.h"
#include "tensorflow/core/util/mirror_pad_mode.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/util/mirror_pad_mode.h"
#include "tensorflow/core/util/tensor_format.h"

#include "approximate_selector.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("AxConv2D")
    .Input("input: Tinput")
    .Input("filter: Tfilter")
    .Input("min_input: float")
    .Input("max_input: float")
    .Input("min_filter: float")
    .Input("max_filter: float")
    .Output("output: out_type")
    .Output("min_output: float")
    .Output("max_output: float")
    .Attr("Tinput: quantizedtype")
    .Attr("AxMult: string = 'mul8u_1JFF'")
    .Attr("AxTune: bool = true")
    .Attr("Tfilter: quantizedtype")
    .Attr("out_type: quantizedtype = DT_QINT32")
    .Attr("strides: list(int)")
    .Attr("padding: {'SAME', 'VALID'}")
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(shape_inference::Conv2DShape(c));
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &unused));
      c->set_output(1, c->Scalar());
      c->set_output(2, c->Scalar());
      return Status::OK();
    });



// This functor implements the convolution operation in as simple a form as
// possible. It won't give great performance, but it is very useful for
// stepping through and instrumenting for debugging, creating minimal benchmarks
// to prototype with, and sharing with teams that want to run this outside of
// our environment.
// With that in mind, I've avoided using anything except pretty standard C++
// types. This is especially noticeable in the data access through raw array
// indexing. It's deliberate in this case though, since it makes the underlying
// memory order very explicit, which is important for both inspecting memory
// contents during debugging and for specifying what we expect to others.
// The memory layout of the data is, from biggest stride to smallest:
// input_data = [input_batches, input_height, input_width, input_depth]
// filter_data = [filter_height, filter_width, input_depth, filter_count]
// output_data = [input_batches, output_height, output_width, filter_count]
template <class T1, class T2, class T3>
class ReferenceAxConvFunctor { 
 public:
  
  ReferenceAxConvFunctor(ApproximateSelector & approx_op) : approx_op_(approx_op)
  {
  }

  void operator()(OpKernelContext* context, const T1* input_data,
                  int input_batches, int input_height, int input_width,
                  int input_depth, int input_offset, const T2* filter_data,
                  int filter_height, int filter_width, int filter_count,
                  int filter_offset, int stride, Padding padding,
                  T3* output_data, int output_height, int output_width,
                  int output_shift, int output_offset, int output_mult) {
    // Set up some constants we need for the output down-shifting and
    // saturation.
    const int32 highest = static_cast<int32>(Eigen::NumTraits<T3>::highest());
    const int32 lowest = static_cast<int32>(Eigen::NumTraits<T3>::lowest());

    // When we're converting the 32 bit accumulator to a lower bit depth, we
    // need to add on 0.5 in fixed-point terms to make the operation round half
    // up towards positive infinity, rather than a floor.
    // We also need to watch out for the case when there's no down shift,
    // because a left shift by a negative number gives undefined results.
    const int32 rounding = (output_shift < 1) ? 0 : (1 << (output_shift - 1));

    // The two different padding modes we support can be a bit confusing. SAME
    // means we're trying to produce an output image that's the same size as the
    // input. It's complicated by stride, which shrinks the output image by a
    // a factor, but it means we end up sampling from outside the borders of the
    // input. These out-of-bounds values are read as zeroes. VALID means only
    // produce output values where the filters can read all their values from
    // within the input image. It effectively removes the margins of the output
    // image compared to the one produced by SAME. Stride complicates this
    // definition though, because it can result in the right and bottom filter
    // patches sampling from outside the borders if it's greater than 1.
    // Most of the logic for sorting this all out is done before this function,
    // when we calculate the output size, but the positioning of the origin of
    // the filters is different between the two modes, since SAME positions the
    // first filter off the edge of the input.
    int filter_left_offset;
    int filter_top_offset;
    if (padding == VALID) {
      filter_left_offset =
          ((output_width - 1) * stride + filter_width - input_width + 1) / 2;
      filter_top_offset =
          ((output_height - 1) * stride + filter_height - input_height + 1) / 2;
    } else {
      filter_left_offset =
          ((output_width - 1) * stride + filter_width - input_width) / 2;
      filter_top_offset =
          ((output_height - 1) * stride + filter_height - input_height) / 2;
    }

    // If we've got multiple images in our input, work through each of them.
    for (int batch = 0; batch < input_batches; ++batch) {
      // Walk through all the output image values, sliding the filter to
      // different
      // positions in the input.
      for (int out_y = 0; out_y < output_height; ++out_y) {
        for (int out_x = 0; out_x < output_width; ++out_x) {
          // Each filter kernel produces one output channel.
          for (int out_channel = 0; out_channel < filter_count; ++out_channel) {
            // We're going to calculate a single output value, which means we
            // need to multiply a three dimensional kernel of weights against
            // the current location within the input image.
            /*
              *-------------------------------...
              |\ ^
              | \in_depth
              |  \ v
              |   *-------------------------------...
              |   |            ^
              |   |       in_y_origin
              |   |            v   \
              |   |<in_x_origin>*---*^
              |   |            \|   |filter_height
              .   |             *---*v
              .   |             <--->
                  .         filter_width
                  .
            */
            const int in_x_origin = (out_x * stride) - filter_left_offset;
            const int in_y_origin = (out_y * stride) - filter_top_offset;
            int32 total = 0;
            for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
              for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                for (int in_channel = 0; in_channel < input_depth;
                     ++in_channel) {
                  const int in_x = in_x_origin + filter_x;
                  const int in_y = in_y_origin + filter_y;
                  int32 input_value;
                  // If the location is outside the bounds of the input image,
                  // use zero as a default value.
                  T1 input_source_value;
                  if ((in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                      (in_y < input_height)) {
                    /*const T1*/ input_source_value =
                        input_data[(batch * input_height * input_width *
                                    input_depth) +
                                   (in_y * input_width * input_depth) +
                                   (in_x * input_depth) + in_channel];
                    // We're promoting the T1 type to a higher bit depth here as
                    // we do the subtraction.
                    input_value =
                        static_cast<int32>(input_source_value) - input_offset;
                  } else {
                    input_value = 0;
                    input_source_value = input_offset;
                  }
                  const T2 filter_source_value =
                      filter_data[(filter_y * filter_width * input_depth *
                                   filter_count) +
                                  (filter_x * input_depth * filter_count) +
                                  (in_channel * filter_count) + out_channel];
                  // Another promotion to 32 bit, as above.
                  const int32 filter_value =
                      static_cast<int32>(filter_source_value) - filter_offset;
                  #if 0
                  total += (input_value * filter_value);
                  #endif

                  assert(static_cast<int32>(input_source_value) >= 0  );
                  assert(static_cast<int32>(input_source_value) <= 255  );
                  assert(static_cast<int32>(filter_source_value) >= 0  );
                  assert(static_cast<int32>(filter_source_value) <= 255  );

                  //std::cerr << static_cast<int32>(input_source_value) << " times " << static_cast<int32>(filter_source_value) << std::endl;
                  //int32 axm =  axtable[static_cast<int32>(input_source_value) * 256 + static_cast<int32>(filter_source_value)]; 
                  int32 axm =  approx_op_.multiplicate(static_cast<int32>(input_source_value), static_cast<int32>(filter_source_value));

#if 0
                if( static_cast<int32>(filter_source_value)  == filter_offset || static_cast<int32>(input_source_value) == input_offset) {
                    axm = static_cast<int32>(input_source_value) * static_cast<int32>(filter_source_value);
                }
#endif
                //else {
                    
                  total += 
                    axm -
                    static_cast<int32>(input_source_value) * filter_offset -
                    static_cast<int32>(filter_source_value) * input_offset +
                    input_offset * filter_offset;    
                //}
                  //assert(input_value >= -128);
                  //assert(input_value <= 128);
                  //if(filter_value < -128 || filter_value > 127 || input_value < -128 || input_value > 127)
                  //  std::cout << input_source_value << " (" << input_value << ") * " << filter_source_value << "(" << filter_value << ")" << std::endl;
                  //assert(filter_value >= -128);
                  //assert(filter_value <= 128);
                  //std::cout << input_value << " * " << filter_value << std::endl;
                }
              }
            }
            // Here we're applying scale factors to compress the 32 bit
            // accumulated total to a potentially lower bit depth.
            const int32_t output =
                ((((total + output_offset) * output_mult) + rounding) >>
                 output_shift);
            // We need to saturate the results against the largest and smallest
            // values that can be represented in this type.
            const int32 top_clamped_output = std::min(output, highest);
            const int32 clamped_output = std::max(top_clamped_output, lowest);
            output_data[(batch * output_height * output_width * filter_count) +
                        (out_y * output_width * filter_count) +
                        (out_x * filter_count) + out_channel] = clamped_output;
          }
        }
      }
    }
  }
private:
    ApproximateSelector & approx_op_;
};

// We don't want to allocate a buffer to hold all the patches if the size is
// going to be extremely large, so break it into chunks if it's bigger than
// a limit. Each chunk will be processed serially, so we can refill the
// buffer for the next chunk and reuse it, keeping maximum memory size down.
// In this case, we've picked 1 megabyte as a reasonable limit, from
// experimentation.
const size_t kMaxChunkSize = (1 * 1024 * 1024);


template <class T1, class T2, class T3,
          template <class TF1, class TF2, class TF3> class ConvFunctor>
class AxQuantizedConv2DOp : public OpKernel {
 public:
  explicit AxQuantizedConv2DOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
    OP_REQUIRES(context, strides_.size() == 4,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES(context, strides_[1] == strides_[2],
                errors::InvalidArgument(
                    "Current implementation only supports equal length "
                    "strides in the row and column dimensions."));
    OP_REQUIRES(
        context, (strides_[0] == 1 && strides_[3] == 1),
        errors::InvalidArgument("Current implementation does not yet support "
                                "strides in the batch and depth dimensions."));
    std::vector<int32> dilations;
    OP_REQUIRES_OK(context, context->GetAttr("dilations", &dilations));
    OP_REQUIRES(context, dilations.size() == 4,
                errors::InvalidArgument("Dilations field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES(context, dilations[1] == 1 && dilations[2] == 1,
                errors::InvalidArgument(
                    "Current implementation only supports dilated rate as 1 "
                    "in the row and column dimensions."));
    OP_REQUIRES(context, (dilations[0] == 1 && dilations[3] == 1),
                errors::InvalidArgument(
                    "Current implementation does not yet support "
                    "dilations in the batch and depth dimensions."));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    
    OP_REQUIRES_OK(context, context->GetAttr("AxMult", &axmult_));

    bool axtune;
    OP_REQUIRES_OK(context, context->GetAttr("AxTune", &axtune));

    //std::cout << "AX mult " << axmult_ << " (" << axtune << ")" << std::endl;

    try {
        approx_op_.init(axmult_.c_str(), axtune);
    } catch (std::invalid_argument & e) {
         OP_REQUIRES(context, false, errors::InvalidArgument(e.what()));
    }
  }

  void Compute(OpKernelContext* context) override {
    // Input tensor is of the following dimensions:
    // [ batch, in_rows, in_cols, in_depth ]
    const Tensor& input = context->input(0);

    // Input filter is of the following dimensions:
    // [ filter_rows, filter_cols, in_depth, out_depth]
    const Tensor& filter = context->input(1);

    // For 2D convolution, there should be 4 dimensions.
    OP_REQUIRES(context, input.dims() == 4,
                errors::InvalidArgument("input must be 4-dimensional",
                                        input.shape().DebugString()));
    OP_REQUIRES(context, filter.dims() == 4,
                errors::InvalidArgument("filter must be 4-dimensional: ",
                                        filter.shape().DebugString()));

    const float min_input = context->input(2).flat<float>()(0);
    const float max_input = context->input(3).flat<float>()(0);
    const float min_filter = context->input(4).flat<float>()(0);
    const float max_filter = context->input(5).flat<float>()(0);
    const int32 offset_input =
        FloatToQuantizedUnclamped<T1>(0.0f, min_input, max_input);
    const int32 offset_filter =
        FloatToQuantizedUnclamped<T2>(0.0f, min_filter, max_filter);
    const int32 offset_output = 0;
    const int32 mult_output = 1;
    const int32 shift_output = 0;

    // The last dimension for input is in_depth. It must be the same as the
    // filter's in_depth.
    const int64 in_depth = input.dim_size(3);
    OP_REQUIRES(context, in_depth == filter.dim_size(2),
                errors::InvalidArgument(
                    "input and filter must have the same depth: ", in_depth,
                    " vs ", filter.dim_size(2)));

    // The last dimension for filter is out_depth.
    const int64 out_depth = filter.dim_size(3);

    // The second dimension for input is rows/height.
    // The first dimension for filter is rows/height.
    const int64 input_rows = input.dim_size(1);
    const int64 filter_rows = filter.dim_size(0);

    // The third dimension for input is columns/width.
    // The second dimension for filter is columns/width.
    const int64 input_cols = input.dim_size(2);
    const int64 filter_cols = filter.dim_size(1);

    // The first dimension for input is batch.
    const int64 batch = input.dim_size(0);

    // For now we take the stride from the second dimension only (we
    // assume row = col stride, and do not support striding on the
    // batch or depth dimension).
    const int stride = strides_[1];

    int64 out_rows = 0, out_cols = 0, pad_rows = 0, pad_cols = 0;
    OP_REQUIRES_OK(context,
                   GetWindowedOutputSize(input_rows, filter_rows, stride,
                                         padding_, &out_rows, &pad_rows));
    OP_REQUIRES_OK(context,
                   GetWindowedOutputSize(input_cols, filter_cols, stride,
                                         padding_, &out_cols, &pad_cols));
    CHECK_GT(batch, 0);
    CHECK_GT(out_rows, 0);
    CHECK_GT(out_cols, 0);
    CHECK_GT(out_depth, 0);
    TensorShape out_shape({batch, out_rows, out_cols, out_depth});

    // Output tensor is of the following dimensions:
    // [ in_batch, out_rows, out_cols, out_depth ]
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));

    // This will call different implementations (e.g. reference or optimized)
    // depending on the template parameter.
    ConvFunctor<T1, T2, T3> conv_functor(approx_op_);
    conv_functor(context, input.flat<T1>().data(), batch, input_rows,
                 input_cols, in_depth, offset_input, filter.flat<T2>().data(),
                 filter_rows, filter_cols, out_depth, offset_filter, stride,
                 padding_, output->flat<T3>().data(), out_rows, out_cols,
                 shift_output, offset_output, mult_output);

    float min_output_value;
    float max_output_value;
    QuantizationRangeForMultiplication<T1, T2, T3>(
        min_input, max_input, min_filter, max_filter, &min_output_value,
        &max_output_value);

    Tensor* output_min = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, {}, &output_min));
    output_min->flat<float>()(0) = min_output_value;

    Tensor* output_max = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(2, {}, &output_max));
    output_max->flat<float>()(0) = max_output_value;
  }

 private:
  std::vector<int32> strides_;
  Padding padding_;
  std::string axmult_;
  ApproximateSelector approx_op_;
};

// Right now we only support taking two eight bit inputs, and returning the
// results as signed 32-bit integers.
REGISTER_KERNEL_BUILDER(
    Name("AxConv2D")
        .Device(DEVICE_CPU)
        .TypeConstraint<quint8>("Tinput")
        .TypeConstraint<quint8>("Tfilter")
        .TypeConstraint<qint32>("out_type"),
    AxQuantizedConv2DOp<quint8, quint8, qint32, ReferenceAxConvFunctor>);

}  // namespace tensorflow
