//========== Copyright (c) 2019, Filip Vaverka, All rights reserved. =========//
//
// Purpose:     Approximate convolution OPs with GPU support
//
// $NoKeywords: $ApproxGPUOpsTF $approx_nn_ops.cpp
// $Date:       $2019-12-19
//============================================================================//

#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/framework/common_shape_fns.h>

using namespace tensorflow;

REGISTER_OP("ExampleOp")
    .Attr("T: {int32, float}")
    .Input("input: T")
    .Output("output: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
        c->set_output(0, c->input(0));
        return Status::OK();
    });

REGISTER_OP("ApproxConv2D")
    .Input("input: T")
    .Input("filter: T")
    .Output("output: T")
    .Attr("T: {half, bfloat16, float, double}")
    .Attr("strides: list(int)")
    /*.Attr("num_bits: int")
    .Attr("mul_map_file: string")*/
    .Attr(GetPaddingAttrStringWithExplicit())
    .Attr(GetExplicitPaddingsAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .SetShapeFn(shape_inference::Conv2DShapeWithExplicitPadding);

REGISTER_OP("ApproxConv2DWithMinMaxVars")
    .Input("input: T")
    .Input("filter: T")
    .Input("input_min: T")
    .Input("input_max: T")
    .Input("filter_min: T")
    .Input("filter_max: T")
    .Output("output: T")
    .Attr("T: {half, bfloat16, float, double}")
    .Attr("strides: list(int)")
    .Attr("num_bits: int")
    .Attr("mul_map_file: string")
    .Attr(GetPaddingAttrStringWithExplicit())
    .Attr(GetExplicitPaddingsAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .SetShapeFn(shape_inference::Conv2DShapeWithExplicitPadding);
