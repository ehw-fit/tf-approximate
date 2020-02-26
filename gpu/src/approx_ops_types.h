//========== Copyright (c) 2019, Filip Vaverka, All rights reserved. =========//
//
// Purpose:     Lookup table approximation of 8-bit MUL operations.
//
// $NoKeywords: $ApproxGPUOpsTF $approx_ops_types.h
// $Date:       $2019-12-19
//============================================================================//

#pragma once

#ifndef APPROX_OPS_TYPES_H
#define APPROX_OPS_TYPES_H

#include <fstream>

#include <tensorflow/core/framework/op_kernel.h>

#define EIGEN_STACK_ALLOCATION_LIMIT 0
#define EIGEN_USE_THREADS
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

// Nudge implementation from "kernels/fake_quant_ops_functor.h" used to compute
// values used for quantization in Fake quantization layer
EIGEN_ALWAYS_INLINE void Nudge(
    const float min, const float max, const int quant_min, const int quant_max,
    float* nudged_min, float* nudged_max, float* scale) {
  const float quant_min_float = static_cast<float>(quant_min);
  const float quant_max_float = static_cast<float>(quant_max);
  *scale = (max - min) / (quant_max_float - quant_min_float);
  const float zero_point_from_min = quant_min_float - min / *scale;
  const uint16 nudged_zero_point = [zero_point_from_min, quant_min,
                                    quant_min_float, quant_max,
                                    quant_max_float] {
    if (zero_point_from_min < quant_min_float) {
      return static_cast<uint16>(quant_min);
    }
    if (zero_point_from_min > quant_max_float) {
      return static_cast<uint16>(quant_max);
    }
    return static_cast<uint16>(std::round(zero_point_from_min));
  }();
  *nudged_min = (quant_min_float - nudged_zero_point) * (*scale);
  *nudged_max = (quant_max_float - nudged_zero_point) * (*scale);
}

}

/**
 * @brief Provider of non-approximated "ApproxOpType" used in construction of
 *        non-approximated layers.
 */
template<typename Device, typename T, typename AT>
struct NullApproxOpType_t {
    NullApproxOpType_t(tensorflow::OpKernelConstruction *) {}

    void Update(tensorflow::OpKernelContext *) {}
};

#define TF_REQUIRES(EXP, STATUS)                \
  do {                                          \
    if (!TF_PREDICT_TRUE(EXP)) return (STATUS); \
} while (false)

/**
 * @brief Base class for 8-bit approximated "ApproxOpType" used in construction
 *        of approximated layers using lookup table stored in file.
 */
template<typename T, typename AT, typename ATVT>
struct BaseTableApproxOpType_t {
    /**
     * @brief The VarQuantProps_t struct describes quantization properties of
     *        single Tensor (input of the operation).
     */
    struct VarQuantProps_t {
        T scale;    ///< Scaling of inputs to integer values.
        T invScale; ///< Invserse of scaling (1.0 / scale).
        T offset;   ///< Offset to allow for unsigned operations.
    };

    /**
     * @brief The OpQuantProps_t struct describes quantization properties needed
     *        by an approximated convolution layer.
     */
    struct OpQuantProps_t {
        VarQuantProps_t input;  ///< Input variable quantization properties.
        VarQuantProps_t filter; ///< Filter variable quantization properties.

        T s1xS2;    ///< Product of input and filter scales.
        T m1;       ///< Input offset.
        T m2;       ///< Filter offset.
        T m1xM2;    ///< Product of input and filter offsets.
    };

    BaseTableApproxOpType_t(tensorflow::OpKernelConstruction *ctx)
        : bitWidth(8)
        , tableFilename("")
        , quantMin(0)
        , quantMax((1 << 8) - 1)
    {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("num_bits", &bitWidth));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("mul_map_file", &tableFilename));

        quantMax = (1 << bitWidth) - 1;

        lookupTable.resize(1 << (2*bitWidth));

        if(tableFilename.empty())
        {
            VLOG(2) << "ApproxTF: Lookup table filename is empty... (using native 8-bit multiplication)";
            for(size_t i = 0; i < lookupTable.size(); ++i)
                lookupTable[i] = ATVT(i >> 8) * ATVT(i & 0xFF);
        }
        else
        {
            OP_REQUIRES_OK(ctx, LoadLookupTable(ctx, tableFilename));
        }
    }

    virtual ~BaseTableApproxOpType_t() {

    }

    /**
     * @brief UpdateVarQuantProps updates quantization properties for single tensor value
     *        given min/max of the tensor.
     * @param minValue      Minimum value in the tensor.
     * @param maxValue      Maximum value in the tensor.
     * @param quantProps    Computed quantization properties.
     */
    void UpdateVarQuantProps(float minValue, float maxValue, VarQuantProps_t &quantProps)
    {
        float nudgedScale    = 0.0f;
        float invNudgedScale = 0.0f;
        float nudgedMin      = 0.0f;
        float nudgedMax      = 0.0f;

        if(minValue != 0.0f || maxValue != 0.0f)
        {
            tensorflow::Nudge(minValue, maxValue, quantMin, quantMax,
                              &nudgedMin, &nudgedMax, &nudgedScale);
            invNudgedScale = 1.0f / nudgedScale;
        }

        quantProps.scale    = nudgedScale;
        quantProps.invScale = invNudgedScale;
        quantProps.offset   = nudgedMin;
    }

    /**
     * @brief Update quantization properties for all inputs
     * @param ctx
     */
    void Update(tensorflow::OpKernelContext *ctx)
    {
        {
            // Input value quantization properties
            const tensorflow::Tensor &min = ctx->input(2);
            const tensorflow::Tensor &max = ctx->input(3);

            float minValue = min.scalar<float>()();
            float maxValue = max.scalar<float>()();

            UpdateVarQuantProps(minValue, maxValue, quantProps.input);
//            std::cout << "INPUT_RANGE: " << minValue << " " << maxValue << std::endl;
        }

        {
            // Filter value quantization properties
            const tensorflow::Tensor &min = ctx->input(4);
            const tensorflow::Tensor &max = ctx->input(5);

            float minValue = min.scalar<float>()();
            float maxValue = max.scalar<float>()();

            UpdateVarQuantProps(minValue, maxValue, quantProps.filter);
//            std::cout << "FILTER_RANGE: " << minValue << " " << maxValue << std::endl;
        }

        quantProps.s1xS2 = quantProps.input.scale * quantProps.filter.scale;
        quantProps.m1    = quantProps.input.offset;
        quantProps.m2    = quantProps.filter.offset;
        quantProps.m1xM2 = quantProps.m1 * quantProps.m2;

//        std::cout << "QUANT_PROPS: " << std::endl
//                  << ", num_bits: "        << bitWidth << std::endl
//                  << ", input_scale: "     << quantProps.input.scale << std::endl
//                  << ", input_inv_scale: " << quantProps.input.invScale << std::endl
//                  << ", input_offset: "    << quantProps.input.offset << std::endl
//                  << ", filter_scale: "     << quantProps.filter.scale << std::endl
//                  << ", filter_inv_scale: " << quantProps.filter.invScale << std::endl
//                  << ", filter_offset: "    << quantProps.filter.offset << std::endl;
    }

    /**
     * @brief LoadLookupTable Read lookup table from simple binary file.
     * @param ctx
     * @param filename  Name of the binary file to load.
     * @return Returns Status::OK() or throws exception.
     */
    tensorflow::Status LoadLookupTable(tensorflow::OpKernelConstruction *ctx, const std::string &filename)
    {
        std::fill(lookupTable.begin(), lookupTable.end(), ATVT(0));

        std::fstream file(filename.c_str(), std::ios::in | std::ios::binary);
        file.read(reinterpret_cast<char *>(lookupTable.data()), lookupTable.size() * sizeof(ATVT));

        VLOG(2) << "ApproxTF: Reading lookup table file: " << filename;
        TF_REQUIRES(file, tensorflow::errors::InvalidArgument("ApproxTF: Couldn't read the lookup table!"));

        return tensorflow::Status::OK();
    }

    int bitWidth;                   ///< Number of bits in approximation (only 8 bit supported).
    std::string tableFilename;      ///< Name of the binary file containing the lookup table.
    int quantMin, quantMax;         ///< Min/Max of quantized value (0 - 2^8-1).
    OpQuantProps_t quantProps;      ///< Quantization properties of the inputs.
    std::vector<ATVT> lookupTable;  ///< Approximate OP lookup table data.
};

#undef TF_REQUIRES

/**
 * Template for approximated operations on both CPU and GPU.
 */
template<typename Device, typename T, typename AT>
struct TableApproxOpType_t : public BaseTableApproxOpType_t<T, AT, tensorflow::uint16>
{
    TableApproxOpType_t(tensorflow::OpKernelConstruction *ctx);
};

/**
 * Approximated operation for CPU.
 */
template<typename T, typename AT>
struct TableApproxOpType_t<Eigen::ThreadPoolDevice, T, AT> : public BaseTableApproxOpType_t<T, AT, tensorflow::uint16>
{
    using Base_t = BaseTableApproxOpType_t<T, AT, tensorflow::uint16>;

    TableApproxOpType_t(tensorflow::OpKernelConstruction *ctx) : Base_t(ctx) {

    }

    const tensorflow::uint16 *GetLookupData() const { return Base_t::lookupTable.data(); }
};

#ifdef GOOGLE_CUDA

typedef unsigned long long cudaTextureObject_t;

/**
 * Approximated operation for GPU (CUDA) using texture memory to store the
 * lookup table.
 */
template<typename T, typename AT>
struct TableApproxOpType_t<Eigen::GpuDevice, T, AT> : public BaseTableApproxOpType_t<T, AT, tensorflow::uint16>
{
    using Base_t = BaseTableApproxOpType_t<T, AT, tensorflow::uint16>;

    TableApproxOpType_t(tensorflow::OpKernelConstruction *ctx);
    virtual ~TableApproxOpType_t();

    cudaTextureObject_t GetLookupData() const { return m_lookupTexture; }

protected:
    cudaTextureObject_t m_lookupTexture;        ///< Lookup table (texture) handle.
    tensorflow::uint16 *m_pLookupTextureData;   ///< Lookup table data.
};

#endif // GOOGLE_CUDA

#endif // APPROX_OPS_TYPES_H
