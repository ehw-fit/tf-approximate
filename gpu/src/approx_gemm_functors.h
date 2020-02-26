//========== Copyright (c) 2019, Filip Vaverka, All rights reserved. =========//
//
// Purpose:     Simple implementation of standard GEMM
//
// $NoKeywords: $ApproxGPUOpsTF $approx_gemm_functors.h
// $Date:       $2019-12-19
//============================================================================//

#pragma once

#ifndef APPROX_GEMM_FUNCTORS_H
#define APPROX_GEMM_FUNCTORS_H

#include <tensorflow/core/framework/op_kernel.h>

/**
 * @brief Reference Matrix-Matrix multiplication OP.
 */
template<class T1, class T2, class T3>
class ReferenceGemmFunctor {
public:
    void operator()(tensorflow::OpKernelContext *ctx, size_t m, size_t n,
                    size_t k, const T1 *a, size_t lda, const T2 *b, size_t ldb,
                    T3 *c, size_t ldc)
    {
        const size_t aIStride = lda;
        const size_t aLStride = 1;
        const size_t bJStride = 1;
        const size_t bLStride = ldb;
        const size_t cIStride = ldc;
        const size_t cJStride = 1;

        for(size_t j = 0; j < n; ++j)
        {
            for(size_t i = 0; i < m; ++i)
            {
                T3 total(0);

                for(size_t l = 0; l < k; ++l)
                {
                    const size_t aIndex = ((i * aIStride) + (l * aLStride));
                    const T1 aValue = a[aIndex];

                    const size_t bIndex = ((j * bJStride) + (l * bLStride));
                    const T2 bValue = b[bIndex];

                    total += (aValue * bValue);
                }

                const size_t cIndex = ((i * cIStride) + (j * cJStride));
                c[cIndex] = total;
            }
        }
    }
};

#endif // APPROX_GEMM_FUNCTORS_H
