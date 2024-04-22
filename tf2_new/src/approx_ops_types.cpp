//========== Copyright (c) 2020, Filip Vaverka, All rights reserved. =========//
//
// Purpose:     Lookup table approximation of 8-bit MUL operations.
//
// $NoKeywords: $ApproxGPUOpsTF $approx_ops_types.cpp
// $Date:       $2020-09-06
//============================================================================//

#define APPROX_OPS_TYPES_H_COMMON_DEFINITIONS
#define EIGEN_USE_THREADS

#include "approx_ops_types.h"

template struct ApproxFilterCorrCoeff<Eigen::ThreadPoolDevice, Eigen::half, Eigen::half, NullApproxOpType_t>;
template struct ApproxFilterCorrCoeff<Eigen::ThreadPoolDevice, float, float, NullApproxOpType_t>;
template struct ApproxFilterCorrCoeff<Eigen::ThreadPoolDevice, double, double, NullApproxOpType_t>;

template struct ApproxFilterCorrCoeff<Eigen::ThreadPoolDevice, Eigen::half, unsigned char, TableApproxOpType_t>;
template struct ApproxFilterCorrCoeff<Eigen::ThreadPoolDevice, float, unsigned char, TableApproxOpType_t>;
template struct ApproxFilterCorrCoeff<Eigen::ThreadPoolDevice, double, unsigned char, TableApproxOpType_t>;