//========== Copyright (c) 2021, Filip Vaverka, All rights reserved. =========//
//
// Purpose:     Quantization data for approximate operation.
//
// $NoKeywords: $ApproxGPUOpsTF $approx_ops_quant_data.cu
// $Date:       $2021-05-30
//============================================================================//

#define EIGEN_USE_GPU
#define EIGEN_STACK_ALLOCATION_LIMIT 0

#include "approx_ops_quant_data.h"

using GPUDevice = Eigen::GpuDevice;
using namespace tensorflow;

#if GOOGLE_CUDA

template class ApproxConvOpQuantData<TableApproxOpType_t<Eigen::GpuDevice, float, uint8_t> >;
//template class ApproxConvOpQuantData<GPUDevice, double, uint8_t, TableApproxOpType_t>;

template<>
void CopyToDeviceSync(const GPUDevice &d, void *pDst, const void *pSrc, size_t size) {
    d.memcpyHostToDevice(pDst, pSrc, size);
    d.synchronize();
}

#endif // GOOGLE_CUDA