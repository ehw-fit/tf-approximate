//========== Copyright (c) 2019, Filip Vaverka, All rights reserved. =========//
//
// Purpose:     Lookup table approximation of 8-bit MUL operations.
//
// $NoKeywords: $ApproxGPUOpsTF $approx_ops_types.cu
// $Date:       $2019-12-19
//============================================================================//

#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "gpu_kernel_helper.h"

#include "approx_ops_types.h"

using GPUDevice = Eigen::GpuDevice;

using namespace tensorflow;

template<typename T, typename AT>
TableApproxOpType_t<Eigen::GpuDevice, T, AT>::TableApproxOpType_t(OpKernelConstruction *ctx)
    : Base_t(ctx)
    , m_lookupTexture(0)
    , m_pLookupTextureData(nullptr)
{
    cudaMalloc(&m_pLookupTextureData, Base_t::lookupTable.size() * sizeof(uint16));
    cudaMemcpy(m_pLookupTextureData, Base_t::lookupTable.data(), Base_t::lookupTable.size() * sizeof(uint16), cudaMemcpyHostToDevice);
    
    cudaResourceDesc lookupTableResDesc;
    memset(&lookupTableResDesc, 0, sizeof(cudaResourceDesc));
    lookupTableResDesc.resType = cudaResourceTypeLinear;
    lookupTableResDesc.res.linear.devPtr = m_pLookupTextureData;
    lookupTableResDesc.res.linear.desc.f = cudaChannelFormatKindUnsigned;
    lookupTableResDesc.res.linear.desc.x = 16;
    lookupTableResDesc.res.linear.sizeInBytes = Base_t::lookupTable.size() * sizeof(uint16);
    
    cudaTextureDesc lookupTableTexDesc;
    memset(&lookupTableTexDesc, 0, sizeof(cudaTextureDesc));
    lookupTableTexDesc.readMode = cudaReadModeElementType;
    
    cudaCreateTextureObject(&m_lookupTexture, &lookupTableResDesc, &lookupTableTexDesc, nullptr);
}

template<typename T, typename AT>
TableApproxOpType_t<Eigen::GpuDevice, T, AT>::~TableApproxOpType_t()
{
    cudaDestroyTextureObject(m_lookupTexture);
    cudaFree(m_pLookupTextureData);
}

template struct TableApproxOpType_t<Eigen::GpuDevice, float, uint8>;

#endif // GOOGLE_CUDA
