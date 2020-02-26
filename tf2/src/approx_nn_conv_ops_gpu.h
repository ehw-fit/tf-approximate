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

#ifndef APPROX_NN_CONV_OPS_GPU_H
#define APPROX_NN_CONV_OPS_GPU_H

#if GOOGLE_CUDA

#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/stream_executor/stream_executor.h>
#include <tensorflow/stream_executor/scratch_allocator.h>
#include "gpu_utils.h"

namespace tensorflow {

class DnnScratchAllocator : public se::ScratchAllocator {
public:
    virtual ~DnnScratchAllocator() {}

    DnnScratchAllocator(int64 memoryLimit, OpKernelContext *ctx)
        : m_memoryLimit(memoryLimit), m_totalByteSize(0), m_context(ctx)
    {
    }

    se::port::StatusOr<se::DeviceMemory<uint8> > AllocateBytes(
            se::Stream *stream, int64 byteSize) override
    {
        Tensor temporaryMemory;
        if(byteSize < 0)
            return se::port::Status{se::port::error::INVALID_ARGUMENT,
                "Requested negative byte size!"};

        if(byteSize > m_memoryLimit)
            return se::port::StatusOr<se::DeviceMemory<uint8> >();

        AllocationAttributes allocationAttr;
        allocationAttr.no_retry_on_failure = true;
        Status allocationStatus(m_context->allocate_temp(
                                    DT_UINT8, TensorShape({byteSize}), &temporaryMemory,
                                    AllocatorAttributes(), allocationAttr));
        if(!allocationStatus.ok())
            return se::port::StatusOr<se::DeviceMemory<uint8> >();

        m_allocatedTensors.push_back(temporaryMemory);
        m_totalByteSize += byteSize;

        return se::port::StatusOr<se::DeviceMemory<uint8> >(
                    AsDeviceMemory(temporaryMemory.flat<uint8>().data(),
                                   temporaryMemory.flat<uint8>().size()));
    }

    int64 TotalByteSize() { return m_totalByteSize; }

private:
    int64 m_memoryLimit;
    int64 m_totalByteSize;
    OpKernelContext *m_context;
    std::vector<Tensor> m_allocatedTensors;
};

}

#endif // GOOGLE_CUDA

#endif // APPROX_NN_CONV_OPS_GPU_H
