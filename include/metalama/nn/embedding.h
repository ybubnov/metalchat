#pragma once


#include <metalama/device.h>
#include <metalama/nn/operation.h>
#include <metalama/tensor.h>


namespace metalama {
namespace nn {


class embedding : public operation {
public:
    embedding(const std::string& opname, device& device)
    : operation(opname, device)
    {}

    template <
        typename T,
        template <typename U>
        class InputRef,
        template <typename V>
        class WeightRef>
    tensor<T, 2, device_ref>
    operator()(const tensor<int32_t, 1, InputRef>& input, const tensor<T, 2, WeightRef>& weight)
    {
        auto op_kernel = this->m_device.make_fn(this->name());

        NS::Error* error = nullptr;
        auto pipeline
            = NS::TransferPtr(this->m_device->newComputePipelineState(op_kernel.get(), &error));
        if (error != nullptr) {
            throw std::runtime_error("failed to create compute pipeline");
        }

        auto command_queue = NS::TransferPtr(this->m_device->newCommandQueue());

        auto result = empty<T>({input.size(0), weight.size(1)}, m_device);

        auto input_buf = this->make_buf(input);
        auto weight_buf = this->make_buf(weight);
        auto result_buf = this->make_buf(result);
        auto weight_stride = weight.stride(0);

        auto command_buf = NS::TransferPtr(command_queue->commandBuffer());
        auto command_encoder = NS::TransferPtr(command_buf->computeCommandEncoder());

        command_encoder->setComputePipelineState(pipeline.get());
        command_encoder->setBuffer(input_buf.get(), 0, 0);
        command_encoder->setBuffer(weight_buf.get(), 0, 1);
        command_encoder->setBytes(&weight_stride, sizeof(weight_stride), 0, 2);
        command_encoder->setBuffer(result_buf.get(), 0, 3);

        MTL::Size grid_size(input.size(0), weight.size(1), 1);
        MTL::Size thread_group_size(1, 1, 1);
        command_encoder->dispatchThreadgroups(grid_size, thread_group_size);

        command_encoder->endEncoding();
        command_buf->commit();
        command_buf->waitUntilCompleted();

        return result;
    }
};


} // namespace nn
} // namespace metalama
