#pragma once


#include <metalchat/device.h>
#include <metalchat/nn/operation.h>
#include <metalchat/tensor.h>


namespace metalchat {
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
        auto stride = full<int64_t>({1}, /*fill_value=*/weight.stride(0));
        auto result = empty<T>({input.size(0), weight.size(1)}, m_device);

        auto blocks = dim3(input.size(0), weight.size(1));
        blocking_kernel(blocks, 1, input, weight, stride, result);

        return result;
    }
};


} // namespace nn
} // namespace metalchat
