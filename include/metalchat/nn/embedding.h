#pragma once


#include <metalchat/device.h>
#include <metalchat/kernel.h>
#include <metalchat/tensor.h>


namespace metalchat {
namespace nn {


class embedding : public kernel {
public:
    embedding(const std::string& opname, device& device)
    : kernel(opname, device)
    {}

    template <
        typename T,
        template <typename U> class InputRef,
        template <typename V> class WeightRef>
    auto
    operator()(const tensor<int32_t, 1, InputRef>& input, const tensor<T, 2, WeightRef>& weight)
    {
        auto stride = full<int64_t>({1}, /*fill_value=*/weight.stride(0));
        auto output = empty<T>({input.size(0), weight.size(1)}, m_device);

        auto blocks = dim3(input.size(0), weight.size(1));
        blocking(blocks, 1)(input, weight, stride, output);

        return output;
    }
};


} // namespace nn
} // namespace metalchat
