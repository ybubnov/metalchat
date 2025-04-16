#pragma once


#include <metalchat/device.h>
#include <metalchat/nn/operation.h>
#include <metalchat/tensor.h>


namespace metalchat {
namespace nn {


class rmsnorm : public operation {
public:
    rmsnorm(const std::string& opname, device& device)
    : operation(opname, device)
    {}

    template <
        typename T,
        template <typename U>
        class InputRef,
        template <typename V>
        class WeightRef>
    void
    operator()(const tensor<T, 1, InputRef>& input, const tensor<T, 1, WeightRef>& weight)
    {}
};


} // namespace nn
} // namespace metalchat
