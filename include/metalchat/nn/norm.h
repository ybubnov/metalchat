#pragma once


#include <metalchat/device.h>
#include <metalchat/kernel.h>
#include <metalchat/tensor.h>


namespace metalchat {
namespace nn {


class rmsnorm : public kernel {
public:
    rmsnorm(const std::string& opname, device& device)
    : kernel(opname, device)
    {}

    template <
        typename T,
        template <typename U> class InputRef,
        template <typename V> class WeightRef>
    tensor<T, 1, device_ref>
    operator()(
        const tensor<T, 1, InputRef>& input,
        const tensor<T, 1, WeightRef>& weight,
        const T eps = T(1e-5)
    )
    {
        auto output = empty<T>({input.size(0)}, m_device);

        auto eps_ = full<T>({1}, eps);
        auto input_size = full<int32_t>({1}, int32_t(input.size(0)));

        auto groups = dim3(1);
        auto threads = dim3(input.size(0) / 4);

        blocking_kernel(groups, threads, input, weight, eps_, input_size, output);
        return output;
    }
};


} // namespace nn
} // namespace metalchat
