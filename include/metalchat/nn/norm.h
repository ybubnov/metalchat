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
    void
    operator()(
        const tensor<T, 1, InputRef>& input,
        const tensor<T, 1, WeightRef>& weight,
        const T eps = T(1e-5)
    )
    {
        auto output = empty<T>({input.size(0)}, m_device);
        auto eps_ = fill<T>({1}, eps);
        auto blocks = dim3(input.size(0));

        blocking_kernel(blocks, 1, input, weight, eps_, output);
        return output;
    }
};


} // namespace nn
} // namespace metalchat
