#pragma once

#include <format>

#include <metalchat/device.h>
#include <metalchat/dtype.h>
#include <metalchat/kernel.h>
#include <metalchat/tensor.h>


namespace metalchat {


template <typename T> class rmsnorm : public kernel {
private:
    inline static const std::string operation_name = "rmsnorm";

public:
    rmsnorm(device& device)
    : kernel(std::format("{}_{}", operation_name, type_traits<T>::name()), device)
    {}

    template <ContiguousContainer InputContainer, ContiguousContainer WeightContainer>
    tensor<T, 1, device_ref>
    operator()(
        const tensor<T, 1, InputContainer>& input,
        const tensor<T, 1, WeightContainer>& weight,
        const T eps = T(1e-5)
    )
    {
        auto output = empty<T>({input.size(0)}, m_device);

        auto eps_ = scalar<T>(eps);
        auto input_size = scalar<int32_t>(input.size(0));

        auto groups = dim3(1);
        auto threads = dim3(input.size(0) / 4);

        blocking(groups, threads)(input, weight, eps_, input_size, output);
        return output;
    }
};


} // namespace metalchat
