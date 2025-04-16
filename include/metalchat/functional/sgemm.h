#pragma once


#include <metalchat/device.h>
#include <metalchat/dtype.h>
#include <metalchat/kernel.h>
#include <metalchat/tensor.h>


namespace metalchat {


template <typename T> class sgemm : public kernel {
private:
    inline static const std::string operation_name = "sgemm";

public:
    sgemm(device& device)
    : kernel(operation_name, type_traits<T>::name(), device)
    {}

    template <ContiguousContainer InputContainer, ContiguousContainer WeightContainer>
    auto
    operator()(
        const tensor<T, 2, InputContainer>& input, const tensor<T, 2, WeightContainer>& weight
    )
    {
        assert(input.size(1) == weight.size(0));

        auto output = empty<T>({input.size(0), weight.size(1)}, m_device);

        auto m = scalar<int32_t>(input.size(0));
        auto k = scalar<int32_t>(input.size(1));
        auto n = scalar<int32_t>(weight.size(1));

        auto groups = dim3(ceil_div(input.size(0), 32), ceil_div(weight.size(1), 32), 1);
        auto threads = dim3(32, 32, 1);

        blocking(groups, threads)(m, n, k, input, weight, output);
        return output;
    }
};


} // namespace metalchat
