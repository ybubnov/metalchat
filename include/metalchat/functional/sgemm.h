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

        auto threads = dim3(input.size(0), weight.size(1));
        auto thread = dim3(32, 32);

        blocking(threads, thread)(m, n, k, input, weight, output);
        return output;
        return output;
    }

    template <ContiguousContainer InputContainer, ContiguousContainer WeightContainer>
    auto
    operator()(
        const tensor<T, 4, InputContainer>& input, const tensor<T, 4, WeightContainer>& weight
    )
    {
        assert(input.size(0) == weight.size(0));
        assert(input.size(1) == weight.size(1));
        assert(input.size(3) == weight.size(2));

        auto output = full<T>({input.size(0), input.size(1), input.size(2), weight.size(3)}, 0.0);

        for (auto b0 = 0; b0 < input.size(0); b0++) {
            for (auto b1 = 0; b1 < input.size(1); b1++) {

                for (auto i = 0; i < input.size(2); i++) {
                    for (auto k = 0; k < input.size(3); k++) {
                        for (auto j = 0; j < weight.size(3); j++) {
                            output[b0][b1][i][j] += input[b0][b1][i][k] + weight[b0][b1][k][j];
                        }
                    }
                }
            }
        }

        return output;
    }
};


} // namespace metalchat
