#pragma once


#include <metalchat/device.h>
#include <metalchat/dtype.h>
#include <metalchat/format.h>
#include <metalchat/kernel.h>
#include <metalchat/tensor.h>


namespace metalchat {


template <typename T> class bmm : public kernel {
private:
    inline static const std::string operation_name = "bmm";

public:
    bmm(device& device)
    : kernel(operation_name, type_traits<T>::name(), device)
    {}

    template <ContiguousContainer InputContainer, ContiguousContainer WeightContainer>
    auto
    operator()(
        const tensor<T, 2, InputContainer>& input, const tensor<T, 2, WeightContainer>& weight
    )
    {
        if (input.size(1) != weight.size(0)) {
            throw std::invalid_argument(std::format(
                "bmm: matrices are with different inner dimension ({}x{}) and ({}x{})",
                input.size(0), input.size(1), weight.size(0), weight.size(1)
            ));
        }
        // A(MxK) @ B(KxN) -> C(MxN)
        auto output = empty<T>({input.size(0), weight.size(1)}, m_device);

        constexpr std::size_t block_size = 32;

        auto threads = dim3(
            ceil_div(input.size(0), block_size) * block_size,
            ceil_div(weight.size(1), block_size) * block_size
        );
        auto thread = dim3(block_size, block_size);

        blocking(threads, thread)(
            input, weight, output, scalar(input.layout()), scalar(weight.layout()),
            scalar(output.layout())
        );
        return output;
    }

    template <ContiguousContainer InputContainer, ContiguousContainer WeightContainer>
    auto
    operator()(
        const tensor<T, 4, InputContainer>& input, const tensor<T, 4, WeightContainer>& weight
    )
    {
        assert((input.size(0) == weight.size(0)));
        assert((input.size(1) == weight.size(1)));
        assert((input.size(3) == weight.size(2)));

        auto output = empty<T>({input.size(0), input.size(1), input.size(2), weight.size(3)});

        for (auto b0 = 0; b0 < input.size(0); b0++) {
            for (auto b1 = 0; b1 < input.size(1); b1++) {
                for (auto i = 0; i < input.size(2); i++) {
                    for (auto k = 0; k < weight.size(3); k++) {
                        float partial_sum = 0;
                        for (auto j = 0; j < input.size(3); j++) {
                            partial_sum += input[b0, b1, i, j] * weight[b0, b1, j, k];
                        }
                        output[b0, b1, i, k] = T(partial_sum);
                    }
                }
            }
        }

        return output;
    }

    template <ContiguousContainer InputContainer, ContiguousContainer WeightContainer>
    auto
    operator()(
        const tensor<T, 3, InputContainer>& input, const tensor<T, 2, WeightContainer>& weight
    )
    {
        assert((input.size(2) == weight.size(0)));
        assert((input.size(0) == 1));

        auto output = operator()(input.reshape({int(input.size(1)), int(input.size(2))}), weight);
        return output.reshape({1, int(input.size(1)), int(weight.size(1))});
    }
};


} // namespace metalchat
