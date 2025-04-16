#pragma once

#include <format>

#include <metalchat/device.h>
#include <metalchat/dtype.h>
#include <metalchat/kernel.h>
#include <metalchat/tensor.h>


namespace metalchat {


template <typename T> class sum : public kernel {
private:
    inline static const std::string operation_name = "sum";

public:
    sum(device& device)
    : kernel(operation_name, type_traits<T>::name(), device)
    {}

    template <
        std::size_t M,
        std::size_t N,
        ContiguousContainer Input1Container,
        ContiguousContainer Input2Container>
    auto
    operator()(
        const tensor<T, M, Input1Container>& input1, const tensor<T, N, Input2Container>& input2
    )
    {
        if (input1.numel() != input2.numel()) {
            throw std::invalid_argument(std::format(
                "kernel::sum: tensor1 {} and tensor2 {} cannot be broadcasted", input1.sizes(),
                input2.sizes()
            ));
        }

        auto output = empty_like(input1, m_device);
        auto n = scalar<int32_t>(input1.numel());

        auto threads = dim3(input1.numel());
        // TODO: change the size of a threadgroup to the maximum.
        auto thread = dim3(32);

        blocking(threads, thread)(n, input1, input2, output);
        return output;
    }
};


} // namespace metalchat
