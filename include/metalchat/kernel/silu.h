#pragma once


#include <metalchat/device.h>
#include <metalchat/dtype.h>
#include <metalchat/kernel.h>
#include <metalchat/tensor.h>


namespace metalchat {


template <typename T> class silu : public kernel {
private:
    inline static const std::string operation_name = "silu";

public:
    silu(device& device)
    : kernel(operation_name, type_traits<T>::name(), device)
    {}

    template <std::size_t N, ContiguousContainer InputContainer>
    auto
    operator()(const tensor<T, N, InputContainer>& input)
    {
        constexpr std::size_t block_size = 32;

        auto data_size = input.numel();
        auto dim_size = input.sizes().back();
        auto num_rows = data_size / dim_size;

        auto output = empty_like(input, m_device);

        auto thread_size = ceil_div(dim_size, block_size);
        auto thread = dim3(thread_size);
        auto threads = dim3(thread_size * num_rows);

        blocking(threads, thread)(scalar<int32_t>(dim_size), input, output);
        return output;
    }
};


} // namespace metalchat
