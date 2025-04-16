#pragma once


#include <metalchat/device.h>
#include <metalchat/dtype.h>
#include <metalchat/format.h>
#include <metalchat/kernel.h>
#include <metalchat/tensor.h>


namespace metalchat {


template <typename T> class softmax : public kernel {
private:
    inline static const std::string operation_name = "softmax";

public:
    softmax(device& device)
    : kernel(operation_name, type_traits<T>::name(), device)
    {}

    template <std::size_t N, ContiguousContainer InputContainer>
    auto
    operator()(const tensor<T, N, InputContainer>& input)
    {
        auto data_size = input.numel();
        auto dim_size = input.sizes().back();
        auto num_rows = data_size / dim_size;

        constexpr std::size_t simd_size = 32;
        constexpr std::size_t block_size = 4;

        assert((dim_size <= block_size * 1024)); // 1024 = max total threads per tg.

        auto thread_size = ceil_div(dim_size, block_size);
        auto thread_simd_size = ceil_div(thread_size, simd_size);

        auto thread = dim3(thread_size * thread_simd_size);
        auto threads = dim3(thread_size * thread_simd_size * num_rows);

        auto output = empty_like(input, m_device);

        blocking(threads, thread)(scalar<int32_t>(dim_size), input, output);
        return output;
    }
};


} // namespace metalchat
