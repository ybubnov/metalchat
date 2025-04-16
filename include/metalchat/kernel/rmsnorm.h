#pragma once


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
    : kernel(operation_name, type_traits<T>::name(), device)
    {}

    template <std::size_t N, ContiguousContainer InputContainer, ContiguousContainer WeightContainer>
    auto
    operator()(
        const tensor<T, N, InputContainer>& input,
        const tensor<T, 1, WeightContainer>& weight,
        const T eps = T(1e-5)
    )
    {
        auto data_size = input.numel();
        auto dim_size = input.sizes().back();
        auto num_rows = data_size / dim_size;

        assert(dim_size == weight.size(0));

        constexpr std::size_t simd_size = 32;
        constexpr std::size_t block_size = 4;

        assert(dim_size <= block_size * 1024); // 1024 = max total threads per tg.

        auto thread_size = ceil_div(dim_size, block_size);
        auto thread_simd_size = ceil_div(thread_size, simd_size);

        auto thread = dim3(thread_size * thread_simd_size);
        auto threads = dim3(thread_size * thread_simd_size * num_rows);

        auto output = empty_like<T>(input, m_device);
        auto eps_ = scalar<T>(eps);

        blocking(threads, thread)(scalar<int32_t>(dim_size), input, weight, eps_, output);
        return output;
    }
};


} // namespace metalchat
