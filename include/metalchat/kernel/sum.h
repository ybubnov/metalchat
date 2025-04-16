#pragma once

#include <format>

#include <metalchat/device.h>
#include <metalchat/dtype.h>
#include <metalchat/kernel.h>
#include <metalchat/tensor.h>


namespace metalchat {


template <typename T> class sum : public base_kernel {
private:
    inline static const std::string operation_name = "sum";

public:
    sum(device& device)
    : base_kernel(operation_name, type_traits<T>::name(), device)
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
        constexpr std::size_t block_size = 32;

        auto data_size = input1.numel();
        auto dim_size = input1.sizes().back();
        auto num_rows = data_size / dim_size;

        if (auto dim_size2 = input2.sizes().back(); dim_size != dim_size2) {
            throw std::invalid_argument(std::format(
                "sum: last dimension should be the same for both tensors {} != {}", dim_size,
                dim_size2
            ));
        }

        if (auto data_size2 = input2.numel(); data_size != data_size2) {
            throw std::invalid_argument(std::format(
                "sum: data size should be the same for both tensors {} != {}", data_size, data_size2
            ));
        }

        auto output = empty_like(input1, m_device);

        auto thread_size = ceil_div(dim_size, block_size);
        auto thread = dim3(thread_size);
        auto threads = dim3(thread_size * num_rows);

        blocking(threads, thread)(scalar<int32_t>(dim_size), input1, input2, output);
        return output;
    }
};


template <typename T> class sum2 : public base_kernel {
private:
    inline static const std::string operation_name = "sum2";

public:
    sum2(device& device)
    : base_kernel(operation_name, type_traits<T>::name(), device)
    {}

    template <
        std::size_t M,
        ContiguousContainer Input1Container,
        ContiguousContainer Input2Container>
    requires(M >= 2)
    auto
    operator()(
        const tensor<T, M, Input1Container>& input1, const tensor<T, 2, Input2Container>& input2
    )
    {
        constexpr std::size_t block_size_x = 4;
        constexpr std::size_t block_size_y = 4;

        auto data_size = input1.numel();
        auto dim0_size = input2.size(0);
        auto dim1_size = input2.size(1);
        auto num_rows = data_size / (dim0_size * dim1_size);
        // auto num_cols = dim1_size /

        if (dim0_size != input1.size(M - 2) || dim1_size != input1.size(M - 1)) {
            throw std::invalid_argument(std::format(
                "sum2: last dimensions should be the same for both tensors {}x{} != {}x{}",
                input1.size(M - 2), input1.size(M - 1), dim0_size, dim1_size
            ));
        }

        auto output = empty_like(input1, m_device);

        auto x_size = ceil_div(dim0_size, block_size_x);
        auto y_size = ceil_div(dim1_size, block_size_y);
        auto thread = dim3(x_size, y_size);
        auto threads = dim3(x_size * num_rows, y_size);

        blocking(threads, thread)(
            scalar<int32_t>(dim0_size), scalar<int32_t>(dim1_size), input1, input2, output
        );
        return output;
    }
};


} // namespace metalchat
