#pragma once

#include <format>

#include <metalchat/device.h>
#include <metalchat/dtype.h>
#include <metalchat/kernel.h>
#include <metalchat/kernel_task.h>
#include <metalchat/tensor.h>
#include <metalchat/tensor_future.h>


namespace metalchat {


template <typename T> class sum {
private:
    inline static const std::string operation_name = "sum";

    kernel_base _m_kernel;

public:
    sum(device& device)
    : _m_kernel(device.load(operation_name, type_traits<T>::name()))
    {}

    template <
        std::size_t M,
        std::size_t N,
        ContiguousContainer Input1Container,
        ContiguousContainer Input2Container>
    auto
    operator()(
        shared_tensor<T, M, Input1Container> input1, shared_tensor<T, N, Input2Container> input2
    )
    {
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

        constexpr std::size_t block_size = 32;
        auto [grid, thread] = make_kernel_grid_1d(input1, block_size);

        auto input1_view = input1.view({-1, int(dim_size)});
        auto input2_view = input2.view({-1, int(dim_size)});

        auto task = kernel_task(_m_kernel, grid, thread);
        auto fn = task.bind_back(input1_view, input2_view);

        auto output = empty_future<T>({num_rows, dim_size}, std::move(fn));
        int output_sizes[N];
        for (auto i = 0; i < N; i++) {
            output_sizes[i] = input1.size(i);
        }

        return output.view(std::move(output_sizes));
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
