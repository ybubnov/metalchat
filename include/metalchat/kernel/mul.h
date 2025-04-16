#pragma once


#include <metalchat/device.h>
#include <metalchat/dtype.h>
#include <metalchat/kernel.h>
#include <metalchat/kernel_task.h>
#include <metalchat/tensor.h>
#include <metalchat/tensor_future.h>


namespace metalchat {


template <typename T> class hadamard : public base_kernel {
private:
    inline static const std::string operation_name = "hadamard";

public:
    hadamard(device& device)
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
                "hadamard: last dimension size should be the same for both tensors {} != {}",
                dim_size, dim_size2
            ));
        }

        if (auto data_size2 = input2.numel(); data_size != data_size2) {
            throw std::invalid_argument(std::format(
                "hadamard: data size should be the same for both tensors {} != {}", data_size,
                data_size2
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


template <typename T> class scalar_mul {
private:
    inline static const std::string operation_name = "scalar_mul";

    kernel_base _m_kernel;

public:
    scalar_mul(device& device)
    : _m_kernel(device.load(operation_name, type_traits<T>::name()))
    {}

    template <
        std::size_t N,
        ContiguousContainer InputContainer,
        ContiguousContainer MultiplierContainer>
    auto
    operator()(
        shared_tensor<T, N, InputContainer> input,
        shared_tensor<T, 0, MultiplierContainer> multiplier
    )
    {
        constexpr std::size_t block_size = 32;

        auto dim_size = input.sizes().back();
        auto num_rows = input.numel() / dim_size;
        auto input_view = input.view({-1, int(dim_size)});

        auto [grid, thread] = make_kernel_grid_1d(input, block_size);

        auto task = kernel_task(_m_kernel, grid, thread);
        auto fn = task.bind_back(input_view, multiplier);

        auto output = empty_future<T>({num_rows, dim_size}, std::move(fn));
        int output_sizes[N];
        for (auto i = 0; i < N; i++) {
            output_sizes[i] = input.size(i);
        }

        return output.view(std::move(output_sizes));
    }

    template <std::size_t N, ContiguousContainer InputContainer>
    auto
    operator()(shared_tensor<T, N, InputContainer> input, const T multiplier)
    {
        return operator()(input, shared_tensor(scalar(multiplier)));
    }
};


} // namespace metalchat
