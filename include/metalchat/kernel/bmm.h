#pragma once


#include <metalchat/device.h>
#include <metalchat/dtype.h>
#include <metalchat/format.h>
#include <metalchat/kernel.h>
#include <metalchat/tensor.h>


namespace metalchat {


template <typename T> class bmm : public base_kernel {
private:
    inline static const std::string operation_name = "bmm";

public:
    bmm(device& device)
    : base_kernel(operation_name, type_traits<T>::name(), device)
    {}

    template <ContiguousContainer InputContainer, ContiguousContainer WeightContainer>
    auto
    operator()(
        const tensor<T, 3, InputContainer>& input, const tensor<T, 3, WeightContainer>& weight
    )
    {
        auto num_batches = input.size(0);

        // Batched matmul does not support broadcasting operations, therefore throw an
        // exception, when the number of batches for input tensors are different.
        if (auto num_weight_batches = weight.size(0); num_batches != num_weight_batches) {
            throw std::invalid_argument(std::format(
                "kernel::bmm: batches of the input tensors should be the same {} != {}",
                num_batches, num_weight_batches
            ));
        }

        if (input.size(2) != weight.size(1)) {
            throw std::invalid_argument(std::format(
                "kernel::bmm: matrices are with different inner dimension ({}x{}) and ({}x{})",
                input.size(1), input.size(2), weight.size(1), weight.size(2)
            ));
        }
        // A(MxK) @ B(KxN) -> C(MxN)
        auto output = empty<T>({num_batches, input.size(1), weight.size(2)}, m_device);

        constexpr std::size_t block_size = 32;

        auto threads = dim3(
            ceil_div(input.size(1), block_size) * block_size,
            ceil_div(weight.size(2), block_size) * block_size, num_batches
        );
        auto thread = dim3(block_size, block_size);

        blocking(threads, thread)(
            scalar(input.layout()), scalar(weight.layout()), scalar(output.layout()), input, weight,
            output
        );
        return output;
    }

    template <ContiguousContainer InputContainer, ContiguousContainer WeightContainer>
    auto
    operator()(
        const tensor<T, 2, InputContainer>& input, const tensor<T, 2, WeightContainer>& weight
    )
    {
        int input_size0 = input.size(0);
        int input_size1 = input.size(1);
        int weight_size0 = weight.size(0);
        int weight_size1 = weight.size(1);

        auto input_view = input.view({1, input_size0, input_size1});
        auto weight_view = weight.view({1, weight_size0, weight_size1});
        auto output = operator()(input_view, weight_view);

        return output.view({input_size0, weight_size1});
    }

    template <
        std::size_t N,
        ContiguousContainer InputContainer,
        ContiguousContainer WeightContainer>
    auto
    operator()(
        const tensor<T, N, InputContainer>& input, const tensor<T, N, WeightContainer>& weight
    ) requires(N > 3)
    {
        int input_size0 = input.size(N - 2);
        int input_size1 = input.size(N - 1);
        int weight_size0 = weight.size(N - 2);
        int weight_size1 = weight.size(N - 1);

        int output_sizes[N];
        output_sizes[N - 2] = input_size0;
        output_sizes[N - 1] = weight_size1;
        for (std::size_t i = 0; i < N - 2; i++) {
            output_sizes[i] = input.size(i);
        }

        auto input_view = input.view({-1, input_size0, input_size1});
        auto weight_view = weight.view({-1, weight_size0, weight_size1});

        auto output = operator()(input_view, weight_view);
        return output.view(std::move(output_sizes));
    }

    template <ContiguousContainer InputContainer, ContiguousContainer WeightContainer>
    auto
    operator()(
        const tensor<T, 3, InputContainer>& input, const tensor<T, 2, WeightContainer>& weight
    )
    {
        int weight_size0 = weight.size(0);
        int weight_size1 = weight.size(1);

        // TODO: does it make sense to call repeat_interleave for the number of batches > 1?
        auto weight_view = weight.view({1, weight_size0, weight_size1});
        return operator()(input, weight_view);
    }

    template <ContiguousContainer InputContainer, ContiguousContainer WeightContainer>
    auto
    maybe_compute(
        const tensor<T, 3, InputContainer>& input, const tensor<T, 2, WeightContainer>& weight
    )
    {
        constexpr std::size_t block_size = 32;

        auto num_batches = input.size(0);
        int weight_size0 = weight.size(0);
        int weight_size1 = weight.size(1);

        auto weight_view = weight.view({1, weight_size0, weight_size1});

        // A(MxK) @ B(KxN) -> C(MxN)
        std::size_t output_sizes[3] = {num_batches, input.size(1), weight_view.size(2)};
        auto output
            = std::make_shared<tensor<T, 3, device_ref<T>>>(std::move(output_sizes), m_device);
        auto cb = unsequenced_policy_callback<T, 3>{
            .output = output,
            .promise = std::make_shared<std::promise<bool>>()
        };

        auto threads = dim3(
            ceil_div(input.size(1), block_size) * block_size,
            ceil_div(weight_view.size(2), block_size) * block_size, num_batches
        );
        auto thread = dim3(block_size, block_size);

        auto policy = nonblocking(threads, thread, std::move(cb));
        policy(
            scalar(input.layout()), scalar(weight_view.layout()), scalar(output->layout()), input,
            weight_view, *output
        );
        return policy;
    }
};


} // namespace metalchat
