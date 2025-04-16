#pragma once

#include <cmath>
#include <concepts>

#include <metalchat/device.h>
#include <metalchat/dtype.h>
#include <metalchat/format.h>
#include <metalchat/kernel.h>
#include <metalchat/tensor.h>
#include <metalchat/tensor_future.h>
#include <metalchat/tensor_shared.h>


namespace metalchat {


template <typename T> class embedding : public base_kernel {
private:
    inline static const std::string operation_name = "embedding";

public:
    embedding(device& device)
    : base_kernel(operation_name, type_traits<T>::name(), device)
    {}

    template <
        integral IndexType,
        ContiguousContainer InputContainer,
        ContiguousContainer WeightContainer>
    auto
    operator()(
        const tensor<IndexType, 2, InputContainer>& input,
        const tensor<T, 2, WeightContainer>& weight
    )
    {
        auto output = empty<T>({input.size(0), input.size(1), weight.size(1)}, m_device);

        auto data_size = input.numel();
        auto emb_size = weight.sizes().back();
        auto dim_size = input.sizes().back();
        auto num_rows = data_size / dim_size;

        constexpr std::size_t block_size = 4;
        constexpr std::size_t eblock_size = 128;

        auto x_size = ceil_div(dim_size, block_size);
        auto y_size = ceil_div(emb_size, eblock_size);
        auto thread = dim3(x_size, y_size);
        auto threads = dim3(x_size * num_rows, y_size);

        blocking(threads, thread)(
            scalar<IndexType>(dim_size), scalar<IndexType>(emb_size), input, weight, output
        );
        return output;
    }
};


template <typename T> class rope {
private:
    inline static const std::string operation_name = "rope";

    kernel_base _m_kernel;

public:
    rope(device& device)
    : _m_kernel(device.load(operation_name, type_traits<T>::name()))
    {}

    template <ContiguousContainer InputContainer, ContiguousContainer FrequenciesContainer>
    auto
    operator()(
        shared_tensor<T, 4, InputContainer> input,
        shared_tensor<float, 2, FrequenciesContainer> freqs_cos,
        shared_tensor<float, 2, FrequenciesContainer> freqs_sin,
        std::size_t start_pos
    )
    {
        constexpr std::size_t block_size = 32;

        auto bs = input.size(0);
        auto n_head = input.size(2);

        auto data_size = input.numel();
        auto dim_size = input.sizes().back();
        auto num_rows = data_size / dim_size;

        if (auto head_dim = freqs_cos.size(1); dim_size != head_dim * 2) {
            throw std::invalid_argument(std::format(
                "kernel::rope: the last dimension of the input should be {}, but received {}",
                head_dim * 2, dim_size
            ));
        }

        auto [grid, thread] = make_kernel_grid_1d(input, block_size);

        auto task = kernel_task(_m_kernel, grid, thread);
        auto fn = task.bind_front(
            shared_tensor(input.view({-1, int(dim_size)})), freqs_cos, freqs_sin,
            shared_tensor(scalar<int32_t>(bs)), shared_tensor(scalar<int32_t>(n_head)),
            shared_tensor(scalar<int32_t>(start_pos))
        );

        auto output = empty_future<T>({num_rows, dim_size}, std::move(fn));
        return output.view(input.sizes());
    }
};


} // namespace metalchat
