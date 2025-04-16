#pragma once

#include <cmath>
#include <concepts>
#include <type_traits>

#include <metalchat/device.h>
#include <metalchat/dtype.h>
#include <metalchat/format.h>
#include <metalchat/kernel.h>
#include <metalchat/kernel_task.h>
#include <metalchat/tensor_future.h>


namespace metalchat {


template <typename T, std::size_t BlockSize = 16, std::size_t EmbeddingBlockSize = 64>
class embedding {
private:
    inline static const std::string operation_name
        = "embedding_" + std::to_string(BlockSize) + "x" + std::to_string(EmbeddingBlockSize);

    kernel_base _m_kernel;

public:
    embedding(device& device)
    : _m_kernel(device.load(operation_name, type_traits<T>::name()))
    {}

    template <immutable_tensor2_t<int32_t> Input, immutable_tensor2_t<T> WeightTensor>
    auto
    operator()(Input input, WeightTensor weight)
    {
        auto data_size = input.numel();
        auto emb_size = weight.sizes().back();
        auto dim_size = input.sizes().back();
        auto num_rows = data_size / dim_size;

        auto thread_size_x = ceil_div(dim_size, BlockSize);
        auto thread_size_y = ceil_div(emb_size, EmbeddingBlockSize);
        auto thread = dim3(thread_size_x, thread_size_y);
        auto grid = dim3(thread_size_x * num_rows, thread_size_y);

        auto task = kernel_task(_m_kernel, grid, thread);
        auto fn = task.bind_back(input, weight);

        return empty_future<T>({input.size(0), dim_size, emb_size}, std::move(fn));
    }
};


template <typename T, std::size_t BlockSize = 16> class rope {
private:
    inline static const std::string operation_name = "rope_" + std::to_string(BlockSize);

    kernel_base _m_kernel;

public:
    rope(device& device)
    : _m_kernel(device.load(operation_name, type_traits<T>::name()))
    {}

    template <
        immutable_tensor4_t<T> Input,
        immutable_tensor2_t<float> Cosines,
        immutable_tensor2_t<float> Sines>
    auto
    debug_rope(Input input, Cosines freqs_cos, Sines freqs_sin, std::size_t start_pos)
    {
        return operator()(input, freqs_cos, freqs_sin, start_pos);
    }

    template <
        immutable_tensor4_t<T> Input,
        immutable_tensor2_t<float> Cosines,
        immutable_tensor2_t<float> Sines>
    auto
    operator()(Input input, Cosines freqs_cos, Sines freqs_sin, std::size_t start_pos)
    {
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

        auto [grid, thread] = make_kernel_grid_1d(input, BlockSize);

        auto task = kernel_task(_m_kernel, grid, thread);
        auto fn = task.bind_front(
            input.view({-1, int(dim_size)}), freqs_cos, freqs_sin,
            shared_tensor(scalar<int32_t>(bs)), shared_tensor(scalar<int32_t>(n_head)),
            shared_tensor(scalar<int32_t>(start_pos))
        );

        auto output = empty_future<T>({num_rows, dim_size}, std::move(fn));
        return output.view(input.sizes());
    }
};


} // namespace metalchat
