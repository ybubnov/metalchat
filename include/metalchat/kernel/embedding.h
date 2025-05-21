#pragma once

#include <cmath>
#include <concepts>
#include <type_traits>

#include <metalchat/accelerator.h>
#include <metalchat/dtype.h>
#include <metalchat/format.h>
#include <metalchat/kernel.h>
#include <metalchat/kernel_task.h>
#include <metalchat/tensor/future.h>


namespace metalchat {
namespace kernel {


template <typename T, std::size_t BlockSize = 16, std::size_t EmbeddingBlockSize = 64>
class embedding {
private:
    basic_kernel _m_kernel;

public:
    embedding(hardware_accelerator& gpu)
    : _m_kernel(gpu.load<T, BlockSize>("embedding"))
    {}

    template <immutable_tensor2_t<int32_t> Input, immutable_tensor2_t<T> WeightTensor>
    auto
    operator()(Input input, WeightTensor weight)
    {
        auto data_size = input.numel();
        auto emb_size = weight.sizes().back();
        auto dim_size = input.sizes().back();
        auto num_rows = data_size / dim_size;

        auto output
            = shared_empty<T>({input.size(0), dim_size, emb_size}, _m_kernel.get_allocator());

        auto thread_size_x = ceil_div(dim_size, BlockSize);
        auto thread_size_y = ceil_div(emb_size, EmbeddingBlockSize);
        auto thread = dim3(thread_size_x, thread_size_y);
        auto grid = dim3(thread_size_x * num_rows, thread_size_y, EmbeddingBlockSize);

        auto task = kernel_task(_m_kernel, grid, thread);
        auto task_future = task.bind_front(output, input, weight);

        return future_tensor(output, std::move(task_future));
    }
};


template <typename T, std::size_t BlockSize = 16> class rope {
private:
    basic_kernel _m_kernel;

public:
    rope(hardware_accelerator& gpu)
    : _m_kernel(gpu.load<T, BlockSize>("rope"))
    {}

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

        if (start_pos >= freqs_cos.size(0)) {
            throw std::invalid_argument(std::format(
                "kernel::rope: start position ({}) exceeds the size of frequencies tensor ({})",
                start_pos, freqs_cos.size(0)
            ));
        }

        if (auto head_dim = freqs_cos.size(1); dim_size != head_dim * 2) {
            throw std::invalid_argument(std::format(
                "kernel::rope: the last dimension of the input should be {}, but received {}",
                head_dim * 2, dim_size
            ));
        }

        auto input_view = flatten<2>(input);
        auto output_view = shared_empty_like<T>(input_view, _m_kernel.get_allocator());

        auto [grid, thread] = make_kernel_grid_2d(input, BlockSize);

        auto task = kernel_task(_m_kernel, grid, thread);
        auto task_future = task.bind_front(
            output_view, input_view, freqs_cos, freqs_sin, shared_tensor(scalar<int32_t>(bs)),
            shared_tensor(scalar<int32_t>(n_head)), shared_tensor(scalar<int32_t>(start_pos))
        );

        auto output = future_tensor(output_view, std::move(task_future));
        return output.view(input.shape());
    }
};


} // namespace kernel
} // namespace metalchat
