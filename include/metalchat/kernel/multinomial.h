#pragma once

#include <metalchat/accelerator.h>
#include <metalchat/dtype.h>
#include <metalchat/kernel.h>
#include <metalchat/tensor/future.h>


namespace metalchat {
namespace kernel {


template <typename T, std::size_t BlockSize = 32> class multinomial {
private:
    basic_kernel _m_kernel;

    std::random_device _m_random_device;
    std::mt19937 _m_generator;
    std::uniform_int_distribution<uint64_t> _m_seed;

public:
    multinomial(hardware_accelerator& gpu)
    : _m_kernel(gpu.load<T, BlockSize>("multinomial")),
      _m_random_device(),
      _m_generator(_m_random_device()),
      _m_seed(0, std::numeric_limits<uint64_t>::max())
    {}

    template <immutable_tensor2_t<T> Input>
    auto
    operator()(Input input, std::size_t sample_size)
    {
        auto num_rows = input.size(0);
        auto dim_size = sample_size;

        auto thread_size = ceil_div(dim_size, BlockSize);
        auto thread = dim3(thread_size);
        auto grid = dim3(thread_size * num_rows, BlockSize);

        auto init_state = _m_seed(_m_generator);
        auto init_seq = _m_seed(_m_generator);

        auto output = shared_empty<int32_t>({num_rows, sample_size}, _m_kernel.get_allocator());

        auto task = kernel_task(_m_kernel, grid, thread);
        auto task_future = task.bind_front(output, input, scalar(init_state), scalar(init_seq));

        return future_tensor(output, std::move(task_future));
    }
};


} // namespace kernel
} // namespace metalchat
