#pragma once

#include <metalchat/device.h>
#include <metalchat/dtype.h>
#include <metalchat/kernel.h>
#include <metalchat/kernel_task.h>
#include <metalchat/tensor_future.h>


namespace metalchat {


template <typename T, std::size_t BlockSize = 16> class multinomial {
private:
    inline static const std::string operation_name = "multinomial_" + std::to_string(BlockSize);

    kernel_base _m_kernel;

    std::random_device _m_random_device;
    std::mt19937 _m_generator;
    std::uniform_int_distribution<uint64_t> _m_seed;

public:
    multinomial(device& device)
    : _m_kernel(device.load(operation_name, type_traits<T>::name())),
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
        auto grid = dim3(thread_size * num_rows);

        auto init_state = _m_seed(_m_generator);
        auto init_seq = _m_seed(_m_generator);

        auto task = kernel_task(_m_kernel, grid, thread);
        auto fn = task.bind_back(
            input, shared_tensor(scalar(init_state)), shared_tensor(scalar(init_seq))
        );

        return empty_future<int32_t>({num_rows, sample_size}, std::move(fn));
    }
};


} // namespace metalchat
