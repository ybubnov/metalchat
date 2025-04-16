#pragma once

#include <metalchat/kernel.h>
#include <metalchat/tensor.h>
#include <metalchat/tensor_shared.h>


namespace metalchat {


template <typename T, std::size_t N, ContiguousContainer Container>
NS::SharedPtr<MTL::Buffer>
make_buffer(MTL::Device* device, const shared_tensor<T, N, Container>& t)
{
    auto size = t.numel() * sizeof(T);
    return NS::TransferPtr(device->newBuffer(t.data_ptr(), size, MTL::ResourceStorageModeShared));
}

template <typename T, std::size_t N>
NS::SharedPtr<MTL::Buffer>
make_buffer(MTL::Device* device, const shared_tensor<T, N, device_ref<T>>& t)
{
    return t.container().storage();
}

template <typename T>
NS::SharedPtr<MTL::Buffer>
make_buffer(MTL::Device* device, const shared_tensor<T, 0, value_ref<T>>& t)
{
    return NS::SharedPtr<MTL::Buffer>();
}


inline std::size_t
ceil_div(std::size_t a, std::size_t b)
{
    return (a + b - 1) / b;
}


template <is_tensor Tensor>
std::tuple<dim3, dim3>
make_kernel_grid_1d(const Tensor& t, std::size_t block_size)
{
    auto data_size = t.numel();
    auto dim_size = t.sizes().back();
    auto num_rows = data_size / dim_size;

    auto thread_size = ceil_div(dim_size, block_size);
    auto thread = dim3(thread_size);
    auto grid = dim3(thread_size * num_rows);

    return std::forward_as_tuple(grid, thread);
}


template <is_tensor... Args> class kernel_task {
private:
    kernel_base _m_kernel;

    std::tuple<Args...> _m_args;

    /// Buffers used to keep encodings of the tensors, they are deleted altogether
    /// with a task object deletion.
    std::vector<NS::SharedPtr<MTL::Buffer>> _m_buffers;

    dim3 _m_grid;
    dim3 _m_thread;

public:
    kernel_task(kernel_task&& task) noexcept = default;

    kernel_task(kernel_base kernel, dim3 grid, dim3 thread, std::tuple<Args...>&& args)
    : _m_kernel(kernel),
      _m_args(args),
      _m_buffers(),
      _m_grid(grid),
      _m_thread(thread)
    {
        auto max_threads = kernel.max_threads_per_threadgroup();
        if (thread.numel() > max_threads) {
            throw std::invalid_argument(std::format(
                "kernel: <{}, {}, {}> exceeds maximum number of threads per group {}", thread.x,
                thread.y, thread.z, max_threads
            ));
        }

        if (grid.numel() < thread.numel()) {
            throw std::invalid_argument(std::format(
                "kernel: there are less threads in grid <{}, {}, {}> than in group <{}, {}, {}>",
                grid.x, grid.y, grid.z, thread.x, thread.y, thread.z
            ));
        }
    }

    kernel_task(kernel_base kernel, dim3 grid, dim3 thread, Args... args)
    : kernel_task(kernel, grid, thread, std::make_tuple(args...))
    {}

    MTL::Device*
    device()
    {
        return _m_kernel.device();
    }

    template <std::size_t... Indices>
    void
    operator()(std::shared_ptr<std::promise<void>> promise, std::index_sequence<Indices...>)
    {
        // Clear the buffers used from the previous execution of the kernel task.
        //
        // Usually there is no reason of executing the kernel task twice, since it's tightly
        // coupled with a tensor future, but kernel task could still be used independently
        // from the tensor future, prepare a clean run of the kernel.
        _m_buffers.clear();

        auto device_ptr = device();
        auto command_buf = _m_kernel.make_buffer();
        auto command_encoder = command_buf->computeCommandEncoder(MTL::DispatchTypeConcurrent);

        command_encoder->setComputePipelineState(_m_kernel.pipeline());

        // Move the tensors to a command encoder one-by-one. Prepend each non-scalar tensor
        // (regular multi-dimensional tensors) with a layout: offsets, strides, sizes. This
        // operation is automatic therefore kernels should be prepared to accept layout
        // followed by a pointer to the tensor's data.
        std::size_t i = 0;

        ([&] {
            auto& arg = std::get<Indices>(_m_args);
            if (auto buf = make_buffer(device_ptr, arg); buf) {
                auto layout = arg.layout();
                command_encoder->setBytes(&layout, sizeof(layout), i++);

                _m_buffers.push_back(buf);
                command_encoder->setBuffer(buf.get(), 0, i);
            } else {
                const void* data_ptr = arg.data_ptr();
                std::size_t data_size = sizeof(typename Args::value_type);
                command_encoder->setBytes(data_ptr, data_size, i);
            }
            i++;
        }(), ...);

        MTL::Size threads_per_grid(_m_grid.x, _m_grid.y, _m_grid.z);
        MTL::Size threads_per_group(_m_thread.x, _m_thread.y, _m_thread.z);
        command_encoder->dispatchThreads(threads_per_grid, threads_per_group);
        command_encoder->endEncoding();

        // After the completion of the kernel execution, release the promise and all blocks
        // waiting for the completion of this kernel.
        command_buf->addCompletedHandler([promise = promise](const MTL::CommandBuffer* buf) {
            promise->set_value();
        });
        command_buf->commit();
    }

    void
    operator()(std::shared_ptr<std::promise<void>> promise)
    {
        operator()(promise, std::index_sequence_for<Args...>{});
    }

    template <is_tensor... FrontArgs>
    kernel_task<FrontArgs..., Args...>
    bind_front(FrontArgs... front_args)
    {
        return kernel_task<FrontArgs..., Args...>(
            _m_kernel, _m_grid, _m_thread, std::tuple_cat(std::make_tuple(front_args...), _m_args)
        );
    }

    template <is_tensor... BackArgs>
    kernel_task<Args..., BackArgs...>
    bind_back(BackArgs... back_args)
    {
        return kernel_task<Args..., BackArgs...>(
            _m_kernel, _m_grid, _m_thread, std::tuple_cat(_m_args, std::make_tuple(back_args...))
        );
    }
};


} // namespace metalchat
