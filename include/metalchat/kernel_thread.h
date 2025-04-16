#pragma once

#include <concepts>
#include <cstdlib>
#include <format>
#include <future>
#include <memory>

#include <metalchat/allocator.h>
#include <metalchat/metal.h>
#include <metalchat/tensor_concept.h>


namespace metalchat {


struct dim3 {
    const std::size_t x, y, z;

    constexpr dim3(std::size_t x_, std::size_t y_ = 1, std::size_t z_ = 1)
    : x(x_),
      y(y_),
      z(z_)
    {}

    friend std::ostream&
    operator<<(std::ostream& os, const dim3& d)
    {
        os << "<" << d.x << "," << d.y << "," << d.z << ">";
        return os;
    }

    std::size_t
    numel() const
    {
        return x * y * z;
    }
};


template <hardware_allocator_t<void> Allocator = hardware_memory_allocator<void>>
class hardware_function_encoder {
private:
    MTL::ComputeCommandEncoder* _m_encoder;
    Allocator _m_allocator;
    std::size_t _m_buffer;

public:
    hardware_function_encoder(MTL::ComputeCommandEncoder* encoder, Allocator alloc)
    : _m_encoder(encoder),
      _m_allocator(alloc),
      _m_buffer(0)
    {}

    void
    initialize(MTL::ComputePipelineState* pipeline)
    {
        _m_encoder->setComputePipelineState(pipeline);
    }

    template <typename T, immutable_scalar_t<T> Scalar>
    void
    encode(const Scalar& scalar)
    {
        const void* data_ptr = scalar.data_ptr();
        std::size_t data_size = sizeof(typename Scalar::value_type);
        _m_encoder->setBytes(data_ptr, data_size, _m_buffer++);
    }

    template <typename T, immutable_hardware_tensor_t<T> Tensor>
    void
    encode(const Tensor& tensor)
    {
        auto layout = tensor.layout();
        _m_encoder->setBytes(&layout, sizeof(layout), _m_buffer++);
        _m_encoder->setBuffer(tensor.container().storage().get(), 0, _m_buffer++);
    }

    template <typename T, immutable_tensor_t<T> Tensor>
    void
    encode(const Tensor& tensor)
    {
        auto alloc = rebind_hardware_allocator<T, Allocator>(_m_allocator);
        auto container = alloc.allocate(tensor.data_ptr(), tensor.numel());

        auto layout = tensor.layout();
        _m_encoder->setBytes(&layout, sizeof(layout), _m_buffer++);
        _m_encoder->setBuffer(container->storage().get(), 0, _m_buffer++);
    }

    void
    dispatch(dim3 grid, dim3 group)
    {
        MTL::Size threads_per_grid(grid.x, grid.y, grid.z);
        MTL::Size threads_per_group(group.x, group.y, group.z);
        _m_encoder->dispatchThreads(threads_per_grid, threads_per_group);
    }
};


template <typename T>
concept hardware_encodable_function
    = requires(std::remove_reference_t<T> t, hardware_function_encoder<> encoder) {
          { t.encode(encoder) } -> std::same_as<void>;
      };


// template <hardware_allocator_t<void> Allocator = hardware_memory_allocator<void>>
class kernel_thread {
private:
    using promise_type = std::promise<void>;

    NS::SharedPtr<MTL::CommandBuffer> _m_commands;
    hardware_memory_allocator<void> _m_allocator;

    std::shared_ptr<promise_type> _m_promise;
    std::shared_future<void> _m_future;

    std::size_t _m_size;
    std::size_t _m_capacity;

    bool _m_committed;
    int _m_id = std::rand();

public:
    using callback_type = std::function<void()>;

    kernel_thread(const kernel_thread&) noexcept = default;

    kernel_thread(
        NS::SharedPtr<MTL::CommandQueue> queue,
        std::size_t capacity,
        hardware_memory_allocator<void> alloc
    )
    : _m_commands(NS::TransferPtr(queue->commandBuffer())),
      _m_allocator(alloc),
      _m_promise(std::make_shared<promise_type>()),
      _m_future(_m_promise->get_future()),
      _m_size(0),
      _m_capacity(capacity),
      _m_committed(false)
    {
        // After the completion of the kernel execution, release the promise and all blocks
        // waiting for the completion of this kernel.
        _m_commands->addCompletedHandler([promise = _m_promise](const MTL::CommandBuffer*) {
            // TODO: handle the error.
            promise->set_value();
        });
    }

    ~kernel_thread()
    {
        //  If thread was completed, the code below does absolutely nothing, otherwise,
        //  on object deletion all commands are committed to the device.
        make_ready_at_thread_exit();
    }

    std::size_t
    size() const
    {
        return _m_size;
    }

    std::size_t
    capacity() const
    {
        return _m_capacity;
    }

    bool
    joinable() const
    {
        return (!_m_committed) && (_m_size < _m_capacity);
    }

    template <hardware_encodable_function F>
    std::shared_future<void>
    push(F& f, std::optional<callback_type> callback = std::nullopt)
    {
        if (!joinable()) {
            throw std::runtime_error(
                std::format("thread: thread is either committed or reached its capacity")
            );
        }
        if (callback.has_value()) {
            _m_commands->addCompletedHandler([callback = callback.value(
                                              )](const MTL::CommandBuffer*) { callback(); });
        }

        auto encoder = _m_commands->computeCommandEncoder(MTL::DispatchTypeSerial);
        f.encode(hardware_function_encoder(encoder, _m_allocator));
        encoder->endEncoding();

        _m_size++;

        if (_m_size == _m_capacity) {
            make_ready_at_thread_exit();
        }
        return _m_future;
    }

    void
    make_ready_at_thread_exit()
    {
        if (!_m_committed) {
            auto label = std::format("metalchat commands (size={})", _m_size);
            auto commands_label
                = NS::TransferPtr(NS::String::string(label.c_str(), NS::UTF8StringEncoding));
            _m_commands->setLabel(commands_label.get());

            _m_commands->commit();
            _m_committed = true;
        }
    }
};


class shared_kernel_thread {
private:
    using allocator_type = hardware_memory_allocator<void>;

    NS::SharedPtr<MTL::CommandQueue> _m_queue;
    std::shared_ptr<kernel_thread> _m_this_thread;
    allocator_type _m_allocator;
    std::size_t _m_thread_capacity;

public:
    shared_kernel_thread(const shared_kernel_thread&) noexcept = default;

    shared_kernel_thread(
        NS::SharedPtr<MTL::CommandQueue> queue,
        std::size_t thread_capacity,
        hardware_memory_allocator<void> alloc
    )
    : _m_queue(queue),
      _m_this_thread(std::make_shared<kernel_thread>(queue, thread_capacity, alloc)),
      _m_allocator(alloc),
      _m_thread_capacity(thread_capacity)
    {}

    allocator_type
    allocator()
    {
        return _m_allocator;
    }

    std::shared_ptr<kernel_thread>
    get_this_thread()
    {
        if (!_m_this_thread->joinable()) {
            auto thread
                = std::make_shared<kernel_thread>(_m_queue, _m_thread_capacity, _m_allocator);
            _m_this_thread.swap(thread);
        }
        return _m_this_thread;
    }
};


} // namespace metalchat
