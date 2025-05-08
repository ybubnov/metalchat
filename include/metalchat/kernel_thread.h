#pragma once

#include <concepts>
#include <cstdlib>
#include <format>
#include <future>
#include <memory>

#include <metalchat/allocator.h>
#include <metalchat/metal.h>
#include <metalchat/tensor/concept.h>


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


class hardware_function_encoder {
public:
    using allocator_type = polymorphic_hardware_memory_allocator<void>;

private:
    NS::SharedPtr<MTL::ComputeCommandEncoder> _m_encoder;
    allocator_type _m_allocator;
    std::size_t _m_buffer;
    std::string _m_name;

public:
    hardware_function_encoder(
        NS::SharedPtr<MTL::ComputeCommandEncoder> encoder, allocator_type alloc
    )
    : _m_encoder(encoder),
      _m_allocator(alloc),
      _m_buffer(0),
      _m_name()
    {}

    void
    initialize(std::string name, MTL::ComputePipelineState* pipeline)
    {
        _m_name = name;
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

        // Mark all hardware-allocated tensors of the command as memory barriers,
        // so that kernel waits until previous kernels stop writing to that memory,
        // before running the current kernel.
        const MTL::Resource* resources[1] = {tensor.container().storage().get()};
        _m_encoder->memoryBarrier(resources, 1);
    }

    template <typename T, immutable_tensor_t<T> Tensor>
    void
    encode(const Tensor& tensor)
    {
        auto alloc = rebind_hardware_allocator<T, allocator_type>(_m_allocator);
        auto container = alloc.allocate(tensor.data_ptr(), tensor.numel());

        auto layout = tensor.layout();
        _m_encoder->setBytes(&layout, sizeof(layout), _m_buffer++);
        _m_encoder->setBuffer(container->storage().get(), 0, _m_buffer++);
    }

    void
    dispatch(dim3 grid, dim3 group)
    {
        std::stringstream command_name_stream;
        command_name_stream << _m_name << "<" << grid << "," << group << ">" << std::endl;

        auto command_name = command_name_stream.str();
        auto command_name_ptr
            = NS::TransferPtr(NS::String::string(command_name.c_str(), NS::UTF8StringEncoding));
        _m_encoder->setLabel(command_name_ptr.get());

        MTL::Size threads_per_grid(grid.x, grid.y, grid.z);
        MTL::Size threads_per_group(group.x, group.y, group.z);
        _m_encoder->dispatchThreads(threads_per_grid, threads_per_group);
    }
};


template <typename T>
concept hardware_encodable_function
    = requires(std::remove_reference_t<T> t, hardware_function_encoder encoder) {
          { t.encode(encoder) } -> std::same_as<void>;
      };


class kernel_thread {
public:
    using callback_type = std::function<void()>;
    using allocator_type = polymorphic_hardware_memory_allocator<void>;

private:
    using promise_type = std::promise<void>;

    NS::SharedPtr<MTL::CommandBuffer> _m_commands;
    NS::SharedPtr<MTL::ComputeCommandEncoder> _m_encoder;
    NS::SharedPtr<MTL::Event> _m_event;

    allocator_type _m_allocator;
    std::shared_ptr<promise_type> _m_promise;
    std::shared_future<void> _m_future;

    std::size_t _m_id;
    std::size_t _m_size;
    std::size_t _m_capacity;

    bool _m_committed;

public:
    kernel_thread(const kernel_thread&) noexcept = default;

    kernel_thread(
        NS::SharedPtr<MTL::CommandQueue> queue,
        NS::SharedPtr<MTL::Event> event,
        std::size_t id,
        std::size_t capacity,
        allocator_type alloc
    )
    : _m_commands(NS::TransferPtr(queue->commandBuffer())),
      _m_event(event),
      _m_allocator(alloc),
      _m_promise(std::make_shared<promise_type>()),
      _m_future(_m_promise->get_future()),
      _m_id(id),
      _m_size(0),
      _m_capacity(capacity),
      _m_committed(false)
    {
        _m_commands->enqueue();

        if (_m_id > 0) {
            _m_commands->encodeWait(_m_event.get(), _m_id);
        }

        _m_encoder
            = NS::TransferPtr(_m_commands->computeCommandEncoder(MTL::DispatchTypeConcurrent));

        // After the completion of the kernel execution, release the promise and all blocks
        // waiting for the completion of this kernel.
        _m_commands->addCompletedHandler([promise = _m_promise](const MTL::CommandBuffer* buffer) {
            if (buffer->error() != nullptr) {
                auto failure_reason = buffer->error()->localizedDescription();
                auto exception_ptr
                    = std::make_exception_ptr(std::runtime_error(failure_reason->utf8String()));
                promise->set_exception(exception_ptr);
            } else {
                promise->set_value();
            }
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
                                              )](const MTL::CommandBuffer* buf) { callback(); });
        }

        f.encode(hardware_function_encoder(_m_encoder, _m_allocator));

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
            auto label_ptr = NS::String::string(label.c_str(), NS::UTF8StringEncoding);
            auto commands_label = NS::TransferPtr(label_ptr);

            _m_encoder->endEncoding();

            _m_commands->setLabel(commands_label.get());
            _m_commands->encodeSignalEvent(_m_event.get(), _m_id + 1);

            _m_commands->commit();
            _m_committed = true;
        }
    }
};


class kernel_thread_group {
public:
    using allocator_type = polymorphic_hardware_memory_allocator<void>;

private:
    NS::SharedPtr<MTL::CommandQueue> _m_queue;
    NS::SharedPtr<MTL::Event> _m_event;

    std::size_t _m_thread_id;
    std::size_t _m_thread_capacity;

    std::shared_ptr<kernel_thread> _m_this_thread;
    allocator_type _m_allocator;

public:
    kernel_thread_group(const kernel_thread_group&) noexcept = default;

    kernel_thread_group(
        NS::SharedPtr<MTL::CommandQueue> queue, std::size_t thread_capacity, allocator_type alloc
    )
    : _m_queue(queue),
      _m_event(NS::TransferPtr(queue->device()->newEvent())),
      _m_thread_id(0),
      _m_thread_capacity(thread_capacity),
      _m_this_thread(std::make_shared<kernel_thread>(
          _m_queue, _m_event, _m_thread_id, _m_thread_capacity, alloc
      )),
      _m_allocator(alloc)
    {}

    allocator_type
    get_allocator() const
    {
        return _m_allocator;
    }

    void
    set_allocator(std::shared_ptr<allocator_type::outer_allocator_type> alloc)
    {
        _m_allocator = allocator_type(alloc);
    }

    void
    set_allocator(allocator_type alloc)
    {
        _m_allocator = alloc;
    }

    template <basic_hardware_allocator_t<void> Allocator>
    void
    set_allocator(Allocator&& alloc)
    {
        set_allocator(std::make_shared<Allocator>(std::move(alloc)));
    }

    std::shared_ptr<kernel_thread>
    get_this_thread()
    {
        if (!_m_this_thread->joinable()) {
            auto thread = std::make_shared<kernel_thread>(
                _m_queue, _m_event, ++_m_thread_id, _m_thread_capacity, _m_allocator
            );

            _m_this_thread.swap(thread);
        }
        return _m_this_thread;
    }
};


} // namespace metalchat
