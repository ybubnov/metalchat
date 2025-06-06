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

    void
    encode(const void* data, std::size_t size);

    void
    encode(metal::shared_buffer buffer);

    void
    encode_memory_barrier(metal::shared_buffer buffer);

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
    initialize(const std::string& name, const metal::shared_kernel kernel);

    template <typename T, immutable_scalar_t<T> Scalar>
    void
    encode(const Scalar& scalar)
    {
        encode(scalar.data_ptr(), sizeof(typename Scalar::value_type));
    }

    template <typename T, immutable_hardware_tensor_t<T> Tensor>
    void
    encode(const Tensor& tensor)
    {
        auto layout = tensor.layout();
        auto buffer = tensor.container().storage();

        encode(&layout, sizeof(layout));
        encode(buffer);

        // Mark all hardware-allocated tensors of the command as memory barriers,
        // so that kernel waits until previous kernels stop writing to that memory,
        // before running the current kernel.
        encode_memory_barrier(buffer);
    }

    template <typename T, immutable_tensor_t<T> Tensor>
    void
    encode(const Tensor& tensor)
    {
        auto alloc = rebind_hardware_allocator<T, allocator_type>(_m_allocator);
        auto container = alloc.allocate(tensor.data_ptr(), tensor.numel());

        auto layout = tensor.layout();
        encode(&layout, sizeof(layout));
        encode(container->storage());
    }

    void
    dispatch(dim3 grid, dim3 group);
};


template <typename T>
concept hardware_encodable_function
    = requires(std::remove_reference_t<T> t, hardware_function_encoder encoder) {
          { t.encode(encoder) } -> std::same_as<void>;
      };


struct kernel_queue;


class kernel_thread {
public:
    using callback_type = std::function<void()>;
    using allocator_type = polymorphic_hardware_memory_allocator<void>;

private:
    using promise_type = std::promise<void>;

    allocator_type _m_allocator;

    std::shared_ptr<kernel_queue> _m_queue;
    std::shared_ptr<promise_type> _m_promise;
    std::shared_future<void> _m_future;

    std::size_t _m_size;
    std::size_t _m_capacity;

    bool _m_committed;

    /// Register callback, which will be executed on a command completion.
    ///
    /// Completion handler is executed once all commands in a command buffer are
    /// completed. All registered handlers are executed.
    void
    on_completed(callback_type callback);

    hardware_function_encoder
    get_hardware_function_encoder();

public:
    kernel_thread(const kernel_thread&) noexcept = default;

    kernel_thread(const kernel_queue& queue, std::size_t capacity, allocator_type alloc);

    ~kernel_thread();

    std::size_t
    size() const;

    std::size_t
    capacity() const;

    /// Checks if the `kernel_thread` object identifies an active thread of execution.
    ///
    /// Specifically, returns true if the kernel thread is not committed and there are
    /// open slots available to encode new functions.
    bool
    joinable() const;

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
            on_completed(callback.value());
        }

        f.encode(get_hardware_function_encoder());

        _m_size++;

        if (_m_size == _m_capacity) {
            make_ready_at_thread_exit();
        }
        return _m_future;
    }

    void
    make_ready_at_thread_exit();
};


class recursive_kernel_thread {
public:
    using allocator_type = polymorphic_hardware_memory_allocator<void>;

private:
    allocator_type _m_allocator;

    std::shared_ptr<kernel_queue> _m_queue;
    std::shared_ptr<kernel_thread> _m_thread;
    std::size_t _m_thread_capacity;

public:
    recursive_kernel_thread(const recursive_kernel_thread&) noexcept = default;

    recursive_kernel_thread(metal::shared_device device, std::size_t thread_capacity);

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
    get_this_thread();
};


} // namespace metalchat
