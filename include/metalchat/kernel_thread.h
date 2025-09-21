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


struct kernel_queue;
using kernel_callback_type = std::function<void()>;


/// The type that is used to specify dimension of the GPU compute grid (thread group). When
/// defining variable of type \ref dim3, any values left unspecified is initialized to 1.
struct dim3 {
    /// X value of a 3-dimensional vector.
    const std::size_t x;
    /// Y value of a 3-dimensional vector.
    const std::size_t y;
    /// Z value of a 3-dimensional vector.
    const std::size_t z;

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
    using allocator_type = polymorphic_hardware_allocator<void>;

private:
    allocator_type _M_allocator;
    std::shared_ptr<kernel_queue> _M_queue;
    std::size_t _M_buffer;
    std::string _M_name;

    /// Encode the specified raw block of memory as bytes.
    ///
    /// This method effectively is used to encoder kernel parameters that are passed by value
    /// as opposed to buffers, which are passed by a pointer.
    ///
    /// Consider the following kernel example, where the first parameter is passed by a value
    /// and the second by a pointer to the device data:
    /// ```c++
    /// kernel void greater_than_value(
    ///     device float* input, constant float& value, device float* output
    /// );
    /// ```
    ///
    /// To encoder parameters of this kernel, use `encode(const void*, std::size_t)` in order
    /// to pass the `value` parameter and `encode(metal::shared_buffer, std::size_t offset)` to
    /// encode `input` and `output` parameters.
    ///
    /// \param data a pointer to the raw data to copy.
    /// \param size a size of the data to copy in bytes.
    void
    encode(const void* data, std::size_t size);

    void
    encode(metal::shared_buffer buffer, std::size_t offset);

    void
    encode_memory_barrier(metal::shared_buffer buffer);

    void
    on_completed(kernel_callback_type callback);

public:
    hardware_function_encoder(std::shared_ptr<kernel_queue> queue, allocator_type alloc);

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
        auto container = tensor.container();

        encode(&layout, sizeof(layout));
        encode(container.storage(), container.storage_offset());

        // Mark all hardware-allocated tensors of the command as memory barriers,
        // so that kernel waits until previous kernels stop writing to that memory,
        // before running the current kernel.
        encode_memory_barrier(container.storage());
    }

    template <typename T, immutable_tensor_t<T> Tensor>
    void
    encode(const Tensor& tensor)
    {
        auto alloc = rebind_allocator<T, allocator_type>(_M_allocator);
        auto container = alloc.allocate(tensor.data_ptr(), tensor.numel());

        auto layout = tensor.layout();
        encode(&layout, sizeof(layout));
        encode(container->storage(), container->storage_offset());
    }

    template <typename T, immutable_filebuf_tensor_t<T> Tensor>
    void
    encode(const Tensor& tensor)
    {
        using container_type = std::remove_cvref_t<typename Tensor::container_type>;
        container_type container = tensor.container();

        on_completed([container = container]() { container.park(); });

        auto alloc = rebind_allocator<T, allocator_type>(_M_allocator);
        auto container_ptr = alloc.allocate(tensor.data_ptr(), tensor.numel());

        auto layout = tensor.layout();
        encode(&layout, sizeof(layout));
        encode(container_ptr->storage(), container_ptr->storage_offset());
    }

    void
    dispatch(dim3 grid, dim3 group);
};


template <typename T>
concept hardware_encodable_function
    = requires(std::remove_reference_t<T> t, hardware_function_encoder encoder) {
          { t.encode(encoder) } -> std::same_as<void>;
      };


class kernel_thread {
public:
    using allocator_type = polymorphic_hardware_allocator<void>;

private:
    using promise_type = std::promise<void>;

    allocator_type _M_allocator;

    std::shared_ptr<kernel_queue> _M_queue;
    std::shared_ptr<promise_type> _M_promise;
    std::shared_future<void> _M_future;

    std::size_t _M_size;
    std::size_t _M_capacity;

    bool _M_committed;

    /// Register callback, which will be executed on a command completion.
    ///
    /// Completion handler is executed once all commands in a command buffer are
    /// completed. All registered handlers are executed.
    void
    on_completed(kernel_callback_type callback);

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
    push(F& f, std::optional<kernel_callback_type> callback = std::nullopt)
    {
        if (!joinable()) {
            throw std::runtime_error(
                std::format("thread: thread is either committed or reached its capacity")
            );
        }
        if (callback.has_value()) {
            on_completed(callback.value());
        }

        f.encode(hardware_function_encoder(_M_queue, _M_allocator));

        _M_size++;

        if (_M_size == _M_capacity) {
            make_ready_at_thread_exit();
        }
        return _M_future;
    }

    void
    make_ready_at_thread_exit();
};


class recursive_kernel_thread {
public:
    using allocator_type = polymorphic_hardware_allocator<void>;

private:
    allocator_type _M_allocator;

    std::shared_ptr<kernel_queue> _M_queue;
    std::shared_ptr<kernel_thread> _M_thread;
    std::size_t _M_thread_capacity;

public:
    recursive_kernel_thread(const recursive_kernel_thread&) noexcept = default;

    recursive_kernel_thread(metal::shared_device device, std::size_t thread_capacity);

    allocator_type
    get_allocator() const
    {
        return _M_allocator;
    }

    void
    set_allocator(std::shared_ptr<allocator_type::outer_allocator_type> alloc)
    {
        _M_allocator = allocator_type(alloc);
    }

    void
    set_allocator(allocator_type alloc)
    {
        _M_allocator = alloc;
    }

    template <hardware_allocator_t<void> Allocator>
    void
    set_allocator(Allocator&& alloc)
    {
        using allocator_type = hardware_allocator_wrapper<Allocator>;
        set_allocator(std::make_shared<allocator_type>(std::move(alloc)));
    }

    std::shared_ptr<kernel_thread>
    get_this_thread();
};


} // namespace metalchat
