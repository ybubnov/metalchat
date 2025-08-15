#pragma once

#include <filesystem>
#include <unordered_map>

#include <metalchat/dtype.h>
#include <metalchat/kernel_thread.h>
#include <metalchat/metal.h>


namespace metalchat {


static const std::string framework_identifier = "com.cmake.metalchat";


struct _StringHash {
    std::size_t
    operator()(const std::string& s) const noexcept;
};


class basic_kernel;


/// Hardware accelerator is an abstraction of the kernel execution pipeline.
///
/// Accelerator is responsible of whole Metal kernels lifecycle: creation of kernels from a
/// library, execution and scheduling of kernels, and allocation of tensors within a GPU memory.
///
/// The hardware accelerator can be copied. Modification of the allocator are distributed to
/// all copies of the hardware accelerator.
class hardware_accelerator {
public:
    /// A type of the hardware memory allocator used to either allocate or transfer memory
    /// of tensors within a running kernel thread.
    using allocator_type = polymorphic_hardware_memory_allocator<void>;

private:
    metal::shared_device _M_device;
    metal::shared_library _M_library;

    std::unordered_map<std::string, basic_kernel> _M_kernels;
    std::shared_ptr<recursive_kernel_thread> _M_thread;

public:
    hardware_accelerator(const hardware_accelerator& accelerator) = default;

    /// Create hardware accelerator from the kernel (shader) library.
    ///
    /// You can create a new hardware accelerator in the following way:
    /// ```cpp
    /// auto gpu = hardware_accelerator("metalchat.metallib");
    /// ```
    ///
    /// \param path Specifies a location of the compiled Metal shaders library.
    ///
    /// \param thread_capacity Specifies the size of the command buffer. Commands are executed
    ///     in batches of `thread_capacity` size. Kernel won't be scheduled until the buffer is
    ///     filled with the configured number of kernels, or when the execution is explicitly
    ///     triggered (usually by calling `future_tensor::get` method).
    hardware_accelerator(const std::filesystem::path& path, std::size_t thread_capacity = 64);

    /// Create hardware accelerator from within a bundle.
    ///
    /// When the library is distributed as a bundle, then it's possible to load the shader
    /// library from the bundle. This constructor performs lookup of the distribution bundle
    /// and loads shader library named `metalchat.metallib`.
    ///
    explicit hardware_accelerator(std::size_t thread_capacity = 64);

    /// Get name of the hardware accelerator.
    std::string
    name() const;

    std::size_t
    max_buffer_size() const;

    std::shared_ptr<kernel_thread>
    get_this_thread();

    /// Return a shared pointer to the underlying Metal Device.
    metal::shared_device
    get_metal_device();

    /// Return an allocator associated with the current thread.
    ///
    /// Use `hardware_accelerator::set_allocator` method to set a new allocator to the currently
    /// running thread.
    allocator_type
    get_allocator() const;

    /// Set allocator to the current thread.
    ///
    /// Hardware accelerator uses a polymorphic allocator in order to provide an option to
    /// change the implementation during kernel queue scheduling. The allocator essentially
    /// is used to transfer all tensors allocated outside of the GPU memory to GPU memory.
    ///
    /// \note You can explore a variety of different allocators in
    /// \verbatim embed:rst:inline :doc:`../allocator` \endverbatim.
    void
    set_allocator(allocator_type alloc);

    /// Set allocator to the current thread.
    template <basic_hardware_allocator_t<void> Allocator>
    void
    set_allocator(Allocator&& alloc)
    {
        _M_thread->set_allocator(std::move(alloc));
    }

    /// Load the kernel from the kernel library.
    ///
    /// Accelerator caches kernels, so kernel is loaded only once on the first call. A kernel
    /// returned from this method is attached to a `metalchat::recursive_kernel_thread`, and
    /// could be used to create a kernel task.
    ///
    /// Example:
    /// ```cpp
    /// using namespace metalchat;
    ///
    /// auto gpu = hardware_accelerator();
    /// auto kernel = gpu.load<float, 16>("hadamard");
    ///
    /// auto output = future_tensor(empty<float>({32}, gpu));
    /// auto input1 = future_tensor(rand<float>({32}, gpu));
    /// auto input2 = future_tensor(rand<float>({32}, gpu));
    ///
    /// // Schedule a kernel task with 2 thread groups, each of 16 threads size.
    /// auto task = kernel_task(kernel, dim3(32), dim3(16));
    ///
    /// // This kernel expects output tensor as the first argument.
    /// auto packaged_task = task.bind_front(output, input1, input2);
    /// auto result = future_tensor(output, std::move(packaged_task));
    ///
    /// // Block the current thread, until the result is ready.
    /// result.get();
    /// ```
    const basic_kernel&
    load(const std::string& name);

    /// Load the kernel from kernel library.
    ///
    /// This is a convenience method that appends to the kernel name it's type: `name_type`, so
    /// users won't need to format kernel name manually.
    const basic_kernel&
    load(const std::string& name, const std::string& type);

    /// Load the kernel from kernel library.
    ///
    /// This is a convenience method that loads kernels with names in the following format:
    /// `{name}_{block_size}_{data_type}`.
    template <typename T, std::size_t BlockSize>
    const basic_kernel&
    load(const std::string_view& name)
    {
        auto kernel_name = std::format("{}_{}_{}", name, BlockSize, type_traits<T>::name());
        return load(kernel_name);
    }
};


} // namespace metalchat
