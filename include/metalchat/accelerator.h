#pragma once

#include <filesystem>
#include <iostream>
#include <unordered_map>

#include <metalchat/dtype.h>
#include <metalchat/kernel.h>
#include <metalchat/kernel_thread.h>
#include <metalchat/metal.h>


namespace metalchat {


static const std::string framework_identifier = "com.cmake.metalchat";


/// Hardware accelerator is an abstraction of the kernel execution pipeline.
///
/// Accelerator is responsible of whole Metal kernels lifecycle: creation of kernels from a
/// library, execution and scheduling of kernels, and allocation of tensors within a GPU memory.
///
/// The hardware accelerator cannot be copied, only moved.
class hardware_accelerator {
public:
    using allocator_type = polymorphic_hardware_memory_allocator<void>;

private:
    NS::SharedPtr<MTL::Device> _m_device;
    NS::SharedPtr<MTL::Library> _m_library;

    std::unordered_map<std::string, basic_kernel> _m_kernels;
    std::shared_ptr<kernel_thread_group> _m_this_thread_group;

    NS::SharedPtr<MTL::Device>
    _m_make_device()
    {
        return NS::TransferPtr(MTL::CreateSystemDefaultDevice());
    }

    std::shared_ptr<kernel_thread_group>
    _m_make_kernel_thread_group(std::size_t thread_capacity)
    {
        auto queue = NS::TransferPtr(_m_device->newCommandQueue());
        auto label = NS::TransferPtr(NS::String::string("metalchat", NS::UTF8StringEncoding));
        queue->setLabel(label.get());

        auto alloc_ptr = std::make_shared<hardware_memory_allocator<void>>(_m_device);
        auto alloc = allocator_type(alloc_ptr);

        return std::make_shared<kernel_thread_group>(queue, thread_capacity, alloc);
    }

    NS::SharedPtr<MTL::Library>
    _m_make_library(const NS::URL* library_path)
    {
        NS::SharedPtr<NS::Error> error = NS::TransferPtr(NS::Error::alloc());
        NS::Error* error_ptr = error.get();

        auto library = NS::TransferPtr(_m_device->newLibrary(library_path, &error_ptr));
        if (!library) {
            auto failure_reason = error_ptr->localizedDescription();
            throw std::runtime_error(failure_reason->utf8String());
        }
        return library;
    }

public:
    /// Create hardware accelerator from the kernel (shader) library.
    ///
    /// You can create a new hardware accelerator in the following way:
    /// ```cpp
    /// auto gpu = hardware_accelerator("metalchat.metallib");
    /// ```
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
    hardware_accelerator(std::size_t thread_capacity = 64);

    /// Get name of the hardware accelerator.
    std::string
    name() const;

    /// Return a shared pointer to the underlying Metal Device.
    inline NS::SharedPtr<MTL::Device>
    get_hardware_device()
    {
        return _m_device;
    }

    /// Return an allocator associated with the current thread.
    ///
    /// Use `set_allocator` method to set a new allocator to the currently running thread.
    allocator_type
    get_allocator() const;

    /// Set allocator to the current thread.
    ///
    /// Hardware accelerator uses a polymorphic allocator in order to provide an option to
    /// change the implementation during kernel queue scheduling. The allocator essentially
    /// is used to transfer all tensors allocated outside of the GPU memory to GPU memory.
    ///
    /// \note You can explore a variety of different allocators in
    /// \verbatim embed:rst:inline :doc:`allocator` \endverbatim.
    void
    set_allocator(allocator_type alloc);

    /// Set allocator to the current thread.
    template <basic_hardware_allocator_t<void> Allocator>
    void
    set_allocator(Allocator&& alloc)
    {
        _m_this_thread_group->set_allocator(std::move(alloc));
    }

    /// Load the kernel from the kernel library.
    ///
    /// Accelerator caches kernels, so kernel is loaded only once on the first call. A kernel
    /// returned from this method is attached to a `shared_kernel_thread`, and could be used
    /// to create a kernel task.
    basic_kernel
    load(const std::string& name);

    /// Load the kernel from kernel library.
    ///
    /// This is a convenience method that appends to the kernel name it's type: `name_type`, so
    /// users won't need to format kernel name manually.
    basic_kernel
    load(const std::string& name, const std::string& type);

    /// Load the kernel from kernel library.
    ///
    /// This is a convenience method that loads kernels with names in the following format:
    /// `{name}_{block_size}_{data_type}`.
    template <typename T, std::size_t BlockSize>
    basic_kernel
    load(const std::string_view& name)
    {
        auto kernel_name = std::format("{}_{}_{}", name, BlockSize, type_traits<T>::name());
        return load(kernel_name);
    }
};


} // namespace metalchat
