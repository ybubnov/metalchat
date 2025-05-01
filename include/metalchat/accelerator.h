#pragma once

#include <filesystem>
#include <iostream>
#include <unordered_map>

#include <metalchat/kernel.h>
#include <metalchat/kernel_thread.h>
#include <metalchat/metal.h>


namespace metalchat {


/// Hardware accelerator is an abstraction of the kernel execution pipeline.
///
/// Accelerator is responsible of whole Metal kernels lifecycle: creation of kernels from a
/// library, execution and scheduling of kernels, and allocation of tensors within a GPU memory.
class hardware_accelerator {
public:
    using allocator_type = polymorphic_hardware_memory_allocator<void>;

private:
    NS::SharedPtr<MTL::Device> _m_device;
    NS::SharedPtr<MTL::Library> _m_library;

    std::unordered_map<std::string, basic_kernel> _m_kernels;
    shared_kernel_thread _m_this_thread;

    NS::SharedPtr<MTL::Device>
    _m_make_device()
    {
        return NS::TransferPtr(MTL::CreateSystemDefaultDevice());
    }

    shared_kernel_thread
    _m_make_kernel_thread(std::size_t thread_capacity)
    {
        auto queue = NS::TransferPtr(_m_device->newCommandQueue());
        auto label = NS::TransferPtr(NS::String::string("metalchat", NS::UTF8StringEncoding));
        queue->setLabel(label.get());

        auto alloc_ptr = std::make_shared<hardware_memory_allocator<void>>(_m_device);

        return shared_kernel_thread(queue, thread_capacity, allocator_type(alloc_ptr));
    }

public:
    hardware_accelerator(hardware_accelerator&&) noexcept = default;
    hardware_accelerator(const hardware_accelerator&) = delete;

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

    /// Get name of the hardware accelerator.
    std::string
    name() const;

    inline NS::SharedPtr<MTL::Device>
    get_hardware_device()
    {
        return _m_device;
    }

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
        _m_this_thread.set_allocator(std::move(alloc));
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
};


} // namespace metalchat
