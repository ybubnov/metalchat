#pragma once

#include <format>
#include <future>
#include <tuple>

#include <metalchat/container.h>
#include <metalchat/kernel_thread.h>
#include <metalchat/tensor/concept.h>


namespace metalchat {


class basic_kernel {
private:
    std::string _m_name;
    metal::shared_kernel _m_kernel;

    std::shared_ptr<kernel_thread_group> _m_kernel_thread_group;

public:
    using allocator_type = polymorphic_hardware_memory_allocator<void>;

    basic_kernel(metal::shared_kernel kernel, std::shared_ptr<kernel_thread_group> group);

    std::string
    name() const;

    const metal::shared_kernel
    get_metal_kernel() const;

    allocator_type
    get_allocator()
    {
        return _m_kernel_thread_group->get_allocator();
    }

    std::size_t
    max_threads_per_threadgroup();

    std::shared_ptr<kernel_thread>
    get_this_thread()
    {
        return _m_kernel_thread_group->get_this_thread();
    }
};


} // namespace metalchat
